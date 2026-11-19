"""A read-only model listing must not delay durable admission on another route."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session_authorities import active_authority
from hermes_cli import model_switch_providers
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.test_messaging_session_picks import _compressed_chat


@pytest.mark.asyncio
@pytest.mark.parametrize("persist_by_default", [False, True])
async def test_picker_delivery_does_not_delay_an_unrelated_native_admission(tmp_path, monkeypatch, persist_by_default):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "5151")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: picker-fixture\n  provider: openrouter\n"
        f"  persist_switch_by_default: {str(persist_by_default).lower()}\n",
        encoding="utf-8",
    )
    runner, source, _route, viewer, original, _ = await _compressed_chat(tmp_path)
    authority = active_authority(runner)
    entered, release = asyncio.Event(), asyncio.Event()

    class PickerAdapter:
        async def send_model_picker(self, **kwargs):
            entered.set()
            await release.wait()
            return SimpleNamespace(success=True)

    telegram = TelegramAdapter(PlatformConfig(enabled=True, token="fixture-token"))
    runner.adapters = {source.platform: telegram}
    picker = PickerAdapter()
    lookup = runner._delivery_adapter_for
    monkeypatch.setattr(runner, "_delivery_adapter_for",
                        lambda origin: picker if origin.chat_id == source.chat_id else lookup(origin))
    monkeypatch.setattr(model_switch_providers, "list_picker_providers", lambda **kwargs: [
        {"provider": "openrouter", "models": [{"id": "picker-fixture"}]}])
    switch = AsyncMock(side_effect=AssertionError("Opening a picker cannot publish a model switch"))
    monkeypatch.setattr(runner, "_perform_model_switch", switch)
    monkeypatch.setattr(authority, "_schedule_admitted", lambda *_args: None)

    listing = asyncio.create_task(runner._handle_model_command(MessageEvent(text="/model", source=source)))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        other = telegram.build_source(chat_id="5151", user_id="5151", chat_type="dm")
        receipt = await asyncio.wait_for(authority.admit_native(
            MessageEvent(text="admit this unrelated turn", source=other, message_id="other-admission")
        ), 5)
        assert receipt.status == "queued"
        assert receipt.ref.session_id != original
        assert not listing.done()
        switch.assert_not_awaited()
    finally:
        release.set()
        await asyncio.gather(listing, return_exceptions=True)
        await viewer.close()
        runner.session_store.close_all_db_handles()
