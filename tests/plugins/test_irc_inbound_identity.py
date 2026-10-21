"""Distinct IRC lines must remain distinct inputs in the gateway's durable FIFO."""

import asyncio
from types import SimpleNamespace

import pytest

from gateway import platform_registry
from gateway.config import GatewayConfig, PlatformConfig
from gateway.platform_registry import PlatformEntry, PlatformRegistry
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_ingress_context import native_callback
from hermes_constants import get_hermes_home
from hermes_state_runtime import list_session_admissions
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("texts", [("first", "second"), ("same", "same")])
async def test_same_millisecond_irc_lines_get_separate_admissions(monkeypatch, texts):
    home = get_hermes_home()
    monkeypatch.setenv("IRC_ALLOWED_USERS", "sender")
    module = load_plugin_adapter("irc")
    registry = PlatformRegistry()
    monkeypatch.setattr(platform_registry, "platform_registry", registry)
    module.register(
        SimpleNamespace(
            register_platform=lambda **kwargs: registry.register(
                PlatformEntry(**kwargs)
            )
        )
    )
    # Freeze only this adapter's clock, not asyncio's scheduling clock.
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: 1_700_000_000.0))
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
    adapter = module.IRCAdapter(
        PlatformConfig(
            enabled=True,
            typing_indicator=False,
            extra={
                "server": "irc.example.invalid",
                "nickname": "testbot",
                "allowed_users": ["sender"],
            },
        )
    )
    adapter.gateway_runner = runner
    runner.adapters = {adapter.platform: adapter}
    authority = await initialize_session_authority(
        runner, profile_id="default", instance_id="irc-test"
    )
    runner._wire_adapter_handlers(adapter)
    received, receipts = [], []

    async def admit(event):
        # Use the real admission/SQLite path without starting a model turn.
        adapter._canonicalize(event.source)
        received.append(event)
        with native_callback(runner, event, home):
            receipts.append(await authority.admit_native(event))

    monkeypatch.setattr(adapter, "handle_message", admit)
    reader = asyncio.StreamReader()
    reader.feed_data(
        "".join(
            f":sender!user@host PRIVMSG testbot :{text}\r\n" for text in texts
        ).encode()
    )
    reader.feed_eof()
    adapter._reader = reader

    try:
        await adapter._receive_loop()
        rows = [
            row
            for session_id in authority.sessions
            for row in list_session_admissions(
                authority.db,
                session_id=session_id,
                pending_only=False,
            )
        ]
        assert [row["payload"]["text"] for row in rows] == list(texts)
        assert len({row["request_id"] for row in rows}) == len(texts)
        assert len({receipt.admission_id for receipt in receipts}) == len(texts)
        # Retrying the already captured event still resolves to its original admission.
        with native_callback(runner, received[0], home):
            retried = await authority.admit_native(received[0])
        assert retried.admission_id == receipts[0].admission_id
    finally:
        authority.db.close()
