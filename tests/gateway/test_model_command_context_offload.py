"""``/model`` context-length resolution must not block the gateway event loop.

Behavioral regression tests for the offload of
``resolve_display_context_length`` (blocking provider probe ladder) out of the
async ``/model`` handlers, and for the offload of
``enrich_model_switch_warnings_for_gateway`` (which reaches the same sync
resolver via ``merge_preflight_compression_warning``).

These drive the real ``_handle_model_command`` with a mocked switch pipeline —
no source-reading assertions; reverting either offload makes the corresponding
test fail because the blocking work lands back on the loop thread.
"""

import asyncio
import threading

import pytest
from unittest.mock import AsyncMock, MagicMock

import gateway.slash_commands as slash_commands
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(),
    )


def _runner_with_store(tmp_path, monkeypatch):
    """Minimal GatewayRunner harness driving the real /model handler."""
    import yaml as _yaml

    import gateway.run as gateway_run
    from gateway.run import GatewayRunner
    from hermes_cli.model_switch import ModelSwitchResult

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        _yaml.safe_dump({"model": {"default": "old-model", "provider": "openrouter"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: ModelSwitchResult(
            success=True,
            new_model="gpt-5.5",
            target_provider="openrouter",
            provider_changed=False,
            api_key="sk-test",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            provider_label="OpenRouter",
        ),
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    # No expensive-model confirmation detour.
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning",
        lambda *a, **k: None,
    )

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._pending_one_turn_model_restores = {}
    runner._running_agents = {}
    _store = MagicMock()
    _store.set_model_override = AsyncMock()
    _store._store = None
    runner.session_store = None
    runner._async_session_store = _store
    return runner


@pytest.mark.asyncio
async def test_context_resolution_runs_off_the_loop_thread(tmp_path, monkeypatch):
    """The sync resolver must execute on a worker thread when the /model
    handler resolves the display context length for the switch reply."""
    from hermes_cli import model_switch

    seen = {}
    loop_thread = threading.current_thread()

    def _recording_resolver(model, provider, **kwargs):
        seen.setdefault("threads", []).append(threading.current_thread())
        return 128000

    monkeypatch.setattr(
        model_switch, "resolve_display_context_length", _recording_resolver
    )

    runner = _runner_with_store(tmp_path, monkeypatch)
    result = await runner._handle_model_command(_event("/model gpt-5.5"))

    assert result is not None and "gpt-5.5" in result
    assert seen.get("threads"), "handler never resolved the context length"
    assert all(th is not loop_thread for th in seen["threads"]), (
        "resolve_display_context_length ran on the event loop thread — "
        "the /model handler must offload it via "
        "resolve_display_context_length_async"
    )


@pytest.mark.asyncio
async def test_warning_enrichment_is_offloaded(tmp_path, monkeypatch):
    """enrich_model_switch_warnings_for_gateway reaches the same sync resolver
    via merge_preflight_compression_warning, so the handler must dispatch it
    through asyncio.to_thread rather than calling it inline on the loop."""
    from hermes_cli import context_switch_guard

    offloaded = []
    real_to_thread = asyncio.to_thread

    async def _spy_to_thread(func, /, *args, **kwargs):
        offloaded.append(func)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(slash_commands.asyncio, "to_thread", _spy_to_thread)

    runner = _runner_with_store(tmp_path, monkeypatch)
    result = await runner._handle_model_command(_event("/model gpt-5.5"))

    assert result is not None and "gpt-5.5" in result
    assert context_switch_guard.enrich_model_switch_warnings_for_gateway in offloaded, (
        "enrich_model_switch_warnings_for_gateway must be dispatched via "
        "asyncio.to_thread (it was called inline on the event loop instead)"
    )
