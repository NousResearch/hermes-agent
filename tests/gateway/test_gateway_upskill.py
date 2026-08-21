"""Gateway parity test for /upskill — the rewrite-and-fall-through contract.

/upskill is supported on the gateway (Telegram/Discord/etc.) via the
``canonical == "upskill"`` branch in ``GatewayRunner._handle_message``, which
rewrites ``event.text`` to the call to ``build_upskill_prompt(...)`` and then
falls through to normal agent processing (like /learn and /init).

The classic CLI got sentinel-based tests for its handler; this mirrors them on
the gateway side so a future "clean-up" that drops the mutation (or breaks the
fall-through) is caught here rather than invisibly. Drives the real dispatcher
per AGENTS.md's E2E-validation bar.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    """Mirror tests/gateway/test_unknown_command.py::_make_runner."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_upskill_rewrites_event_text_and_falls_through(monkeypatch):
    """/upskill on the gateway must rewrite event.text to the sweep prompt and
    fall through to agent processing — not be flagged unknown or short-circuit.
    """
    import gateway.run as gateway_run

    runner = _make_runner()
    # If the rewrite/fall-through is broken, the event.text would either stay
    # "/upskill" (reaching the unknown guard) or the branch would return the
    # "Could not start" error. We assert the rewrite happened AND that the
    # agent path actually ran (await asserted) — honest fall-through proof.
    runner._run_agent = AsyncMock(return_value={"final_response": "ok"})

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    event = _make_event("/upskill")
    result = await runner._handle_message(event)

    # The sweep prompt was injected into event.text (the load-bearing mutation).
    assert event.text.startswith("[/upskill]")
    assert "propose" in event.text.lower()
    # Neither the unknown-command guard nor the error path returned.
    assert result is not None
    assert "Unknown command" not in repr(result)
    assert "Could not start /upskill" not in str(result)
    # Positive proof of fall-through: the agent path was actually invoked.
    runner._run_agent.assert_awaited()


@pytest.mark.asyncio
async def test_upskill_scoped_arg_is_embedded_in_rewritten_text(monkeypatch):
    """A scope hint after /upskill should be preserved into the sweep prompt."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._run_agent = AsyncMock(return_value="agent-ran")

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    event = _make_event("/upskill focus on the WiNG console")
    await runner._handle_message(event)

    assert event.text.startswith("[/upskill]")
    assert "WiNG console" in event.text