"""Regression: /footer must dispatch to its handler while an agent is running.

The gateway runner's running-agent block (``gateway/run.py``) routes the
"session-level toggles that are safe to run mid-agent" through a membership
set before the catch-all that rejects everything else with
``"Agent is running — /<cmd> can't run mid-turn"``.

``/footer`` is a pure display toggle: like its sibling ``/verbose`` it only
writes a ``display.*`` config key and returns a status string (see
``_handle_footer_command`` / ``_handle_verbose_command``), and the runner
already ships a dedicated mid-run dispatch branch for it. But the gate set
listed only ``{"yolo", "verbose"}``, so the ``footer`` branch was
unreachable: ``/footer`` fell through to the catch-all and was rejected,
forcing the user to ``/stop`` a running agent just to toggle a footer.

These tests pin that ``/footer`` (and, as a parity guard, ``/verbose``)
reach their handlers mid-run instead of the busy catch-all rejection.

Mirrors the runner-construction pattern of ``test_steer_command.py`` so the
same proven path through ``_handle_message`` reaches the running-agent
command dispatch.
"""
from __future__ import annotations

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
    return MessageEvent(
        text=text,
        source=_make_source(),
        message_id="m1",
    )


def _make_runner(session_entry: SessionEntry):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._pending_messages = {}
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
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
    return runner, adapter


def _session_entry() -> SessionEntry:
    return SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )


_BUSY_REJECTION = "can't run"


@pytest.mark.asyncio
async def test_footer_dispatches_to_handler_when_agent_running():
    """/footer mid-run reaches _handle_footer_command, not the busy catch-all."""
    runner, _adapter = _make_runner(_session_entry())
    sk = build_session_key(_make_source())
    runner._running_agents[sk] = MagicMock()

    handler = AsyncMock(return_value="footer toggled")
    runner._handle_footer_command = handler

    result = await runner._handle_message(_make_event("/footer"))

    handler.assert_awaited_once()
    assert result == "footer toggled"
    assert _BUSY_REJECTION not in (result or ""), (
        "/footer hit the busy catch-all instead of dispatching to its handler"
    )


@pytest.mark.asyncio
async def test_footer_with_arg_dispatches_when_agent_running():
    """/footer on must also dispatch (argument form, same routing)."""
    runner, _adapter = _make_runner(_session_entry())
    sk = build_session_key(_make_source())
    runner._running_agents[sk] = MagicMock()

    handler = AsyncMock(return_value="footer on")
    runner._handle_footer_command = handler

    result = await runner._handle_message(_make_event("/footer on"))

    handler.assert_awaited_once()
    assert result == "footer on"


@pytest.mark.asyncio
async def test_verbose_sibling_still_dispatches_when_agent_running():
    """Parity guard for the safe-toggle set: the documented sibling /verbose
    also dispatches mid-run, proving the set routes its members rather than
    rejecting them. Guards against a regression that drops the whole set."""
    runner, _adapter = _make_runner(_session_entry())
    sk = build_session_key(_make_source())
    runner._running_agents[sk] = MagicMock()

    handler = AsyncMock(return_value="verbose cycled")
    runner._handle_verbose_command = handler

    result = await runner._handle_message(_make_event("/verbose"))

    handler.assert_awaited_once()
    assert result == "verbose cycled"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
