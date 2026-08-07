"""Regression tests: /yolo and /verbose dispatch mid-agent-run.

When an agent is running, the gateway's running-agent guard rejects most
slash commands with "⏳ Agent is running — /{cmd} can't run mid-turn"
(PR #12334). A small allowlist bypasses that and actually dispatches:

  * /yolo — toggles the session yolo flag; useful to pre-approve a
    pending approval prompt without waiting for the agent to finish.
  * /verbose — cycles the per-platform tool-progress display mode;
    affects the ongoing stream.

Commands whose handlers say "takes effect on next message" stay on the
catch-all by design:

  * /fast — writes config.yaml only
  * /reasoning — writes config.yaml only

These tests lock in both behaviors so the allowlist doesn't silently
grow or shrink.
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
    """Minimal GatewayRunner with an active running agent for this session."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)

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
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._service_tier = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()

    # Simulate agent actively running for this session so the guard fires.
    # Note: the stale-eviction branch calls agent.get_activity_summary() and
    # compares seconds_since_activity against HERMES_AGENT_TIMEOUT. Return a
    # dict with recent activity so the eviction path doesn't clear our
    # fake running agent before the toggle guard runs.
    import time
    sk = build_session_key(_make_source())
    agent_mock = MagicMock()
    agent_mock.get_activity_summary.return_value = {
        "seconds_since_activity": 0.0,
        "last_activity_desc": "api_call",
        "api_call_count": 1,
        "max_iterations": 60,
    }
    runner._running_agents[sk] = agent_mock
    runner._running_agents_ts[sk] = time.time()
    return runner


@pytest.mark.asyncio
async def test_yolo_dispatches_mid_run(monkeypatch):
    """/yolo mid-run must dispatch to its handler, not hit the catch-all."""
    runner = _make_runner()
    runner._handle_yolo_command = AsyncMock(return_value="⚡ YOLO mode **ON** for this session")

    result = await runner._handle_message(_make_event("/yolo"))

    runner._handle_yolo_command.assert_awaited_once()
    assert result == "⚡ YOLO mode **ON** for this session"
    assert "can't run mid-turn" not in (result or "")


@pytest.mark.asyncio
async def test_verbose_dispatches_mid_run(monkeypatch):
    """/verbose mid-run must dispatch to its handler, not hit the catch-all."""
    runner = _make_runner()
    runner._handle_verbose_command = AsyncMock(return_value="tool progress: new")

    result = await runner._handle_message(_make_event("/verbose"))

    runner._handle_verbose_command.assert_awaited_once()
    assert result == "tool progress: new"
    assert "can't run mid-turn" not in (result or "")


class TestBusySlashCommandRejectionPrefixed:
    """Regression tests (review of #75436): the shared busy-command
    rejection path (_dispatch_busy_slash_command, "Guard 2" -- routes
    active-session recognized commands, distinct from
    _handle_active_session_busy_message()'s own message-selection logic
    and the separate cold-path external-drain gate both already covered)
    still returned bare, unprefixed text for its catch-all and its named
    _BUSY_REJECT_TEXT entries. Drives the real _handle_message() end to
    end via the same active-session runner fixture used above.
    """

    @pytest.mark.asyncio
    async def test_catch_all_reject_is_prefixed(self):
        """A recognized command with no special mid-run handler and no
        dispatch policy (e.g. /reasoning) hits the catch-all reject."""
        runner = _make_runner()

        result = await runner._handle_message(_make_event("/reasoning"))

        assert result is not None
        assert "can't run mid-turn" in result
        assert result.startswith("[System] "), (
            f"Catch-all busy-command reject must carry the [System] "
            f"prefix like every other busy-ack message: {result!r}"
        )

    @pytest.mark.asyncio
    async def test_named_reject_text_is_prefixed(self):
        """/model mid-run hits the named _BUSY_REJECT_TEXT entry, a
        separate dict from the catch-all string above -- both needed
        the prefix independently."""
        runner = _make_runner()

        result = await runner._handle_message(_make_event("/model gpt-5"))

        assert result is not None
        assert "switch models" in result
        assert result.startswith("[System] "), (
            f"Named busy-command reject text must carry the [System] "
            f"prefix like every other busy-ack message: {result!r}"
        )

    @pytest.mark.asyncio
    async def test_goal_set_new_text_reject_is_prefixed(self):
        """/goal <new goal text> mid-run hits _busy_goal_command's own
        non-control rejection (review of #75599) -- a third, separate
        return site distinct from both the catch-all and the
        _BUSY_REJECT_TEXT dict, reached via the same Guard-2 dispatcher.
        /goal's CONTROL verbs (status/pause/clear/wait/etc.) dispatch
        normally and must NOT be rejected -- only setting new goal text
        mid-run is."""
        runner = _make_runner()

        result = await runner._handle_message(_make_event("/goal ship the release"))

        assert result is not None
        assert "before setting a new goal" in result
        assert result.startswith("[System] "), (
            f"The /goal <text> mid-run reject must carry the [System] "
            f"prefix like every other busy-ack message: {result!r}"
        )



