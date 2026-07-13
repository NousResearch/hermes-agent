"""Tests for memory.review_on_reset / review_on_session_end / review_on_compression (#31597).

Each flag gates a call to ``agent.background_review.maybe_spawn_boundary_review``
at the matching session boundary:

- ``reset``        → CLI ``HermesCLI.new_session``, TUI ``_reset_session_agent``,
                     gateway ``_handle_reset_command``
- ``session_end``  → CLI ``_run_cleanup`` (exit, bounded join),
                     TUI ``_finalize_session``, gateway session expiry
- ``compression``  → ``compress_context``, only AFTER the abort / no-op guards,
                     with a snapshot captured BEFORE ``compress()``

Gating (flags, empty snapshots, missing memory store/tool) is centralised in
the helper and unit-tested here; the per-surface tests exercise the *real*
wiring code paths and assert the helper is invoked with the right trigger and
snapshot.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


HISTORY = [
    {"role": "user", "content": "remember my dog is named Biscuit"},
    {"role": "assistant", "content": "Got it — Biscuit."},
]


# ---------------------------------------------------------------------------
# maybe_spawn_boundary_review — central gate
# ---------------------------------------------------------------------------

class TestMaybeSpawnBoundaryReviewGate:
    """The helper owns all gating: flags, snapshot, store/tool availability."""

    def _agent(self, **overrides):
        agent = SimpleNamespace(
            _memory_review_on_reset=False,
            _memory_review_on_session_end=False,
            _memory_review_on_compression=False,
            _memory_store=object(),
            valid_tool_names=["memory", "task"],
        )
        for key, value in overrides.items():
            setattr(agent, key, value)
        return agent

    def _patch_spawn(self, monkeypatch, captured):
        import agent.background_review as br_mod

        def fake_spawn(agent, messages_snapshot, review_memory=False, review_skills=False):
            captured["agent"] = agent
            captured["messages"] = messages_snapshot
            captured["review_memory"] = review_memory
            captured["ran"] = threading.Event()
            return captured["ran"].set, "prompt"

        monkeypatch.setattr(br_mod, "spawn_background_review_thread", fake_spawn)

    def test_spawns_named_daemon_thread_per_trigger(self, monkeypatch):
        from agent.background_review import maybe_spawn_boundary_review

        flag_by_trigger = {
            "reset": "_memory_review_on_reset",
            "session_end": "_memory_review_on_session_end",
            "compression": "_memory_review_on_compression",
        }
        for trigger, flag in flag_by_trigger.items():
            captured = {}
            self._patch_spawn(monkeypatch, captured)
            agent = self._agent(**{flag: True})

            thread = maybe_spawn_boundary_review(agent, HISTORY, trigger=trigger)

            assert thread is not None, f"trigger {trigger!r} should spawn"
            assert thread.daemon is True
            assert thread.name == f"bg-review-{trigger}"
            thread.join(timeout=5)
            assert captured["ran"].is_set()
            assert captured["review_memory"] is True
            assert captured["agent"] is agent

    def test_flag_off_spawns_nothing(self, monkeypatch):
        from agent.background_review import maybe_spawn_boundary_review

        captured = {}
        self._patch_spawn(monkeypatch, captured)
        agent = self._agent()  # all flags False

        for trigger in ("reset", "session_end", "compression"):
            assert maybe_spawn_boundary_review(agent, HISTORY, trigger=trigger) is None
        assert captured == {}

    def test_unknown_trigger_spawns_nothing(self, monkeypatch):
        from agent.background_review import maybe_spawn_boundary_review

        captured = {}
        self._patch_spawn(monkeypatch, captured)
        agent = self._agent(_memory_review_on_reset=True)

        assert maybe_spawn_boundary_review(agent, HISTORY, trigger="bogus") is None
        assert captured == {}

    def test_empty_snapshot_spawns_nothing(self, monkeypatch):
        from agent.background_review import maybe_spawn_boundary_review

        captured = {}
        self._patch_spawn(monkeypatch, captured)
        agent = self._agent(_memory_review_on_reset=True)

        assert maybe_spawn_boundary_review(agent, [], trigger="reset") is None
        assert maybe_spawn_boundary_review(agent, None, trigger="reset") is None
        assert captured == {}

    def test_missing_memory_store_spawns_nothing(self, monkeypatch):
        from agent.background_review import maybe_spawn_boundary_review

        captured = {}
        self._patch_spawn(monkeypatch, captured)
        agent = self._agent(_memory_review_on_reset=True, _memory_store=None)

        assert maybe_spawn_boundary_review(agent, HISTORY, trigger="reset") is None
        assert captured == {}

    def test_missing_memory_tool_spawns_nothing(self, monkeypatch):
        from agent.background_review import maybe_spawn_boundary_review

        captured = {}
        self._patch_spawn(monkeypatch, captured)
        agent = self._agent(
            _memory_review_on_reset=True, valid_tool_names=["task", "terminal"]
        )

        assert maybe_spawn_boundary_review(agent, HISTORY, trigger="reset") is None
        assert captured == {}

    def test_snapshot_is_copied(self, monkeypatch):
        from agent.background_review import maybe_spawn_boundary_review

        captured = {}
        self._patch_spawn(monkeypatch, captured)
        agent = self._agent(_memory_review_on_reset=True)
        original = list(HISTORY)

        thread = maybe_spawn_boundary_review(agent, original, trigger="reset")

        assert thread is not None
        thread.join(timeout=5)
        assert captured["messages"] == original
        assert captured["messages"] is not original

    def test_spawn_failure_returns_none_without_raising(self, monkeypatch):
        import agent.background_review as br_mod
        from agent.background_review import maybe_spawn_boundary_review

        monkeypatch.setattr(
            br_mod,
            "spawn_background_review_thread",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        agent = self._agent(_memory_review_on_reset=True)

        assert maybe_spawn_boundary_review(agent, HISTORY, trigger="reset") is None


# ---------------------------------------------------------------------------
# review_on_compression — compress_context wiring
# ---------------------------------------------------------------------------

class TestReviewOnCompression:
    """Fires only after the abort / no-op guards, with a pre-compression snapshot."""

    def _make_agent(self):
        import os

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=None,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
            )
        return agent

    def _stub_compressor(self, agent, *, compress_side_effect, aborted=False):
        compressor = MagicMock()
        compressor.compress.side_effect = compress_side_effect
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = aborted
        compressor._last_aux_model_failure_model = None
        compressor._last_aux_model_failure_error = None
        agent.context_compressor = compressor
        return compressor

    def test_successful_compression_fires_with_pre_compression_snapshot(self):
        agent = self._make_agent()
        agent._memory_review_on_compression = True
        self._stub_compressor(
            agent,
            compress_side_effect=lambda msgs, **kw: [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                {"role": "user", "content": "tail question"},
            ],
        )
        messages = [{"role": "user", "content": f"m{i}"} for i in range(10)]
        original = list(messages)

        with patch("agent.background_review.maybe_spawn_boundary_review") as spawn:
            agent._compress_context(messages, "sys", approx_tokens=10_000)

        spawn.assert_called_once()
        args, kwargs = spawn.call_args
        assert args[0] is agent
        assert kwargs.get("trigger") == "compression"
        # The snapshot is the PRE-compression transcript, captured as a copy
        # so the compressor rewriting `messages` cannot mutate it.
        assert args[1] == original
        assert args[1] is not messages

    def test_aborted_compression_does_not_fire(self):
        """An aborted compression retains the transcript — no context-loss
        boundary, so the opt-in review cost must not be paid (sweeper review
        of the original patch)."""
        agent = self._make_agent()
        agent._memory_review_on_compression = True
        self._stub_compressor(
            agent,
            compress_side_effect=lambda msgs, **kw: msgs,
            aborted=True,
        )
        messages = [{"role": "user", "content": "lots of context"}]

        with patch("agent.background_review.maybe_spawn_boundary_review") as spawn:
            result, _sp = agent._compress_context(messages, "sys", approx_tokens=10_000)

        spawn.assert_not_called()
        assert result is messages

    def test_noop_compression_does_not_fire(self):
        """A compressor returning the input object made no structural progress
        — nothing was discarded, so no review fires."""
        agent = self._make_agent()
        agent._memory_review_on_compression = True
        self._stub_compressor(agent, compress_side_effect=lambda msgs, **kw: msgs)
        messages = [{"role": "user", "content": "x"}]

        with patch("agent.background_review.maybe_spawn_boundary_review") as spawn:
            result, _sp = agent._compress_context(messages, "sys", approx_tokens=10_000)

        spawn.assert_not_called()
        assert result is messages

    def test_flag_off_does_not_fire_or_snapshot(self):
        agent = self._make_agent()
        agent._memory_review_on_compression = False
        self._stub_compressor(
            agent,
            compress_side_effect=lambda msgs, **kw: [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"}
            ],
        )
        messages = [{"role": "user", "content": f"m{i}"} for i in range(10)]

        with patch("agent.background_review.maybe_spawn_boundary_review") as spawn:
            agent._compress_context(messages, "sys", approx_tokens=10_000)

        spawn.assert_not_called()


# ---------------------------------------------------------------------------
# review_on_reset — CLI HermesCLI.new_session
# ---------------------------------------------------------------------------

class TestReviewOnResetCLI:
    """/new hands the helper a copy of the history being rotated away,
    alongside (not inside) the serialized provider end→switch task (#16454)."""

    @patch("hermes_cli.plugins.invoke_hook")
    def test_new_session_fires_reset_review(self, _mock_hook):
        from cli import HermesCLI

        cli = HermesCLI()
        cli.agent = MagicMock()
        cli.agent.session_id = "old-session-id"
        cli.conversation_history = list(HISTORY)

        with patch("agent.background_review.maybe_spawn_boundary_review") as spawn:
            cli.new_session(silent=True)

        spawn.assert_called_once()
        args, kwargs = spawn.call_args
        assert args[0] is cli.agent
        assert kwargs.get("trigger") == "reset"
        # Copy of the pre-reset history — new_session clears the live list.
        assert args[1] == HISTORY
        assert cli.conversation_history == []
        # The provider end→switch ordering contract (#16454) is untouched:
        # extraction + rebinding still go through the memory manager's
        # serialized boundary task, not through the review thread.
        cli.agent._memory_manager.commit_session_boundary_async.assert_called_once()

    @patch("hermes_cli.plugins.invoke_hook")
    def test_new_session_without_history_does_not_fire(self, _mock_hook):
        from cli import HermesCLI

        cli = HermesCLI()
        cli.agent = MagicMock()
        cli.agent.session_id = "old-session-id"
        cli.conversation_history = []

        with patch("agent.background_review.maybe_spawn_boundary_review") as spawn:
            cli.new_session(silent=True)

        spawn.assert_not_called()


# ---------------------------------------------------------------------------
# review_on_session_end — CLI exit (_run_cleanup)
# ---------------------------------------------------------------------------

class TestReviewOnSessionEndCLIExit:
    """CLI exit spawns the review before the provider drain and joins it
    (bounded) after provider shutdown, so exit is never wedged."""

    def _run_cleanup_with(self, agent, spawn_return):
        import cli as cli_mod

        with patch(
            "agent.background_review.maybe_spawn_boundary_review",
            return_value=spawn_return,
        ) as spawn:
            cli_mod._active_agent_ref = agent
            cli_mod._cleanup_done = False
            try:
                cli_mod._run_cleanup()
            finally:
                cli_mod._active_agent_ref = None
                cli_mod._cleanup_done = False
        return spawn

    @patch("hermes_cli.plugins.invoke_hook")
    def test_cleanup_fires_session_end_review_and_joins(self, _mock_hook):
        from agent.background_review import SESSION_END_REVIEW_JOIN_TIMEOUT_S

        agent = MagicMock()
        agent.session_id = "cli-session-id"
        agent._session_messages = list(HISTORY)
        fake_thread = MagicMock()

        spawn = self._run_cleanup_with(agent, fake_thread)

        spawn.assert_called_once()
        args, kwargs = spawn.call_args
        assert args[0] is agent
        assert args[1] == HISTORY
        assert kwargs.get("trigger") == "session_end"
        # Review overlaps the provider drain, then gets a bounded join.
        fake_thread.join.assert_called_once_with(
            timeout=SESSION_END_REVIEW_JOIN_TIMEOUT_S
        )
        # Provider shutdown still ran.
        agent.shutdown_memory_provider.assert_called_once()

    @patch("hermes_cli.plugins.invoke_hook")
    def test_cleanup_non_list_messages_pass_empty_snapshot(self, _mock_hook):
        agent = MagicMock()
        agent.session_id = "cli-session-id"
        # MagicMock auto-synthesises _session_messages as a non-list mock.

        spawn = self._run_cleanup_with(agent, None)

        spawn.assert_called_once()
        assert spawn.call_args.args[1] == []

    @patch("hermes_cli.plugins.invoke_hook")
    def test_cleanup_survives_no_spawned_thread(self, _mock_hook):
        """helper returning None (flag off / gated) must not break cleanup."""
        agent = MagicMock()
        agent.session_id = "cli-session-id"
        agent._session_messages = list(HISTORY)

        self._run_cleanup_with(agent, None)  # must not raise

        agent.shutdown_memory_provider.assert_called_once()


# ---------------------------------------------------------------------------
# review_on_reset / review_on_session_end — TUI server
# ---------------------------------------------------------------------------

class TestReviewOnTUIBoundaries:
    def _make_session(self, agent, history):
        return {
            "session_key": "tui-key-001",
            "agent": agent,
            "history": list(history),
            "history_lock": threading.Lock(),
            "model_override": None,
        }

    @patch("hermes_cli.plugins.invoke_hook")
    def test_finalize_session_fires_session_end_review(self, _mock_hook):
        from tui_gateway.server import _finalize_session

        agent = MagicMock()
        agent.session_id = "tui-session-id"
        agent._session_messages = None
        session = self._make_session(agent, HISTORY)

        with patch("agent.background_review.maybe_spawn_boundary_review") as spawn:
            _finalize_session(session, end_reason="tui_close")

        spawn.assert_called_once()
        args, kwargs = spawn.call_args
        assert args[0] is agent
        assert args[1] == HISTORY
        assert kwargs.get("trigger") == "session_end"
        agent.commit_memory_session.assert_called_once()

    @patch("hermes_cli.plugins.invoke_hook")
    def test_finalize_session_empty_history_does_not_fire(self, _mock_hook):
        from tui_gateway.server import _finalize_session

        agent = MagicMock()
        agent.session_id = "tui-session-id"
        agent._session_messages = None
        session = self._make_session(agent, [])

        with patch("agent.background_review.maybe_spawn_boundary_review") as spawn:
            _finalize_session(session, end_reason="tui_close")

        spawn.assert_not_called()

    def test_reset_session_agent_fires_reset_review_with_old_history(self):
        import tui_gateway.server as tui_server

        old_agent = MagicMock()
        session = self._make_session(old_agent, HISTORY)

        with patch.object(tui_server, "_set_session_context", return_value=[]), \
                patch.object(tui_server, "_clear_session_context"), \
                patch.object(tui_server, "_make_agent", return_value=MagicMock()), \
                patch.object(tui_server, "_config_model_target", return_value="m"), \
                patch.object(tui_server, "_load_show_reasoning", return_value=False), \
                patch.object(tui_server, "_load_tool_progress_mode", return_value="all"), \
                patch.object(tui_server, "_session_info", return_value={}), \
                patch.object(tui_server, "_emit"), \
                patch.object(tui_server, "_restart_slash_worker"), \
                patch("agent.background_review.maybe_spawn_boundary_review") as spawn:
            tui_server._reset_session_agent("sid-1", session)

        spawn.assert_called_once()
        args, kwargs = spawn.call_args
        assert args[0] is old_agent
        assert kwargs.get("trigger") == "reset"
        # Snapshot of the history that the reset then cleared.
        assert args[1] == HISTORY
        assert session["history"] == []


# ---------------------------------------------------------------------------
# review_on_reset — gateway /new (_handle_reset_command)
# ---------------------------------------------------------------------------

class TestReviewOnGatewayReset:
    def test_reset_command_fires_reset_review_before_teardown(self):
        from gateway.slash_commands import GatewaySlashCommandsMixin

        stub_agent = SimpleNamespace(_session_messages=list(HISTORY))
        gateway = MagicMock()
        gateway._agent_cache.get.return_value = (stub_agent, object())
        gateway._run_in_executor_with_context = AsyncMock()
        event = MagicMock()

        async def _invoke():
            # The tail of the reset handler touches live-gateway internals the
            # MagicMock self doesn't model; the review hook fires before them.
            try:
                await GatewaySlashCommandsMixin._handle_reset_command(gateway, event)
            except Exception:
                pass

        with patch("agent.background_review.maybe_spawn_boundary_review") as spawn:
            asyncio.run(_invoke())

        spawn.assert_called_once()
        args, kwargs = spawn.call_args
        assert args[0] is stub_agent
        assert args[1] == HISTORY
        assert kwargs.get("trigger") == "reset"


# ---------------------------------------------------------------------------
# review_on_session_end — gateway session expiry (_session_expiry_watcher)
# ---------------------------------------------------------------------------

class TestReviewOnGatewayExpiry:
    """Mirrors the #14981 expiry-watcher fixture from
    tests/gateway/test_session_boundary_hooks.py."""

    @pytest.mark.asyncio
    @patch("hermes_cli.plugins.invoke_hook")
    async def test_session_expiry_fires_session_end_review(self, mock_invoke_hook):
        from datetime import datetime, timedelta

        from gateway.config import Platform
        from gateway.run import GatewayRunner
        from gateway.session import SessionEntry

        runner = object.__new__(GatewayRunner)
        runner._running = True
        runner._running_agents = {}
        runner._last_session_store_prune_ts = 0.0

        stub_agent = SimpleNamespace(_session_messages=list(HISTORY))
        session_key = "agent:main:telegram:dm:42"
        runner._agent_cache = {session_key: (stub_agent, object())}
        runner._agent_cache_lock = threading.Lock()

        expired_entry = SessionEntry(
            session_key=session_key,
            session_id="sess-expired",
            created_at=datetime.now() - timedelta(hours=2),
            updated_at=datetime.now() - timedelta(hours=2),
            platform=Platform.TELEGRAM,
            chat_type="dm",
        )
        expired_entry.expiry_finalized = False

        runner.session_store = MagicMock()
        runner.session_store._ensure_loaded = MagicMock()
        runner.session_store._entries = {session_key: expired_entry}
        runner.session_store._is_session_expired = MagicMock(return_value=True)
        runner.session_store._lock = MagicMock()
        runner.session_store._lock.__enter__ = MagicMock(return_value=None)
        runner.session_store._lock.__exit__ = MagicMock(return_value=None)
        runner.session_store._save = MagicMock()

        runner._evict_cached_agent = MagicMock()
        runner._cleanup_agent_resources = MagicMock()
        runner._cleanup_agent_resources_off_loop = AsyncMock()
        runner._sweep_idle_cached_agents = MagicMock(return_value=0)

        # Make the watcher's initial 60s sleep instant, and stop the loop
        # after the first expiry pass fires its plugin hook.
        _orig_sleep = asyncio.sleep

        async def _fast_sleep(_):
            await _orig_sleep(0)

        def _hook_and_stop(*_a, **_kw):
            runner._running = False
            return None

        mock_invoke_hook.side_effect = _hook_and_stop

        with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep), \
                patch("agent.background_review.maybe_spawn_boundary_review") as spawn:
            await runner._session_expiry_watcher(interval=0)

        spawn.assert_called_once()
        args, kwargs = spawn.call_args
        assert args[0] is stub_agent
        assert args[1] == HISTORY
        assert kwargs.get("trigger") == "session_end"
        # The review is scheduled before resource teardown, which still runs.
        runner._cleanup_agent_resources_off_loop.assert_awaited()
