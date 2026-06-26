"""Tests for the async-delivery capability gate (issue #10760).

Stateless request/response adapters (the API server / WebUI path) cannot route
a background completion back to the agent after a turn ends — there is no
persistent channel and ``APIServerAdapter.send()`` is a no-op stub. So tools
that promise async delivery (``terminal`` notify_on_complete / watch_patterns,
``delegate_task`` background=True) must refuse the promise on that path instead
of silently registering a watcher that never fires.

This is wired through:
  - ``BasePlatformAdapter.supports_async_delivery`` (default True)
  - ``APIServerAdapter.supports_async_delivery = False``
  - ``gateway.session_context._SESSION_ASYNC_DELIVERY`` contextvar +
    ``async_delivery_supported()`` helper, bound per-session.

These are behavior/invariant tests (how the capability relates to the channel),
not snapshots of a current value.
"""

import json
from types import SimpleNamespace

import pytest

from gateway.session_context import (
    async_delivery_supported,
    clear_session_vars,
    get_session_env,
    set_session_vars,
)


# ---------------------------------------------------------------------------
# Capability helper
# ---------------------------------------------------------------------------

class TestAsyncDeliverySupported:
    def test_default_unbound_is_supported(self):
        """CLI / cron / unaware paths never bind the var -> supported."""
        assert async_delivery_supported() is True

    def test_set_true_is_supported(self):
        tokens = set_session_vars(
            platform="telegram",
            chat_id="123",
            session_key="telegram:private:123",
            async_delivery=True,
        )
        try:
            assert async_delivery_supported() is True
            # Platform metadata stays readable alongside the capability.
            assert get_session_env("HERMES_SESSION_PLATFORM") == "telegram"
        finally:
            clear_session_vars(tokens)

    def test_set_false_is_unsupported(self):
        tokens = set_session_vars(
            platform="api_server",
            chat_id="sess1",
            session_key="sess1",
            async_delivery=False,
        )
        try:
            assert async_delivery_supported() is False
            # Platform must still be readable for routing/diagnostics even
            # though delivery is unsupported.
            assert get_session_env("HERMES_SESSION_PLATFORM") == "api_server"
        finally:
            clear_session_vars(tokens)

    def test_omitted_arg_defaults_supported(self):
        """Back-compat: callers that don't pass async_delivery stay supported."""
        tokens = set_session_vars(platform="discord", chat_id="9")
        try:
            assert async_delivery_supported() is True
        finally:
            clear_session_vars(tokens)

    def test_clear_resets_to_default_supported(self):
        """A cleared context must fall back to default-supported, NOT be
        mistaken for an opted-out stateless adapter."""
        tokens = set_session_vars(
            platform="api_server", session_key="s1", async_delivery=False
        )
        assert async_delivery_supported() is False
        clear_session_vars(tokens)
        assert async_delivery_supported() is True


# ---------------------------------------------------------------------------
# Adapter capability flag
# ---------------------------------------------------------------------------

class TestAdapterCapabilityFlag:
    def test_base_default_true(self):
        from gateway.platforms.base import BasePlatformAdapter

        assert BasePlatformAdapter.supports_async_delivery is True

    def test_api_server_false(self):
        from gateway.platforms.api_server import APIServerAdapter

        assert APIServerAdapter.supports_async_delivery is False

    def test_api_server_bind_chokepoint_hardwires_no_delivery(self):
        """Every API-server agent-entry path binds through
        _bind_api_server_session, which hardwires async_delivery=False — a new
        route physically cannot reintroduce the silent no-op (#10760)."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.session_context import clear_session_vars, get_session_env

        tokens = APIServerAdapter._bind_api_server_session(
            chat_id="c1", session_key="sk1", session_id="sid1"
        )
        try:
            assert async_delivery_supported() is False
            assert get_session_env("HERMES_SESSION_PLATFORM") == "api_server"
        finally:
            clear_session_vars(tokens)

    def test_api_server_binding_does_not_outlive_turn(self):
        """The no-delivery decision is request-scoped, NOT stuck to the session.
        After clear, a session resumed on a delivering interface re-binds fresh
        and is NOT blocked."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.session_context import clear_session_vars

        # Turn 1: same session over the API server -> blocked.
        tokens = APIServerAdapter._bind_api_server_session(session_key="shared-key")
        assert async_delivery_supported() is False
        clear_session_vars(tokens)

        # Turn 2: SAME session_key resumed on a delivering interface (CLI/gateway)
        # -> supported. The earlier False did not follow the session.
        tokens = set_session_vars(
            platform="telegram",
            session_key="shared-key",
            async_delivery=True,
        )
        try:
            assert async_delivery_supported() is True
        finally:
            clear_session_vars(tokens)

    def test_cron_session_binds_no_async_delivery(self):
        """Cron jobs are detached scheduler runs with no completion-queue
        re-entry path, so they must opt out of background delivery promises.
        """
        tokens = set_session_vars(
            platform="",
            chat_id="",
            chat_name="",
            async_delivery=False,
        )
        try:
            assert async_delivery_supported() is False
            assert get_session_env("HERMES_SESSION_PLATFORM") == ""
        finally:
            clear_session_vars(tokens)


# ---------------------------------------------------------------------------
# terminal_tool: refuses to register a watcher on unsupported sessions
# ---------------------------------------------------------------------------

class TestTerminalNotifyGate:
    @pytest.fixture(autouse=True)
    def _clean_watchers(self):
        from tools.process_registry import process_registry

        process_registry.pending_watchers = []
        yield
        process_registry.pending_watchers = []

    def _run_bg(self, command):
        from tools.terminal_tool import terminal_tool

        return json.loads(
            terminal_tool(command=command, background=True, notify_on_complete=True)
        )

    def test_api_server_skips_watcher_and_notes(self):
        from tools.process_registry import process_registry

        tokens = set_session_vars(
            platform="api_server", chat_id="s1", session_key="s1", async_delivery=False
        )
        try:
            d = self._run_bg("sleep 30 && echo DONE")
        finally:
            clear_session_vars(tokens)

        assert d.get("notify_on_complete") is False
        assert d.get("notify_unsupported"), "must explain the limitation"
        assert "poll" in d["notify_unsupported"].lower()
        assert len(process_registry.pending_watchers) == 0

    def test_gateway_registers_watcher(self):
        from tools.process_registry import process_registry

        tokens = set_session_vars(
            platform="telegram",
            chat_id="123",
            thread_id="7",
            user_id="u1",
            session_key="telegram:private:123",
            async_delivery=True,
        )
        try:
            d = self._run_bg("sleep 30 && echo DONE")
        finally:
            clear_session_vars(tokens)

        assert d.get("notify_on_complete") is True
        assert not d.get("notify_unsupported")
        assert len(process_registry.pending_watchers) == 1
        assert process_registry.pending_watchers[0]["platform"] == "telegram"

    def test_cli_stays_supported(self):
        """CLI delivers via the in-process completion_queue: notify stays on,
        no false 'unsupported' note, and no pending_watcher (empty platform)."""
        from tools.process_registry import process_registry

        d = self._run_bg("sleep 30 && echo DONE")
        assert d.get("notify_on_complete") is True
        assert not d.get("notify_unsupported")
        # No platform bound -> no gateway watcher, but completion_queue still fires.
        assert len(process_registry.pending_watchers) == 0


class TestDelegateTaskBackgroundGate:
    def test_cron_context_falls_back_to_sync(self, monkeypatch):
        """Cron jobs must not dispatch detached background subagents because
        no cron drain re-enters async-delegation completions after the turn.
        """
        import tools.delegate_tool as dt

        parent = pytest.importorskip("unittest.mock").MagicMock()
        parent._delegate_depth = 0
        parent.session_id = "cron-session"
        parent._interrupt_requested = False
        parent._active_children = []
        parent._active_children_lock = None

        creds = {
            "model": "m",
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "command": None,
            "args": None,
        }

        monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
        monkeypatch.setattr(
            dt,
            "_build_child_agent",
            lambda **kw: SimpleNamespace(
                _delegate_saved_tool_names=[],
                _delegate_role="leaf",
            ),
        )
        monkeypatch.setattr(
            dt,
            "_run_single_child",
            lambda task_index, goal, **kw: {
                "task_index": task_index,
                "status": "completed",
                "summary": f"done: {goal}",
                "api_calls": 1,
                "duration_seconds": 0.1,
                "model": "m",
                "exit_reason": "completed",
            },
        )

        tokens = set_session_vars(
            platform="",
            chat_id="",
            chat_name="",
            async_delivery=False,
        )
        try:
            out = json.loads(
                dt.delegate_task(
                    tasks=[{"goal": "a"}, {"goal": "b"}],
                    background=True,
                    parent_agent=parent,
                )
            )
        finally:
            clear_session_vars(tokens)

        assert out["results"][0]["summary"] == "done: a"
        assert out["results"][1]["summary"] == "done: b"
        assert "SYNCHRONOUSLY" in out["note"]
