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

import pytest

from gateway.session_context import (
    async_delivery_supported,
    clear_session_vars,
    get_session_env,
    reset_session_vars,
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
# Stateless runners — issues #53027 / #63142
# ---------------------------------------------------------------------------

class TestDeclareStatelessChannel:
    """``hermes -z`` and cron cannot receive a completion after their turn ends.

    Cron clears the ``HERMES_SESSION_*`` routing keys, so an async delegation's
    completion event carries ``session_key=""`` and the gateway watcher drops it
    for lack of routing metadata; either way the job's final response has already
    shipped. One-shot simply exits. Both must bind the capability, or
    ``delegate_task`` is forced background and every subagent result is lost.
    """

    def test_declare_stateless_channel_disables_async_delivery(self):
        from gateway.session_context import declare_stateless_channel

        reset_session_vars()  # don't assume ambient contextvar state
        assert async_delivery_supported() is True
        try:
            declare_stateless_channel()
            assert async_delivery_supported() is False
        finally:
            reset_session_vars()

    def test_declare_does_not_engage_full_session_context(self):
        """The helper binds ONLY the capability.

        ``set_session_vars`` latches ``_session_context_engaged``, which flips the
        subprocess env bridge to ContextVar-authoritative. A pure single-process
        one-shot must not trigger that as a side effect of declaring a capability.
        """
        from gateway import session_context as sc

        reset_session_vars()
        engaged_before = sc._session_context_engaged
        try:
            sc.declare_stateless_channel()
            assert sc._session_context_engaged is engaged_before
        finally:
            reset_session_vars()


class TestStatelessChannelForcesSyncDelegation:
    """The behavioral contract: a stateless channel must run delegations INLINE.

    This is the regression that #53027 / #63142 describe — a background dispatch
    on a channel that can never deliver the completion.
    """

    def test_background_delegation_runs_inline_when_channel_is_stateless(
        self, monkeypatch
    ):
        import tools.delegate_tool as dt
        from gateway.session_context import declare_stateless_channel

        class _Parent:
            _delegate_depth = 0
            _subagent_id = None

        fake_child = type("C", (), {"_subagent_id": "s1"})()
        dispatched = []

        def _fake_dispatch(*a, **kw):
            dispatched.append(kw)
            return {"delegation_id": "deleg_x"}

        def _child(task_index, goal, child=None, parent_agent=None, **kw):
            return {
                "task_index": 0, "status": "completed", "summary": f"done: {goal}",
                "api_calls": 1, "duration_seconds": 0.1, "model": "m",
                "exit_reason": "completed",
            }

        creds = {
            "model": "m", "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "command": None, "args": None,
        }
        monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
        monkeypatch.setattr(dt, "_run_single_child", _child)
        monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
        monkeypatch.setattr(
            "tools.async_delegation.dispatch_async_delegation_batch", _fake_dispatch
        )

        reset_session_vars()
        try:
            declare_stateless_channel()
            out = dt.delegate_task(
                goal="review the spec", background=True, parent_agent=_Parent()
            )
        finally:
            reset_session_vars()

        parsed = json.loads(out)
        # The whole point: NOT dispatched to a channel that can't deliver.
        assert not dispatched, "stateless channel must not dispatch a detached child"
        assert parsed.get("status") != "dispatched"
        # The caller gets the actual work product, in-turn.
        assert "results" in parsed
        assert "done: review the spec" in json.dumps(parsed)


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
