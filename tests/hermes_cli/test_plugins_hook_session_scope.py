"""Bounded plugin-hook dispatch is scoped per session.

A hook callback is shared by every gateway session. Timeout suppression and the
abandoned-worker cap must therefore be keyed per session, or one session's hung
pre_tool_call worker fail-closes every other session's tool calls. The cap that
prevents the worker-thread leak (#98382) still holds per session, with a
per-callback backstop across sessions.
"""

import threading

from hermes_cli.plugins import PluginManager
from tests.waiting import wait_until


class TestHookSuppressionSessionScope:
    def test_timeout_in_one_session_does_not_block_another_session(self, monkeypatch):
        """A callback is shared by every gateway session: session A's hung pre_tool_call worker
        must not fail-close session B's tool calls (observed: one slow shell hook blocked every
        gateway session for minutes). Session A itself stays suppressed and fail-closed."""
        from hermes_cli.plugins import _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE

        monkeypatch.setattr(
            "hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.1
        )
        hold = threading.Event()
        starts = []

        def guard(**kwargs):
            starts.append(kwargs.get("session_id"))
            if kwargs.get("session_id") == "sess-a":
                hold.wait(timeout=10.0)  # session A's invocation hangs

        mgr = PluginManager()  # default 60s suppression window
        mgr._hooks["pre_tool_call"] = [guard]
        blocked = [{"action": "block", "message": _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE}]

        def fire(session_id, call_id):
            return mgr.invoke_hook("pre_tool_call", tool_name="read_file", args={},
                                   session_id=session_id, tool_call_id=call_id)

        try:
            assert fire("sess-a", "a-1") == blocked  # timed out
            assert fire("sess-b", "b-1") == []  # other session: runs and allows
            assert fire("sess-a", "a-2") == blocked  # A still suppressed (no new worker)
            assert fire("sess-b", "b-2") == []
            assert starts == ["sess-a", "sess-b", "sess-b"]
        finally:
            hold.set()

    def test_abandoned_worker_cap_is_per_session_with_global_backstop(self, monkeypatch):
        """The live-abandoned-worker cap that prevents a thread leak (#98382) applies per
        session; a separate per-callback backstop still bounds the total when many sessions
        hang at once."""
        import hermes_cli.plugins_dispatch as dispatch
        from hermes_cli.plugins import _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE

        monkeypatch.setattr(
            "hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.05
        )
        hold = threading.Event()
        starts = []

        def blocker(**kwargs):
            starts.append(kwargs.get("session_id"))
            hold.wait(timeout=10.0)

        mgr = PluginManager()
        mgr._hook_timeout_suppression_seconds = 0.0  # isolate the caps from suppression
        mgr._hooks["pre_tool_call"] = [blocker]
        blocked = [{"action": "block", "message": _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE}]
        per_session = dispatch._HOOK_MAX_ABANDONED_WORKERS
        backstop = dispatch._HOOK_MAX_ABANDONED_WORKERS_PER_CALLBACK
        assert backstop > per_session

        def fire(session_id, call_id):
            return mgr.invoke_hook("pre_tool_call", tool_name="terminal", args={},
                                   session_id=session_id, tool_call_id=call_id)

        try:
            for i in range(per_session + 2):  # session A hits its own cap
                assert fire("sess-a", f"a-{i}") == blocked
            wait_until(lambda: starts.count("sess-a") >= per_session, timeout=5,
                       what=f"{per_session} session-A workers to start")
            assert starts.count("sess-a") == per_session
            assert fire("sess-b", "b-0") == blocked  # B still gets a worker of its own
            wait_until(lambda: "sess-b" in starts, timeout=5, what="the session-B worker to start")
            assert starts.count("sess-b") == 1

            # Many hung sessions: the per-callback backstop stops new workers. Firing `backstop`
            # fresh sessions is more than enough to reach it from the workers already live.
            for n in range(backstop):
                assert fire(f"sess-{n}", f"x-{n}") == blocked
            wait_until(lambda: len(starts) >= backstop, timeout=5,
                       what=f"{backstop} live workers across sessions")
            assert len(starts) == backstop
            assert fire("sess-fresh", "fresh-0") == blocked
            assert len(starts) == backstop
        finally:
            hold.set()
