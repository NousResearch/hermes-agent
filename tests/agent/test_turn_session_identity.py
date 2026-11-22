"""Cached native turns publish their current identity through real worker/export bridges.

The conversation-loop seam prevents inference; the facade, scope, tool-thread propagation,
compression caller rebind, environment exporter and subprocess are production implementations.
"""
import asyncio
import contextvars
import json
import os
import socket
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.compression_facade import _rebind_caller_session_context
from gateway import session_context as sc
from run_agent import AIAgent
from tests.agent.test_cross_process_turn_lease import _agent_with_db
from tools.environments.local import hermes_subprocess_env
from tools.thread_context import propagate_context_to_thread

KEYS = ("HERMES_SESSION_ID", "HERMES_SESSION_KEY", "HERMES_UI_SESSION_ID")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network/inference forbidden in identity tests")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def _bridge():
    """Real propagated tool worker -> real exporter -> executable child, no bridge mocks."""
    def worker():
        from tools.approval_context import get_current_session_key
        env = hermes_subprocess_env(base_env={"PATH": os.defpath})
        output = subprocess.check_output(
            [sys.executable, "-I", "-c",
             "import json,os; print(json.dumps([os.environ.get(k) for k in "
             + repr(KEYS) + "]))"], env=env, text=True, timeout=10)
        return [sc.get_session_env(k) for k in KEYS], json.loads(output), get_current_session_key()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(propagate_context_to_thread(worker)).result(timeout=15)


def _scope(route):
    sc.set_session_vars(session_id="stale-ui-root", session_key=route, ui_session_id=route)
    from tools.approval_context import set_current_session_key
    set_current_session_key(route)


def _assert_bridge(native, route):
    direct, child, approval = _bridge()
    assert direct == [native, route, route]
    assert child == [native, route, route]
    assert approval == route


@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
def test_cached_turn_binds_current_native_tip_and_restores_caller(monkeypatch, failure):
    def scenario():
        route = "browser-route"
        _scope(route)
        agent = _agent_with_db(None, session_id="native-tip-1", platform="webui")
        mirror = os.environ.get("HERMES_SESSION_ID")
        seen = []

        def loop(current, *args, **kwargs):
            _assert_bridge(current.session_id, route)
            seen.append(current.session_id)
            if failure:
                raise failure("synthetic exit")
            return {"final_response": "ok", "messages": []}

        monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
        for tip in ("native-tip-1", "native-tip-2"):
            agent.session_id = tip  # same cached object, no constructor re-publication
            if failure:
                with pytest.raises(failure):
                    agent.run_conversation("synthetic prompt")
            else:
                agent.run_conversation("synthetic prompt")
            assert sc.get_session_env("HERMES_SESSION_ID") == "stale-ui-root"
            assert os.environ.get("HERMES_SESSION_ID") == mirror
        assert seen == ["native-tip-1", "native-tip-2"]
    contextvars.Context().run(scenario)


def test_compression_tip_wins_for_later_tools_and_next_cached_turn(monkeypatch):
    def scenario():
        route = "browser-route"
        _scope(route)
        agent = _agent_with_db(None, session_id="native-before", platform="webui")
        seen = []
        def loop(current, *args, **kwargs):
            _assert_bridge(current.session_id, route)
            seen.append(current.session_id)
            current.session_id = "native-after"
            # Existing compressor rebind, not a test-only setter. Delegated scope prevents its
            # legacy os.environ mirror from changing the process while preserving native behavior.
            from agent.delegation_context import _DELEGATED_CHILD_CONTEXT
            token = _DELEGATED_CHILD_CONTEXT.set(True)
            try:
                _rebind_caller_session_context(current)
            finally:
                _DELEGATED_CHILD_CONTEXT.reset(token)
            _assert_bridge("native-after", route)
            return {"final_response": "ok", "messages": []}
        monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
        agent.run_conversation("compress")
        assert sc.get_session_env("HERMES_SESSION_ID") == "stale-ui-root"
        agent.run_conversation("reuse")
        assert seen == ["native-before", "native-after"]
    contextvars.Context().run(scenario)


def test_parallel_cached_sessions_keep_native_route_and_ui_isolated(monkeypatch):
    barrier = threading.Barrier(2, timeout=15)
    def loop(current, *args, **kwargs):
        barrier.wait()
        _assert_bridge(current.session_id, "route-" + current.session_id)
        return {"final_response": "ok", "messages": []}
    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    def scenario(native):
        _scope("route-" + native)
        agent = _agent_with_db(None, session_id=native, platform="webui")
        agent.run_conversation("parallel")
        assert sc.get_session_env("HERMES_SESSION_ID") == "stale-ui-root"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(contextvars.Context().run, scenario, native) for native in ("alpha", "beta")]
        for future in futures:
            future.result(timeout=30)
