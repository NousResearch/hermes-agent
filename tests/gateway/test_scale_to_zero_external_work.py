"""Separate dashboard work must survive client disconnect and parent completion."""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def _dashboard_backend(home, mode, outcome, channel):
    """Run the real dashboard turn and delegation lifecycles without model/network calls."""
    os.environ["HERMES_HOME"] = home
    os.environ["HERMES_TEST_ISOLATION"] = "1"

    def no_network(event, _args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
            raise RuntimeError("network forbidden in isolated backend test")

    sys.addaudithook(no_network)
    from agent import relay_runtime
    from hermes_state import SessionDB
    from tests.agent.test_cross_process_turn_lease import _agent_with_db
    from tests.tui_gateway.test_auto_continue import _session
    from tools import async_delegation
    from tui_gateway import server

    release = threading.Event()
    entered = threading.Event()
    db = SessionDB(Path(home) / "state.db")
    parent = _agent_with_db(db, session_id="parent", platform="desktop")
    child = _agent_with_db(db, session_id="child", platform="subagent")
    child._parent_session_id = "parent"
    db.create_session("parent", source="desktop")
    db.create_session("child", source="subagent", parent_session_id="parent")
    session = _session(
        agent=parent, running=True, profile_home=home, session_key="parent"
    )

    def child_turn():
        if mode == "queued":
            entered.set()
            if not release.wait(30):
                raise TimeoutError("test did not release the queued worker")
        return child.run_conversation("inert task")

    def model_call(agent, *_args, **_kwargs):
        if mode != "dashboard" and agent is parent:
            handle = async_delegation.dispatch_async_delegation(
                goal="inert task",
                context=None,
                toolsets=None,
                role="worker",
                model=None,
                session_key="",
                parent_session_id="parent",
                runner=child_turn,
            )
            assert handle["status"] == "dispatched"
        else:
            entered.set()
            if not release.wait(30):
                raise TimeoutError("test did not release the model stub")
            if outcome == "error":
                raise RuntimeError("synthetic model failure")
        return {
            "final_response": "done",
            "messages": [],
            "failed": False,
            "interrupted": outcome == "interrupted",
        }

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                relay_runtime.HOST_REGISTRY,
                "for_profile",
                lambda profile_key=None, **_k: relay_runtime.NoopRelayRuntime(
                    profile_key=profile_key or home, reason="offline lifecycle test"
                ),
            )
            patch.setattr("agent.conversation_loop.run_conversation", model_call)
            patch.setattr(server, "_hermes_home", Path(home))
            patch.setattr(server, "write_json", lambda _frame: True)
            patch.setattr(server, "_wire_callbacks", lambda _sid: None)
            patch.setattr(server, "_sync_agent_model_with_config", lambda *_a: None)
            patch.setattr(server, "_session_cwd", lambda _session: home)
            patch.setattr(server, "_register_session_cwd", lambda _session: None)
            patch.setattr(server, "_tts_stream_begin", lambda: None)
            patch.setattr(
                server, "_sync_session_key_after_compress", lambda *_a, **_k: None
            )
            patch.setattr(server, "_get_usage", lambda _agent: {})
            patch.setattr(
                server,
                "_start_usage_ticker",
                lambda *_a: (threading.Event(), SimpleNamespace(join=lambda: None)),
            )
            submit = vars(server)["_run_prompt_submit"]
            assert submit("request", "view", session, "inert task")
            assert entered.wait(15), "real AIAgent turn never reached the model stub"
            if mode != "dashboard":
                from tests.waiting import wait_until

                assert wait_until(lambda: not session["running"], timeout=10)
            channel.send({"active": True, "parent_running": session["running"]})
            assert channel.recv() == "finish"
            release.set()
            from tests.waiting import wait_until

            assert wait_until(
                lambda: not session["running"] and async_delegation.active_count() == 0,
                timeout=15,
            )
            channel.send("finished")
            channel.recv()  # Keep the idle backend alive until the gateway checked cleanup.
    except BaseException as exc:
        channel.send({"error": repr(exc)})
        raise
    finally:
        release.set()
        async_delegation._reset_for_tests()
        db.close()
        channel.close()


def _idle_runner():
    return SimpleNamespace(
        _last_inbound_at=time.time() - 3600,
        _running_agent_count=lambda: 0,
        _running_cron_job_count=lambda: 0,
        _api_server_hook=lambda _name: 0,
        _relay_adapter_for_dormancy=lambda: None,
        _scale_to_zero_idle_timeout_seconds=lambda: 120,
        _scale_to_zero_has_live_background_work=lambda: False,
    )


@pytest.mark.parametrize("mode", ["dashboard", "delegated", "queued"])
@pytest.mark.parametrize("outcome", ["success", "error", "interrupted", "crash"])
def test_detached_backend_work_blocks_idle_until_finished(
    tmp_path, monkeypatch, mode, outcome
):
    from gateway.scale_to_zero import gateway_is_idle

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    context = multiprocessing.get_context("spawn")
    channel, worker_channel = context.Pipe()
    backend = context.Process(
        target=_dashboard_backend, args=(str(tmp_path), mode, outcome, worker_channel)
    )
    backend.start()
    worker_channel.close()
    try:
        assert channel.poll(25), "dashboard backend failed to start"
        state = channel.recv()
        assert state.get("active"), state
        assert state["parent_running"] is (mode == "dashboard")
        # No client heartbeat or gateway-owned work exists. The actual backend turn is live.
        busy = not gateway_is_idle(_idle_runner())
        if outcome == "crash":
            backend.terminate()
            backend.join(10)
            assert not backend.is_alive()
        else:
            channel.send("finish")
            assert channel.poll(20), "dashboard turn failed to settle"
            assert channel.recv() == "finished"
        assert gateway_is_idle(_idle_runner()), (
            "an idle live backend must not pin the machine"
        )
        assert busy, "gateway allowed suspend while a separate backend turn was running"
    finally:
        if backend.is_alive():
            channel.send("exit")
        backend.join(10)
        if backend.is_alive():
            backend.terminate()
            backend.join(5)
        channel.close()
    if outcome != "crash":
        assert backend.exitcode == 0
