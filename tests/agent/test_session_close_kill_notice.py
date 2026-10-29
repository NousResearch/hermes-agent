"""A session-level close that kills a job the agent promised to report on must say so.

``tui_gateway._teardown_session`` stamps the end reason on the agent before ``close()``. The kill of a
``notify_on_complete`` job then keeps its completion event (instead of consuming it) so the session is
told the job was stopped unfinished, and every lifecycle kill is logged with its reason. A close with
no reason (a delegated child's teardown) keeps consuming: the child's summary is the deliverable.
"""
import json
import logging
import shlex
import sys

import pytest

from agent.turn_context import _bind_turn_identity
from run_agent import AIAgent
from tools.process_registry import ProcessRegistry
from tools.process_registry_notifications import format_process_notification
from tools.terminal_tool import terminal_tool


def _agent():
    return AIAgent(api_key="test", base_url="http://127.0.0.1:9/v1",
                   provider="openai-compat", model="test", enabled_toolsets=[],
                   quiet_mode=True, skip_context_files=True, skip_memory=True)


def _spawn(agent, task_id, tmp_path, *, notify=True):
    _bind_turn_identity(agent, task_id, None, None, None, None)
    command = shlex.quote(sys.executable) + " -c " + shlex.quote("import time; time.sleep(60)")
    result = json.loads(terminal_tool(command, background=True, task_id=task_id,
                                     workdir=str(tmp_path), notify_on_complete=notify))
    return result["session_id"]


@pytest.fixture()
def registry(monkeypatch):
    import tools.process_registry as processes
    reg = ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", reg)
    yield reg
    reg.kill_all()


def _queued(registry):
    events = []
    while not registry.completion_queue.empty():
        events.append(registry.completion_queue.get_nowait())
    return events


def test_close_keeps_notice_only_for_a_session_end_of_a_notify_job(registry, tmp_path, caplog):
    # One test, agents spawned up front: like test_owned_process_cleanup, the agents share one
    # terminal environment, and a previous test's environment teardown would reap these processes.
    session, quiet, child = _agent(), _agent(), _agent()
    try:
        promised = _spawn(session, "session-owner", tmp_path)
        unpromised = _spawn(quiet, "quiet-owner", tmp_path, notify=False)
        delegated = _spawn(child, "sa-child-owner", tmp_path)
        session._close_reason = "lru_evict"
        quiet._close_reason = "tui_close"
        with caplog.at_level(logging.INFO, logger="run_agent"):
            for agent in (session, quiet, child):
                agent.close()

        assert all(registry.poll(p)["status"] != "running" for p in (promised, unpromised, delegated))
        # Session end of a promised job: the completion survives and says it did not finish.
        assert promised not in registry._completion_consumed
        [event] = [e for e in _queued(registry) if e.get("session_id") == promised]
        assert event["termination_source"] == "agent_close:lru_evict"
        text = format_process_notification(event) or ""
        assert "was STOPPED by Hermes before it finished" in text and "lru_evict" in text
        assert "did NOT finish" in text and "Stopped:" in text
        # Nothing promised, or a delegated child's teardown: consumed as before.
        assert {unpromised, delegated} <= registry._completion_consumed
        logs = [r.getMessage() for r in caplog.records if "Lifecycle kill" in r.getMessage()]
        assert any(promised in m and "reason=lru_evict" in m and "notice_kept=True" in m for m in logs)
        assert any(unpromised in m and "reason=tui_close" in m and "notice_kept=False" in m for m in logs)
        assert any(delegated in m and "reason=-" in m and "notice_kept=False" in m for m in logs)
    finally:
        for agent in (session, quiet, child):
            agent.close()


def test_other_kill_sources_keep_their_wording():
    event = {"session_id": "proc_x", "command": "make", "exit_code": -15,
             "completion_reason": "killed", "termination_source": "process.kill"}
    text = format_process_notification(event) or ""
    assert "terminated by process.kill" in text and "did NOT finish" not in text
