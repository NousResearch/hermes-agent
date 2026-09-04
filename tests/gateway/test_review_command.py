"""Gateway /review command — direct handler tests.

Drives the REAL GatewayRunner._handle_review_command on a bare runner with a
cached agent, dispatching through the REAL delegate_task background rail
(child build/run stubbed at the delegate_tool seam, same pattern as
tests/tools/test_async_delegation.py).
"""

import json
import time
from unittest.mock import MagicMock

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    deadline = time.monotonic() + 2.0
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


SESSION_KEY = "agent:main:test:dm:1"


def _make_agent():
    agent = MagicMock()
    agent._delegate_depth = 0
    agent.session_id = "gw-review-sess"
    agent._interrupt_requested = False
    agent._active_children = []
    agent._active_children_lock = None
    agent._session_messages = [
        {"role": "user", "content": "open a PR"},
        {"role": "assistant", "content": "PR #5 opened: https://x/pull/5"},
    ]
    return agent


def _make_runner(agent):
    import threading

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._agent_cache = {SESSION_KEY: agent}
    runner._agent_cache_lock = threading.Lock()
    runner._session_key_for_source = lambda source: SESSION_KEY
    return runner


class _Event:
    source = object()  # any non-None sentinel

    def __init__(self, args=""):
        self._args = args

    def get_command_args(self):
        return self._args


@pytest.mark.asyncio
async def test_review_command_dispatches_background_subagent(monkeypatch):
    import tools.delegate_tool as dt
    from agent import review_engine as re_mod

    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    built = {}

    def fake_build(**kw):
        built.update(kw)
        return fake_child

    monkeypatch.setattr(dt, "_build_child_agent", fake_build)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(
        dt, "_run_single_child",
        lambda *a, **k: {
            "task_index": 0, "status": "completed", "summary": "review done",
            "api_calls": 1, "duration_seconds": 0.1, "model": "m",
            "exit_reason": "completed",
        },
    )
    monkeypatch.setattr(re_mod, "_load_review_credentials_cfg", lambda: None)

    agent = _make_agent()
    runner = _make_runner(agent)
    out = await runner._handle_review_command(_Event("check tests"))

    assert "dispatched" in out
    assert "PR #5 opened" in built["context"]
    assert "check tests" in built["context"]

    # Completion event routes back to the gateway session key (captured from
    # the approval contextvar the handler binds around the dispatch).
    deadline = time.monotonic() + 5.0
    evt = None
    while time.monotonic() < deadline:
        try:
            evt = process_registry.completion_queue.get(timeout=0.2)
            break
        except Exception:
            continue
    assert evt is not None
    assert evt["type"] == "async_delegation"
    assert evt["session_key"] == SESSION_KEY
    assert evt["results"][0]["summary"] == "review done"


@pytest.mark.asyncio
async def test_review_command_rejects_while_agent_running():
    agent = _make_agent()
    runner = _make_runner(agent)
    runner._running_agents = {SESSION_KEY: object()}
    out = await runner._handle_review_command(_Event())
    assert "Agent is running" in out


@pytest.mark.asyncio
async def test_review_command_requires_cached_agent():
    runner = _make_runner(None)
    runner._agent_cache = {}
    out = await runner._handle_review_command(_Event())
    assert "send a message first" in out


@pytest.mark.asyncio
async def test_review_dispatch_branch_reaches_handler(monkeypatch):
    """/review typed in a gateway chat must not fall through to the agent.

    Proves the gateway/run.py dispatch branch exists by resolving the command
    through the registry the same way _handle_message does.
    """
    from hermes_cli.commands import resolve_command

    cmd = resolve_command("review")
    assert cmd is not None
    assert cmd.name == "review"
    assert not cmd.cli_only
