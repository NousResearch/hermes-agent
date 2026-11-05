"""The interactive CLI must release the old AIAgent's LLM clients when it drops the instance
for a rebuild (/personality, /reasoning, /fast, model/route/credential change, MoA one-shot):
on the codex_app_server route the app-server child belongs to that instance and ``self.agent =
None`` alone orphans it for the CLI process lifetime (#72548)."""

import threading
from types import SimpleNamespace
from unittest.mock import patch

from agent import review_idle_queue
from hermes_cli.cli_agent_setup_mixin import _retire_agent
from hermes_cli.cli_commands_mixin import CLICommandsMixin


class _FakeAgent:
    def __init__(self):
        self.release_calls = 0

    def release_clients(self):
        self.release_calls += 1


def test_reasoning_command_releases_old_agent_clients_before_rebuild():
    agent = _FakeAgent()
    stub = SimpleNamespace(
        reasoning_config={"enabled": True, "effort": "medium"},
        show_reasoning=False,
        agent=agent,
    )
    with patch("cli.save_config_value"), patch("cli._cprint"):
        CLICommandsMixin._handle_reasoning_command(stub, "/reasoning high")
    assert stub.reasoning_config == {"enabled": True, "effort": "high"}
    assert stub.agent is None
    assert agent.release_calls == 1


def _review_parent():
    agent = _FakeAgent()
    agent._background_review_lock = threading.Lock()
    agent._background_review_run = None
    agent._background_review_agent = None
    return agent


def test_rebuild_retires_the_old_agents_deferred_reviews(monkeypatch):
    queue = review_idle_queue.ReviewIdleQueue()
    monkeypatch.setattr(queue, "_ensure_thread", lambda: None)
    monkeypatch.setattr(review_idle_queue, "QUEUE", queue)
    old = _review_parent()
    queue.enqueue(old, "cli-session", {})
    stub = SimpleNamespace(agent=old)
    _retire_agent(stub)
    assert stub.agent is None and old.release_calls == 1
    assert queue.pending_count() == 0, (
        "The discarded agent's queued review must be purged"
    )
    queue.enqueue(old, "cli-session", {})  # late preemption requeue from its worker
    assert queue.pending_count() == 0, "A discarded agent must not re-enter the queue"


def test_rebuild_still_releases_clients_when_review_retirement_fails(monkeypatch):
    def broken_purge(agent):
        raise RuntimeError("queue purge failed")

    monkeypatch.setattr(review_idle_queue.QUEUE, "discard_parent", broken_purge)
    old = _review_parent()
    stub = SimpleNamespace(agent=old)
    _retire_agent(stub)
    assert stub.agent is None and old.release_calls == 1
