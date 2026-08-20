from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast


def _agent_stub(decision, policy):
    agent = SimpleNamespace(
        session_id="bg-lifecycle-test",
        _self_improvement_decision=decision,
        session_write_policy=policy,
    )
    return agent


def _assert_thread_not_constructed(monkeypatch, agent):
    import run_agent
    from run_agent import AIAgent

    def _fail_thread(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("threading.Thread must not be constructed on DENY")

    monkeypatch.setattr(run_agent.threading, "Thread", _fail_thread)
    AIAgent._spawn_background_review(
        cast(Any, agent),
        messages_snapshot=[{"role": "user", "content": "probe"}],
        review_memory=True,
        review_skills=True,
    )


def test_env_disabled_does_not_construct_thread(monkeypatch):
    from agent.self_improvement_policy import DENY_ENV_DISABLED, Decision
    from agent.session_write_policy import SessionWritePolicy

    agent = _agent_stub(
        Decision(result=DENY_ENV_DISABLED, reason="env disabled"),
        SessionWritePolicy.normal(session_id="env", origin="test"),
    )
    _assert_thread_not_constructed(monkeypatch, agent)


def test_read_only_does_not_construct_thread(monkeypatch):
    from agent.self_improvement_policy import DENY_READ_ONLY_SESSION, Decision
    from agent.session_write_policy import SessionWritePolicy

    agent = _agent_stub(
        Decision(result=DENY_READ_ONLY_SESSION, reason="read only"),
        SessionWritePolicy.normal(session_id="ro", origin="test"),
    )
    _assert_thread_not_constructed(monkeypatch, agent)


def test_deny_all_session_policy_does_not_construct_thread(monkeypatch):
    from agent.self_improvement_policy import ALLOW, Decision
    from agent.session_write_policy import SessionWritePolicy

    agent = _agent_stub(
        Decision(result=ALLOW, reason="allow"),
        SessionWritePolicy.deny_all(session_id="deny", origin="test"),
    )
    _assert_thread_not_constructed(monkeypatch, agent)


def test_missing_context_does_not_construct_thread(monkeypatch):
    agent = SimpleNamespace(session_id="missing")
    _assert_thread_not_constructed(monkeypatch, agent)


def test_allow_constructs_and_starts_thread_once(monkeypatch):
    import run_agent
    from agent.self_improvement_policy import ALLOW, Decision
    from agent.session_write_policy import SessionWritePolicy
    from run_agent import AIAgent

    constructed = []
    starts = []

    class FakeThread:
        def __init__(self, *, target, daemon, name):
            constructed.append({"target": target, "daemon": daemon, "name": name})

        def start(self):
            starts.append(True)

    monkeypatch.setattr(run_agent.threading, "Thread", FakeThread)
    agent = _agent_stub(
        Decision(result=ALLOW, reason="allow"),
        SessionWritePolicy.normal(session_id="allow", origin="test"),
    )

    AIAgent._spawn_background_review(
        cast(Any, agent),
        messages_snapshot=[{"role": "user", "content": "probe"}],
        review_memory=True,
        review_skills=True,
    )

    assert len(constructed) == 1
    assert constructed[0]["daemon"] is True
    assert constructed[0]["name"] == "bg-review"
    assert callable(constructed[0]["target"])
    assert len(starts) == 1
