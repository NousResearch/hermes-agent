"""One-shot runs (``hermes chat -q``/``-Q``) keep their post-turn review (#126417).

Covers: the exit linger that lets an in-flight review land, the deferred-queue flush at exit, the
opt-in one-shot learning mode (skill_manage kept, memory reviewed after a substantive turn), and ``-Q`` stdout staying clean when the review now
finishes inside the process.
"""

from __future__ import annotations

import threading
import types

import pytest

from agent import background_review as br
from agent import oneshot_footprint


@pytest.fixture
def oneshot(monkeypatch):
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")


@pytest.fixture
def interactive(monkeypatch):
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)


def _cfg(monkeypatch, **task):
    enabled = task.pop("enabled", True)
    monkeypatch.setattr(br, "load_background_review_settings", lambda: (enabled, dict(task)))
    monkeypatch.setattr(br, "_background_review_task_config", lambda task_cfg=None: task_cfg if isinstance(task_cfg, dict) else dict(task))


def _tools(*names):
    return [{"type": "function", "function": {"name": n}} for n in names]


# ── exit linger ───────────────────────────────────────────────────────────────


def test_drain_waits_for_in_flight_review_and_reports_completion(monkeypatch):
    _cfg(monkeypatch, linger_timeout_s=5)
    agent = types.SimpleNamespace(session_id="s1", _background_review_run=None)
    run = br.prepare_background_review_run(agent)
    threading.Timer(0.05, br.finish_background_review_run, args=(agent, run)).start()
    assert br.drain_background_review(agent) is True
    assert agent._background_review_run is None


def test_drain_is_bounded_and_zero_disables_it(monkeypatch):
    _cfg(monkeypatch, linger_timeout_s=0)
    agent = types.SimpleNamespace(session_id="s2", _background_review_run=None)
    br.prepare_background_review_run(agent)
    assert br.drain_background_review(agent) is False           # 0 = do not linger
    assert br.drain_background_review(agent, timeout=0.05) is False  # a stuck review is abandoned


def test_drain_without_a_review_returns_immediately(monkeypatch):
    _cfg(monkeypatch)
    assert br.drain_background_review(types.SimpleNamespace(session_id="s3", _background_review_run=None)) is False
    assert br.drain_background_review(None) is False


def test_drain_flushes_a_deferred_review_before_waiting(monkeypatch):
    """A review parked in the idle queue lives only in this process: dispatch it at exit."""
    from agent.review_idle_queue import QUEUE

    _cfg(monkeypatch, linger_timeout_s=1)
    spawned = []
    agent = types.SimpleNamespace(session_id="s4", _background_review_run=None)
    agent._spawn_background_review_now = lambda **kw: spawned.append(kw)
    item = types.SimpleNamespace(session_key="s4", agent=agent, kwargs={"review_memory": True})
    with QUEUE._lock:
        QUEUE._pending["s4"] = item
    monkeypatch.setattr(QUEUE, "_still_enabled", lambda _item: True)
    br.drain_background_review(agent)
    assert spawned == [{"review_memory": True}]
    assert QUEUE.pending_count() == 0


def test_hermes_z_teardown_waits_for_the_review_before_closing_the_agent(monkeypatch):
    """``hermes -z`` tears down in ``oneshot._close_agent`` and hard-exits, not via cli._run_cleanup."""
    import hermes_cli.oneshot as oneshot

    _cfg(monkeypatch, linger_timeout_s=5)
    monkeypatch.setattr(oneshot, "_linger_for_background_completions", lambda: None)
    order = []
    agent = types.SimpleNamespace(session_id="s5", _background_review_run=None)
    agent.shutdown_memory_provider = lambda *a: order.append("memory")
    agent.close = lambda: order.append("close")
    run = br.prepare_background_review_run(agent)

    def _finish():
        order.append("review")
        br.finish_background_review_run(agent, run)

    threading.Timer(0.05, _finish).start()
    oneshot._close_agent(agent, None)
    assert order == ["review", "memory", "close"]


# ── opt-in one-shot learning ──────────────────────────────────────────────────


def test_oneshot_keeps_skill_manage_only_when_learning_is_opted_in(oneshot, monkeypatch):
    names = lambda: {t["function"]["name"] for t in oneshot_footprint.prune_oneshot_tools(_tools("skill_manage", "terminal"))}
    _cfg(monkeypatch)
    assert names() == {"terminal"}
    _cfg(monkeypatch, oneshot_learning=True)
    assert names() == {"skill_manage", "terminal"}
    assert "skill_manage(action='patch')" in oneshot_footprint.oneshot_skills_guidance()


def test_interactive_sessions_are_untouched(interactive, monkeypatch):
    _cfg(monkeypatch, oneshot_learning=True)
    assert oneshot_footprint.oneshot_skills_guidance() == ""
    assert len(oneshot_footprint.prune_oneshot_tools(_tools("skill_manage"))) == 1
    assert br.oneshot_memory_review_due(types.SimpleNamespace(_skill_nudge_interval=10), 50) is False


@pytest.mark.parametrize(("cfg", "calls", "due"), [
    ({}, 40, False),                                                       # default: off
    ({"oneshot_learning": True}, 9, False),                                # trivial run: no fork
    ({"oneshot_learning": True}, 10, True),                                # substantive run
    ({"oneshot_learning": True, "oneshot_min_tool_calls": 3}, 3, True),
    ({"oneshot_learning": True, "enabled": False}, 40, False),             # reviews disabled wins
])
def test_oneshot_memory_review_due(oneshot, monkeypatch, cfg, calls, due):
    _cfg(monkeypatch, **cfg)
    assert br.oneshot_memory_review_due(types.SimpleNamespace(_skill_nudge_interval=10), calls) is due


def test_turn_tool_call_count_counts_batched_calls_of_this_turn_only():
    earlier = [{"role": "user", "content": "old"}, {"role": "assistant", "tool_calls": [1]},
               {"role": "tool", "content": "x"}, {"role": "assistant", "content": "done"}]
    batched = [{"role": "user", "content": "now"},
               {"role": "assistant", "tool_calls": list(range(11))}]
    batched += [{"role": "tool", "content": str(i)} for i in range(11)]
    batched += [{"role": "assistant", "content": "answer"}]
    assert br.turn_tool_call_count(earlier + batched) == 11   # one iteration, eleven calls
    assert br.turn_tool_call_count(earlier) == 1
    assert br.turn_tool_call_count([]) == 0


# ── -Q stdout stays clean ─────────────────────────────────────────────────────


def test_quiet_run_logs_review_summary_instead_of_printing(caplog):
    printed = []
    agent = types.SimpleNamespace(suppress_status_output=True, background_review_callback=None,
                                  _safe_print=lambda *a, **k: printed.append(a))
    with caplog.at_level("INFO", logger=br.logger.name):
        br._publish_review_summary(agent, ["Memory updated"])
    assert printed == [] and "Memory updated" in caplog.text
    agent.suppress_status_output = False
    br._publish_review_summary(agent, ["Memory updated"])
    assert len(printed) == 1
