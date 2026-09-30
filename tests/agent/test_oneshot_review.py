"""One-shot runs (``hermes chat -q``/``-Q``) keep their post-turn review (#126417).

Covers: the exit linger that lets an in-flight review land, the deferred-queue flush at exit, and ``-Q`` stdout staying clean when the review now
finishes inside the process.
"""

from __future__ import annotations

import threading
import types

import pytest

from agent import background_review as br


def _cfg(monkeypatch, **task):
    enabled = task.pop("enabled", True)
    monkeypatch.setattr(br, "load_background_review_settings", lambda: (enabled, dict(task)))
    monkeypatch.setattr(br, "_background_review_task_config", lambda task_cfg=None: task_cfg if isinstance(task_cfg, dict) else dict(task))


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
