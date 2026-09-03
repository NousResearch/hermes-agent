"""Deferred background review on the managed local runtime.

Behavior contracts for agent/review_idle_queue.py and the decision
wrapper in run_agent.AIAgent._spawn_background_review:

- defer: auto + review runtime == managed local  -> queued, not spawned
- defer: never, or non-managed runtime, or /refine -> immediate spawn
- queue coalesces per session (newest snapshot wins, age preserved)
- dispatch requires sustained process-quiet AND server idle
- aged-out items dispatch regardless of idleness (delay, never lose)
- preempted deferred reviews requeue with a bounded attempt cap
"""

import threading
import time
import types

import pytest

from agent.review_idle_queue import (
    ReviewIdleQueue,
    _IDLE_SETTLE_S,
    defer_max_age_s,
    defer_mode,
)


# ── config parsing ───────────────────────────────────────────────


def test_defer_mode_values():
    assert defer_mode(None) == "auto"
    assert defer_mode({}) == "auto"
    assert defer_mode({"defer": "auto"}) == "auto"
    assert defer_mode({"defer": "never"}) == "never"
    assert defer_mode({"defer": "NEVER"}) == "never"
    # Unknown values fall back to auto (the safe, documented default).
    assert defer_mode({"defer": "sometimes"}) == "auto"
    assert defer_mode({"defer": 3}) == "auto"


def test_defer_max_age_parsing():
    assert defer_max_age_s(None) == 30 * 60
    assert defer_max_age_s({"defer_max_age_s": 120}) == 120.0
    assert defer_max_age_s({"defer_max_age_s": "600"}) == 600.0
    # Nonsense and non-positive fall back to the default.
    assert defer_max_age_s({"defer_max_age_s": "soon"}) == 30 * 60
    assert defer_max_age_s({"defer_max_age_s": 0}) == 30 * 60
    assert defer_max_age_s({"defer_max_age_s": -5}) == 30 * 60


# ── queue harness ────────────────────────────────────────────────


class _FakeAgent:
    def __init__(self):
        self.spawned = []
        self.session_id = "sess-x"

    def _spawn_background_review_now(self, **kwargs):
        self.spawned.append(kwargs)


def _make_queue(now=None, server_idle=True):
    q = ReviewIdleQueue()
    clock = {"t": 0.0}
    if now is None:
        q._now = lambda: clock["t"]
    else:
        q._now = now
    q._server_idle = lambda: server_idle
    # Never start the real dispatcher thread in unit tests.
    q._ensure_thread = lambda: None
    return q, clock


def test_enqueue_coalesces_per_session_newest_wins_oldest_age():
    q, clock = _make_queue()
    agent = _FakeAgent()

    clock["t"] = 100.0
    q.enqueue(agent, "s1", {"messages_snapshot": ["old"], "task_cfg": {}})
    clock["t"] = 200.0
    q.enqueue(agent, "s1", {"messages_snapshot": ["new"], "task_cfg": {}})
    q.enqueue(agent, "s2", {"messages_snapshot": ["other"], "task_cfg": {}})

    assert q.pending_count() == 2
    with q._lock:
        item = q._pending["s1"]
    # Newest snapshot won, but the age clock kept the ORIGINAL enqueue
    # time so a busy session cannot push its own age-out forever.
    assert item.kwargs["messages_snapshot"] == ["new"]
    assert item.enqueued_at == 100.0


def test_dispatch_waits_for_sustained_quiet():
    q, clock = _make_queue()
    agent = _FakeAgent()
    q.enqueue(agent, "s1", {"task_cfg": {}})

    # A live turn: nothing dispatches.
    q.note_turn_started()
    assert q._pop_dispatchable() is None

    # Turn finished, but the settle window hasn't elapsed.
    q.note_turn_finished()
    assert q._pop_dispatchable() is None

    # Quiet long enough -> dispatchable.
    clock["t"] += _IDLE_SETTLE_S + 1
    item = q._pop_dispatchable()
    assert item is not None and item.session_key == "s1"
    assert q.pending_count() == 0


def test_dispatch_blocked_by_busy_server():
    q, clock = _make_queue(server_idle=False)
    agent = _FakeAgent()
    q.enqueue(agent, "s1", {"task_cfg": {}})
    q.note_turn_started()
    q.note_turn_finished()
    clock["t"] += _IDLE_SETTLE_S + 1
    # Process is quiet but the managed server has a processing slot
    # (another profile's session, a live prefill): hold.
    assert q._pop_dispatchable() is None
    assert q.pending_count() == 1


def test_aged_out_item_dispatches_despite_busy_server():
    q, clock = _make_queue(server_idle=False)
    agent = _FakeAgent()
    q.enqueue(agent, "s1", {"task_cfg": {"defer_max_age_s": 60}})
    q.note_turn_started()  # never goes quiet
    clock["t"] += 61
    item = q._pop_dispatchable()
    assert item is not None
    assert item.session_key == "s1"


def test_new_turn_resets_the_quiet_clock():
    q, clock = _make_queue()
    agent = _FakeAgent()
    q.enqueue(agent, "s1", {"task_cfg": {}})
    q.note_turn_started()
    q.note_turn_finished()
    clock["t"] += _IDLE_SETTLE_S - 2
    # A new prompt arrives just before the settle window closes.
    q.note_turn_started()
    clock["t"] += 30
    assert q._pop_dispatchable() is None  # still live
    q.note_turn_finished()
    assert q._pop_dispatchable() is None  # settle restarts
    clock["t"] += _IDLE_SETTLE_S + 1
    assert q._pop_dispatchable() is not None


def test_nested_turns_require_all_to_finish():
    q, clock = _make_queue()
    agent = _FakeAgent()
    q.enqueue(agent, "s1", {"task_cfg": {}})
    q.note_turn_started()
    q.note_turn_started()
    q.note_turn_finished()
    clock["t"] += _IDLE_SETTLE_S + 1
    assert q._pop_dispatchable() is None  # one turn still live
    q.note_turn_finished()
    clock["t"] += _IDLE_SETTLE_S + 1
    assert q._pop_dispatchable() is not None


# ── the decision wrapper ─────────────────────────────────────────


def _wrapper_agent(monkeypatch, defer="auto", managed=True):
    """A minimal object wearing the real _spawn_background_review."""
    import run_agent
    from agent import review_idle_queue as riq

    agent = _FakeAgent()
    agent._delegate_depth = 0
    calls = {"enqueued": [], "spawned": []}

    monkeypatch.setattr(
        "agent.background_review.load_background_review_settings",
        lambda: (True, {"defer": defer}),
    )
    monkeypatch.setattr(
        riq, "review_targets_managed_local", lambda a, cfg: managed
    )
    monkeypatch.setattr(
        riq.QUEUE, "enqueue",
        lambda a, key, kw: calls["enqueued"].append((key, kw)),
    )
    agent._spawn_background_review_now = (
        lambda **kw: calls["spawned"].append(kw)
    )
    bound = types.MethodType(run_agent.AIAgent._spawn_background_review, agent)
    return bound, calls


def test_wrapper_defers_managed_local_auto(monkeypatch):
    spawn, calls = _wrapper_agent(monkeypatch, defer="auto", managed=True)
    spawn([{"role": "user", "content": "hi"}], review_memory=True)
    assert len(calls["enqueued"]) == 1
    assert calls["spawned"] == []
    key, kwargs = calls["enqueued"][0]
    assert key == "sess-x"
    assert kwargs["review_memory"] is True


def test_wrapper_spawns_immediately_for_non_managed(monkeypatch):
    spawn, calls = _wrapper_agent(monkeypatch, defer="auto", managed=False)
    spawn([{"role": "user", "content": "hi"}], review_skills=True)
    assert calls["enqueued"] == []
    assert len(calls["spawned"]) == 1


def test_wrapper_defer_never_is_old_behavior(monkeypatch):
    spawn, calls = _wrapper_agent(monkeypatch, defer="never", managed=True)
    spawn([{"role": "user", "content": "hi"}], review_memory=True)
    assert calls["enqueued"] == []
    assert len(calls["spawned"]) == 1


def test_wrapper_refine_bypasses_queue(monkeypatch):
    spawn, calls = _wrapper_agent(monkeypatch, defer="auto", managed=True)
    spawn([{"role": "user", "content": "hi"}], review_memory=True,
          focus="save the deploy workflow")
    assert calls["enqueued"] == []
    assert len(calls["spawned"]) == 1
    assert calls["spawned"][0]["focus"] == "save the deploy workflow"


def test_wrapper_bare_refine_bypasses_queue(monkeypatch):
    """/refine with no focus text is still explicit: never deferred."""
    spawn, calls = _wrapper_agent(monkeypatch, defer="auto", managed=True)
    spawn([{"role": "user", "content": "hi"}], review_memory=True,
          focus=None, explicit=True)
    assert calls["enqueued"] == []
    assert len(calls["spawned"]) == 1


def test_wrapper_cloud_fast_path_skips_runtime_resolution(monkeypatch):
    """No managed server on the machine -> the classifier answers from the
    TTL-cached netloc probe alone, without resolving the review runtime.
    Guards the cloud-only turn tail from growing new work."""
    from agent import review_idle_queue as riq

    resolved = {"count": 0}

    def _explode(agent, cfg):
        resolved["count"] += 1
        raise AssertionError("runtime resolution must not run")

    monkeypatch.setattr(
        "agent.auxiliary_client._managed_local_netloc", lambda: "")
    monkeypatch.setattr(
        "agent.background_review._resolve_review_runtime", _explode)
    assert riq.review_targets_managed_local(object(), {}) is False
    assert resolved["count"] == 0


def test_dispatcher_rechecks_enabled_gate(monkeypatch):
    """A review disabled while queued must not be resurrected at dispatch."""
    from agent import review_idle_queue as riq

    q, clock = _make_queue()
    agent = _FakeAgent()
    q.enqueue(agent, "s1", {"task_cfg": {}})
    monkeypatch.setattr(
        "agent.background_review.load_background_review_settings",
        lambda: (False, {}),
    )
    item = None
    clock["t"] += _IDLE_SETTLE_S + 1
    q.note_turn_started()
    q.note_turn_finished()
    clock["t"] += _IDLE_SETTLE_S + 1
    item = q._pop_dispatchable()
    assert item is not None
    assert q._still_enabled(item) is False


# ── requeue on preemption ────────────────────────────────────────


class _Run:
    def __init__(self, cancelled):
        self.cancel_requested = threading.Event()
        if cancelled:
            self.cancel_requested.set()


def _requeue_agent(monkeypatch, managed=True):
    import run_agent
    from agent import review_idle_queue as riq

    agent = _FakeAgent()
    calls = {"enqueued": []}
    monkeypatch.setattr(
        riq, "review_targets_managed_local", lambda a, cfg: managed
    )
    monkeypatch.setattr(
        riq.QUEUE, "enqueue",
        lambda a, key, kw: calls["enqueued"].append(kw),
    )
    agent._REVIEW_REQUEUE_MAX_ATTEMPTS = (
        run_agent.AIAgent._REVIEW_REQUEUE_MAX_ATTEMPTS
    )
    bound = types.MethodType(
        run_agent.AIAgent._maybe_requeue_preempted_review, agent
    )
    return bound, calls


def test_preempted_review_requeues(monkeypatch):
    requeue, calls = _requeue_agent(monkeypatch)
    requeue(_Run(cancelled=True),
            {"task_cfg": {"defer": "auto"}, "focus": None,
             "_requeue_attempts": 1})
    assert len(calls["enqueued"]) == 1
    # The attempt counter rides along so the cap survives the round trip.
    assert calls["enqueued"][0]["_requeue_attempts"] == 1


def test_completed_review_does_not_requeue(monkeypatch):
    requeue, calls = _requeue_agent(monkeypatch)
    requeue(_Run(cancelled=False),
            {"task_cfg": {"defer": "auto"}, "focus": None,
             "_requeue_attempts": 1})
    assert calls["enqueued"] == []


def test_requeue_attempt_cap(monkeypatch):
    requeue, calls = _requeue_agent(monkeypatch)
    requeue(_Run(cancelled=True),
            {"task_cfg": {"defer": "auto"}, "focus": None,
             "_requeue_attempts": 4})
    assert calls["enqueued"] == []


def test_requeue_skips_non_managed(monkeypatch):
    requeue, calls = _requeue_agent(monkeypatch, managed=False)
    requeue(_Run(cancelled=True),
            {"task_cfg": {"defer": "auto"}, "focus": None,
             "_requeue_attempts": 1})
    assert calls["enqueued"] == []
