"""Blockers from maintainer review 5252598219 on PR #92494 head 9d6d337.

Two reported task-settlement defects, each exercised through the real
consumer seam (not source-shape assertions):

1. ``_durable_complete_pending`` durably publishes COMPLETED but resolves
   the waiter only after the synchronous post-commit tail (callback
   delivery).  A callback that blocks past the waiter's reply deadline
   lets the waiter finalize FAILED against already-durable COMPLETED and
   surface a false ``DurablePublishError`` for a valid result.

2. Bounded terminal retention evicts the durable row; a stale second
   ``TaskStore`` then re-adds its stale WORKING copy during merge and can
   complete the task again with a different reply, repeating terminal
   side effects.  Proven with the reported ``_MAX_TERMINAL=1``
   interleaving (complete t, complete u to evict t, publish from stale
   store B) plus restart/readback.
"""

from __future__ import annotations

import json
import shutil
import threading
import time

from gateway.config import PlatformConfig
from plugins.platforms.a2a import protocol
from plugins.platforms.a2a.adapter import A2AAdapter


def _working_record(task_id: str, context_id: str, *, created_at: float | None = None) -> dict:
    return {
        "task_id": task_id,
        "context_id": context_id,
        "peer": "peer-a",
        "agent_slug": "",
        "tenant": "",
        "state": protocol.STATE_WORKING,
        "reply": "",
        "created_at": time.time() if created_at is None else created_at,
        "created_iso": protocol.now_iso(),
        "push_url": "",
        "push_config_id": "",
    }


def _completed_candidate(base: dict, reply: str, completed_at: float) -> dict:
    candidate = dict(base)
    candidate["state"] = protocol.STATE_COMPLETED
    candidate["reply"] = reply
    candidate["completed_at"] = completed_at
    return candidate


def _settle_eviction_fixture(ledger, tmp_path) -> tuple[dict, dict, "protocol.TaskStore"]:
    """Build the reported interleaving: t completed, then u completed so the
    ``_MAX_TERMINAL=1`` window evicts t's durable row.  Returns
    ``(t_working, stale_snapshot_path, store_a)`` where the snapshot is a
    point-in-time ledger copy that still holds t as WORKING."""
    now = time.time()
    stale_snapshot = tmp_path / "stale_a2a_task_ledger.json"
    store_a = protocol.TaskStore()

    t_working = _working_record("t", "ctx-t", created_at=now - 100)
    assert store_a.publish_durable(ledger, "t", t_working).published
    shutil.copy(ledger, stale_snapshot)
    assert store_a.publish_durable(
        ledger, "t", _completed_candidate(t_working, "original answer", now - 10)
    ).published

    u_working = _working_record("u", "ctx-u", created_at=now - 90)
    assert store_a.publish_durable(ledger, "u", u_working).published
    assert store_a.publish_durable(
        ledger, "u", _completed_candidate(u_working, "u reply", now - 5)
    ).published
    return t_working, stale_snapshot, store_a


def test_delayed_callback_past_waiter_deadline_still_settles_successfully(
    monkeypatch, tmp_path
) -> None:
    """A callback blocking past the waiter deadline must not turn durable
    COMPLETED into a false timeout/FAILED/DurablePublishError."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("A2A_REPLY_TIMEOUT", "1")

    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
    ledger = tmp_path / "a2a_task_ledger.json"
    violations: list[str] = []

    record = _working_record("task-slow-callback", "ctx-task-slow", created_at=time.time() - 0.95)
    assert adapter.tasks.publish_durable(ledger, record["task_id"], record).published
    future = adapter._add_pending(record["task_id"], record["context_id"])

    callback_started = threading.Event()
    release_callback = threading.Event()

    def blocking_push(task_id, context_id, reply, state) -> None:
        if not future.done():
            violations.append(
                "waiter future still unresolved when callback delivery started "
                f"(durable {state!r} already committed)"
            )
        callback_started.set()
        release_callback.wait(timeout=10)

    monkeypatch.setattr(adapter, "_send_push_notification", blocking_push)
    monkeypatch.setattr(protocol, "persist_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(adapter, "_audit_safe", lambda *args, **kwargs: None, raising=False)

    outcome: dict = {}

    def complete() -> None:
        outcome["ok"], outcome["err"] = adapter._durable_complete_pending(
            record["task_id"], record["context_id"], "real answer", "message-1"
        )

    worker = threading.Thread(target=complete, daemon=True, name="durable-complete")
    try:
        worker.start()
        assert callback_started.wait(timeout=10), "callback delivery never started"

        # The waiter's seam: _rpc_message_send awaits, then finalizes whatever
        # _await_reply returns — exactly the path that produced the false error.
        waiter_pending = {
            "task_id": record["task_id"],
            "context_id": record["context_id"],
            "peer": record["peer"],
            "future": future,
            "created_iso": record["created_iso"],
            "started": record["created_at"],
        }
        state, reply, _oob, _defer = adapter._await_reply(waiter_pending, patience=None)
        if state != protocol.STATE_COMPLETED or reply != "real answer":
            violations.append(
                f"waiter settled as {state!r}/{reply!r} instead of the durable success"
            )
        try:
            finalized = adapter._finalize_task(waiter_pending, state, reply)
        except protocol.DurablePublishError as exc:
            violations.append(f"false DurablePublishError after durable COMPLETED: {exc}")
            finalized = None
        if finalized is not None and finalized != (protocol.STATE_COMPLETED, "real answer"):
            violations.append(f"waiter finalization returned {finalized!r}")

        durable = adapter.tasks.get(record["task_id"])
        if (
            durable is None
            or durable["state"] != protocol.STATE_COMPLETED
            or durable["reply"] != "real answer"
        ):
            violations.append(f"durable record is not the valid success: {durable!r}")
    finally:
        release_callback.set()
        worker.join(timeout=10)

    if not outcome.get("ok", False):
        violations.append(f"durable completion reported failure: {outcome.get('err')!r}")

    assert violations == []


def test_evicted_terminal_task_cannot_resurrect_or_complete_divergently(
    monkeypatch, tmp_path
) -> None:
    """After the terminal row leaves the bounded window, a stale store must
    not resurrect it as WORKING or settle it with a different reply —
    including across a restart."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(protocol.TaskStore, "_MAX_TERMINAL", 1)
    ledger = tmp_path / "a2a_task_ledger.json"
    violations: list[str] = []
    now = time.time()

    t_working, stale_snapshot, _store_a = _settle_eviction_fixture(ledger, tmp_path)

    disk = json.loads(ledger.read_text())
    if "t" in disk:
        violations.append(f"t was not evicted by the _MAX_TERMINAL=1 window: {disk['t']!r}")

    # Stale store B holds t=WORKING via the real restore seam.
    store_b = protocol.TaskStore()
    store_b.restore(stale_snapshot)
    b_rec = store_b.get("t")
    if b_rec is None or b_rec["state"] != protocol.STATE_WORKING:
        violations.append(f"stale store B should hold t=WORKING, got {b_rec!r}")

    # Positive control: a genuinely new task from B must still publish, and
    # that publish must not re-add B's stale copy of the evicted t.
    v_working = _working_record("v", "ctx-v", created_at=now - 1)
    if not store_b.publish_durable(ledger, "v", v_working).published:
        violations.append("genuine new task v from stale store failed to publish")
    disk = json.loads(ledger.read_text())
    if "v" not in disk:
        violations.append("genuine new task v missing from durable ledger")
    stale_t = disk.get("t")
    if stale_t is not None and stale_t.get("state") not in protocol.TERMINAL_STATES:
        violations.append(
            f"evicted terminal t resurrected as {stale_t.get('state')!r} by stale store publish"
        )

    # B must not settle t again with a divergent reply.
    divergent = _completed_candidate(t_working, "different answer", now - 4)
    outcome = store_b.publish_durable(ledger, "t", divergent)
    if outcome.published:
        violations.append("stale store durably re-completed evicted task t with a different reply")
    if outcome.durable_state != protocol.STATE_COMPLETED:
        violations.append(
            f"divergent completion reported durable_state={outcome.durable_state!r}, want COMPLETED"
        )
    disk = json.loads(ledger.read_text())
    stale_t = disk.get("t")
    if stale_t is not None and (
        stale_t.get("state") not in protocol.TERMINAL_STATES
        or stale_t.get("reply") != "original answer"
    ):
        violations.append(f"durable ledger now holds divergent t: {stale_t!r}")

    # Restart/readback: the authoritative side restarts from the surviving ledger.
    store_c = protocol.TaskStore()
    store_c.restore(ledger)
    rec_c = store_c.get("t")
    if rec_c is not None and rec_c["state"] not in protocol.TERMINAL_STATES:
        violations.append(f"t returned to {rec_c['state']!r} after restart")

    # A still-stale store stays locked out after the restart, too.
    store_d = protocol.TaskStore()
    store_d.restore(stale_snapshot)
    outcome_d = store_d.publish_durable(ledger, "t", divergent)
    if outcome_d.published:
        violations.append("post-restart stale store re-completed evicted task t")
    w_working = _working_record("w", "ctx-w", created_at=now)
    store_d.publish_durable(ledger, "w", w_working)
    disk = json.loads(ledger.read_text())
    stale_t = disk.get("t")
    if stale_t is not None and stale_t.get("state") not in protocol.TERMINAL_STATES:
        violations.append(
            f"post-restart publish resurrected t as {stale_t.get('state')!r}"
        )

    assert violations == []


def test_stale_adapter_terminal_finalize_after_eviction_is_rejected_without_side_effects(
    monkeypatch, tmp_path
) -> None:
    """The consumer seam: an adapter still holding t=WORKING finalizes t with
    a different reply after eviction — it must surface the terminal conflict
    and repeat no terminal side effects."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(protocol.TaskStore, "_MAX_TERMINAL", 1)
    ledger = tmp_path / "a2a_task_ledger.json"
    violations: list[str] = []
    now = time.time()

    t_working, stale_snapshot, _store_a = _settle_eviction_fixture(ledger, tmp_path)
    if "t" in json.loads(ledger.read_text()):
        violations.append("fixture did not evict t from the durable ledger")

    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
    # Adapter startup restores the authoritative ledger; the extra restore of
    # the stale snapshot is the same seam a gateway that started earlier uses.
    adapter.tasks.restore(stale_snapshot)
    held = adapter.tasks.get("t")
    if held is None or held["state"] != protocol.STATE_WORKING:
        violations.append(f"adapter should hold stale t=WORKING, got {held!r}")

    pushes: list = []
    persists: list = []
    audits: list = []
    monkeypatch.setattr(adapter, "_send_push_notification", lambda *args: pushes.append(args))
    monkeypatch.setattr(protocol, "persist_message", lambda *args, **kwargs: persists.append(args))
    monkeypatch.setattr(adapter, "_audit_safe", lambda *args, **kwargs: audits.append(args), raising=False)
    completed_before = protocol.metrics.tasks_completed

    pending = {
        "task_id": "t",
        "context_id": "ctx-t",
        "peer": t_working["peer"],
        "started": t_working["created_at"],
        "created_iso": t_working["created_iso"],
    }
    rejected = False
    try:
        adapter._finalize_task(pending, protocol.STATE_COMPLETED, "different answer")
    except protocol.DurablePublishError:
        rejected = True
    if not rejected:
        violations.append(
            "stale adapter durably re-completed evicted task t without a terminal conflict"
        )
    metrics_delta = protocol.metrics.tasks_completed - completed_before
    if pushes or persists or audits or metrics_delta:
        violations.append(
            "terminal side effects repeated: "
            f"pushes={len(pushes)} persists={len(persists)} audits={len(audits)} "
            f"tasks_completed_delta={metrics_delta}"
        )

    disk = json.loads(ledger.read_text())
    stale_t = disk.get("t")
    if stale_t is not None and (
        stale_t.get("state") not in protocol.TERMINAL_STATES
        or stale_t.get("reply") != "original answer"
    ):
        violations.append(f"durable ledger now holds divergent t: {stale_t!r}")

    assert violations == []
