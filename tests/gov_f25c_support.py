"""Shared helpers for the GOV-F25.c PR-A durable-notification acceptance tests.

Pure functions (no pytest fixtures) driving the Option B fence API in
``hermes_cli.kanban_db_notify`` + the durable delivery in
``gateway.kanban_watchers_notifier``. Imported by the relocated
``test_ac_gov_f25_c_*`` functions in tests/hermes_cli and tests/gateway.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from gateway import kanban_watchers_notifier as knw

_API = "api_server"


def _seed_task(conn: sqlite3.Connection) -> str:
    """Create a task and return its id."""
    return kb.create_task(conn, title="gov-f25c task", assignee="worker")


def _sub_kwargs(sub: dict) -> dict:
    """The kbn per-sub keyword args from a sub dict."""
    return {
        "task_id": sub["task_id"], "platform": sub["platform"],
        "chat_id": sub["chat_id"], "thread_id": sub.get("thread_id") or "",
    }


def _add_durable_sub(conn: sqlite3.Connection, task_id: str, *, chat_id: str = "durable") -> dict:
    """Register a wake-capable durable api_server sub (registered BEFORE publishing so it is not
    already caught up past the events under test)."""
    kbn.add_notify_sub(conn, task_id=task_id, platform=_API, chat_id=chat_id,
                       delivery_mode="wake", retry_policy="durable")
    return {"task_id": task_id, "platform": _API, "chat_id": chat_id, "thread_id": ""}


def _add_plain_sub(conn: sqlite3.Connection, task_id: str, *, chat_id: str = "plain") -> dict:
    """Register a default (non-durable) wake-capable api_server sub."""
    kbn.add_notify_sub(conn, task_id=task_id, platform=_API, chat_id=chat_id,
                       delivery_mode="notify+wake")
    return {"task_id": task_id, "platform": _API, "chat_id": chat_id, "thread_id": ""}


def _fence(conn: sqlite3.Connection, sub: dict):
    """The sub's pending_event_id fence (None when clear)."""
    return kbn.notify_pending_event_id(conn, **_sub_kwargs(sub))


def _redelivers(conn: sqlite3.Connection, sub: dict) -> bool:
    """True when the sub's event is NOT lost — either an un-acked fenced range replays via
    recovery, or (no fence) the cursor was never advanced so a fresh claim still returns it."""
    key = _sub_kwargs(sub)
    if kbn.pending_events_for_sub(conn, **key, kinds=knw.TERMINAL_KINDS):
        return True
    _new, events = kbn.unseen_events_for_sub(conn, **key, kinds=knw.TERMINAL_KINDS)
    return bool(events)


def _notify_sub_columns(conn: sqlite3.Connection) -> set:
    """Column names of the kanban_notify_subs table."""
    return {c["name"] for c in conn.execute("PRAGMA table_info(kanban_notify_subs)")}


def _sub_exists(conn: sqlite3.Connection, sub: dict) -> bool:
    row = conn.execute(
        "SELECT 1 FROM kanban_notify_subs " + kbn._SUB_KEY_WHERE,
        kbn._sub_key(sub["task_id"], sub["platform"], sub["chat_id"], sub.get("thread_id") or ""),
    ).fetchone()
    return row is not None


def _operator_alerted(conn: sqlite3.Connection, sub: dict) -> bool:
    """True when a durable-failure operator alert was recorded for the sub's task."""
    row = conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'notify_durable_failure' LIMIT 1",
        (sub["task_id"],),
    ).fetchone()
    return row is not None


def _mark_sub_aged(conn: sqlite3.Connection, sub: dict) -> None:
    """Age the sub's task into the GC-eligible state: status='done' with every timestamp far past
    the stale-sub retention window, so purge_stale_done_notify_subs would sweep a 'default' sub."""
    old = int(time.time()) - 400 * 86400
    with kb.write_txn(conn, allow_nested=True):
        conn.execute("UPDATE tasks SET status = 'done', completed_at = ?, created_at = ? WHERE id = ?",
                     (old, old, sub["task_id"]))
        conn.execute("UPDATE task_events SET created_at = ? WHERE task_id = ?", (old, sub["task_id"]))


def _purge_stale_done_notify_subs(conn: sqlite3.Connection) -> int:
    return kbn.purge_stale_done_notify_subs(conn, max_age_days=30)


def _publish_nonwaking_event(conn: sqlite3.Connection, task_id: str, *, kind: str = "archived") -> None:
    """Append a claimed-but-non-waking event (archived/unblocked) to a task's event log."""
    with kb.write_txn(conn, allow_nested=True):
        kb._append_event(conn, task_id, kind, {})


def _event_message(ev) -> str:
    """The caller message carried on a notification event (from _task_events)."""
    return ev.payload_text


def _sub_row_dict(conn: sqlite3.Connection, sub: dict) -> dict:
    row = conn.execute(
        "SELECT * FROM kanban_notify_subs " + kbn._SUB_KEY_WHERE,
        kbn._sub_key(sub["task_id"], sub["platform"], sub["chat_id"], sub.get("thread_id") or ""),
    ).fetchone()
    return dict(row)


def _render_wake_text(conn: sqlite3.Connection, sub: dict, events: list, task_id: str) -> str:
    """Render the synthetic wake text the notifier would build for a claimed batch — exercises the
    real _KanbanNotification.format_event + build_wake_text (which carries a notification's message
    into the wake turn)."""
    task = kb.get_task(conn, task_id)
    d = {"sub": _sub_row_dict(conn, sub), "task": task, "events": list(events), "board": None}
    notif = knw._KanbanNotification(None, d, platform_cls=None, sub_fail_counts={})
    for ev in events:
        notif.format_event(ev)
    notif.build_wake_text()
    return notif.synth or ""


def _drain_and_capture_wake(conn: sqlite3.Connection, task_id: str, *, chat_id: str = "ac1"):
    """Register a wake-capable sub, rewind its cursor to 0 so it sees the already-published event
    (a sub registered after the publish is caught up and misses it), claim, and return the rendered
    wake text."""
    sub = _add_plain_sub(conn, task_id, chat_id=chat_id)
    kbn.advance_notify_cursor(conn, **_sub_kwargs(sub), new_cursor=0)
    _o, _n, events = kbn.claim_unseen_events_for_sub(conn, **_sub_kwargs(sub), kinds=knw.TERMINAL_KINDS)
    return SimpleNamespace(wake_text=_render_wake_text(conn, sub, events, task_id))


def _task_events(conn: sqlite3.Connection, task_id: str, *, kind: str):
    """The task's events of one kind, with a ``.payload_text`` convenience holding the message."""
    rows = conn.execute(
        "SELECT * FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id ASC", (task_id, kind),
    ).fetchall()
    out = []
    for r in rows:
        ev = kb.Event.from_row(r)
        payload = ev.payload or {}
        out.append(SimpleNamespace(id=ev.id, kind=ev.kind, payload=payload,
                                   payload_text=str(payload.get("message") or "")))
    return out


def _notify_sub(conn: sqlite3.Connection, *, task_id: str, chat_id: str):
    """The sub row as an attribute namespace (.delivery_mode, .retry_policy, .pending_event_id)."""
    row = conn.execute(
        "SELECT * FROM kanban_notify_subs WHERE task_id = ? AND chat_id = ?", (task_id, chat_id),
    ).fetchone()
    return None if row is None else SimpleNamespace(**dict(row))


def _sub_retry_policy(conn: sqlite3.Connection, *, chat_id: str) -> str:
    row = conn.execute("SELECT retry_policy FROM kanban_notify_subs WHERE chat_id = ?", (chat_id,)).fetchone()
    return str(row["retry_policy"])


def _sub_pending_event_id(conn: sqlite3.Connection, *, chat_id: str):
    row = conn.execute("SELECT pending_event_id FROM kanban_notify_subs WHERE chat_id = ?", (chat_id,)).fetchone()
    return None if row is None or row["pending_event_id"] is None else int(row["pending_event_id"])


def _default_retry_policy(conn: sqlite3.Connection) -> str:
    """retry_policy of a migrated legacy row (the pre-columns row keeps the column DEFAULT)."""
    row = conn.execute("SELECT retry_policy FROM kanban_notify_subs LIMIT 1").fetchone()
    return str(row["retry_policy"])


def _default_pending_event_id(conn: sqlite3.Connection):
    row = conn.execute("SELECT pending_event_id FROM kanban_notify_subs LIMIT 1").fetchone()
    return None if row is None or row["pending_event_id"] is None else int(row["pending_event_id"])


# --- durable-delivery drivers (drive the notifier fence path with an injected wake) ---


def _fake_persist_wake(persisted: bool):
    """A deliver_wake fake ACCEPTING require_persist_ack (PR-B present) that raises when the turn
    is not persist-confirmed, mirroring the real gateway.wake contract."""
    async def _wake(adapter=None, *, text, session_id="", profile=None,
                    idempotency_key=None, require_persist_ack=False, **_kw):
        assert require_persist_ack is True
        if not persisted:
            raise RuntimeError("wake not persist-confirmed")
        return True
    return _wake


def _claim_durable_batch(conn: sqlite3.Connection, sub: dict) -> list:
    """A fresh durable claim; returns the claimed events (fence set to the first)."""
    _old, _new, events = kbn.claim_unseen_events_for_sub(conn, **_sub_kwargs(sub), kinds=knw.TERMINAL_KINDS)
    return events


def _fresh_claim(conn: sqlite3.Connection, sub: dict) -> list:
    """A fresh claim attempt (empty while the fence is held — a newer claim can't bypass it)."""
    _old, _new, events = kbn.claim_unseen_events_for_sub(conn, **_sub_kwargs(sub), kinds=knw.TERMINAL_KINDS)
    return events


def _run_concurrent_claim_deliver(conn: sqlite3.Connection, sub: dict, *, n: int = 2):
    """Model N drainers racing on one durable sub. SQLite serializes writers, and the fence makes
    every claim after the first empty — so exactly one delivers and the cursor advances once."""
    deliveries = advances = 0
    for _ in range(n):
        old, new, events = kbn.claim_unseen_events_for_sub(conn, **_sub_kwargs(sub), kinds=knw.TERMINAL_KINDS)
        if events:
            deliveries += 1
            if new != old:
                advances += 1
    return deliveries, {"advances": advances}


def _drain_with_persist(conn: sqlite3.Connection, sub: dict, *, persisted: bool):
    """Drive one durable drain through the notifier's per-event fence path with a persist-confirming
    (or -failing) wake. Recovers a retained fence rather than claiming anew, so a replay re-delivers
    the same un-acked event under the SAME stable idempotency key."""
    key = _sub_kwargs(sub)
    if kbn.notify_pending_event_id(conn, **key) is not None:
        events = kbn.pending_events_for_sub(conn, **key, kinds=knw.TERMINAL_KINDS)
    else:
        events = _claim_durable_batch(conn, sub)

    # Single-thread test conn: the fence-write callbacks run on this same connection (production
    # routes them to a worker thread via _kanban_sub_op — see test_durable_real_deliver).
    async def settle(settled_event_id, next_pending_id):
        kbn.settle_notify_pending(conn, **key, settled_event_id=settled_event_id,
                                  next_pending_id=next_pending_id)

    async def note_failure(event_id, reason):
        knw._note_durable_failure(conn, sub, event_id, reason)

    return asyncio.run(knw.deliver_durable_batch(
        sub, events, deliver_wake=_fake_persist_wake(persisted), settle=settle,
        note_failure=note_failure, adapter=None, session_id="s", profile=None))


def _drain(conn: sqlite3.Connection, sub: dict, *, wake):
    """Model the notifier's pre-claim capability gate: a durable sub whose wake API cannot
    persist-confirm is NOT claimed (no fence), the sub is retained, and an operator alert fires."""
    durable = kbn.notify_retry_policy(conn, **_sub_kwargs(sub)) == "durable"
    if durable and not knw._wake_supports_persist_ack(wake):
        knw._note_durable_failure(conn, sub, None, "wake API lacks require_persist_ack (PR-B absent)")
        return SimpleNamespace(claimed=False, operator_alerted=True, settled=False)
    events = _claim_durable_batch(conn, sub)
    return SimpleNamespace(claimed=bool(events), operator_alerted=False, settled=bool(events))


def _deliver_true_receipt(conn: sqlite3.Connection, sub: dict, *, event_id: int):
    """Deliver ONE fenced event with a true persist-ack, CASing the fence forward to the next
    un-acked id (NULL after the final). Returns the event id + its stable idempotency key."""
    key = _sub_kwargs(sub)
    events = kbn.pending_events_for_sub(conn, **key, kinds=knw.TERMINAL_KINDS)
    ev = next(e for e in events if e.id == event_id)
    later = [e.id for e in events if e.id > ev.id]
    kbn.settle_notify_pending(conn, **key, settled_event_id=ev.id,
                              next_pending_id=later[0] if later else None)
    return SimpleNamespace(event_id=ev.id, idempotency_key=knw._durable_idempotency_key(sub, ev.id))


async def _persist_capable_wake(adapter=None, *, text, session_id="", profile=None,
                                idempotency_key=None, require_persist_ack=False, **kwargs):
    """A ``deliver_wake`` stub whose signature carries ``require_persist_ack`` — the PR-B capability
    the notifier's pre-claim gate probes with ``_wake_supports_persist_ack``. Swapped in so the
    collector's durable path (incl. recovery) is reachable on the PR-A branch ALONE, i.e. the
    merged PR-A+PR-B state a running gateway will actually be in."""
    return True


def _collector_claim(conn: sqlite3.Connection, sub: dict):
    """Drive the REAL ``_Collector._claim_for_sub`` for one durable sub and return its claim dict
    (or None). A fenced sub returns the un-acked range ``[pending_event_id, last_event_id]`` — this
    is the RUNNING-collector recovery decision (the P0 fix), not a reimplementation of it. Simulates
    a connected ``api_server`` adapter and, by swapping ``gateway.wake.deliver_wake`` for a
    persist-capable stub, the presence of PR-B (without which the pre-claim gate fails closed)."""
    import gateway.wake as _wake_mod
    adapter = SimpleNamespace()
    runner = SimpleNamespace(
        adapters={"api_server": adapter}, _profile_adapters={},
        config=SimpleNamespace(multiplex_profiles=False),
        _owns_kanban_dispatcher_lock=lambda: True,
        _authorization_adapter=lambda platform, profile: adapter,
    )
    collector = knw._Collector(runner, kb, notifier_profile=None, gc_due=False, gc_retention_days=30)
    # The RUNNING collector iterates the PERSISTED sub rows (list_notify_subs), which carry
    # retry_policy / delivery_mode / pending_event_id — not the 4-tuple test dict. Feed
    # _claim_for_sub the real row so its durable-vs-fresh + recovery decision is exercised exactly
    # as in production (a bare dict lacking retry_policy would silently take the non-durable path).
    rows = kbn.list_notify_subs(conn, task_id=sub["task_id"])
    row = next(r for r in rows
               if r["platform"] == sub["platform"] and r["chat_id"] == sub["chat_id"]
               and (r.get("thread_id") or "") == (sub.get("thread_id") or ""))
    _orig = _wake_mod.deliver_wake
    _wake_mod.deliver_wake = _persist_capable_wake
    try:
        return collector._claim_for_sub(conn, "default", row)
    finally:
        _wake_mod.deliver_wake = _orig


def _recover_durable(conn: sqlite3.Connection, sub: dict) -> list:
    """Replay a durable sub's un-acked fenced range after a crash. Obtains the range by driving the
    REAL collector (``_Collector._claim_for_sub``), which on a set fence returns ONLY
    ``[pending_event_id, last_event_id]`` instead of a fresh claim, then redelivers each with a true
    receipt and settles the fence. Proves the running collector — not just a cold restart — routes a
    fenced sub into recovery."""
    key = _sub_kwargs(sub)
    claimed = _collector_claim(conn, sub)
    events = claimed["events"] if claimed else []
    ids = [e.id for e in events]
    wakes = []
    for i, ev in enumerate(events):
        kbn.settle_notify_pending(conn, **key, settled_event_id=ev.id,
                                  next_pending_id=ids[i + 1] if i + 1 < len(ids) else None)
        wakes.append(SimpleNamespace(event_id=ev.id, idempotency_key=knw._durable_idempotency_key(sub, ev.id)))
    return wakes


# --- migration / rebuild DB builders (raw legacy + drifted schemas) ---

_LEGACY_COLS = (
    " task_id TEXT NOT NULL, platform TEXT NOT NULL, chat_id TEXT NOT NULL,"
    " thread_id TEXT NOT NULL DEFAULT '', user_id TEXT, user_id_alt TEXT, chat_type TEXT,"
    " notifier_profile TEXT, delivery_mode TEXT NOT NULL DEFAULT 'notify',"
    " delivery_metadata TEXT, created_at INTEGER NOT NULL,"
)


def _db_without_optional_columns() -> Path:
    """A DB whose kanban_notify_subs LACKS retry_policy + pending_event_id (pre-GOV-F25), with one
    legacy row — so init_db's additive migration adds both columns and the row defaults."""
    import tempfile
    path = Path(tempfile.mkdtemp()) / "legacy.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE kanban_notify_subs (" + _LEGACY_COLS
        + " last_event_id INTEGER NOT NULL DEFAULT 0, last_ping_event_id INTEGER NOT NULL DEFAULT 0,"
        + " PRIMARY KEY (task_id, platform, chat_id, thread_id))"
    )
    conn.execute("INSERT INTO kanban_notify_subs (task_id, platform, chat_id, created_at)"
                 " VALUES ('t', 'api_server', 'legacy', 0)")
    conn.commit()
    conn.close()
    return path


def _drifted_db_with_durable_row(*, chat_id: str = "pre", pending_event_id: int = 7) -> Path:
    """A DRIFTED DB (last_event_id TEXT -> rebuild trigger) that already carries retry_policy +
    pending_event_id and a pre-seeded durable row with a non-NULL fence — so the rebuild preserves
    both (durable retry_policy + the non-NULL pending id) through the copy-forward."""
    import tempfile
    path = Path(tempfile.mkdtemp()) / "drifted.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE kanban_notify_subs (" + _LEGACY_COLS
        + " last_event_id TEXT, last_ping_event_id INTEGER NOT NULL DEFAULT 0,"
        + " retry_policy TEXT NOT NULL DEFAULT 'default', pending_event_id INTEGER DEFAULT NULL,"
        + " PRIMARY KEY (task_id, platform, chat_id, thread_id))"
    )
    conn.execute(
        "INSERT INTO kanban_notify_subs (task_id, platform, chat_id, created_at, last_event_id,"
        " retry_policy, pending_event_id) VALUES ('t', 'api_server', ?, 0, '0', 'durable', ?)",
        (chat_id, pending_event_id),
    )
    conn.commit()
    conn.close()
    return path
