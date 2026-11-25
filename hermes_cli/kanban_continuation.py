"""One-use operator continuation grants, bound to an existing owner's exact attempt.

The local operator invokes this API explicitly. Comment text never grants authority.
The event and consumption receipt live on the board; no schema migration is needed.
"""
import hashlib
import json
import os
import time


_ACCEPTANCE_FIELDS = (
    "title", "body", "assignee", "tenant", "skills", "completion_contract",
    "model_override", "provider_override", "workspace_kind", "workspace_path",
)


def _acceptance(task):
    return hashlib.sha256(json.dumps({k: getattr(task, k) for k in _ACCEPTANCE_FIELDS},
                                    sort_keys=True).encode()).hexdigest()


def _latest_comment(conn, task_id):
    row = conn.execute("SELECT MAX(id) FROM task_comments WHERE task_id=?", (task_id,)).fetchone()
    return row[0]


def authorize_continuation(conn, task_id, *, profile, expected_run_id, authority_comment_id,
                           actor, reason, ttl_seconds=3600):
    """Authorize ONE amendment attempt; neither unblocks nor claims/spawns the task.

    This is the same trusted-local-operator boundary as assign/unblock. The actor and
    reason are audit attribution, not an authentication mechanism. The exact newest
    comment is an evidence/version anchor, not interpreted approval prose.
    """
    from hermes_cli import kanban_db as kb
    if os.environ.get("HERMES_KANBAN_TASK") == task_id:
        raise PermissionError("a worker cannot self-authorize its own continuation")
    if not actor.strip() or not reason.strip() or type(ttl_seconds) is not int or not 1 <= ttl_seconds <= 86400:
        raise ValueError("actor, reason and a 1–86400 second lifetime are required")
    with kb.write_txn(conn):
        task = kb.get_task(conn, task_id)
        source = conn.execute("SELECT * FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
        if (task is None or source is None or source["id"] != expected_run_id
                or task.assignee != profile or source["profile"] != profile
                or task.status not in {"ready", "blocked", "running"}
                or authority_comment_id is None or _latest_comment(conn, task_id) != authority_comment_id):
            raise ValueError("stale continuation authority, owner or attempt")
        kb._append_event(conn, task_id, "continuation_authorized", {
            "profile": profile, "source_run_id": expected_run_id, "source_claim_lock": source["claim_lock"],
            "authority_comment_id": authority_comment_id, "actor": actor, "reason": reason,
            "acceptance": _acceptance(task), "expires_at": int(time.time()) + ttl_seconds,
        })
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def pending_continuation(conn, task_id):
    """Return an unconsumed, current ready-task grant, or None. Read-only."""
    from hermes_cli import kanban_db as kb
    event = conn.execute("SELECT id,payload FROM task_events WHERE task_id=? AND kind='continuation_authorized' ORDER BY id DESC LIMIT 1",
                         (task_id,)).fetchone()
    if event is None:
        return None
    grant = kb._json_dict(event["payload"])
    task = kb.get_task(conn, task_id)
    source = conn.execute("SELECT * FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    if (task is None or task.status != "ready" or task.claim_lock is not None or source is None
            or source["ended_at"] is None or source["id"] != grant.get("source_run_id")
            or source["claim_lock"] != grant.get("source_claim_lock")
            or task.assignee != grant.get("profile") or source["profile"] != task.assignee
            or grant.get("expires_at", 0) <= int(time.time()) or _acceptance(task) != grant.get("acceptance")
            or _latest_comment(conn, task_id) != grant.get("authority_comment_id")):
        return None
    invalidation = conn.execute("SELECT 1 FROM task_events WHERE task_id=? AND id>? AND kind IN "
                                "('continuation_consumed','assigned','review_requested','changes_requested','review_reopened') LIMIT 1",
                                (task_id, event["id"])).fetchone()
    return None if invalidation else event["id"]


def is_handoff_event(kind, payload):
    """Only a real different-owner assignment lifts the historical PR hold.

    Older events without `from`, no-op/unassign and automatic default filling are
    not a handoff. Review verdict/reopen events retain their existing semantics.
    """
    from hermes_cli import kanban_db as kb
    if kind != "assigned":
        return True
    data = kb._json_or(payload, {})
    if not isinstance(data, dict) or data.get("source") == "kanban.default_assignee":
        return False
    to = data.get("assignee")
    return bool(to) and "from" in data and data["from"] != to


def resolve_respawn_admission(conn, task_id, lane, guard_reason):
    """Carry the exact eligible grant to the atomic claim, never override other guards."""
    if lane == "ready" and guard_reason in {None, "active_pr", "recent_success"}:
        grant = pending_continuation(conn, task_id)
        if grant is not None:
            return None, grant
    return guard_reason, None


def claim_ready_task(conn, task_id, *, ttl_seconds=None, claimer=None, expect_assignee=None,
                     continuation_id=None):
    """Native claim transaction, with optional exact-grant CAS and atomic consumption."""
    from hermes_cli import kanban_db as kb
    now = int(time.time())
    lock = claimer or kb._claimer_id()
    expires = now + kb._resolve_claim_ttl_seconds(ttl_seconds)
    with kb.write_txn(conn):
        if continuation_id is not None:
            from hermes_cli.kanban_db_dispatch import check_respawn_guard
            if (pending_continuation(conn, task_id) != continuation_id
                    or check_respawn_guard(conn, task_id) not in {None, "active_pr", "recent_success"}):
                return None
        if not kb._parents_satisfied(conn, task_id):
            conn.execute("UPDATE tasks SET status='todo' WHERE id=? AND status='ready'", (task_id,))
            kb._append_event(conn, task_id, "claim_rejected", {"reason": "parents_not_done"})
            return None
        kb._reclaim_dangling_run(conn, task_id, statuses=("ready",), now=now, note="invariant recovery on re-claim")
        run_id = kb._claim_and_open_run(conn, task_id, "ready", lock, expires, now,
                                        expect_assignee=expect_assignee)
        if run_id is None:
            return None
        if continuation_id is not None:
            kb._append_event(conn, task_id, "continuation_consumed", {"authorization_id": continuation_id}, run_id=run_id)
        claimed = kb.get_task(conn, task_id)
    kb._fire_task_hook("kanban_task_claimed", claimed, task_id, run_id)
    return claimed
