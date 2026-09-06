
from hermes_cli import kanban_db_connect
"""Bounded, opt-in handoffs to an existing review child, owned by dispatch.

No tool gains cross-card authority. See docs/kanban-delivery-review.md for the
structured run-metadata contract. Free text, titles and PR URLs are not authority.
"""
import json
import re
import time

from hermes_cli import kanban_db as kb


def _object(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def _latest(conn, tid):
    return conn.execute("SELECT * FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)).fetchone()


def _event(conn, tid, kind, key):
    return conn.execute(
        "SELECT * FROM task_events WHERE task_id=? AND kind=? AND json_valid(payload) "
        "AND json_extract(payload, '$.transition')=? ORDER BY id DESC LIMIT 1",
        (tid, kind, key),
    ).fetchone()


def _emit_once(conn, tid, kind, key, payload, run_id):
    if _event(conn, tid, kind, key):
        return False
    kb._append_event(conn, tid, kind, {"transition": key, **payload}, run_id=run_id)
    return True


def _claimed(conn, run):
    return run is not None and conn.execute(
        "SELECT 1 FROM task_events WHERE task_id=? AND run_id=? AND kind='claimed'",
        (run["task_id"], run["id"]),
    ).fetchone() is not None


def _healthy(task):
    return (task["claim_lock"] is None and task["worker_pid"] is None
            and not task["consecutive_failures"] and not task["last_failure_error"])


def reconcile(conn):
    """One atomic controller pass. Return externally-managed review parents.

    A submitted implementation phase may be released without the worker calling
    complete on its parent (which otherwise deadlocks its review child). This is
    NOT delivery acceptance: only delivery_accepted denotes exact gate PASS.
    No retry is generated from a crash, timeout, exhaustion, or blocked verdict.
    """
    managed = set()
    holds = []
    with kanban_db_connect.write_txn(conn):
        rows = conn.execute(
            "SELECT t.*, r.metadata AS handoff_metadata, r.id AS handoff_run "
            "FROM tasks t JOIN task_runs r ON r.id=(SELECT MAX(id) FROM task_runs WHERE task_id=t.id) "
            "WHERE t.status IN ('review', 'done') AND r.ended_at IS NOT NULL "
            "AND r.outcome IN ('completed', 'review_requested')"
        ).fetchall()
        for parent in rows:
            metadata = _object(parent["handoff_metadata"])
            if "delivery_review" not in metadata:
                continue
            spec = _object(metadata["delivery_review"])
            # Verdict-only child metadata is not an implementation handoff.
            if "verdict" in spec and not parent["review_requirement"]:
                continue
            tid, rid = parent["id"], parent["handoff_run"]
            managed.add(tid)  # malformed external handoffs fail closed too
            requirement = _object(parent["review_requirement"])
            head, child_id = spec.get("head"), requirement.get("review_task_id")
            key = f"{tid}:{child_id}:{head}"
            payload = {"head": head, "review_task": child_id, "implementation_task": tid}

            def hold(reason):
                holds.append((tid, reason))
                _emit_once(conn, tid, "delivery_review_hold", key + ":" + reason,
                           {**payload, "reason": reason}, rid)

            if requirement.get("required") is not True or not requirement.get("owner"):
                hold("canonical review_requirement required")
                continue
            if not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{40}", head):
                hold("full exact commit SHA required")
                continue
            children = conn.execute(
                "SELECT t.* FROM tasks t JOIN task_links l ON l.child_id=t.id WHERE l.parent_id=?",
                (tid,),
            ).fetchall()
            if (len(children) != 1 or children[0]["id"] != child_id
                    or not children[0]["assignee"] or children[0]["assignee"] == parent["assignee"]
                    or children[0]["tenant"] != parent["tenant"]):
                hold("handoff must name the existing sole independent review child")
                continue
            child = children[0]
            if conn.execute("SELECT COUNT(*) FROM task_links WHERE child_id=?", (child_id,)).fetchone()[0] != 1:
                hold("review child has additional dependencies")
                continue
            run = _latest(conn, tid)
            if not _claimed(conn, run) or run["id"] != rid or run["profile"] != parent["assignee"]:
                hold("owned implementation run required; do not reassign parent to reviewer")
                continue
            evidence = _object(spec.get("evidence"))
            target = evidence.get("validation_target")
            if target is not None and (not isinstance(target, dict)
                    or target.get("kind") != "pr_head" or target.get("commit") != head):
                hold("PR-head acceptance requires matching pr_head validation, not merged-checkout or deployed claims")
                continue
            artifact = evidence.get("artifact")
            if not isinstance(artifact, str) or not artifact.strip():
                hold("stable PR or repository/branch artifact identity required")
                continue
            previous = conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='delivery_phase_completed' ORDER BY id LIMIT 1",
                (tid,),
            ).fetchone()
            if previous and _object(_object(previous["payload"]).get("evidence")).get("artifact") != artifact:
                hold("artifact identity changed; continue the same PR/branch")
                continue
            checks = evidence.get("checks")
            if not isinstance(checks, list) or not checks or not all(
                isinstance(c, dict) and c.get("command") and c.get("result") == "passed" for c in checks
            ):
                hold("focused check evidence required")
                continue
            if not _healthy(parent):
                hold("implementation has unresolved failure or claim")
                continue
            sent = _event(conn, tid, "delivery_phase_completed", key)
            if sent is None:
                if child["status"] not in {"todo", "ready", "done"} or not _healthy(child):
                    hold("reviewer parked or owned; operator decision required")
                    continue
                last_child_run = _latest(conn, child_id)
                if last_child_run is not None and last_child_run["outcome"] not in {"completed", "review_requested"}:
                    hold("reviewer did not finish normally; operator decision required")
                    continue
                # Phase completion and exact-head review enqueue are one commit.
                # No completed event, success counter reset, new card, or notice
                # subscription is synthesized. The existing graph is preserved.
                conn.execute("UPDATE tasks SET status='done', completed_at=? WHERE id=?", (int(time.time()), tid))
                conn.execute("UPDATE tasks SET status='ready', completed_at=NULL, result=NULL WHERE id=?", (child_id,))
                _emit_once(conn, tid, "delivery_phase_completed", key,
                           {**payload, "acceptance": "pending", "evidence": evidence}, rid)
                _emit_once(conn, child_id, "delivery_review_enqueued", key,
                           {**payload, "implementation_run": rid, "status": "ready"}, rid)
                # Durable exact candidate takes precedence over stale reviewer
                # body text; retain original IDs and past generations for audit.
                kb._append_event(conn, child_id, "delivery_review_candidate", {
                    **payload, "transition": key, "evidence": evidence,
                    "instruction": "Review this exact head, not the superseded head in the opening body. "
                                   "Return delivery_review metadata with head, verdict PASS/BLOCK, and findings.",
                }, run_id=rid)
                continue
            if sent["run_id"] != rid:
                hold("this exact head was already consumed; submit a new generation")
                continue
            if child["status"] != "done" or not _healthy(child):
                if child["status"] in {"blocked", "triage"} or not _healthy(child) and child["claim_lock"] is None:
                    hold("reviewer parked or failed; operator decision required")
                continue
            reviewed = _latest(conn, child_id)
            if not _claimed(conn, reviewed) or reviewed["outcome"] != "completed" or reviewed["profile"] != child["assignee"]:
                continue
            claim = conn.execute("SELECT id FROM task_events WHERE task_id=? AND run_id=? AND kind='claimed'",
                                 (child_id, reviewed["id"])).fetchone()
            if claim["id"] <= sent["id"]:
                continue  # a prior review is never evidence for a new candidate
            review = _object(_object(reviewed["metadata"]).get("delivery_review"))
            if review.get("head") != head:
                hold("review head missing or stale")
                continue
            if review.get("verdict") == "BLOCK" and isinstance(review.get("findings"), str) and review["findings"].strip():
                if _emit_once(conn, tid, "delivery_changes_requested", key,
                              {**payload, "status": "ready", "review_run": reviewed["id"],
                               "findings": kb.redact_review_value(review["findings"])}, rid):
                    conn.execute("UPDATE tasks SET status='ready', completed_at=NULL, result=NULL WHERE id=?", (tid,))
                continue
            if review.get("verdict") != "PASS":
                hold("review has no exact PASS or actionable BLOCK")
                continue
            if not (evidence.get("ci_head") == head and evidence.get("ci") == "success"
                    and evidence.get("draft") is False and evidence.get("proof_head") == head
                    and evidence.get("proof") == "passed"):
                hold("review passed; CI, nonDraft or proof gate missing/failed/stale")
                continue
            _emit_once(conn, tid, "delivery_accepted", key,
                       {**payload, "review_run": reviewed["id"], "acceptance": "ready",
                        "evidence": evidence}, rid)
    return managed, holds
