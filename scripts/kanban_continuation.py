"""Opt-in, allowlisted Kanban continuation; no inference in the supervisor."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from hermes_cli import kanban_db as kb


def capacity_decision(snapshot, now, minimum_remaining_percent=10):
    epoch = int(now.timestamp())
    unknown = (False, epoch + 900)
    if not snapshot or not snapshot.available or not snapshot.windows:
        return unknown
    age = (now - snapshot.fetched_at).total_seconds()
    if not -60 <= age <= 120:
        return unknown
    values = [w.used_percent for w in snapshot.windows]
    if any(type(v) not in (int, float) or not 0 <= v <= 100 for v in values):
        return unknown
    limiting = [w for w in snapshot.windows
                if w.used_percent >= 100 or 100 - w.used_percent < minimum_remaining_percent]
    if not limiting:
        return True, epoch
    resets = []
    for w in limiting:
        if not isinstance(w.reset_at, datetime) or w.reset_at.tzinfo is None:
            return unknown
        reset = int(w.reset_at.timestamp())
        if reset <= epoch:
            return unknown
        resets.append(reset)
    return False, max(resets) + 60


def acceptance_gaps(metadata, requirements, evidence_root):
    acceptance = metadata.get("acceptance", {}) if isinstance(metadata, dict) else {}
    if not isinstance(acceptance, dict):
        return list(requirements)
    root = Path(evidence_root).resolve()
    gaps = []
    for key in requirements:
        item = acceptance.get(key)
        valid = False
        if isinstance(item, dict) and item.get("status") == "PASS":
            try:
                path = Path(item.get("evidence", "")).resolve()
                valid = path.is_relative_to(root) and path.is_file() and path.stat().st_size > 0
            except (TypeError, ValueError, OSError):
                pass
        if not valid:
            gaps.append(key)
    return gaps


def reopen_incomplete(conn, task_id, run_id, gaps, max_reopens=2, *, reviewer=None):
    # CLI block/unblock cannot reopen done. Use an audited, CAS-guarded
    # administrative repair, matching dashboard parent invalidation semantics.
    # No run history is rewritten and no active worker may be reclaimed here.
    with kb.write_txn(conn):
        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)
        if not task or task.status != "done" or task.current_run_id or (run.id if run else None) != run_id:
            return None
        if reviewer:
            review_id, expected_review_run, expected_summary = reviewer
            review_task = kb.get_task(conn, review_id)
            review_run = kb.latest_run(conn, review_id)
            if (not review_task or review_task.status != "blocked" or review_task.current_run_id
                    or not review_run or review_run.id != expected_review_run
                    or review_run.summary != expected_summary):
                return None
        active_child = conn.execute(
            "SELECT 1 FROM task_links l JOIN tasks t ON t.id=l.child_id "
            "WHERE l.parent_id=? AND t.status IN ('running','review') LIMIT 1", (task_id,),
        ).fetchone()
        if active_child:
            return None
        count = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='continuation_reopened'",
            (task_id,),
        ).fetchone()[0]
        status = "scheduled" if count < max_reopens else "triage"
        conn.execute(
            "UPDATE tasks SET status=?,completed_at=NULL,result=NULL WHERE id=? AND status='done'",
            (status, task_id),
        )
        kb._append_event(conn, task_id, "continuation_reopened",
                         {"reviewed_run_id": run_id, "gaps": gaps, "status": status})
        if reviewer and status == "scheduled":
            conn.execute("UPDATE tasks SET status='scheduled' WHERE id=?", (reviewer[0],))
            kb._append_event(conn, reviewer[0], "continuation_review_requeued",
                             {"review_run_id": reviewer[1], "parent": task_id, "parent_run_id": run_id})
        for row in conn.execute("SELECT child_id FROM task_links WHERE parent_id=?", (task_id,)):
            child = row[0]
            changed = conn.execute("UPDATE tasks SET status='todo' WHERE id=? AND status='ready'", (child,))
            if changed.rowcount:
                kb._append_event(conn, child, "status", {"status": "todo", "reason": "parent_reopened", "parent": task_id})
    kb.add_comment(conn, task_id, author="continuation-controller",
                   body="전체 완료 반려: " + ", ".join(gaps) +
                   ". 원본 카드/이력 유지. 증거 보완 후 재검증; 미검증을 PASS로 처리하지 말 것.")
    return status


def set_waiting_state(conn, task_id, status, candidate=None):
    """CAS transitions for idle allowlisted work; preserve all failure counters."""
    with kb.write_txn(conn):
        task = kb.get_task(conn, task_id)
        permitted = {"todo", "ready", "scheduled"}
        if not task or task.status not in permitted or task.current_run_id:
            return False
        if status == "ready":
            if conn.execute(
                "SELECT 1 FROM task_links l JOIN tasks p ON p.id=l.parent_id "
                "WHERE l.child_id=? AND p.status!='done' LIMIT 1", (task_id,),
            ).fetchone():
                status = "scheduled"
        pin_changed = bool(candidate and status == "ready" and
                           (task.model_override != candidate["model"] or
                            task.provider_override != candidate["provider"]))
        if task.status == status and not pin_changed:
            return False
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
        if candidate and status == "ready":
            conn.execute("UPDATE tasks SET model_override=?,provider_override=? WHERE id=?",
                         (candidate["model"], candidate["provider"], task_id))
        kb._append_event(conn, task_id, "continuation_capacity",
                         {"status": status, "candidate": candidate if status == "ready" else None})
        return True


def tick(conn, manifest, state, fetch_usage, now, *, persist_state=None):
    """One bounded tick. No worker creation, inference, or blanket unblock."""
    task_specs = manifest["tasks"]
    by_id = {s["id"]: s for s in task_specs}
    changes = []
    for spec in sorted(task_specs, key=lambda s: not bool(s.get("review_of"))):
        task = kb.get_task(conn, spec["id"])
        if not task:
            raise ValueError("allowlisted task missing: " + spec["id"])
        if task.workspace_kind != "dir" or not task.workspace_path or Path(task.workspace_path).resolve() != Path(manifest["workspace"]).resolve():
            raise ValueError("allowlisted workspace changed: " + task.id)
        run = kb.latest_run(conn, task.id)
        if task.status == "blocked" and spec.get("review_of") in by_id and run:
            try:
                verdict = json.loads(run.summary or "{}")
            except (TypeError, ValueError):
                verdict = {}
            parent_id = spec["review_of"]
            parent_run = kb.latest_run(conn, parent_id)
            if (isinstance(verdict, dict) and verdict.get("verdict") == "REJECT"
                    and verdict.get("reviewed_task_id") == parent_id and parent_run
                    and verdict.get("reviewed_run_id") == parent_run.id
                    and isinstance(verdict.get("missing"), list) and verdict["missing"]
                    and all(isinstance(s, str) for s in verdict["missing"])
                    and verdict.get("needs_user_choice") is False):
                reopened = reopen_incomplete(conn, parent_id, parent_run.id, verdict["missing"],
                                             reviewer=(task.id, run.id, run.summary))
                if reopened == "scheduled":
                    changes.append(parent_id + ": 독립 리뷰 반려 → 원본 재개")
        if task.status == "done":
            metadata = run.metadata if run else None
            gaps = acceptance_gaps(metadata, spec["requirements"], manifest["evidence_root"])
            if spec.get("review_of"):
                parent_run = kb.latest_run(conn, spec["review_of"])
                if not parent_run or (metadata or {}).get("reviewed_run_id") != parent_run.id:
                    gaps.append("latest_parent_run")
            if gaps:
                status = reopen_incomplete(conn, task.id, run.id if run else None, gaps)
                if status:
                    changes.append(task.id + ": 완료 증거 누락 → " + status)

    tasks = [t for s in task_specs if (t := kb.get_task(conn, s["id"])) is not None]
    if len(tasks) != len(task_specs):
        raise ValueError("allowlisted task disappeared")
    # Wait for active workers, real QA blocks, cancelled work and exhausted
    # review budgets. No model calls or meter polling when nothing can run.
    idle = [t for t in tasks if t.status in ("ready", "scheduled", "todo")]
    if not idle or any(t.status in ("running", "review") for t in tasks):
        return {"statuses": {t.id: t.status for t in tasks}, "changes": changes}
    if not any(not conn.execute(
            "SELECT 1 FROM task_links l JOIN tasks p ON p.id=l.parent_id "
            "WHERE l.child_id=? AND p.status!='done' LIMIT 1", (t.id,),
    ).fetchone() for t in idle):
        return {"statuses": {t.id: t.status for t in tasks}, "changes": changes}
    epoch = int(now.timestamp())
    gates = state.setdefault("providers", {})
    selected = None
    for candidate in manifest["candidates"]:
        provider = candidate["provider"]
        prior = gates.get(provider, {})
        # An exhausted bucket is NOT re-probed before its advertised recovery.
        if prior.get("allowed") is False and prior.get("next_at", 0) > epoch:
            continue
        try:
            snap = fetch_usage(provider)
            allowed, next_at = capacity_decision(snap, now, manifest.get("minimum_remaining_percent", 10))
        except Exception:
            allowed, next_at = False, epoch + 900
        gates[provider] = {"allowed": allowed, "next_at": next_at}
        if persist_state:
            persist_state()
        if allowed:
            selected = candidate
            break
    for task in idle:
        if set_waiting_state(conn, task.id, "ready" if selected else "scheduled", selected):
            changes.append(task.id + (": 용량 확인 → ready" if selected else ": 용량 대기 → scheduled"))
    statuses = {t.id: t.status for s in task_specs if (t := kb.get_task(conn, s["id"])) is not None}
    result = {"statuses": statuses, "changes": changes, "selected": selected}
    if selected is None:
        result["next_check_at"] = min(gates[c["provider"]]["next_at"] for c in manifest["candidates"])
    return result


def prepare(conn, manifest, contract):
    """One-time operator setup. Leave all work scheduled until enabled."""
    spec = manifest["tasks"][0]
    root = kb.get_task(conn, spec["id"])
    if not root or root.workspace_kind != "dir" or not root.workspace_path or Path(root.workspace_path).resolve() != Path(manifest["workspace"]).resolve():
        raise ValueError("Original QA workspace does not match approved manifest")
    if root.status not in ("done", "scheduled") or root.current_run_id:
        raise ValueError("Original QA is active or requires manual disposition")
    if root.status == "done":
        run = kb.latest_run(conn, root.id)
        gaps = acceptance_gaps(run.metadata if run else None, spec["requirements"], manifest["evidence_root"])
        if gaps and reopen_incomplete(conn, root.id, run.id if run else None, gaps) != "scheduled":
            raise ValueError("Original QA cannot be automatically reopened")
    marker = "\n\n## continuation-acceptance-contract\n"
    with kb.write_txn(conn):
        current = kb.get_task(conn, root.id)
        if not current or current.status not in ("done", "scheduled") or current.current_run_id:
            raise ValueError("Original QA changed during setup")
        if marker not in (current.body or ""):
            conn.execute("UPDATE tasks SET body=? WHERE id=?", ((current.body or "") + marker + contract, root.id))
            kb._append_event(conn, root.id, "continuation_contract_added", {"requirements": spec["requirements"]})
    # Recover a partially completed setup by durable identity, not a new card
    # on every invocation. This is a review child, never a replacement QA card.
    rows = conn.execute(
        "SELECT t.id FROM tasks t JOIN task_links l ON l.child_id=t.id "
        "WHERE l.parent_id=? AND t.created_by='continuation-setup'", (root.id,),
    ).fetchall()
    if len(rows) > 1:
        raise ValueError("Multiple independent review cards; require operator choice")
    review_id = rows[0][0] if rows else kb.create_task(
        conn, title="뭉게 원본 Alpha QA의 독립 완료 판정", assignee="amuro", triage=True,
        body="브라이트의 원본 QA " + root.id + " 최신 run을 독립 검토하라.\n" + contract,
        parents=[root.id], workspace_kind="dir", workspace_path=manifest["workspace"],
        created_by="continuation-setup",
    )
    with kb.write_txn(conn):
        if conn.execute("UPDATE tasks SET status='scheduled' WHERE id=? AND status='triage' AND current_run_id IS NULL", (review_id,)).rowcount:
            kb._append_event(conn, review_id, "continuation_review_prepared", {"parent": root.id})
    manifest["tasks"] = [spec, {"id": review_id, "review_of": root.id,
                                "requirements": ["independent_review"]}]
    return review_id


def main():
    import argparse
    import fcntl
    import os
    import subprocess
    import sys

    from agent.account_usage import fetch_account_usage
    from hermes_constants import get_hermes_home

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--enable", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if Path(manifest["hermes_home"]).resolve() != get_hermes_home().resolve():
        raise ValueError("Wrong Hermes profile; refusing board mutation")
    if args.prepare or args.enable:
        kb._assert_not_delegated_child_mutation()
        if args.prepare:
            if manifest.get("enabled"):
                raise ValueError("Already enabled; refusing to modify live setup")
            with kb.connect(board=manifest["board"]) as conn:
                import sqlite3
                backup = args.manifest.with_name("mungge-before-continuation.sqlite")
                if not backup.exists():
                    with sqlite3.connect(backup) as dst:
                        conn.backup(dst)
                review_id = prepare(conn, manifest, Path(manifest["acceptance_contract"]).read_text(encoding="utf-8"))
            print("Prepared independent review:", review_id)
        else:
            if len(manifest["tasks"]) != 2:
                raise ValueError("Prepare the independent review before enabling")
            manifest["enabled"] = True
        temp = args.manifest.with_suffix(".prepared.tmp")
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(args.manifest)
        return
    if not manifest.get("enabled"):
        return
    state_file = args.manifest.with_suffix(".state.json")
    with args.manifest.with_suffix(".lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
        def save():
            temp = state_file.with_suffix(".tmp")
            with temp.open("w", encoding="utf-8") as out:
                json.dump(state, out, ensure_ascii=False, indent=2)
                out.flush()
                os.fsync(out.fileno())
            temp.replace(state_file)
        with kb.connect(board=manifest["board"]) as conn:
            result = tick(conn, manifest, state, fetch_account_usage, datetime.now(timezone.utc), persist_state=save)
        # Gates survive later DB / delivery failures; notification state only
        # advances after ACK so failed reports are retried independently.
        signature = json.dumps({k: v for k, v in result.items() if k != "changes"}, sort_keys=True)
        pending = state.get("notified") != signature
        state["last_result"] = result
        save()
        if pending:
            message = "[Kanban 자동 이어가기] " + ", ".join(f"{k}={v}" for k, v in result["statuses"].items())
            if result.get("changes"):
                message += "\n" + "\n".join(result["changes"])
            if "next_check_at" in result:
                when = datetime.fromtimestamp(result["next_check_at"], timezone.utc).astimezone()
                message += "\n복구 재확인: " + when.isoformat(timespec="seconds") + " (그 전 worker 호출 없음)"
            if manifest.get("notify_target"):
                delivery = subprocess.run(
                    [sys.executable, "-m", "hermes_cli.main", "-p", "amuro", "send", "--to",
                     manifest["notify_target"], message], capture_output=True, text=True, timeout=60,
                )
                if delivery.returncode or "sent" not in delivery.stdout.lower():
                    raise RuntimeError("Discord delivery not acknowledged; will retry next tick")
            state["notified"] = signature
            save()
            print(message)


if __name__ == "__main__":
    main()
