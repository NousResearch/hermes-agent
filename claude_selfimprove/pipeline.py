"""Orchestration: the nightly scan pass and the weekly consolidation pass.

``scan()`` is the cheap, frequent pass: read new transcript turns
incrementally, flag candidates, classify them, and record evidence in the
candidate store. It never writes to any Claude Code target file.

``consolidate()`` is the pass that can actually change George's files: it
walks pending candidates, excludes repo-scoped ones, checks evidence
thresholds, checks for conflicts with what is already applied, and — only
for what clears every gate — writes through ``writers.py`` and notifies
through ``notify.py``.

Splitting these matters for two reasons: a scan can run every night cheaply
(mostly regex work, one batched model call only when something new was
flagged), while consolidation's conflict check and file writes are worth
running less often once evidence has had time to accumulate across several
nights. Both passes are independently safe to run at any time — consolidate
simply does nothing when nothing is eligible yet.

Both passes take an exclusive lock (mkdir-based, matching the same
technique ``self_improvement_watch.sh`` uses) so an overlapping run — a
manual ``--scan`` while the nightly cron is still running, for instance —
exits cleanly instead of racing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import audit, heuristics, llm, notify, paths, routing, scanner, store, thresholds, writers

_STALE_LOCK_SECONDS = 600


class LockBusyError(Exception):
    pass


def _acquire_lock(lock_path: Path) -> None:
    try:
        lock_path.mkdir(parents=True)
        return
    except FileExistsError:
        pass

    try:
        age = time.time() - lock_path.stat().st_mtime
    except OSError:
        age = 0.0
    if age > _STALE_LOCK_SECONDS:
        try:
            lock_path.rmdir()
        except OSError:
            pass
        try:
            lock_path.mkdir(parents=True)
            return
        except FileExistsError:
            pass
    raise LockBusyError(f"another claude-selfimprove run holds the lock at {lock_path}")


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.rmdir()
    except OSError:
        pass


@dataclass
class SeedResult:
    files_seeded: int = 0
    already_seeded: bool = False
    skipped_busy: bool = False
    error: str | None = None


@dataclass
class ScanResult:
    turns_scanned: int = 0
    files_touched: int = 0
    candidates_flagged: int = 0
    candidates_classified: int = 0
    skipped_busy: bool = False
    error: str | None = None


@dataclass
class ConsolidateResult:
    evaluated: int = 0
    applied: list[str] = field(default_factory=list)
    conflict_blocked: list[str] = field(default_factory=list)
    excluded_repo_scope: list[str] = field(default_factory=list)
    not_yet_eligible: int = 0
    failed: list[str] = field(default_factory=list)
    skipped_busy: bool = False
    error: str | None = None


def seed(*, force: bool = False, dry_run: bool = False) -> SeedResult:
    """One-time: mark all existing transcript content as already seen.

    Refuses to run when a checkpoint file already exists, unless
    ``force=True`` — protects a profile that has already been scanning for
    a while from silently losing its place. Safe and idempotent to call on
    a genuinely fresh profile (no checkpoint file yet), which is the only
    case this pipeline's install step calls it for.
    """
    paths.ensure_state_dirs()
    lock_path = paths.lock_dir_path()
    try:
        _acquire_lock(lock_path)
    except LockBusyError:
        return SeedResult(skipped_busy=True)

    try:
        if paths.checkpoints_path().exists() and not force:
            return SeedResult(already_seeded=True)

        checkpoints = scanner.CheckpointStore()
        count = scanner.seed_all(checkpoints)
        if not dry_run:
            checkpoints.save()
            audit.record("checkpoints_seeded", files_seeded=count)
        return SeedResult(files_seeded=count)
    except Exception as exc:  # outermost fail-safe
        audit.record("seed_failed", error=str(exc))
        return SeedResult(error=str(exc))
    finally:
        _release_lock(lock_path)


def scan(
    *,
    model: str | None = None,
    provider: str | None = None,
    llm_runner=None,
    dry_run: bool = False,
) -> ScanResult:
    paths.ensure_state_dirs()
    lock_path = paths.lock_dir_path()
    try:
        _acquire_lock(lock_path)
    except LockBusyError:
        return ScanResult(skipped_busy=True)

    try:
        checkpoints = scanner.CheckpointStore()
        turns = list(scanner.scan_new_turns(checkpoints))

        raw_candidates = heuristics.mine_candidates(turns)
        classified = (
            llm.classify_all(raw_candidates, runner=llm_runner, model=model, provider=provider)
            if raw_candidates
            else []
        )

        if not dry_run:
            with store.CandidateStore() as db:
                for c in classified:
                    source_hash_value = store.source_hash(c.raw.file_path, c.raw.session_id, c.raw.text)
                    cid = db.upsert_occurrence(
                        canonical_key=c.canonical_key,
                        category=c.raw.category,
                        scope=c.scope,
                        target_kind=c.target_kind,
                        target_path=None,
                        title=c.title,
                        body=c.body,
                        confidence=c.confidence,
                        session_id=c.raw.session_id,
                        source_hash_value=source_hash_value,
                    )
                    audit.record(
                        "candidate_occurrence",
                        candidate_id=cid,
                        canonical_key=c.canonical_key,
                        category=c.raw.category,
                        scope=c.scope,
                    )
            checkpoints.save()

        return ScanResult(
            turns_scanned=len(turns),
            files_touched=checkpoints.known_file_count(),
            candidates_flagged=len(raw_candidates),
            candidates_classified=len(classified),
        )
    except Exception as exc:  # outermost fail-safe: a scan bug must not corrupt state
        audit.record("scan_failed", error=str(exc))
        return ScanResult(error=str(exc))
    finally:
        _release_lock(lock_path)


def _apply_one(db: store.CandidateStore, row: dict, decision: routing.RoutingDecision) -> writers.WriteResult:
    if decision.kind == "claude_md_block":
        applied = db.list_by_status("applied")
        entries = [
            {"title": r["title"], "body": r["body"]}
            for r in applied
            if r["target_kind"] == "claude_md_block"
        ]
        entries.append({"title": row["title"], "body": row["body"]})
        write_result = writers.write_claude_md_block(entries)
        artifact = str(paths.claude_md_path())
    elif decision.kind == "rule":
        write_result = writers.write_rule_file(decision.path, title=row["title"], body=row["body"])
        artifact = str(decision.path)
    elif decision.kind == "skill":
        write_result = writers.write_skill_file(
            decision.path,
            slug=decision.path.parent.name,
            title=row["title"],
            body=row["body"],
            canonical_key=row["canonical_key"],
        )
        artifact = str(decision.path)
    else:  # pragma: no cover - resolve_target only returns known kinds
        return writers.WriteResult(False, None, "unknown target kind")

    session_id = row["session_ids"][-1] if row["session_ids"] else ""
    if write_result.success:
        db.mark_applied(row["id"], target_kind=decision.kind, target_path=str(decision.path) if decision.path else None)
        notify.notify_applied(artifact=artifact, label=row["title"], improvement=row["body"], session_id=session_id)
        audit.record("candidate_applied", candidate_id=row["id"], artifact=artifact, target_kind=decision.kind)
    else:
        notify.notify_failure(artifact=artifact, label=row["title"], reason=write_result.reason)
        audit.record("candidate_apply_failed", candidate_id=row["id"], artifact=artifact, reason=write_result.reason)
    return write_result


def consolidate(
    *,
    dry_run: bool = False,
    conflict_runner=None,
    model: str | None = None,
    provider: str | None = None,
) -> ConsolidateResult:
    paths.ensure_state_dirs()
    lock_path = paths.lock_dir_path()
    try:
        _acquire_lock(lock_path)
    except LockBusyError:
        return ConsolidateResult(skipped_busy=True)

    result = ConsolidateResult()
    try:
        with store.CandidateStore() as db:
            pending = db.list_by_status("pending")

            repo_scoped = [r for r in pending if r["scope"] != "global"]
            for row in repo_scoped:
                if not dry_run:
                    db.set_status(row["id"], "excluded_repo_scope")
                    audit.record("candidate_excluded_repo_scope", candidate_id=row["id"])
                result.excluded_repo_scope.append(row["id"])

            pending_global = [r for r in pending if r["scope"] == "global"]
            result.evaluated = len(pending_global)

            eligible_rows = []
            for row in pending_global:
                ok, _reason = thresholds.is_eligible(row)
                if ok:
                    eligible_rows.append(row)
                else:
                    result.not_yet_eligible += 1

            if not eligible_rows:
                return result

            applied_rows = db.list_by_status("applied")
            existing_entries = [{"title": r["title"], "body": r["body"]} for r in applied_rows]

            conflict_results = llm.check_conflicts(
                eligible_rows, existing_entries, runner=conflict_runner, model=model, provider=provider,
            )

            current_block_chars = len(writers.read_managed_block_body())

            for row in eligible_rows:
                conflicts, reason = conflict_results.get(row["id"], (True, "conflict check unavailable"))
                if conflicts:
                    if not dry_run:
                        db.set_status(row["id"], "conflict_blocked", reason=reason)
                        notify.notify_conflict_blocked(
                            artifact=row.get("target_path") or str(paths.claude_md_path()),
                            label=row["title"],
                            reason=reason,
                        )
                        audit.record("candidate_conflict_blocked", candidate_id=row["id"], reason=reason)
                    result.conflict_blocked.append(row["id"])
                    continue

                decision = routing.resolve_target(row, current_claude_md_block_chars=current_block_chars)
                if decision is None:
                    result.excluded_repo_scope.append(row["id"])
                    continue

                if dry_run:
                    result.applied.append(row["id"])
                    continue

                write_result = _apply_one(db, row, decision)
                if write_result.success:
                    result.applied.append(row["id"])
                    if decision.kind == "claude_md_block":
                        current_block_chars += len(row["body"]) + len(row["title"]) + 10
                else:
                    result.failed.append(row["id"])

        return result
    except Exception as exc:  # outermost fail-safe
        audit.record("consolidate_failed", error=str(exc))
        result.error = str(exc)
        return result
    finally:
        _release_lock(lock_path)
