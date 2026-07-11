#!/usr/bin/env python3
"""Pure policy helpers for the fork-only upstream sync routine."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
STATE_SCHEMA = "muncho-auto-sync-blocker-dedupe.v1"
DEFAULT_REPEAT_AFTER_SECONDS = 24 * 60 * 60


def _valid_sha(value: str | None) -> bool:
    return bool(value and SHA_RE.fullmatch(value))


def classify_stale_sync_pr(
    *,
    automation_owned: bool,
    head_already_in_fork_main: bool,
    upstream_snapshot_sha: str | None,
    upstream_snapshot_in_fork_merge_base: bool,
    current_upstream_sha: str | None,
    current_upstream_contains_snapshot: bool,
) -> str | None:
    """Return a bounded stale reason for a provably automation-owned PR.

    A newer upstream ref is not sufficient on its own: ancestry must prove
    that the current upstream still contains the snapshot embedded in the PR.
    This avoids closing a PR after an upstream force-push or unrelated ref.
    """

    if not automation_owned:
        return None
    if head_already_in_fork_main:
        return "head_already_in_fork_main"
    if upstream_snapshot_in_fork_merge_base:
        return "upstream_snapshot_already_in_fork_merge_base"
    if not (
        _valid_sha(upstream_snapshot_sha)
        and _valid_sha(current_upstream_sha)
        and upstream_snapshot_sha != current_upstream_sha
        and current_upstream_contains_snapshot
    ):
        return None
    return "upstream_snapshot_superseded"


def blocker_fingerprint(
    *,
    status: str,
    pr_number: int | None,
    head_sha: str | None,
    blockers: Iterable[str],
    failed_checks: Iterable[Mapping[str, Any]],
) -> str:
    """Build a stable fingerprint without storing raw logs or PR bodies."""

    normalized_checks = sorted(
        {
            (
                str(row.get("name") or "unknown")[:160],
                str(row.get("conclusion") or "unknown")[:40].upper(),
            )
            for row in failed_checks
        }
    )
    payload = {
        "status": str(status)[:120],
        "pr_number": int(pr_number) if pr_number is not None else None,
        "head_sha": head_sha if _valid_sha(head_sha) else None,
        "blockers": sorted({str(item)[:120] for item in blockers}),
        "failed_checks": normalized_checks,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(now: datetime) -> str:
    return now.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict) or data.get("schema") != STATE_SCHEMA:
        return {}
    return data


def _atomic_write_private(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    finally:
        tmp.unlink(missing_ok=True)


def decide_blocker_delivery(
    state_path: Path,
    *,
    fingerprint: str,
    now: datetime | None = None,
    repeat_after_seconds: int = DEFAULT_REPEAT_AFTER_SECONDS,
) -> dict[str, Any]:
    """Persist and return whether an unchanged blocker should notify.

    First sight, a changed fingerprint, and a 24-hour reminder emit. Identical
    three-hour cron runs are recorded as suppressed and return success/no
    output at the caller, preventing repetitive Discord failure messages.
    """

    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("invalid blocker fingerprint")
    if repeat_after_seconds < 60:
        raise ValueError("repeat_after_seconds must be at least 60")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    state = _load_state(state_path)
    previous = state.get("fingerprint")
    last_emitted = _parse_utc(state.get("last_emitted_at"))
    age = (current - last_emitted).total_seconds() if last_emitted else None

    if previous != fingerprint:
        emit, reason = True, "new_or_changed_blocker"
        suppressed_runs = 0
    elif age is None or age >= repeat_after_seconds:
        emit, reason = True, "repeat_window_elapsed"
        suppressed_runs = int(state.get("suppressed_runs") or 0)
    else:
        emit, reason = False, "unchanged_blocker_suppressed"
        suppressed_runs = int(state.get("suppressed_runs") or 0) + 1

    payload = {
        "schema": STATE_SCHEMA,
        "active": True,
        "fingerprint": fingerprint,
        "last_seen_at": _iso(current),
        "last_emitted_at": _iso(current) if emit else state.get("last_emitted_at"),
        "suppressed_runs": suppressed_runs,
    }
    _atomic_write_private(state_path, payload)
    return {
        "emit": emit,
        "reason": reason,
        "suppressed_runs": suppressed_runs,
        "repeat_after_seconds": repeat_after_seconds,
    }


def clear_blocker_delivery_state(
    state_path: Path, *, now: datetime | None = None
) -> None:
    """Mark the blocker inactive so recurrence is treated as a new event."""

    if not state_path.exists():
        return
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    _atomic_write_private(
        state_path,
        {"schema": STATE_SCHEMA, "active": False, "cleared_at": _iso(current)},
    )
