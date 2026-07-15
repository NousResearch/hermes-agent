#!/usr/bin/env python3
"""Pure policy helpers for the fork-only upstream sync routine."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

try:  # Linux/macOS production path; harmless fallback keeps Windows imports safe.
    import fcntl
except ImportError:  # pragma: no cover - Windows does not run the Cloud helper.
    fcntl = None  # type: ignore[assignment]

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
STATE_SCHEMA = "muncho-auto-sync-blocker-dedupe.v2"
DEFAULT_REPEAT_AFTER_SECONDS = 24 * 60 * 60
_DELIVERY_OBSERVATIONS = frozenset({"none", "confirmed", "failed"})


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
    if type(data.get("active")) is not bool:
        return {}
    if data["active"] is False:
        return (
            data
            if set(data) == {"schema", "active", "cleared_at"}
            and _parse_utc(data.get("cleared_at")) is not None
            else {}
        )
    if set(data) != {
        "schema",
        "active",
        "fingerprint",
        "last_seen_at",
        "last_selected_for_delivery_at",
        "last_delivery_confirmed_at",
        "pending_delivery",
        "suppressed_runs",
    }:
        return {}
    fingerprint = data.get("fingerprint")
    selected_at = data.get("last_selected_for_delivery_at")
    confirmed_at = data.get("last_delivery_confirmed_at")
    suppressed_runs = data.get("suppressed_runs")
    pending = data.get("pending_delivery")
    if (
        not isinstance(fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
        or _parse_utc(data.get("last_seen_at")) is None
        or _parse_utc(selected_at) is None
        or (confirmed_at is not None and _parse_utc(confirmed_at) is None)
        or type(suppressed_runs) is not int
        or suppressed_runs < 0
    ):
        return {}
    if pending is not None:
        if not isinstance(pending, dict) or set(pending) != {
            "fingerprint",
            "selected_at",
            "observed_previous_run_at",
        }:
            return {}
        baseline = pending.get("observed_previous_run_at")
        if (
            pending.get("fingerprint") != fingerprint
            or _parse_utc(pending.get("selected_at")) is None
            or (
                baseline is not None
                and (not isinstance(baseline, str) or not (0 < len(baseline) <= 80))
            )
        ):
            return {}
    return data


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _locked_state(path: Path) -> Iterator[None]:
    """Serialize the state read/modify/write cycle across cron processes."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    lock_path = path.with_name(f".{path.name}.lock")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError("invalid blocker state lock")
        os.fchmod(descriptor, 0o600)
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _atomic_write_private(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    raw = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    )
    descriptor = -1
    try:
        descriptor = os.open(tmp, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        os.chmod(path, 0o600)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        tmp.unlink(missing_ok=True)


def decide_blocker_delivery(
    state_path: Path,
    *,
    fingerprint: str,
    now: datetime | None = None,
    repeat_after_seconds: int = DEFAULT_REPEAT_AFTER_SECONDS,
    observed_previous_run_at: str | None = None,
    previous_delivery_status: str | None = None,
) -> dict[str, Any]:
    """Persist and return whether this run should print a blocker notification.

    ``emit`` means only "selected for downstream delivery".  A delivery is
    recorded as confirmed on the *next* cron invocation, after the scheduler's
    run-bound ``last_delivery_status`` / ``last_delivery_confirmed_at`` pair
    proves that every resolved target returned an explicit success receipt.
    This prevents state from claiming a Discord delivery merely because no
    transport exception was observed.
    """

    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("invalid blocker fingerprint")
    if repeat_after_seconds < 60:
        raise ValueError("repeat_after_seconds must be at least 60")
    if previous_delivery_status not in _DELIVERY_OBSERVATIONS | {None}:
        raise ValueError("invalid previous delivery status")
    if observed_previous_run_at is not None:
        if not isinstance(observed_previous_run_at, str) or not (
            0 < len(observed_previous_run_at) <= 80
        ):
            raise ValueError("invalid observed previous run timestamp")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    with _locked_state(state_path):
        state = _load_state(state_path)
        previous = state.get("fingerprint")
        pending = state.get("pending_delivery")
        if not isinstance(pending, dict):
            pending = None

        confirmed_at = state.get("last_delivery_confirmed_at")
        prior_delivery_failed = False
        prior_delivery_reconciled = False
        instrumented = previous_delivery_status in _DELIVERY_OBSERVATIONS
        if pending and pending.get("fingerprint") == previous:
            baseline = pending.get("observed_previous_run_at")
            completion_observed = (
                observed_previous_run_at is not None
                and observed_previous_run_at != baseline
            )
            if completion_observed and previous_delivery_status == "confirmed":
                confirmed_at = _iso(current)
                pending = None
                prior_delivery_reconciled = True
            elif completion_observed and previous_delivery_status == "failed":
                pending = None
                prior_delivery_failed = True
                prior_delivery_reconciled = True

        last_confirmed = _parse_utc(confirmed_at)
        last_selected = _parse_utc(state.get("last_selected_for_delivery_at"))
        confirmed_age = (
            (current - last_confirmed).total_seconds() if last_confirmed else None
        )
        selected_age = (
            (current - last_selected).total_seconds() if last_selected else None
        )

        if previous != fingerprint:
            emit, reason = True, "new_or_changed_blocker"
            suppressed_runs = 0
            confirmed_at = None
            pending = None
        elif prior_delivery_failed:
            emit, reason = True, "previous_delivery_failed_retry"
            suppressed_runs = int(state.get("suppressed_runs") or 0)
        elif pending is not None and instrumented:
            # The scheduler did not persist completion of the selected run.
            # Retry rather than inventing a delivery receipt.
            emit, reason = True, "previous_delivery_unconfirmed_retry"
            suppressed_runs = int(state.get("suppressed_runs") or 0)
        elif confirmed_age is not None and confirmed_age >= repeat_after_seconds:
            emit, reason = True, "repeat_window_elapsed"
            suppressed_runs = int(state.get("suppressed_runs") or 0)
        elif confirmed_age is not None:
            emit, reason = False, "unchanged_delivered_blocker_suppressed"
            suppressed_runs = int(state.get("suppressed_runs") or 0) + 1
        elif selected_age is None or selected_age >= repeat_after_seconds:
            emit, reason = True, "unconfirmed_repeat_window_elapsed"
            suppressed_runs = int(state.get("suppressed_runs") or 0)
        else:
            # Compatibility for a direct/manual invocation that has no generic
            # scheduler observation variables.  It suppresses noise but never
            # records the selection as a real platform delivery.
            emit, reason = False, "unchanged_selection_suppressed_unconfirmed"
            suppressed_runs = int(state.get("suppressed_runs") or 0) + 1

        selected_at = state.get("last_selected_for_delivery_at")
        if emit:
            selected_at = _iso(current)
            pending = {
                "fingerprint": fingerprint,
                "selected_at": selected_at,
                "observed_previous_run_at": observed_previous_run_at,
            }
        payload = {
            "schema": STATE_SCHEMA,
            "active": True,
            "fingerprint": fingerprint,
            "last_seen_at": _iso(current),
            "last_selected_for_delivery_at": selected_at,
            "last_delivery_confirmed_at": confirmed_at,
            "pending_delivery": pending,
            "suppressed_runs": suppressed_runs,
        }
        _atomic_write_private(state_path, payload)
        return {
            "emit": emit,
            "reason": reason,
            "suppressed_runs": suppressed_runs,
            "repeat_after_seconds": repeat_after_seconds,
            "delivery_confirmed_at": confirmed_at,
            "pending_delivery": pending is not None,
            "prior_delivery_reconciled": prior_delivery_reconciled,
        }


def clear_blocker_delivery_state(
    state_path: Path, *, now: datetime | None = None
) -> None:
    """Mark the blocker inactive so recurrence is treated as a new event."""

    if not state_path.exists():
        return
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    with _locked_state(state_path):
        _atomic_write_private(
            state_path,
            {"schema": STATE_SCHEMA, "active": False, "cleared_at": _iso(current)},
        )
