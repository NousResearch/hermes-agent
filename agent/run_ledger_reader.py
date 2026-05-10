"""Read-only retrieval helpers for durable run ledgers."""

from __future__ import annotations

import contextlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

try:  # POSIX shared locks; read paths degrade only where fcntl is unavailable.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


DEFAULT_READ_LOCK_TIMEOUT_SECONDS = 2.0
DEFAULT_EVENT_LIMIT = 200

_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_EVENT_ID_RE = re.compile(r"^evt_(\d{9})$")
_CAPSULE_ID_RE = re.compile(r"^cap_(\d{9})(?:\.json)?$")
_CAPSULE_NAME_RE = re.compile(r"^cap_(\d{9})\.json$")


class RunLedgerReadError(RuntimeError):
    """Raised when a read-only ledger request is invalid or unsafe."""


class RunLedgerLockTimeout(RunLedgerReadError):
    """Raised when a read snapshot cannot acquire an existing shared lock."""


@dataclass(frozen=True)
class RunSpan:
    run_id: str
    start_seq: int | None = None
    end_seq: int | None = None


@dataclass
class _LedgerSnapshot:
    run_id: str
    run_root: Path
    events: list[dict[str, Any]]
    corrupt_lines: list[dict[str, Any]]


def _hermes_home(hermes_home: Path | str | None = None) -> Path:
    return Path(hermes_home) if hermes_home is not None else get_hermes_home()


def _validate_run_id(run_id: str) -> str:
    value = str(run_id or "").strip()
    if not value:
        raise RunLedgerReadError("run id is required")
    if value in {".", ".."} or not _SAFE_SEGMENT_RE.fullmatch(value):
        raise RunLedgerReadError(f"unsafe run id: {value!r}")
    return value


def _seq_from_token(token: str) -> int:
    token = token.strip()
    if not token:
        raise RunLedgerReadError("malformed event span range")
    match = _EVENT_ID_RE.fullmatch(token)
    if match:
        return int(match.group(1))
    if token.isdigit():
        seq = int(token)
        if seq > 0:
            return seq
    raise RunLedgerReadError(f"malformed event span token: {token!r}")


def parse_run_span(handle: str) -> RunSpan:
    """Parse ``RUN_ID[:START..END]`` handles used by recovery prompts."""

    raw = str(handle or "").strip()
    if not raw:
        raise RunLedgerReadError("run span is required")
    run_id, sep, span = raw.partition(":")
    run_id = _validate_run_id(run_id)
    if not sep:
        return RunSpan(run_id=run_id)
    if ".." not in span:
        raise RunLedgerReadError("malformed event span range; expected START..END")
    start_raw, end_raw = span.split("..", 1)
    start = _seq_from_token(start_raw)
    end = _seq_from_token(end_raw)
    if start > end:
        raise RunLedgerReadError("malformed event span range; start is after end")
    return RunSpan(run_id=run_id, start_seq=start, end_seq=end)


def _run_root_for(run_id: str, *, hermes_home: Path | str | None = None) -> Path:
    run_id = _validate_run_id(run_id)
    runs_root = _hermes_home(hermes_home) / "runs"
    if runs_root.is_symlink():
        raise RunLedgerReadError("refusing symlinked runs directory")
    run_root = runs_root / run_id
    if not run_root.exists():
        raise RunLedgerReadError(f"run ledger not found: {run_id}")
    if not run_root.is_dir():
        raise RunLedgerReadError(f"run ledger root is not a directory: {run_id}")
    if run_root.is_symlink():
        raise RunLedgerReadError(f"refusing symlinked run ledger root: {run_id}")
    try:
        run_root.resolve().relative_to(runs_root.resolve())
    except (OSError, ValueError) as exc:
        raise RunLedgerReadError(f"run ledger root escapes runs directory: {run_id}") from exc
    return run_root


@contextlib.contextmanager
def _shared_lock_if_present(lock_path: Path, *, timeout_seconds: float):
    if not lock_path.exists():
        yield
        return
    with lock_path.open("r", encoding="utf-8") as lock_fh:
        if fcntl is not None:
            deadline = time.monotonic() + max(0.0, timeout_seconds)
            while True:
                try:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if time.monotonic() >= deadline:
                        raise RunLedgerLockTimeout(
                            f"timed out acquiring shared run ledger lock: {lock_path}"
                        ) from exc
                    time.sleep(0.05)
        try:
            yield
        finally:
            if fcntl is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _read_snapshot_bytes(run_root: Path, *, lock_timeout_seconds: float) -> bytes:
    events_path = run_root / "events.jsonl"
    lock_path = run_root / "events.lock"
    with _shared_lock_if_present(lock_path, timeout_seconds=lock_timeout_seconds):
        if not events_path.exists():
            return b""
        try:
            return events_path.read_bytes()
        except OSError as exc:
            raise RunLedgerReadError(f"could not read run ledger events: {exc}") from exc


def _parse_jsonl_snapshot(data: bytes) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    corrupt_lines: list[dict[str, Any]] = []
    if not data:
        return events, corrupt_lines

    lines = data.splitlines(keepends=True)
    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        text = raw.decode("utf-8", errors="replace")
        if not raw.endswith(b"\n"):
            corrupt_lines.append(
                {
                    "line_number": line_number,
                    "error": "partial final JSONL line",
                    "preview": text[:200],
                }
            )
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            corrupt_lines.append(
                {
                    "line_number": line_number,
                    "error": str(exc),
                    "preview": text[:200],
                }
            )
            continue
        if not isinstance(parsed, dict):
            corrupt_lines.append(
                {
                    "line_number": line_number,
                    "error": "event is not an object",
                    "preview": text[:200],
                }
            )
            continue
        events.append(parsed)
    return events, corrupt_lines


def _read_ledger_snapshot(
    run_id: str,
    *,
    hermes_home: Path | str | None = None,
    lock_timeout_seconds: float = DEFAULT_READ_LOCK_TIMEOUT_SECONDS,
) -> _LedgerSnapshot:
    run_root = _run_root_for(run_id, hermes_home=hermes_home)
    data = _read_snapshot_bytes(run_root, lock_timeout_seconds=lock_timeout_seconds)
    events, corrupt_lines = _parse_jsonl_snapshot(data)
    return _LedgerSnapshot(
        run_id=run_id,
        run_root=run_root,
        events=events,
        corrupt_lines=corrupt_lines,
    )


def _latest_capsule_relative(run_root: Path) -> str | None:
    capsules_dir = run_root / "capsules"
    if not capsules_dir.is_dir() or capsules_dir.is_symlink():
        return None
    best: tuple[int, Path] | None = None
    for path in capsules_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            continue
        match = _CAPSULE_NAME_RE.fullmatch(path.name)
        if not match:
            continue
        item = (int(match.group(1)), path)
        if best is None or item[0] > best[0]:
            best = item
    if best is None:
        return None
    return best[1].relative_to(run_root).as_posix()


def _recover_from_events(events: list[dict[str, Any]], *, max_completed: int = DEFAULT_EVENT_LIMIT) -> dict[str, Any]:
    in_flight: dict[str, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    artifact_refs: list[dict[str, Any]] = []
    terminal_types = {"tool.finished", "tool.failed", "tool.skipped"}
    for event in events:
        refs = event.get("artifact_refs")
        if isinstance(refs, list):
            artifact_refs.extend(ref for ref in refs if isinstance(ref, dict))
        call_id = event.get("tool_call_id")
        if event.get("type") == "tool.started" and call_id:
            in_flight[str(call_id)] = event
        elif event.get("type") in terminal_types and call_id:
            in_flight.pop(str(call_id), None)
            completed.append(
                {
                    "event_id": event.get("event_id"),
                    "event_seq": event.get("event_seq"),
                    "tool_name": event.get("tool_name"),
                    "tool_call_id": call_id,
                    "status": event.get("status"),
                    "duration_ms": event.get("duration_ms"),
                    "output": event.get("output") or {},
                    "artifact_refs": refs if isinstance(refs, list) else [],
                }
            )
    completed_limit = max(0, int(max_completed))
    return {
        "in_flight": in_flight,
        "recent_completed_tools": completed[-completed_limit:] if completed_limit else [],
        "artifact_refs": artifact_refs,
    }


def list_run_ledgers(
    *,
    hermes_home: Path | str | None = None,
    limit: int | None = None,
    lock_timeout_seconds: float = DEFAULT_READ_LOCK_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """List direct, safe run ledger children under ``HERMES_HOME/runs``."""

    runs_root = _hermes_home(hermes_home) / "runs"
    if not runs_root.is_dir() or runs_root.is_symlink():
        return []
    summaries: list[dict[str, Any]] = []
    for run_root in runs_root.iterdir():
        if run_root.is_symlink() or not run_root.is_dir():
            continue
        run_id = run_root.name
        if run_id in {".", ".."} or not _SAFE_SEGMENT_RE.fullmatch(run_id):
            continue
        snapshot = _read_ledger_snapshot(
            run_id,
            hermes_home=hermes_home,
            lock_timeout_seconds=lock_timeout_seconds,
        )
        last_event = snapshot.events[-1] if snapshot.events else {}
        recovery = _recover_from_events(snapshot.events)
        summaries.append(
            {
                "run_id": run_id,
                "run_root": str(run_root),
                "event_count": len(snapshot.events),
                "last_event_id": last_event.get("event_id"),
                "last_event_seq": last_event.get("event_seq"),
                "last_event_type": last_event.get("type"),
                "last_event_timestamp_utc": last_event.get("timestamp_utc"),
                "latest_capsule": _latest_capsule_relative(run_root),
                "in_flight_count": len(recovery["in_flight"]),
                "corrupt_line_count": len(snapshot.corrupt_lines),
            }
        )
    summaries.sort(
        key=lambda item: (
            str(item.get("last_event_timestamp_utc") or ""),
            int(item.get("last_event_seq") or 0),
            item.get("run_id") or "",
        ),
        reverse=True,
    )
    if limit is not None:
        return summaries[: max(0, limit)]
    return summaries


def _event_matches(event: dict[str, Any], filters: dict[str, str] | None) -> bool:
    if not filters:
        return True
    field_map = {
        "type": "type",
        "tool": "tool_name",
        "tool_name": "tool_name",
        "session": "session_id",
        "session_id": "session_id",
        "status": "status",
    }
    for key, value in filters.items():
        if value in (None, ""):
            continue
        field = field_map.get(key, key)
        if str(event.get(field) or "") != str(value):
            return False
    return True


def fetch_run_events(
    handle: str,
    *,
    hermes_home: Path | str | None = None,
    filters: dict[str, str] | None = None,
    limit: int = DEFAULT_EVENT_LIMIT,
    lock_timeout_seconds: float = DEFAULT_READ_LOCK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    span = parse_run_span(handle)
    snapshot = _read_ledger_snapshot(
        span.run_id,
        hermes_home=hermes_home,
        lock_timeout_seconds=lock_timeout_seconds,
    )
    max_events = max(0, int(limit))
    selected: list[dict[str, Any]] = []
    truncated = False
    next_start: str | None = None
    for event in snapshot.events:
        seq = event.get("event_seq")
        if not isinstance(seq, int):
            continue
        if span.start_seq is not None and seq < span.start_seq:
            continue
        if span.end_seq is not None and seq > span.end_seq:
            continue
        if not _event_matches(event, filters):
            continue
        if len(selected) >= max_events:
            truncated = True
            next_start = event.get("event_id") or (f"evt_{seq:09d}" if seq else None)
            break
        selected.append(event)
    return {
        "run_id": span.run_id,
        "run_root": str(snapshot.run_root),
        "events": selected,
        "matched_count": len(selected),
        "limit": max_events,
        "truncated": truncated,
        "next_start": next_start,
        "corrupt_lines": snapshot.corrupt_lines,
    }


def _first_symlink_component(path: Path) -> Path | None:
    check_path = path if path.is_absolute() else path.absolute()
    current = Path(check_path.anchor) if check_path.anchor else Path()
    parts = check_path.parts[1:] if check_path.anchor else check_path.parts
    for part in parts:
        current = current / part
        if current.is_symlink():
            return current
    return None


def _resolve_capsule_path(run_root: Path, capsule: str | None, *, latest: bool) -> Path:
    capsules_dir = run_root / "capsules"
    if not capsules_dir.is_dir() or capsules_dir.is_symlink():
        raise RunLedgerReadError("capsules directory not found")
    if latest or not capsule:
        candidates: list[tuple[int, Path]] = []
        for path in capsules_dir.iterdir():
            match = _CAPSULE_NAME_RE.fullmatch(path.name)
            if match and path.is_file():
                candidates.append((int(match.group(1)), path))
        if not candidates:
            raise RunLedgerReadError("no state capsules found")
        path = max(candidates, key=lambda item: item[0])[1]
    else:
        raw = str(capsule).strip()
        if not raw:
            raise RunLedgerReadError("capsule path is required")
        match = _CAPSULE_ID_RE.fullmatch(raw)
        if match:
            path = capsules_dir / f"cap_{int(match.group(1)):09d}.json"
        else:
            supplied = Path(raw)
            if supplied in {Path("."), Path("..")} or ".." in supplied.parts:
                raise RunLedgerReadError(f"unsafe capsule path outside capsules directory: {capsule}")
            if supplied.is_absolute():
                path = supplied
            elif supplied.parts and supplied.parts[0] == "capsules":
                path = run_root / supplied
            else:
                path = capsules_dir / supplied
    symlink_component = _first_symlink_component(path)
    if symlink_component is not None:
        raise RunLedgerReadError(f"refusing symlinked capsule path: {symlink_component}")
    try:
        resolved = path.resolve(strict=True)
        capsules_resolved = capsules_dir.resolve(strict=True)
        resolved.relative_to(capsules_resolved)
    except FileNotFoundError as exc:
        raise RunLedgerReadError(f"capsule not found: {capsule or 'latest'}") from exc
    except (OSError, ValueError) as exc:
        raise RunLedgerReadError(f"unsafe capsule path outside capsules directory: {capsule}") from exc
    if not resolved.is_file():
        raise RunLedgerReadError(f"capsule is not a file: {capsule or 'latest'}")
    return resolved


def read_run_capsule(
    run_id: str,
    *,
    hermes_home: Path | str | None = None,
    latest: bool = False,
    capsule: str | None = None,
) -> dict[str, Any]:
    run_root = _run_root_for(run_id, hermes_home=hermes_home)
    path = _resolve_capsule_path(run_root, capsule, latest=latest)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunLedgerReadError(f"capsule is not valid JSON: {path.name}") from exc
    if not isinstance(data, dict):
        raise RunLedgerReadError(f"capsule is not a JSON object: {path.name}")
    return {
        "run_id": run_id,
        "run_root": str(run_root),
        "relative_path": path.relative_to(run_root).as_posix(),
        "capsule": data,
    }


def recover_run(
    run_id: str,
    *,
    hermes_home: Path | str | None = None,
    max_completed: int = DEFAULT_EVENT_LIMIT,
    lock_timeout_seconds: float = DEFAULT_READ_LOCK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    run_id = _validate_run_id(run_id)
    snapshot = _read_ledger_snapshot(
        run_id,
        hermes_home=hermes_home,
        lock_timeout_seconds=lock_timeout_seconds,
    )
    return {
        "run_id": run_id,
        "run_root": str(snapshot.run_root),
        "recovery": _recover_from_events(snapshot.events, max_completed=max_completed),
        "corrupt_lines": snapshot.corrupt_lines,
    }
