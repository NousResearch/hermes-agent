"""Matrix digest-detail reaction registry.

Cron digest jobs can collapse several local source-job reports into one Matrix
message.  This module stores the small, non-secret mapping needed for a later
Matrix 🧾 reaction to resolve that digest event back to the local source reports.
The registry intentionally stores metadata only: job IDs, display names, event
IDs, and paths under ``HERMES_HOME/cron/output``.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

_DETAIL_EMOJI = "🧾"
_DETAIL_TTL_SECONDS = 7 * 24 * 60 * 60
_MAX_DETAIL_CHARS = 3500
_REGISTRY_LOCK_TIMEOUT_SECONDS = 10.0
_REGISTRY_LOCK_POLL_SECONDS = 0.01
_KEYCAPS = (
    "1️⃣",
    "2️⃣",
    "3️⃣",
    "4️⃣",
    "5️⃣",
    "6️⃣",
    "7️⃣",
    "8️⃣",
    "9️⃣",
    "🔟",
)


def _registry_path() -> Path:
    return get_hermes_home() / "state" / "matrix-digest-reactions.json"


def _cron_output_root() -> Path:
    return get_hermes_home() / "cron" / "output"


def _registry_lock_path() -> Path:
    return _registry_path().with_suffix(".lock")


@contextmanager
def _registry_transaction_lock():
    """Serialize registry read-modify-write transactions across processes."""
    path = _registry_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        try:
            path.chmod(0o600)
        except OSError:
            pass
        deadline = time.monotonic() + _REGISTRY_LOCK_TIMEOUT_SECONDS
        acquired = False
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            while not acquired:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("timed out acquiring Matrix digest registry lock")
                    time.sleep(_REGISTRY_LOCK_POLL_SECONDS)
        else:
            import fcntl

            while not acquired:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("timed out acquiring Matrix digest registry lock")
                    time.sleep(_REGISTRY_LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            if acquired:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _load_registry() -> dict[str, dict[str, Any]]:
    path = _registry_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_registry(data: dict[str, dict[str, Any]]) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".matrix_digest_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _record_key(room_id: str, event_id: str) -> str:
    return f"{room_id}\0{event_id}"


def normalize_context_from(value: Any) -> list[str]:
    """Return source job IDs from a cron ``context_from`` value."""
    if value is None:
        return []
    if isinstance(value, str):
        candidates: Iterable[Any] = [value]
    elif isinstance(value, Iterable):
        candidates = value
    else:
        return []
    result: list[str] = []
    for item in candidates:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _safe_output_path(path: str | Path | None) -> str:
    if not path:
        return ""
    try:
        candidate = Path(path).expanduser().resolve()
        candidate.relative_to(_cron_output_root().resolve())
    except (OSError, ValueError):
        return ""
    return str(candidate)


def _latest_output_path(job_id: str) -> str:
    if not job_id:
        return ""
    output_dir = (_cron_output_root() / job_id).resolve()
    try:
        output_dir.relative_to(_cron_output_root().resolve())
    except ValueError:
        return ""
    if not output_dir.is_dir():
        return ""
    try:
        safe_candidates = []
        for path in output_dir.glob("*.md"):
            safe_path = _safe_output_path(path)
            if not safe_path:
                continue
            candidate = Path(safe_path)
            if candidate.is_file():
                safe_candidates.append(candidate)
    except OSError:
        return ""
    if not safe_candidates:
        return ""
    safe_candidates.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return str(safe_candidates[0])


def register_digest_delivery(
    *,
    room_id: str,
    event_id: str,
    digest_job: dict[str, Any],
    source_job_ids: Iterable[str],
    output_file: str | Path | None = None,
    source_names: dict[str, str] | None = None,
    now: float | None = None,
) -> None:
    """Persist metadata for a Matrix digest message that can answer 🧾 details."""
    room_id = str(room_id or "").strip()
    event_id = str(event_id or "").strip()
    sources = normalize_context_from(list(source_job_ids or []))
    if not room_id or not event_id or not sources:
        return

    now_ts = float(time.time() if now is None else now)
    names = source_names or {}
    record = {
        "room_id": room_id,
        "event_id": event_id,
        "created_at": now_ts,
        "expires_at": now_ts + _DETAIL_TTL_SECONDS,
        "digest_job_id": str(digest_job.get("id") or ""),
        "digest_name": str(digest_job.get("name") or digest_job.get("id") or "Digest"),
        "digest_output_path": _safe_output_path(output_file),
        "sources": [
            {
                "job_id": job_id,
                "name": str(names.get(job_id) or job_id),
                "output_path": _latest_output_path(job_id),
            }
            for job_id in sources[: len(_KEYCAPS)]
        ],
    }
    with _registry_transaction_lock():
        data = _load_registry()
        data[_record_key(room_id, event_id)] = record
        # Opportunistic pruning keeps the state file bounded without a background job.
        pruned = {
            key: existing
            for key, existing in data.items()
            if float(existing.get("expires_at") or 0) >= now_ts
        }
        _save_registry(pruned)


def resolve_digest_delivery(
    room_id: str,
    event_id: str,
    *,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Return a digest-detail record for a Matrix event, or None if unavailable."""
    now_ts = float(time.time() if now is None else now)
    key = _record_key(str(room_id or ""), str(event_id or ""))
    with _registry_transaction_lock():
        data = _load_registry()
        record = data.get(key)
        if not record:
            return None
        if float(record.get("expires_at") or 0) < now_ts:
            data.pop(key, None)
            _save_registry(data)
            return None
        return record


def detail_reaction_emoji() -> str:
    return _DETAIL_EMOJI


def selection_reactions(count: int) -> list[str]:
    return list(_KEYCAPS[: max(0, min(count, len(_KEYCAPS)))])


def selection_index_for_reaction(key: str) -> int | None:
    try:
        return _KEYCAPS.index(str(key or ""))
    except ValueError:
        return None


def _extract_response_section(text: str) -> str:
    marker = "\n## Response"
    idx = text.find(marker)
    if idx == -1 and text.startswith("## Response"):
        idx = 0
    if idx == -1:
        return text.strip()
    response = text[idx:].split("\n", 1)
    if len(response) == 1:
        return ""
    # Stop before any later top-level saved-output sections.
    body = response[1]
    for stop in ("\n## Error", "\n## Script Output", "\n## Prompt"):
        stop_idx = body.find(stop)
        if stop_idx != -1:
            body = body[:stop_idx]
    return body.strip()


def _read_source_detail(source: dict[str, Any]) -> str | None:
    path = _safe_output_path(source.get("output_path"))
    if not path:
        path = _latest_output_path(str(source.get("job_id") or ""))
    if not path:
        return None
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    detail = _extract_response_section(text)
    if len(detail) > _MAX_DETAIL_CHARS:
        detail = detail[: _MAX_DETAIL_CHARS - 120].rstrip() + (
            "\n\n… [gekürzt; vollständiger Einzelbericht liegt lokal im Cron-Output]"
        )
    return detail or None


def format_digest_detail_response(record: dict[str, Any], *, source_index: int = 0) -> str:
    """Return a safe Matrix reply for one source report in a registered digest."""
    sources = record.get("sources") if isinstance(record, dict) else None
    if not isinstance(sources, list) or not (0 <= source_index < len(sources)):
        return "⚠️ Der angeforderte Einzelbericht ist nicht mehr verfügbar."
    source = sources[source_index]
    name = str(source.get("name") or source.get("job_id") or "Quelle")
    detail = _read_source_detail(source)
    if not detail:
        return f"⚠️ The detail output is no longer available for **{name}**."
    return f"**🧾 Einzelbericht: {name}**\n\n{detail}"


def format_digest_source_selection(record: dict[str, Any]) -> str:
    sources = record.get("sources") if isinstance(record, dict) else []
    lines = ["**🧾 Mehrere Einzelberichte verfügbar**", "", "Wähle per Reaction:"]
    for emoji, source in zip(selection_reactions(len(sources)), sources):
        lines.append(f"{emoji} {source.get('name') or source.get('job_id') or 'Quelle'}")
    return "\n".join(lines)
