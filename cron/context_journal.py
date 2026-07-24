"""
Cron context journal — lightweight, bounded, on-disk record of cron deliveries.

The journal is a JSON array of recent delivery entries, written atomically and
auto-pruned on every append so the file never grows unbounded. The model reads
it via ``read_file`` when the user references a scheduled task or reminder.

Thread-safe: atomic replace (write tmp → fsync → rename) prevents partial
reads. File lock not needed since the rename is atomic on all target platforms.
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ENTRIES = 20
_HARD_SIZE_LIMIT = 100 * 1024  # 100 KB safety net


def _get_home():
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def _max_entries() -> int:
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) or {}
        journal_cfg = cron_cfg.get("context_journal", {}) or {}
        val = int(journal_cfg.get("max_entries", _DEFAULT_MAX_ENTRIES))
        if val > 0:
            return val
    except Exception:
        pass
    return _DEFAULT_MAX_ENTRIES


def _journal_path() -> Path:
    """Return the profile-aware path to the context journal file."""
    return _get_home() / "cron" / "context_journal.json"


def _read_entries() -> list:
    """Read journal entries from disk. Returns [] on any error."""
    path = _journal_path()
    if not path.exists():
        return []
    try:
        data = path.read_text(encoding="utf-8")
        entries = json.loads(data)
        if isinstance(entries, list):
            return entries
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Failed to read cron context journal: %s", exc)
    return []


def _write_entries(entries: list) -> None:
    """Atomically write entries to the journal file."""
    path = _journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=".context_journal_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _summarize(text: str, max_chars: int = 500) -> str:
    """Extract a concise summary from raw cron output.

    Takes the first meaningful lines up to *max_chars*, appending a "[...]"
    marker when truncated. Empty input returns "".
    """
    text = (text or "").strip()
    if not text:
        return ""
    lines = text.split("\n")
    result = []
    char_count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if char_count + len(stripped) + 1 > max_chars:
            remaining = max_chars - char_count
            if remaining > 20:
                result.append(stripped[:remaining])
            else:
                if result:
                    result[-1] = result[-1] + " […]"
            break
        result.append(stripped)
        char_count += len(stripped) + 1
    return "\n".join(result)


def append_entry(job_id: str, job_name: str, content: str, delivered_at: str = None) -> None:
    """Append a delivery entry to the cron context journal.

    Written atomically and auto-pruned so the file stays bounded. Safe to
    call from any thread (atomic replace, no file lock needed).

    Args:
        job_id: Unique cron job identifier.
        job_name: Human-readable job name for display.
        content: Raw cron output (will be summarized).
        delivered_at: ISO-8601 timestamp. Auto-generated when omitted.
    """
    if not content:
        return
    if delivered_at is None:
        delivered_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    entry = {
        "job_id": job_id,
        "job_name": job_name or job_id,
        "delivered_at": delivered_at,
        "summary": _summarize(content),
    }

    entries = _read_entries()
    entries.append(entry)

    # Count-based prune: keep newest N entries
    max_n = _max_entries()
    if len(entries) > max_n:
        entries = entries[-max_n:]

    # Hard size limit: if the serialised blob somehow exceeds 100 KB, drop
    # to a tighter window. Defends against runaway entries from a bug.
    try:
        blob = json.dumps(entries, ensure_ascii=False)
        if len(blob) > _HARD_SIZE_LIMIT:
            entries = entries[-(max(10, max_n // 2)):]
    except Exception:
        entries = entries[-10:]

    _write_entries(entries)
    logger.debug(
        "Cron context journal: appended delivery for job '%s' (%d entries)",
        job_id, len(entries),
    )


def clear_journal() -> int:
    """Remove all entries from the context journal. Returns count of removed entries."""
    entries = _read_entries()
    count = len(entries)
    if count > 0:
        _write_entries([])
        logger.info("Cron context journal cleared (%d entries removed)", count)
    return count


def read_journal() -> list:
    """Return the current journal entries (for inspection / testing)."""
    return _read_entries()
