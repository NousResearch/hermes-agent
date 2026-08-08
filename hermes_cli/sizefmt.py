"""Small shared size-formatting helpers for CLI/agent output.

Public home for the human-readable byte formatter that previously existed
as five near-identical private copies (``hermes_cli/backup.py``,
``hermes_cli/checkpoints.py``, ``hermes_cli/doctor.py``,
``agent/context_references.py``, ``agent/curator_backup.py``). Sibling of
``hermes_cli.timefmt``, and kept dependency-free for the same reason:
lightweight consumers must not drag in the whole CLI surface.

Two in-repo formatters intentionally do NOT delegate here:

* ``hermes_cli/session_recovery.py`` uses binary suffixes (KiB/MiB/GiB)
  throughout its recovery report — a deliberate, self-consistent style.
* ``gateway/platforms/qqbot/chunked_upload.py`` renders bytes with one
  decimal ("100.0 B", pinned by tests) inside a self-contained upload
  protocol module.
"""

from __future__ import annotations


def format_bytes(n, *, fallback: str = "?") -> str:
    """1234567 -> '1.2 MB' (B/KB/MB/GB/TB; integer bytes, one decimal above).

    Accepts anything ``float()`` accepts; returns *fallback* for None or
    unparseable input so display call sites never raise.
    """
    try:
        size = float(n)
    except (TypeError, ValueError):
        return fallback
    if size < 1024:
        return f"{int(size)} B"
    for unit in ("KB", "MB", "GB", "TB"):
        size /= 1024.0
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
    return f"{size:.1f} TB"  # unreachable; keeps type-checkers satisfied
