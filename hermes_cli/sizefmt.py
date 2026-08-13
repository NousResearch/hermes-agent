"""Small shared size-formatting helpers for CLI/agent output.

Sibling of ``hermes_cli.timefmt`` (same extraction rationale: a tiny
purpose-named module lightweight consumers can import without dragging in
the CLI surface). Replaces six near-identical private byte formatters.

Two in-repo formatters intentionally do NOT delegate here:

* ``hermes_cli/session_recovery.py`` uses binary suffixes (KiB/MiB/GiB)
  throughout its recovery report — a deliberate, self-consistent style.
* ``gateway/platforms/qqbot/chunked_upload.py`` renders bytes with one
  decimal ("100.0 B", pinned by tests) inside a self-contained upload
  protocol module.
"""

from __future__ import annotations


def format_bytes(n) -> str:
    """1234567 -> '1.2 MB' (B/KB/MB/GB/TB; integer bytes, one decimal above).

    Accepts anything ``float()`` accepts; returns ``"?"`` for None or
    unparseable input so display call sites never raise (contract inherited
    from doctor's original copy — its stats dict tolerates None fields).
    """
    try:
        size = float(n)
    except (TypeError, ValueError):
        return "?"
    if size < 1024:
        return f"{int(size)} B"
    for unit in ("KB", "MB", "GB"):
        size /= 1024.0
        if size < 1024:
            return f"{size:.1f} {unit}"
    return f"{size / 1024.0:.1f} TB"
