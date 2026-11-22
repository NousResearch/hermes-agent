"""The logs an upgrade cell's machine writes, and what they gained since a point in the cell.

One reader for every upgrade test that checks log lines (``test_upgrade_path``, the handoff
scenario, the rehearsal cells), so coverage and rotation handling are the same everywhere.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def log_files(homes: Iterable[Path], extra: Iterable[Path] = ()) -> list[Path]:
    """Every log under the participating ``HERMES_HOME``s and the extra process logs that exist.

    Per home: ``logs/*.log``, every profile's ``profiles/<name>/logs/*.log`` and the kanban
    worker logs (``t_*.log``, anywhere below it).
    """
    found: set[Path] = {p for p in extra if p.is_file()}
    for home in homes:
        found.update(home.glob("logs/*.log"))
        found.update(home.glob("profiles/*/logs/*.log"))
        found.update(home.rglob("t_*.log"))
    return sorted(p for p in found if p.is_file())


def log_offsets(homes: Iterable[Path], extra: Iterable[Path] = ()) -> dict[Path, int]:
    """The current size of every log (``log_files``): the point ``log_text_since`` reads from."""
    return {p: p.stat().st_size for p in log_files(homes, extra)}


def log_text_since(offsets: dict[Path, int], homes: Iterable[Path], extra: Iterable[Path] = ()) -> str:
    """What every log gained since ``offsets``, each under a ``--- <dir>/<dir>/<name> ---`` header.

    A log created since then is read whole, and so is one now shorter than its offset (rotated or
    truncated: its current content was all written since).
    """
    parts = []
    for p in log_files(homes, extra):
        start = offsets.get(p, 0)
        with p.open("rb") as fh:
            if fh.seek(0, 2) < start:
                start = 0
            fh.seek(start)
            parts.append(f"--- {'/'.join(p.parts[-3:])} ---\n" + fh.read().decode("utf-8", errors="replace"))
    return "\n".join(parts)
