"""
DREAMS.md writer — appends REM-phase narrative entries.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path


def _dreams_path(hermes_home: str | None = None) -> Path:
    base = Path(hermes_home or os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    return base / "DREAMS.md"


def append_entry(
    narrative: str,
    promoted: list[str],
    skipped_meta: list[str],
    *,
    hermes_home: str | None = None,
) -> Path:
    """Append one dream cycle entry to DREAMS.md. Returns the path written."""
    path = _dreams_path(hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"\n## Dream cycle — {ts}\n", f"{narrative.strip()}\n"]

    if promoted:
        lines.append(f"\n**Promoted to MEMORY.md ({len(promoted)}):**\n")
        for item in promoted:
            lines.append(f"- {item[:120]}\n")

    if skipped_meta:
        lines.append(f"\n**Meta-entries routed to SKILL.md ({len(skipped_meta)}):**\n")
        for item in skipped_meta:
            lines.append(f"- {item[:120]}\n")

    lines.append("\n---\n")

    with path.open("a", encoding="utf-8") as fh:
        fh.writelines(lines)

    return path


def last_entry(hermes_home: str | None = None) -> str:
    """Return the text of the most recent dream cycle entry, or empty string."""
    path = _dreams_path(hermes_home)
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8")
    parts = content.split("## Dream cycle —")
    if len(parts) < 2:
        return ""
    return ("## Dream cycle —" + parts[-1]).strip()
