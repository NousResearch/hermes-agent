"""Release-age exclude upkeep for `hermes pm update --pnpm`."""
from __future__ import annotations

from pathlib import Path
import re


def _locked_versions(lock_text: str) -> dict[str, set[str]]:
    import hermes_yaml

    locked: dict[str, set[str]] = {}
    for key in (hermes_yaml.safe_load(lock_text) or {}).get("packages") or {}:
        name, _, version = str(key).partition("(")[0].rpartition("@")
        if name:
            locked.setdefault(name, set()).add(version)
    return locked


def _kept_versions(entry: str, locked: dict[str, set[str]]) -> list[str] | None:
    """The versions of an exact `name@v1 || v2` entry the lock still holds; None for any other shape."""
    name, _, versions = entry.rpartition("@")
    if not name or "*" in name or not versions:
        return None
    return [v for v in (part.strip() for part in versions.split("||")) if v in locked.get(name, set())]


def prune_excludes(workspace_yaml: Path, lock_yaml: Path) -> list[str]:
    """Drop the minimumReleaseAgeExclude versions `lock_yaml` no longer holds; returns the dropped entries.

    pnpm's own minimumReleaseAgeExcludePrune would run on every install, user builds included, and
    rewrite a tracked file there; this runs only where the lock is refreshed. Each entry is one
    line, `  - name@version  # why`, so a dropped line takes its reason with it.
    """
    import hermes_yaml

    text = workspace_yaml.read_text(encoding="utf-8")
    entries = (hermes_yaml.safe_load(text) or {}).get("minimumReleaseAgeExclude") or []
    locked = _locked_versions(lock_yaml.read_text(encoding="utf-8"))
    dropped, lines, in_block = [], [], False
    for line in text.splitlines(keepends=True):
        if not line.startswith((" ", "\t", "#")) and line.strip():
            in_block = line.startswith("minimumReleaseAgeExclude:")
        item = re.match(r"(\s+- )(['\"]?)([^'\"#\s][^'\"#]*?)\2(\s*(?:#.*)?)$", line.rstrip("\r\n")) if in_block else None
        if item and item.group(3).strip() in entries:
            entry = item.group(3).strip()
            kept = _kept_versions(entry, locked)
            if kept is not None and len(kept) < len(entry.rpartition("@")[2].split("||")):
                dropped.append(entry)
                if not kept:
                    continue
                newline = line[len(line.rstrip("\r\n")):]
                line = f"{item.group(1)}{entry.rpartition('@')[0]}@{' || '.join(kept)}{item.group(4)}{newline}"
        lines.append(line)
    if dropped:
        workspace_yaml.write_text("".join(lines), encoding="utf-8")
    return dropped


