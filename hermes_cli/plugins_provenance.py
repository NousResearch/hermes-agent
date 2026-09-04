"""Plugin provenance: the 2x2 reconciliation of install metadata.

Row-presence says "hermes installed this" (the .install-metadata.json
sidecar); ``.git``-presence cross-checks it. The reconciliation IS the
disambiguation — ``source`` alone does not distinguish a git install
from a manual copy (settled 2026-09-03, plugin-auto-update plan):

  row+git   = git install      — updatable via the recorded source
  neither   = manual drop      — not auto-updatable, listed as such
  git only  = self-cloned      — adoptable; origin URL carried ready
  row only  = provenance drift — flagged with a reinstall remedy

Pure functions over (plugins_dir, sidecar rows) — no globals, no
network, fully testable. The sidecar remains the single authority for
what *hermes* installed; this module only reads it.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class ProvenanceClass(Enum):
    GIT = "git"
    MANUAL = "manual"
    SELF_CLONED = "self-cloned"
    DRIFT = "drift"


@dataclass
class Provenance:
    name: str
    klass: ProvenanceClass
    path: Path
    row: Optional[dict] = None          # the sidecar row, when present
    origin_url: Optional[str] = None    # self-cloned: the .git remote
    saved_update_url: Optional[str] = None  # the row's saved feed tag


def read_sidecar_rows(plugins_dir: Path) -> dict:
    """The install-metadata rows for this plugins dir; {} when absent
    or unreadable (manual-only installs never wrote one)."""
    sidecar = plugins_dir / ".install-metadata.json"
    if not sidecar.is_file():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _git_origin_url(plugin_dir: Path) -> Optional[str]:
    """origin's URL from the installed .git — remote get-url on a real
    repo, config-file parse as the fallback (works for partial/copy
    installs where git refuses)."""
    # fast path: real git repo
    try:
        proc = subprocess.run(
            ["git", "-C", str(plugin_dir), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            url = (proc.stdout or "").strip()
            if url:
                return url
    except (OSError, subprocess.TimeoutExpired):
        pass
    # fallback: parse .git/config directly (a .git dir with a config
    # file is all the adopt path needs)
    config = plugin_dir / ".git" / "config"
    if not config.is_file():
        return None
    try:
        text = config.read_text(encoding="utf-8-sig")
    except OSError:
        return None
    in_origin = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == '[remote "origin"]':
            in_origin = True
            continue
        if stripped.startswith("["):
            in_origin = False
            continue
        if in_origin and stripped.startswith("url"):
            _, _, value = stripped.partition("=")
            url = value.strip()
            return url or None
    return None


def plugins_provenance(plugins_dir: Path) -> list[Provenance]:
    """Classify every plugin dir under plugins_dir per the 2x2."""
    rows = read_sidecar_rows(plugins_dir)
    out: list[Provenance] = []
    if not plugins_dir.is_dir():
        return out
    try:
        entries = sorted(plugins_dir.iterdir(), key=lambda p: p.name)
    except OSError:
        return out
    for plugin_dir in entries:
        if not plugin_dir.is_dir() or plugin_dir.name.startswith("."):
            continue
        has_git = (plugin_dir / ".git").exists()
        row = rows.get(plugin_dir.name)
        row = row if isinstance(row, dict) else None

        if row and has_git:
            klass = ProvenanceClass.GIT
        elif row and not has_git:
            klass = ProvenanceClass.DRIFT
        elif has_git and not row:
            klass = ProvenanceClass.SELF_CLONED
        else:
            klass = ProvenanceClass.MANUAL

        out.append(
            Provenance(
                name=plugin_dir.name,
                klass=klass,
                path=plugin_dir,
                row=row,
                origin_url=_git_origin_url(plugin_dir) if klass is ProvenanceClass.SELF_CLONED else None,
                saved_update_url=(row or {}).get("update_url") or None,
            )
        )
    return out