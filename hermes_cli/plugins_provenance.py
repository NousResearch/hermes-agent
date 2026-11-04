"""Plugin provenance: the reconciliation of install metadata.

Row-presence says "hermes installed this" (the .install-metadata.json
sidecar); ``.git``-presence cross-checks it, and a shipped
``install.json`` (written by the release-archive installer) cross-checks
the artifact half. The reconciliation IS the disambiguation — ``source``
alone does not distinguish a git install from a manual copy (settled
2026-09-03, plugin-auto-update plan):

  row+git                = git install      — updatable via the recorded source
  row+install.json       = artifact-pinned  — class proven; updates are
                                              computed against the catalog
  neither                = manual drop      — not auto-updatable, listed as such
  git only               = self-cloned      — adoptable; origin URL carried ready
  row only, no marker    = provenance drift — flagged with a reinstall remedy

An archive install carries no ``.git`` by design — when the sidecar row
and the bundle's own ``install.json`` agree, the tree is PROVEN, not
drift; classifying it as drift would prescribe a git-clone remedy that
downgrades the install class (UPD-5, est-qrazy).

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
    ARTIFACT = "artifact-pinned"


@dataclass
class Provenance:
    name: str
    klass: ProvenanceClass
    path: Path
    row: Optional[dict] = None          # the sidecar row, when present
    origin_url: Optional[str] = None    # self-cloned: the .git remote
    saved_update_url: Optional[str] = None  # the row's saved feed tag
    install_json: Optional[dict] = None  # artifact-pinned: the shipped marker


def read_sidecar_rows(plugins_dir: Path) -> dict:
    """The install-metadata rows for this plugins dir; {} when absent
    or unreadable (manual-only installs never wrote one)."""
    sidecar = plugins_dir / ".install-metadata.json"
    if not sidecar.is_file():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):  # ValueError: bad JSON and bytes that are not UTF-8
        return {}
    return data if isinstance(data, dict) else {}


def read_install_json(plugin_dir: Path) -> Optional[dict]:
    """The bundle's own ``install.json`` marker; None when absent or not
    a JSON object. Written by the release-archive packer/installer:
    version, source_commit, packaged_at (and source_repo from the
    one-schema pack onward)."""
    marker = plugin_dir / "install.json"
    if not marker.is_file():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _artifact_origin_proven(row: dict, install_json: dict) -> bool:
    """True when the shipped ``install.json`` proves the artifact origin.

    Admits a marker recording a source commit or a version — the fields
    every packer release writes. When the marker names a ``source_repo``
    it must agree with the row's recorded source (repo identity is
    host/path spelling, same normalization as the kill list); a marker
    from a different repo than the row paid for is drift, not proof.
    """
    commit = install_json.get("source_commit")
    version = install_json.get("version")
    has_origin = (isinstance(commit, str) and commit.strip()) or (
        isinstance(version, str) and version.strip()
    )
    if not has_origin:
        return False
    repo = install_json.get("source_repo")
    row_source = row.get("source")
    if isinstance(repo, str) and repo.strip() and isinstance(row_source, str) and row_source.strip():
        from hermes_cli.plugin_catalog import _normalize_repo

        row_repo = row_source.split("#", 1)[0]
        if row_repo.strip() and _normalize_repo(row_repo) != _normalize_repo(repo):
            return False
    return True


def _git_origin_url(plugin_dir: Path) -> Optional[str]:
    """origin's URL from the installed .git — remote get-url on a real
    repo, config-file parse as the fallback (works for partial/copy
    installs where git refuses)."""
    # fast path: real git repo. Same resolver as install/update: PATH may be minimal
    # (gateway service, Windows without Git on PATH); no git → straight to the config parse.
    from hermes_cli.plugins_cmd import _resolve_git_executable

    git = _resolve_git_executable()
    if git is not None:
        try:
            proc = subprocess.run(
                [git, "-C", str(plugin_dir), "remote", "get-url", "origin"],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=10,
                check=False,
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
        install_json = None if has_git else read_install_json(plugin_dir)

        if row and has_git:
            klass = ProvenanceClass.GIT
        elif row and not has_git and install_json is not None and _artifact_origin_proven(row, install_json):
            klass = ProvenanceClass.ARTIFACT
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
                install_json=install_json if klass is ProvenanceClass.ARTIFACT else None,
            )
        )
    return out