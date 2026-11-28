"""Backfill: classify every repo-backed install under the Hermes home.

One normalised row per install, whatever surface it came from:

=====  ============================  ===========================================
skin   source of truth              ``provenance``
=====  ============================  ===========================================
plugin ``plugins/.install-metadata.json``  ``sidecar`` (Hermes installed it)
skill  ``~/.agents/skills/.manifest.json`` ``manifest`` (owner/repo@ref:path)
mcp    ``config.yaml: mcp_servers`` + ``mcp-installs/<name>/.git``
clone  any other ``.git`` under the Hermes home
=====  ============================  ===========================================

Pure reads: no network, no writes, no globals. Rows are returned unsorted-by-
surface then sorted by id so the output is stable across runs.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

SURFACE_PLUGIN = "plugin"
SURFACE_SKILL = "skill"
SURFACE_MCP = "mcp"
SURFACE_CLONE = "clone"

PROV_SIDECAR = "sidecar"
PROV_MANIFEST = "manifest"
PROV_GIT_REMOTE = "git-remote"
PROV_ENDPOINT = "endpoint"
PROV_INFERRED = "inferred"

STATUS_ENABLED = "enabled"
STATUS_DISABLED = "disabled"
STATUS_NOT_ENABLED = "not enabled"
STATUS_PRESENT = "present"


@dataclass
class RepoInstall:
    """One repo-backed install, normalised across surfaces."""

    surface: str
    name: str
    repo: Optional[str] = None
    ref: Optional[str] = None
    sha: Optional[str] = None
    status: str = STATUS_PRESENT
    provenance: str = PROV_INFERRED
    path: Optional[str] = None
    installed_at: Optional[str] = None
    updated_at: Optional[str] = None
    manage: dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.surface}:{self.name}"

    def to_row(self) -> dict[str, Any]:
        """The registry row (``manage`` omitted when empty, so the JSON stays clean)."""
        row: dict[str, Any] = {
            "id": self.id,
            "surface": self.surface,
            "name": self.name,
            "repo": self.repo,
            "ref": self.ref,
            "sha": self.sha,
            "status": self.status,
            "provenance": self.provenance,
            "path": self.path,
            "installed_at": self.installed_at,
            "updated_at": self.updated_at,
        }
        if self.manage:
            row["manage"] = self.manage
        return row


def _git_remote(path: Path) -> Optional[str]:
    """origin's URL for *path* — git first, then a direct .git/config parse."""
    git = None
    for candidate in ("git", "/usr/bin/git"):
        try:
            proc = subprocess.run(
                [candidate, "--version"], capture_output=True, timeout=5, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0:
            git = candidate
            break
    if git is not None:
        try:
            proc = subprocess.run(
                [git, "-C", str(path), "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
            if proc.returncode == 0:
                url = (proc.stdout or "").strip()
                if url:
                    return url
        except (OSError, subprocess.TimeoutExpired):
            pass
    config = path / ".git" / "config"
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
            _key, _sep, value = stripped.partition("=")
            url = value.strip()
            if url:
                return url
    return None


def _read_json(path: Path) -> Any:
    """Parsed JSON at *path*, or None when absent/unreadable/not an object."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def _config() -> dict[str, Any]:
    """The live config, or {} — a broken config must not break the inventory."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
        return config if isinstance(config, dict) else {}
    except Exception:
        return {}


def _plugin_status(name: str, config: dict[str, Any]) -> str:
    plugins = config.get("plugins")
    if not isinstance(plugins, dict):
        return STATUS_NOT_ENABLED
    enabled = plugins.get("enabled")
    disabled = plugins.get("disabled")
    if isinstance(disabled, list) and name in disabled:
        return STATUS_DISABLED
    if isinstance(enabled, list) and name in enabled:
        return STATUS_ENABLED
    return STATUS_NOT_ENABLED


def _collect_plugins(home: Path, config: dict[str, Any]) -> list[RepoInstall]:
    plugins_dir = home / "plugins"
    if not plugins_dir.is_dir():
        return []
    rows = _read_json(plugins_dir / ".install-metadata.json")
    rows = rows if isinstance(rows, dict) else {}
    out: list[RepoInstall] = []
    try:
        entries = sorted(plugins_dir.iterdir(), key=lambda p: p.name)
    except OSError:
        return out
    for plugin_dir in entries:
        if not plugin_dir.is_dir() or plugin_dir.name.startswith("."):
            continue
        row = rows.get(plugin_dir.name)
        row = row if isinstance(row, dict) else None
        has_git = (plugin_dir / ".git").exists()
        if row is None and not has_git:
            # A plain manual drop with no repo and no record: not a repo install.
            continue
        catalog = row.get("catalog") if row else None
        catalog = catalog if isinstance(catalog, dict) else {}
        repo = catalog.get("repo") or (_git_remote(plugin_dir) if has_git else None)
        sha = catalog.get("sha") or row.get("revision") if row else None
        provenance = PROV_SIDECAR if row else PROV_GIT_REMOTE
        out.append(
            RepoInstall(
                surface=SURFACE_PLUGIN,
                name=plugin_dir.name,
                repo=repo,
                sha=sha,
                status=_plugin_status(plugin_dir.name, config),
                provenance=provenance,
                path=str(plugin_dir),
                installed_at=row.get("installed_at") if row else None,
                updated_at=row.get("updated_at") if row else None,
                manage={
                    "enable": f"hermes plugins enable {plugin_dir.name}",
                    "disable": f"hermes plugins disable {plugin_dir.name}",
                    "uninstall": f"hermes plugins remove {plugin_dir.name}",
                    "update": f"hermes plugins update {plugin_dir.name}",
                },
            )
        )
    return out


def _skills_manifest_path(home: Path) -> Path:
    """``~/.agents/skills/.manifest.json`` — the hub-install provenance store."""
    return home.parent / ".agents" / "skills" / ".manifest.json"


def _collect_skills(manifest_path: Path) -> list[RepoInstall]:
    manifest = _read_json(manifest_path)
    skills = manifest.get("skills") if isinstance(manifest, dict) else None
    out: list[RepoInstall] = []
    if not isinstance(skills, dict):
        return out
    for name, entry in skills.items():
        if not isinstance(entry, dict):
            continue
        source = entry.get("source")
        repo = ref = path = None
        if isinstance(source, str) and "@" in source:
            slug, _, rest = source.partition("@")
            repo = f"https://github.com/{slug}"
            ref, _sep, path = rest.partition(":")
        out.append(
            RepoInstall(
                surface=SURFACE_SKILL,
                name=name,
                repo=repo,
                ref=ref or None,
                sha=entry.get("version")
                if isinstance(entry.get("version"), str)
                else None,
                status=STATUS_ENABLED,
                provenance=PROV_MANIFEST,
                path=path,
                installed_at=entry.get("installedAt"),
                updated_at=entry.get("updatedAt"),
                manage={
                    "uninstall": f"hermes skills uninstall {name}",
                    "update": f"hermes skills update {name}",
                },
            )
        )
    return out


def _collect_mcp(home: Path, config: dict[str, Any]) -> list[RepoInstall]:
    servers = config.get("mcp_servers")
    if not isinstance(servers, dict):
        return []
    installs_dir = home / "mcp-installs"
    out: list[RepoInstall] = []
    for name, entry in servers.items():
        if not isinstance(entry, dict):
            continue
        install_dir = installs_dir / name
        repo = _git_remote(install_dir) if (install_dir / ".git").exists() else None
        enabled = entry.get("enabled")
        status = STATUS_DISABLED if enabled is False else STATUS_ENABLED
        out.append(
            RepoInstall(
                surface=SURFACE_MCP,
                name=name,
                repo=repo,
                status=status,
                provenance=PROV_GIT_REMOTE if repo else PROV_ENDPOINT,
                path=str(install_dir) if install_dir.is_dir() else None,
                manage={"uninstall": f"hermes mcp remove {name}"},
            )
        )
    return out


_CLONE_PRUNE = {
    "hermes-agent",
    "node_modules",
    "venv",
    ".venv",
    "logs",
    "state",
    "tools",
}


def _iter_git_dirs(home: Path, *, max_depth: int = 3):
    """Yield ``.git`` dirs under *home* up to *max_depth*, pruning heavy trees.

    A full ``rglob`` is not an option: ``~/.hermes/hermes-agent`` is itself a
    checkout carrying ``node_modules`` and its own nested repos.
    """

    def walk(directory: Path, depth: int) -> Any:
        if depth > max_depth:
            return
        try:
            entries = list(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name in _CLONE_PRUNE:
                continue
            if entry.name == ".git":
                yield entry
                continue
            yield from walk(entry, depth + 1)

    yield from walk(home, 1)


def _collect_clones(home: Path, claimed: set[Path]) -> list[RepoInstall]:
    """Git repos under the Hermes home that no other surface already claimed."""
    out: list[RepoInstall] = []
    if not home.is_dir():
        return out
    for git_dir in sorted(_iter_git_dirs(home), key=str):
        repo_dir = git_dir.parent
        try:
            resolved = repo_dir.resolve()
        except OSError:
            continue
        if resolved in claimed:
            continue
        repo = _git_remote(repo_dir)
        if not repo:
            continue
        claimed.add(resolved)
        out.append(
            RepoInstall(
                surface=SURFACE_CLONE,
                name=repo_dir.name,
                repo=repo,
                status=STATUS_PRESENT,
                provenance=PROV_GIT_REMOTE,
                path=str(repo_dir),
            )
        )
    return out


def collect_installs(home: Path | None = None) -> list[RepoInstall]:
    """Every repo-backed install under the Hermes home, sorted by id.

    *home* is injectable so tests can point the reader at a fixture tree.
    """
    home = get_hermes_home() if home is None else home
    config = _config()
    installs = (
        _collect_plugins(home, config)
        + _collect_skills(_skills_manifest_path(home))
        + _collect_mcp(home, config)
    )
    claimed = {Path(i.path).resolve() for i in installs if i.path}
    installs += _collect_clones(home, claimed)
    return sorted(installs, key=lambda i: i.id)
