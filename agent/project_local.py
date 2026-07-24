"""Project-local ``.hermes`` discovery, identity, and trust state.

This module is intentionally small and side-effect light. It gives CLI and
runtime surfaces one canonical answer for "does this git project define local
Hermes capabilities, and has this user approved the current fingerprint?".
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_json_write

try:  # POSIX only; Windows falls back to the in-process lock.
    import fcntl
except Exception:  # pragma: no cover - platform dependent
    fcntl = None  # type: ignore[assignment]

TRUST_FILENAME = "project-trust.json"
TRUST_VERSION = 1
PROJECT_CONFIG_REL = Path(".hermes") / "config.yaml"
PROJECT_SKILLS_REL = Path(".hermes") / "skills"

_trust_lock = threading.Lock()
_candidate_cache: dict[str, bool] = {}


@dataclass(frozen=True)
class ProjectSkillManifest:
    name: str
    path: str
    sha256: str


@dataclass(frozen=True)
class ProjectMCPManifest:
    config_path: str | None = None
    sha256: str = ""
    servers: tuple[str, ...] = ()
    trusted: bool = False


@dataclass(frozen=True)
class ProjectLocalState:
    canonical_id: str
    project_root: str
    hermes_dir: str
    skill_roots: tuple[str, ...] = ()
    skills: tuple[ProjectSkillManifest, ...] = ()
    skills_trusted: bool = False
    mcp: ProjectMCPManifest = field(default_factory=ProjectMCPManifest)

    @property
    def manifest_hash(self) -> str:
        payload = {
            "canonical_id": self.canonical_id,
            "skills": [s.__dict__ for s in self.skills],
            "mcp": {
                "sha256": self.mcp.sha256,
                "servers": list(self.mcp.servers),
                "trusted": self.mcp.trusted,
            },
            "skills_trusted": self.skills_trusted,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def trust_path() -> Path:
    return get_hermes_home() / TRUST_FILENAME


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out or None


def _real(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def project_identity(
    start: str | os.PathLike[str] | None = None,
) -> tuple[str, Path] | None:
    """Return ``(canonical_id, worktree_root)`` for a cwd inside a git worktree.

    The id is based on ``git rev-parse --git-common-dir`` after resolving both
    symlinks and relative git paths, so linked worktrees of the same repository
    share one stable key.
    """
    cwd = _real(Path(start or os.getcwd()))
    if cwd.is_file():
        cwd = cwd.parent

    root_s = _run_git(["rev-parse", "--show-toplevel"], cwd)
    if not root_s:
        return None
    root = _real(Path(root_s))

    common_s = _run_git(["rev-parse", "--git-common-dir"], root)
    if common_s:
        common = Path(common_s)
        if not common.is_absolute():
            common = root / common
        canonical_id = str(_real(common))
    else:
        canonical_id = str(root)
    return canonical_id, root


def _safe_project_hermes_dir(root: Path) -> Path | None:
    hermes_dir = root / ".hermes"
    try:
        if (
            not hermes_dir.exists()
            or hermes_dir.is_symlink()
            or not hermes_dir.is_dir()
        ):
            return None
    except OSError:
        return None
    return hermes_dir


def _project_config_path(hermes_dir: Path) -> Path | None:
    path = hermes_dir / "config.yaml"
    try:
        if path.exists() and path.is_file() and not path.is_symlink():
            return path
    except OSError:
        return None
    return None


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _project_mcp_servers(config_path: Path | None) -> dict[str, dict]:
    if config_path is None:
        return {}
    servers = _load_yaml_mapping(config_path).get("mcp_servers")
    return servers if isinstance(servers, dict) else {}


def project_config_path(start: str | os.PathLike[str] | None = None) -> Path | None:
    identity = project_identity(start)
    if identity is None:
        return None
    _canonical_id, root = identity
    hermes_dir = _safe_project_hermes_dir(root)
    return _project_config_path(hermes_dir) if hermes_dir else root / PROJECT_CONFIG_REL


def has_project_local_candidate(start: str | os.PathLike[str] | None = None) -> bool:
    identity = project_identity(start)
    if identity is None:
        return False
    canonical_id, root = identity
    cached = _candidate_cache.get(canonical_id)
    if cached is not None:
        return cached

    hermes_dir = _safe_project_hermes_dir(root)
    if hermes_dir is None:
        _candidate_cache[canonical_id] = False
        return False

    result = bool(_skill_manifests(hermes_dir)) or bool(
        _project_mcp_servers(_project_config_path(hermes_dir))
    )
    _candidate_cache[canonical_id] = result
    return result


def clear_project_local_cache() -> None:
    _candidate_cache.clear()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _skill_manifests(hermes_dir: Path) -> tuple[ProjectSkillManifest, ...]:
    skills_dir = hermes_dir / "skills"
    try:
        if (
            not skills_dir.exists()
            or skills_dir.is_symlink()
            or not skills_dir.is_dir()
        ):
            return ()
    except OSError:
        return ()

    manifests: list[ProjectSkillManifest] = []
    for skill_md in sorted(skills_dir.rglob("SKILL.md")):
        with contextlib.suppress(OSError):
            if skill_md.is_symlink() or any(
                part.startswith(".") for part in skill_md.relative_to(skills_dir).parts
            ):
                continue
            root = skill_md.parent
            if root.is_symlink():
                continue
            name = root.name
            manifests.append(
                ProjectSkillManifest(
                    name=name,
                    path=str(_real(root)),
                    sha256=_sha256_file(skill_md),
                )
            )
    return tuple(manifests)


def load_trust() -> dict[str, Any]:
    try:
        raw = json.loads(trust_path().read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    raw.setdefault("version", TRUST_VERSION)
    raw.setdefault("projects", {})
    if not isinstance(raw["projects"], dict):
        raw["projects"] = {}
    return raw


def save_trust(data: dict[str, Any]) -> None:
    data["version"] = TRUST_VERSION
    data.setdefault("projects", {})
    atomic_json_write(trust_path(), data, mode=0o600)


def _locked_update(mutator) -> Any:
    p = trust_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None:
        with _trust_lock:
            data = load_trust()
            result = mutator(data)
            save_trust(data)
            return result
    lock_path = p.with_suffix(p.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            data = load_trust()
            result = mutator(data)
            save_trust(data)
            return result
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _skills_trusted(
    canonical_id: str, skills: tuple[ProjectSkillManifest, ...]
) -> bool:
    if not skills:
        return False
    entry = load_trust().get("projects", {}).get(canonical_id, {})
    trusted = entry.get("skills", {}) if isinstance(entry, dict) else {}
    if not isinstance(trusted, dict):
        return False
    approved = trusted.get("approved")
    fingerprints = trusted.get("fingerprints")
    expected = {s.name: s.sha256 for s in skills}
    return approved is True and fingerprints == expected


def _mcp_trusted(canonical_id: str, config_hash: str, servers: tuple[str, ...]) -> bool:
    if not config_hash or not servers:
        return False
    entry = load_trust().get("projects", {}).get(canonical_id, {})
    trusted = entry.get("mcp", {}) if isinstance(entry, dict) else {}
    if not isinstance(trusted, dict):
        return False
    return (
        trusted.get("approved") is True and trusted.get("config_sha256") == config_hash
    )


def trust_project_mcp(
    start: str | os.PathLike[str] | None = None,
) -> ProjectLocalState | None:
    state = resolve_project_local_state(start)
    if state is None or not state.mcp.sha256:
        return state

    def mutate(data: dict[str, Any]) -> None:
        project = data.setdefault("projects", {}).setdefault(state.canonical_id, {})
        project["mcp"] = {
            "approved": True,
            "config_sha256": state.mcp.sha256,
            "servers": list(state.mcp.servers),
            "approved_at": int(time.time()),
        }

    _locked_update(mutate)
    return resolve_project_local_state(start)


def resolve_project_local_state(
    start: str | os.PathLike[str] | None = None,
) -> ProjectLocalState | None:
    identity = project_identity(start)
    if identity is None:
        return None
    canonical_id, root = identity
    hermes_dir = _safe_project_hermes_dir(root)
    if hermes_dir is None:
        return None

    skills = _skill_manifests(hermes_dir)
    cfg_path = _project_config_path(hermes_dir)
    servers = _project_mcp_servers(cfg_path)
    config_hash = _sha256_file(cfg_path) if cfg_path and servers else ""
    server_names = tuple(sorted(str(name) for name in servers))
    return ProjectLocalState(
        canonical_id=canonical_id,
        project_root=str(root),
        hermes_dir=str(_real(hermes_dir)),
        skill_roots=tuple(s.path for s in skills),
        skills=skills,
        skills_trusted=_skills_trusted(canonical_id, skills),
        mcp=ProjectMCPManifest(
            config_path=str(cfg_path) if cfg_path else None,
            sha256=config_hash,
            servers=server_names,
            trusted=_mcp_trusted(canonical_id, config_hash, server_names),
        ),
    )


def trusted_project_mcp_servers(
    start: str | os.PathLike[str] | None = None,
) -> dict[str, dict]:
    state = resolve_project_local_state(start)
    if state is None or not state.mcp.trusted or not state.mcp.config_path:
        return {}
    return _project_mcp_servers(Path(state.mcp.config_path))
