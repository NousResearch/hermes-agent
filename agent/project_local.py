"""Project-local identity, candidate detection, and trust state.

This module is intentionally state-only. It discovers recognized project-local
surfaces, builds immutable runtime state, and persists operator consent in a
profile-aware sidecar. It does not load project-local skill instructions into a
prompt; later project-local feature phases consume the state explicitly.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

try:  # POSIX only; Windows falls back to the in-process lock.
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger("agent.project_local")

PROJECT_TRUST_FILENAME = "project-trust.json"
PROJECT_TRUST_VERSION = 1

_GIT_TIMEOUT = 2.5
_candidate_cache_lock = threading.Lock()
_CandidateCacheStamp = tuple[tuple[str, int, int, bool], ...]
_candidate_cache: dict[str, tuple[_CandidateCacheStamp, bool, str]] = {}
_trust_write_lock = threading.Lock()


@dataclass(frozen=True)
class ProjectIdentity:
    """Canonical git-backed project identity."""

    canonical_id: str
    worktree_root: str
    git_common_dir: str


@dataclass(frozen=True)
class ProjectSkillManifestEntry:
    """One recognized project-local skill file."""

    name: str
    path: str
    sha256: str


@dataclass(frozen=True)
class ProjectLocalState:
    """Immutable project-local runtime state captured at agent construction."""

    canonical_id: str = ""
    worktree_root: str = ""
    git_common_dir: str = ""
    skill_roots: tuple[str, ...] = ()
    skills_manifest: tuple[ProjectSkillManifestEntry, ...] = ()
    skills_manifest_hash: str = ""
    consent_decision: str = ""
    mcp_manifest: tuple[tuple[str, str], ...] = ()
    mcp_manifest_hash: str = ""
    rejected_reason: str = ""

    @property
    def recognized(self) -> bool:
        return bool(self.skills_manifest or self.mcp_manifest)

    def cache_signature(self) -> dict[str, str]:
        if not (self.recognized or self.rejected_reason):
            return {}
        signature = {
            "project.canonical_id": self.canonical_id,
        }
        if self.skills_manifest_hash:
            signature["project.skills_manifest_hash"] = self.skills_manifest_hash
        if self.mcp_manifest_hash:
            signature["project.mcp_manifest_hash"] = self.mcp_manifest_hash
        if self.rejected_reason:
            signature["project.rejected_reason"] = self.rejected_reason
        return signature


def clear_project_local_cache() -> None:
    with _candidate_cache_lock:
        _candidate_cache.clear()


def project_trust_path() -> Path:
    return get_hermes_home() / PROJECT_TRUST_FILENAME


def _empty_trust() -> dict[str, Any]:
    return {"version": PROJECT_TRUST_VERSION, "projects": {}}


def load_project_trust() -> dict[str, Any]:
    try:
        raw = json.loads(project_trust_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return _empty_trust()
    if not isinstance(raw, dict):
        return _empty_trust()
    if raw.get("version") != PROJECT_TRUST_VERSION:
        return _empty_trust()
    projects = raw.get("projects")
    if not isinstance(projects, dict):
        raw["projects"] = {}
    return raw


def save_project_trust(data: dict[str, Any]) -> None:
    payload = dict(data if isinstance(data, dict) else {})
    payload["version"] = PROJECT_TRUST_VERSION
    if not isinstance(payload.get("projects"), dict):
        payload["projects"] = {}
    atomic_json_write(
        project_trust_path(),
        payload,
        indent=2,
        mode=0o600,
        sort_keys=True,
    )


@contextlib.contextmanager
def locked_project_trust() -> Iterator[dict[str, Any]]:
    path = project_trust_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")

    if fcntl is None:  # pragma: no cover
        with _trust_write_lock:
            data = load_project_trust()
            yield data
            save_project_trust(data)
        return

    with open(lock_path, "a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            data = load_project_trust()
            yield data
            save_project_trust(data)
        finally:
            with contextlib.suppress(OSError, IOError):
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def record_project_skill_consent(
    canonical_id: str,
    *,
    skills_manifest_hash: str,
    decision: str,
) -> None:
    if not canonical_id:
        return
    with locked_project_trust() as trust:
        projects = trust.setdefault("projects", {})
        project = projects.setdefault(canonical_id, {})
        project["skills"] = {
            "manifest_hash": skills_manifest_hash,
            "decision": str(decision or ""),
        }
        project.setdefault("mcp", {})


def _git(
    start: Path,
    *args: str,
) -> str:
    result = subprocess.run(
        ["git", "-C", str(start), *args],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "").strip())
    return result.stdout.strip()


def canonical_project_identity(start: str | Path | None = None) -> Optional[ProjectIdentity]:
    try:
        start_path = Path(start or os.getcwd()).expanduser()
        if start_path.is_file():
            start_path = start_path.parent
        start_path = start_path.resolve()
    except (OSError, RuntimeError, ValueError):
        return None

    try:
        root = Path(_git(start_path, "rev-parse", "--show-toplevel")).expanduser()
        if not root.is_absolute():
            root = start_path / root
        root = root.resolve()
        common = Path(_git(root, "rev-parse", "--git-common-dir")).expanduser()
        if not common.is_absolute():
            common = root / common
        common = common.resolve()
    except Exception:
        return None

    digest = hashlib.sha256(str(common).encode("utf-8")).hexdigest()[:24]
    return ProjectIdentity(
        canonical_id=f"git:{digest}",
        worktree_root=str(root),
        git_common_dir=str(common),
    )


def _safe_lstat(path: Path) -> tuple[int, int, bool]:
    try:
        stat_result = path.lstat()
    except OSError:
        return -1, -1, False
    return stat_result.st_mtime_ns, stat_result.st_size, path.is_symlink()


def _candidate_cache_stamp(hermes_dir: Path) -> _CandidateCacheStamp:
    entries: list[tuple[str, int, int, bool]] = []

    def add(label: str, path: Path) -> None:
        mtime, size, is_symlink = _safe_lstat(path)
        entries.append((label, mtime, size, is_symlink))

    add(".", hermes_dir)
    add("config.yaml", hermes_dir / "config.yaml")

    skills_dir = hermes_dir / "skills"
    add("skills", skills_dir)
    if skills_dir.exists() and not skills_dir.is_symlink():
        try:
            dirs = sorted(
                (
                    path
                    for path in skills_dir.rglob("*")
                    if path.is_dir() or path.is_symlink()
                ),
                key=lambda path: str(path.relative_to(hermes_dir)),
            )
        except OSError:
            dirs = []
        for path in dirs:
            try:
                label = str(path.relative_to(hermes_dir))
            except ValueError:
                label = str(path)
            add(label, path)

    return tuple(entries)


def _has_mcp_config(hermes_dir: Path) -> bool:
    config = hermes_dir / "config.yaml"
    if not config.exists():
        return False
    if config.is_symlink():
        return False
    try:
        import yaml

        data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    except Exception:
        return False
    return isinstance(data, dict) and "mcp_servers" in data


def project_has_recognized_files(identity: ProjectIdentity) -> tuple[bool, str]:
    hermes_dir = Path(identity.worktree_root) / ".hermes"
    cache_stamp = _candidate_cache_stamp(hermes_dir)
    with _candidate_cache_lock:
        cached = _candidate_cache.get(identity.canonical_id)
        if cached is not None and cached[0] == cache_stamp:
            return cached[1], cached[2]

    recognized = False
    rejected_reason = ""
    if hermes_dir.exists():
        if hermes_dir.is_symlink():
            rejected_reason = "symlinked .hermes directory is not trusted"
        elif (
            (hermes_dir / "skills").exists()
            and not (hermes_dir / "skills").is_symlink()
            and any(
                path.name == "SKILL.md" and not path.is_symlink()
                for path in (hermes_dir / "skills").rglob("SKILL.md")
            )
        ):
            recognized = True
        elif _has_mcp_config(hermes_dir):
            recognized = True

    with _candidate_cache_lock:
        _candidate_cache[identity.canonical_id] = (
            cache_stamp,
            recognized,
            rejected_reason,
        )
    return recognized, rejected_reason


def _skill_name(skill_md: Path) -> str:
    try:
        content = skill_md.read_text(encoding="utf-8")
        if content.startswith("---"):
            end = content.find("\n---", 3)
            if end != -1:
                for line in content[3:end].splitlines():
                    key, _, value = line.partition(":")
                    if key.strip() == "name" and value.strip():
                        return value.strip().strip("'\"")[:64]
    except Exception:
        pass
    return skill_md.parent.name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_skills(identity: ProjectIdentity) -> tuple[tuple[str, ...], tuple[ProjectSkillManifestEntry, ...]]:
    skills_root = Path(identity.worktree_root) / ".hermes" / "skills"
    if not skills_root.exists() or skills_root.is_symlink():
        return (), ()

    entries: list[ProjectSkillManifestEntry] = []
    roots: set[str] = set()
    for skill_md in sorted(skills_root.rglob("SKILL.md"), key=lambda p: str(p)):
        if skill_md.is_symlink():
            continue
        try:
            resolved_skill = skill_md.resolve()
            resolved_root = skill_md.parent.resolve()
            roots.add(str(resolved_root))
            entries.append(
                ProjectSkillManifestEntry(
                    name=_skill_name(skill_md),
                    path=str(resolved_skill),
                    sha256=_sha256_file(skill_md),
                )
            )
        except OSError:
            continue
    return tuple(sorted(roots)), tuple(entries)


def _manifest_hash(entries: tuple[ProjectSkillManifestEntry, ...]) -> str:
    if not entries:
        return ""
    payload = [
        {"name": entry.name, "path": entry.path, "sha256": entry.sha256}
        for entry in entries
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mcp_manifest(identity: ProjectIdentity) -> tuple[tuple[str, str], ...]:
    hermes_dir = Path(identity.worktree_root) / ".hermes"
    if not _has_mcp_config(hermes_dir):
        return ()
    config = hermes_dir / "config.yaml"
    try:
        return (("config_sha256", _sha256_file(config)),)
    except OSError:
        return ()


def _mcp_manifest_hash(entries: tuple[tuple[str, str], ...]) -> str:
    if not entries:
        return ""
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _consent_decision(canonical_id: str, manifest_hash: str) -> str:
    if not canonical_id or not manifest_hash:
        return ""
    project = load_project_trust().get("projects", {}).get(canonical_id, {})
    if not isinstance(project, dict):
        return ""
    skills = project.get("skills")
    if not isinstance(skills, dict):
        return ""
    if skills.get("manifest_hash") != manifest_hash:
        return ""
    return str(skills.get("decision") or "")


def resolve_project_local_state(
    cwd: str | Path | None = None,
) -> ProjectLocalState:
    if cwd is None:
        try:
            from agent.runtime_cwd import resolve_agent_cwd

            cwd = resolve_agent_cwd()
        except Exception:
            cwd = None
    identity = canonical_project_identity(cwd)
    if identity is None:
        return ProjectLocalState()

    recognized, rejected_reason = project_has_recognized_files(identity)
    if not recognized:
        return ProjectLocalState(
            canonical_id=identity.canonical_id,
            worktree_root=identity.worktree_root,
            git_common_dir=identity.git_common_dir,
            rejected_reason=rejected_reason,
        )

    skill_roots, skills = _project_skills(identity)
    manifest_hash = _manifest_hash(skills)
    mcp = _mcp_manifest(identity)
    mcp_hash = _mcp_manifest_hash(mcp)
    return ProjectLocalState(
        canonical_id=identity.canonical_id,
        worktree_root=identity.worktree_root,
        git_common_dir=identity.git_common_dir,
        skill_roots=skill_roots,
        skills_manifest=skills,
        skills_manifest_hash=manifest_hash,
        consent_decision=_consent_decision(identity.canonical_id, manifest_hash),
        mcp_manifest=mcp,
        mcp_manifest_hash=mcp_hash,
        rejected_reason=rejected_reason,
    )


__all__ = [
    "PROJECT_TRUST_FILENAME",
    "PROJECT_TRUST_VERSION",
    "ProjectIdentity",
    "ProjectLocalState",
    "ProjectSkillManifestEntry",
    "canonical_project_identity",
    "clear_project_local_cache",
    "load_project_trust",
    "locked_project_trust",
    "project_has_recognized_files",
    "project_trust_path",
    "record_project_skill_consent",
    "resolve_project_local_state",
    "save_project_trust",
]
