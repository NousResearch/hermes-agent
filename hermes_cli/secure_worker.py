"""Fail-closed controls for remote-model sanitized workspaces.

This module is deliberately deterministic and does not call an LLM.  It owns the
boundary between a trusted source checkout and a disposable workspace that may be
shown to a remote inference provider.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


class SecurityBoundaryError(RuntimeError):
    """A deny decision at the secure-worker boundary."""


_SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(rb"\bgh[opusr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        rb"(?i)\b(?:aws_secret_access_key|github_token|api[_-]?key|client[_-]?secret)"
        rb"\s*[:=]\s*[^\s'\"]{16,}"
    ),
)


@dataclass(frozen=True)
class PackPolicy:
    policy_version: str
    max_file_bytes: int
    max_pack_bytes: int
    excluded_segments: tuple[str, ...]

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "PackPolicy":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            policy_version=str(data["policy_version"]),
            max_file_bytes=int(data["max_file_bytes"]),
            max_pack_bytes=int(data["max_pack_bytes"]),
            excluded_segments=tuple(str(item) for item in data["excluded_segments"]),
        )

    def __post_init__(self) -> None:
        if not self.policy_version.strip():
            raise ValueError("policy_version must not be empty")
        if self.max_file_bytes <= 0 or self.max_pack_bytes <= 0:
            raise ValueError("pack budgets must be positive")
        if self.max_file_bytes > self.max_pack_bytes:
            raise ValueError("file budget cannot exceed pack budget")
        if not self.excluded_segments:
            raise ValueError("excluded_segments must not be empty")


@dataclass(frozen=True)
class WorkerImageLock:
    schema: str
    image: str
    dockerfile_sha256: str
    requirements_sha256: str

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "WorkerImageLock":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        lock = cls(
            schema=str(data["schema"]),
            image=str(data["image"]),
            dockerfile_sha256=str(data["dockerfile_sha256"]),
            requirements_sha256=str(data["requirements_sha256"]),
        )
        if lock.schema != "hermes.secure-worker.image-lock.v1":
            raise SecurityBoundaryError("unsupported worker image lock schema")
        if not _PINNED_IMAGE_RE.fullmatch(lock.image):
            raise SecurityBoundaryError("worker image lock is not digest-pinned")
        for label, digest in (
            ("Dockerfile", lock.dockerfile_sha256),
            ("requirements", lock.requirements_sha256),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise SecurityBoundaryError(f"worker image lock {label} digest is invalid")
        return lock


@dataclass(frozen=True)
class ManifestFile:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class PackManifest:
    schema: str
    policy_version: str
    policy_sha256: str
    pack_root: str
    source_commit: str
    source_tree: str
    files: tuple[ManifestFile, ...]
    total_bytes: int

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> "PackManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            schema=str(data["schema"]),
            policy_version=str(data["policy_version"]),
            policy_sha256=str(data["policy_sha256"]),
            pack_root=str(data["pack_root"]),
            source_commit=str(data["source_commit"]),
            source_tree=str(data["source_tree"]),
            files=tuple(ManifestFile(**item) for item in data["files"]),
            total_bytes=int(data["total_bytes"]),
        )


@dataclass(frozen=True)
class VerificationReceipt:
    schema: str
    manifest_sha256: str
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class DiffVerificationReceipt:
    schema: str
    manifest_sha256: str
    diff_sha256: str
    changed_paths: tuple[str, ...]


@dataclass(frozen=True)
class BoundaryReport:
    allowed: bool
    reasons: tuple[str, ...]
    manifest_sha256: str | None = None
    source_commit: str | None = None
    config_sha256: str | None = None
    policy_sha256: str | None = None
    worker_image: str | None = None
    broker_executable_sha256: str | None = None
    broker_module_sha256: str | None = None


@dataclass(frozen=True)
class AdmissionReceipt:
    schema: str
    config_sha256: str
    policy_sha256: str
    manifest_sha256: str
    source_commit: str
    worker_image: str
    broker_executable_sha256: str
    broker_module_sha256: str


_LOCAL_PROVIDERS = {"ollama", "ollama-launch", "custom"}
_REMOTE_PROVIDER_NAMES = {
    "anthropic",
    "azure-foundry",
    "bedrock",
    "gemini",
    "nous",
    "openai",
    "openai-codex",
    "openrouter",
    "vertex",
    "xai",
}
_PINNED_IMAGE_RE = re.compile(r"(?:[^@\s]+@)?sha256:[0-9a-f]{64}")
_NOUS_SECURE_ENDPOINT = "https://inference-api.nousresearch.com/v1"


def _run_git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise SecurityBoundaryError(f"git boundary check failed: {detail.strip()}") from exc
    return result.stdout.strip()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        + b"\n"
    )


def _policy_sha256(policy: PackPolicy) -> str:
    return hashlib.sha256(_canonical_json(asdict(policy))).hexdigest()


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise SecurityBoundaryError(f"admitted executable or module is unsafe: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_external_file(pack: Path, path: Path, label: str) -> None:
    if path == pack or pack in path.parents:
        raise SecurityBoundaryError(f"{label} must remain outside the model-writable pack")
    if path.is_symlink():
        raise SecurityBoundaryError(f"{label} must not be a symlink")


def _normalized_relative_path(raw: str, excluded_segments: Sequence[str]) -> PurePosixPath:
    if not raw or "\\" in raw:
        raise SecurityBoundaryError(f"unsafe selected path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise SecurityBoundaryError(f"path escape or non-canonical path: {raw!r}")
    excluded = {item.casefold() for item in excluded_segments}
    if any(part.casefold() in excluded for part in path.parts):
        raise SecurityBoundaryError(f"excluded path segment in {raw!r}")
    return path


def _assert_clean_repository(source_root: Path) -> None:
    top = Path(_run_git(source_root, "rev-parse", "--show-toplevel")).resolve()
    if top != source_root.resolve():
        raise SecurityBoundaryError("source must be the exact Git worktree root")
    if _run_git(source_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise SecurityBoundaryError("source worktree must be clean, including untracked files")


def _assert_safe_content(path: Path, data: bytes, policy: PackPolicy) -> None:
    if len(data) > policy.max_file_bytes:
        raise SecurityBoundaryError(f"file budget exceeded: {path.name}")
    if b"\x00" in data:
        raise SecurityBoundaryError(f"binary file denied: {path.name}")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SecurityBoundaryError(f"binary or non-UTF-8 file denied: {path.name}") from exc
    if any(pattern.search(data) for pattern in _SECRET_PATTERNS):
        raise SecurityBoundaryError(f"secret-like content denied: {path.name}")


def _validate_selected_files(
    source_root: Path, selected_paths: Iterable[str], policy: PackPolicy
) -> list[tuple[PurePosixPath, bytes]]:
    normalized = sorted(
        {_normalized_relative_path(raw, policy.excluded_segments) for raw in selected_paths},
        key=lambda item: item.as_posix(),
    )
    if not normalized:
        raise SecurityBoundaryError("at least one explicit file is required")

    validated: list[tuple[PurePosixPath, bytes]] = []
    total = 0
    for relative in normalized:
        rel = relative.as_posix()
        _run_git(source_root, "ls-files", "--error-unmatch", "--", rel)
        source = source_root / relative
        try:
            mode = source.lstat().st_mode
        except OSError as exc:
            raise SecurityBoundaryError(f"selected file unavailable: {rel}") from exc
        if stat.S_ISLNK(mode):
            raise SecurityBoundaryError(f"symlink denied: {rel}")
        if not stat.S_ISREG(mode):
            raise SecurityBoundaryError(f"non-regular file denied: {rel}")
        data = source.read_bytes()
        _assert_safe_content(source, data, policy)
        total += len(data)
        if total > policy.max_pack_bytes:
            raise SecurityBoundaryError("total pack budget exceeded")
        validated.append((relative, data))
    return validated


def build_context_pack(
    source_root: str | os.PathLike[str],
    selected_paths: Iterable[str],
    pack_root: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    policy: PackPolicy,
) -> PackManifest:
    """Build a new sanitized Git repository from explicit safe files."""

    source = Path(source_root).resolve()
    pack = Path(pack_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    _assert_clean_repository(source)
    validated = _validate_selected_files(source, selected_paths, policy)

    if pack.exists() or manifest_file.exists():
        raise SecurityBoundaryError("pack and manifest destinations must not already exist")
    if manifest_file == pack or pack in manifest_file.parents:
        raise SecurityBoundaryError("manifest must remain outside the model-writable pack")

    source_commit = _run_git(source, "rev-parse", "HEAD")
    source_tree = _run_git(source, "rev-parse", "HEAD^{tree}")
    entries = tuple(
        ManifestFile(
            path=relative.as_posix(),
            bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        for relative, data in validated
    )
    manifest = PackManifest(
        schema="hermes.secure-worker.pack-manifest.v1",
        policy_version=policy.policy_version,
        policy_sha256=_policy_sha256(policy),
        pack_root=str(pack),
        source_commit=source_commit,
        source_tree=source_tree,
        files=entries,
        total_bytes=sum(entry.bytes for entry in entries),
    )

    try:
        pack.mkdir(parents=True)
        for relative, data in validated:
            destination = pack / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        _run_git(pack, "init", "-q")
        _run_git(pack, "config", "user.email", "secure-worker@localhost")
        _run_git(pack, "config", "user.name", "Hermes Secure Worker")
        _run_git(pack, "add", "--all")
        _run_git(pack, "commit", "-qm", f"sanitized pack from {source_commit[:12]}")
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_bytes(_canonical_json(asdict(manifest)))
    except Exception:
        shutil.rmtree(pack, ignore_errors=True)
        try:
            manifest_file.unlink()
        except FileNotFoundError:
            pass
        raise
    return manifest


def verify_context_pack(
    pack_root: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    policy: PackPolicy,
) -> VerificationReceipt:
    """Verify manifest, file inventory, content hashes, and policy compatibility."""

    pack = Path(pack_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    _assert_external_file(pack, manifest_file, "manifest")
    manifest = PackManifest.from_path(manifest_file)
    if manifest.schema != "hermes.secure-worker.pack-manifest.v1":
        raise SecurityBoundaryError("unsupported manifest schema")
    if manifest.policy_version != policy.policy_version:
        raise SecurityBoundaryError("manifest policy version mismatch")
    if manifest.policy_sha256 != _policy_sha256(policy):
        raise SecurityBoundaryError("manifest policy digest mismatch")
    if manifest.pack_root != str(pack):
        raise SecurityBoundaryError("pack manifest binding mismatch")

    expected = {entry.path: entry for entry in manifest.files}
    actual = {
        path.relative_to(pack).as_posix()
        for path in pack.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(pack).parts
    }
    if actual != set(expected):
        raise SecurityBoundaryError("pack file inventory does not match manifest")

    total = 0
    for rel, entry in expected.items():
        relative = _normalized_relative_path(rel, policy.excluded_segments)
        path = pack / relative
        if path.is_symlink() or not path.is_file():
            raise SecurityBoundaryError(f"unsafe pack file: {rel}")
        data = path.read_bytes()
        _assert_safe_content(path, data, policy)
        if len(data) != entry.bytes or hashlib.sha256(data).hexdigest() != entry.sha256:
            raise SecurityBoundaryError(f"file hash mismatch: {rel}")
        total += len(data)
    if total != manifest.total_bytes or total > policy.max_pack_bytes:
        raise SecurityBoundaryError("manifest total does not match pack")
    return VerificationReceipt(
        schema="hermes.secure-worker.verification-receipt.v1",
        manifest_sha256=hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        file_count=len(expected),
        total_bytes=total,
    )


def _assert_manifest_binding(pack: Path, manifest_file: Path) -> str:
    _assert_external_file(pack, manifest_file, "manifest")
    try:
        manifest = PackManifest.from_path(manifest_file)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise SecurityBoundaryError("pack manifest binding is invalid") from exc
    if manifest.pack_root != str(pack):
        raise SecurityBoundaryError("pack manifest binding mismatch")
    digest = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    return digest


def verify_proposed_diff(
    pack_root: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    policy: PackPolicy,
) -> DiffVerificationReceipt:
    """Verify a model-produced working-tree diff without applying or publishing it."""

    pack = Path(pack_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest_sha = _assert_manifest_binding(pack, manifest_file)
    manifest = PackManifest.from_path(manifest_file)
    if manifest.policy_version != policy.policy_version:
        raise SecurityBoundaryError("manifest policy version mismatch")
    if manifest.policy_sha256 != _policy_sha256(policy):
        raise SecurityBoundaryError("manifest policy digest mismatch")
    expected = {entry.path: entry for entry in manifest.files}
    actual = {
        path.relative_to(pack).as_posix()
        for path in pack.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(pack).parts
    }
    extra = actual - set(expected)
    if extra:
        raise SecurityBoundaryError(f"change outside approved pack: {sorted(extra)[0]}")
    changed: list[str] = []
    change_records: list[dict[str, object]] = []
    total = 0
    for rel, entry in sorted(expected.items()):
        canonical = _normalized_relative_path(rel, policy.excluded_segments).as_posix()
        if canonical not in actual:
            raise SecurityBoundaryError(f"file deletion denied: {canonical}")
        path = pack / canonical
        if path.is_symlink() or not path.is_file():
            raise SecurityBoundaryError(f"unsafe changed file: {canonical}")
        data = path.read_bytes()
        _assert_safe_content(path, data, policy)
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry.sha256 or len(data) != entry.bytes:
            total += len(data)
            if total > policy.max_pack_bytes:
                raise SecurityBoundaryError("changed-file pack budget exceeded")
            changed.append(canonical)
            change_records.append(
                {
                    "path": canonical,
                    "before_sha256": entry.sha256,
                    "after_sha256": digest,
                    "after_bytes": len(data),
                }
            )
    return DiffVerificationReceipt(
        schema="hermes.secure-worker.diff-verification.v1",
        manifest_sha256=manifest_sha,
        diff_sha256=hashlib.sha256(_canonical_json(change_records)).hexdigest(),
        changed_paths=tuple(sorted(changed)),
    )


def destroy_context_pack(
    pack_root: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    *,
    quarantine_root: str | os.PathLike[str],
) -> None:
    """Destroy only a manifest-bound secure-worker pack, never an arbitrary directory."""

    pack = Path(pack_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    quarantine = Path(quarantine_root).resolve()
    if pack in {Path("/").resolve(), Path.home().resolve()} or len(pack.parts) < 4:
        raise SecurityBoundaryError("refusing broad or unsafe destruction target")
    if Path(pack_root).is_symlink():
        raise SecurityBoundaryError("refusing symlink destruction target")
    if receipt_file.exists():
        raise SecurityBoundaryError("destruction receipt destination already exists")
    manifest_sha = _assert_manifest_binding(pack, manifest_file)
    manifest = PackManifest.from_path(manifest_file)
    if manifest.schema != "hermes.secure-worker.pack-manifest.v1":
        raise SecurityBoundaryError("unsupported manifest schema")

    status = "destroyed"
    cleanup_error: OSError | None = None
    try:
        shutil.rmtree(pack)
    except OSError as exc:
        cleanup_error = exc
        status = "quarantined"
        quarantine.mkdir(parents=True, exist_ok=True)
        destination = quarantine / f"pack-{manifest_sha[:16]}"
        if destination.exists():
            raise SecurityBoundaryError("cleanup failed and quarantine destination exists") from exc
        try:
            os.replace(pack, destination)
        except OSError as move_exc:
            raise SecurityBoundaryError("cleanup and quarantine both failed") from move_exc

    receipt = {
        "schema": "hermes.secure-worker.destroy-receipt.v1",
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_commit": manifest.source_commit,
        "manifest_sha256": manifest_sha,
    }
    receipt_file.parent.mkdir(parents=True, exist_ok=True)
    receipt_file.write_bytes(_canonical_json(receipt))
    if cleanup_error is not None:
        raise SecurityBoundaryError("pack cleanup failed; workspace was quarantined") from cleanup_error


def _terminal_boundary(cwd: Path, worker_image: str) -> dict[str, object]:
    if not _PINNED_IMAGE_RE.fullmatch(worker_image):
        raise SecurityBoundaryError("worker image must be pinned by sha256 digest")
    return {
        "backend": "docker",
        "cwd": str(cwd.resolve()),
        "docker_image": worker_image,
        "container_persistent": False,
        "docker_persist_across_processes": False,
        "persistent_shell": False,
        "docker_network": False,
        "docker_isolate_host_data": True,
        "docker_mount_cwd_to_workspace": True,
        "docker_volumes": [],
        "docker_forward_env": [],
        "docker_env": {},
        "docker_run_as_host_user": False,
        "docker_extra_args": [],
        "home_mode": "isolated",
    }


def render_ox_profile(
    pack_root: str | os.PathLike[str],
    broker_python: str,
    staging_owner: str,
    staging_repo: str,
    worker_image: str,
) -> dict[str, object]:
    """Return a minimal remote profile whose only network broker is staging GitHub."""

    if not Path(broker_python).is_absolute():
        raise SecurityBoundaryError("broker Python must be an absolute pinned path")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", staging_owner or ""):
        raise SecurityBoundaryError("invalid staging owner")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", staging_repo or ""):
        raise SecurityBoundaryError("invalid staging repository")
    pack = Path(pack_root).resolve()
    return {
        "model": {
            "provider": "nous",
            "default": "stealth/ox-alpha",
            "base_url": _NOUS_SECURE_ENDPOINT,
        },
        "fallback_providers": [],
        "provider_routing": {"data_collection": "deny"},
        "toolsets": [],
        "terminal": _terminal_boundary(pack, worker_image),
        "platform_toolsets": {"cli": ["terminal", "file"]},
        "agent": {"disabled_toolsets": ["bfl", "kanban"]},
        "skills": {"trusted_project_dirs": []},
        "plugins": {"enabled": [], "disabled": []},
        "mcp_servers": {
            "secure-github-staging": {
                "command": broker_python,
                "args": ["-m", "hermes_cli.secure_github_broker"],
                "env": {
                    "HERMES_STAGING_GITHUB_OWNER": staging_owner,
                    "HERMES_STAGING_GITHUB_REPO": staging_repo,
                    # Resolved at MCP startup; the token value is never
                    # serialized into the profile.
                    "HERMES_STAGING_GITHUB_TOKEN": "${HERMES_STAGING_GITHUB_TOKEN}",
                },
                "enabled": True,
            }
        },
        "checkpoints": {"enabled": False},
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "telemetry": {"shared_metrics": {"enabled": False}},
    }


def render_local_safe_profile(
    source_root: str | os.PathLike[str], model: str, worker_image: str
) -> dict[str, object]:
    """Return a repo-capable local profile with no remote fallback or web path."""

    source = Path(source_root).resolve()
    return {
        "model": {
            "provider": "ollama-launch",
            "default": model,
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
        },
        "fallback_providers": [],
        "toolsets": [],
        "terminal": _terminal_boundary(source, worker_image),
        "platform_toolsets": {"cli": ["terminal", "file"]},
        "agent": {"disabled_toolsets": ["bfl", "kanban"]},
        "mcp_servers": {},
        "telemetry": {"shared_metrics": {"enabled": False}},
    }


def _is_remote_provider(entry: object) -> bool:
    if not isinstance(entry, dict):
        return True
    provider = str(entry.get("provider", "")).strip().casefold()
    base_url = str(entry.get("base_url", entry.get("api", ""))).strip().casefold()
    if provider in _REMOTE_PROVIDER_NAMES:
        return True
    if provider in _LOCAL_PROVIDERS:
        return bool(base_url) and not (
            base_url.startswith("http://127.0.0.1:")
            or base_url.startswith("http://localhost:")
            or base_url.startswith("http://[::1]:")
        )
    return True


def _valid_privacy_attestation(
    value: object, provider: str, now: datetime
) -> tuple[bool, str]:
    if not isinstance(value, dict):
        return False, "privacy attestation is missing"
    if value.get("schema") != "hermes.secure-worker.privacy-attestation.v1":
        return False, "privacy attestation schema is invalid"
    if str(value.get("provider", "")).casefold() != provider.casefold():
        return False, "privacy attestation provider mismatch"
    if value.get("privacy_mode") is not True:
        return False, "privacy attestation does not affirm Privacy Mode"
    try:
        attested = datetime.fromisoformat(str(value["attested_at"]))
        expires = datetime.fromisoformat(str(value["expires_at"]))
        if attested.tzinfo is None or expires.tzinfo is None:
            raise ValueError("timezone required")
    except (KeyError, TypeError, ValueError):
        return False, "privacy attestation timestamps are invalid"
    if attested > now or expires <= now or expires <= attested:
        return False, "privacy attestation is stale or temporally invalid"
    return True, ""


def audit_profile_boundary(
    config: dict[str, object],
    *,
    pack_root: str | os.PathLike[str] | None,
    manifest_path: str | os.PathLike[str] | None,
    policy: PackPolicy | None,
    privacy_attestation: dict[str, object] | None,
    docker_available: bool,
    admitted_worker_image: str | None,
    now: datetime | None = None,
) -> BoundaryReport:
    """Audit a generated or existing repo-capable profile without mutating it."""

    reasons: list[str] = []
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    provider = str(model.get("provider", ""))
    remote = _is_remote_provider(model)
    terminal = config.get("terminal") if isinstance(config.get("terminal"), dict) else {}
    manifest_sha256: str | None = None
    source_commit: str | None = None
    policy_sha256: str | None = None
    broker_executable_sha256: str | None = None
    broker_module_sha256: str | None = None

    if terminal.get("backend") != "docker":
        reasons.append("terminal must use Docker; local/host fallback is denied")
    if not _PINNED_IMAGE_RE.fullmatch(str(terminal.get("docker_image", ""))):
        reasons.append("Docker worker image must be pinned by sha256 digest")
    if not admitted_worker_image or terminal.get("docker_image") != admitted_worker_image:
        reasons.append("Docker worker image does not match the trusted image lock")
    if not docker_available:
        reasons.append("Docker unavailable; host fallback is denied")
    if terminal.get("docker_network") is not False:
        reasons.append("task-container network must be disabled")
    if terminal.get("docker_isolate_host_data") is not True:
        reasons.append("automatic host credential, skill, cache, and proxy mounts must be isolated")
    if terminal.get("container_persistent") is not False or terminal.get("persistent_shell") is not False:
        reasons.append("persistent task state is denied")
    if terminal.get("docker_persist_across_processes") is not False:
        reasons.append("cross-process Docker reuse is denied")
    if terminal.get("docker_mount_cwd_to_workspace") is not True:
        reasons.append("the admitted cwd must be the only workspace mount")
    if terminal.get("docker_volumes") not in ([], None):
        reasons.append("additional Docker volume mounts are denied")
    if terminal.get("docker_forward_env") not in ([], None):
        reasons.append("forwarded task-container environment variables are denied")
    if terminal.get("docker_env") not in ({}, None):
        reasons.append("explicit task-container environment variables are denied")
    if terminal.get("docker_run_as_host_user") is not False:
        reasons.append("running the task container as the host user is denied")
    if terminal.get("docker_extra_args") not in ([], None):
        reasons.append("Docker extra arguments are denied")

    fallbacks = config.get("fallback_providers")
    if not isinstance(fallbacks, list):
        reasons.append("fallback provider configuration must be an explicit empty list")
    elif any(_is_remote_provider(item) for item in fallbacks):
        qualifier = "remote" if remote else "cloud"
        reasons.append(f"{qualifier} fallback crosses the inference boundary")

    if config.get("toolsets") != []:
        reasons.append("top-level toolsets must be an explicit empty list")
    if config.get("platform_toolsets") != {"cli": ["terminal", "file"]}:
        reasons.append("CLI toolset selection must contain only terminal and file")
    if config.get("agent") != {"disabled_toolsets": ["bfl", "kanban"]}:
        reasons.append("agent toolset disables do not match the admitted boundary")
    try:
        from hermes_cli.tools_config import _get_platform_tools

        effective_toolsets = _get_platform_tools(config, "cli")
    except Exception as exc:
        reasons.append(f"effective CLI toolset resolution failed: {exc}")
    else:
        expected_toolsets = {"terminal", "file"}
        if remote:
            expected_toolsets.add("secure-github-staging")
        if effective_toolsets != expected_toolsets:
            reasons.append(
                "effective CLI toolsets exceed the admitted boundary: "
                + ", ".join(sorted(effective_toolsets))
            )

    if remote:
        if config.get("skills") != {"trusted_project_dirs": []}:
            reasons.append("trusted host project directories are denied")
        if config.get("plugins") != {"enabled": [], "disabled": []}:
            reasons.append("plugins must be explicitly empty")
        if config.get("checkpoints") != {"enabled": False}:
            reasons.append("checkpoints must be disabled")
        if config.get("memory") != {
            "memory_enabled": False,
            "user_profile_enabled": False,
        }:
            reasons.append("memory and user-profile injection must be disabled")
        if config.get("telemetry") != {"shared_metrics": {"enabled": False}}:
            reasons.append("shared telemetry must be disabled")
        if provider.casefold() != "nous" or model.get("base_url") != _NOUS_SECURE_ENDPOINT:
            reasons.append("remote model endpoint must be the exact admitted Nous endpoint")
        if pack_root is None or manifest_path is None or policy is None:
            reasons.append("remote profile requires an admitted pack, manifest, and policy")
        else:
            pack = Path(pack_root).resolve()
            manifest_file = Path(manifest_path).resolve()
            expected_cwd = str(pack)
            if str(terminal.get("cwd", "")) != expected_cwd:
                reasons.append("remote profile cwd does not match the admitted pack")
            try:
                receipt = verify_context_pack(pack, manifest_file, policy)
                bound_digest = _assert_manifest_binding(pack, manifest_file)
                if receipt.manifest_sha256 != bound_digest:
                    raise SecurityBoundaryError("pack verification receipt binding mismatch")
                manifest = PackManifest.from_path(manifest_file)
            except (OSError, ValueError, TypeError, KeyError, SecurityBoundaryError) as exc:
                reasons.append(f"admitted pack verification failed: {exc}")
            else:
                manifest_sha256 = receipt.manifest_sha256
                source_commit = manifest.source_commit
                policy_sha256 = _policy_sha256(policy)
        if config.get("provider_routing") != {"data_collection": "deny"}:
            reasons.append("provider data-collection denial is missing")
        valid, attestation_reason = _valid_privacy_attestation(
            privacy_attestation,
            provider,
            now or datetime.now(timezone.utc),
        )
        if not valid:
            reasons.append(attestation_reason)
        servers = config.get("mcp_servers")
        if not isinstance(servers, dict) or set(servers) != {"secure-github-staging"}:
            reasons.append("MCP allowlist must contain only secure-github-staging")
        else:
            server = servers["secure-github-staging"]
            if not isinstance(server, dict):
                reasons.append("MCP broker configuration is invalid")
            else:
                command = Path(str(server.get("command", "")))
                expected_python = Path(sys.executable).resolve()
                if (
                    not command.is_absolute()
                    or command.is_symlink()
                    or command.resolve() != expected_python
                    or server.get("args") != [
                    "-m",
                    "hermes_cli.secure_github_broker",
                    ]
                ):
                    reasons.append(
                        "MCP broker executable is not the exact admitted interpreter"
                    )
                else:
                    try:
                        broker_executable_sha256 = _file_sha256(command)
                        broker_module_sha256 = _file_sha256(
                            Path(__file__).with_name("secure_github_broker.py")
                        )
                    except SecurityBoundaryError as exc:
                        reasons.append(f"MCP broker identity verification failed: {exc}")
                broker_env = server.get("env")
                if (
                    server.get("enabled") is not True
                    or not isinstance(broker_env, dict)
                    or set(broker_env) != {
                        "HERMES_STAGING_GITHUB_OWNER",
                        "HERMES_STAGING_GITHUB_REPO",
                        "HERMES_STAGING_GITHUB_TOKEN",
                    }
                    or not all(
                        re.fullmatch(r"[A-Za-z0-9_.-]+", str(value or ""))
                        for key, value in broker_env.items()
                        if key != "HERMES_STAGING_GITHUB_TOKEN"
                    )
                    or broker_env.get("HERMES_STAGING_GITHUB_TOKEN")
                    != "${HERMES_STAGING_GITHUB_TOKEN}"
                ):
                    reasons.append("MCP broker identity configuration is invalid")

    return BoundaryReport(
        allowed=not reasons,
        reasons=tuple(reasons),
        manifest_sha256=manifest_sha256,
        source_commit=source_commit,
        config_sha256=hashlib.sha256(_canonical_json(config)).hexdigest(),
        policy_sha256=policy_sha256,
        worker_image=str(terminal.get("docker_image", "")),
        broker_executable_sha256=broker_executable_sha256,
        broker_module_sha256=broker_module_sha256,
    )


def admission_receipt_from_report(report: BoundaryReport) -> AdmissionReceipt:
    """Convert one successful effective-runtime audit into immutable launch commitments."""

    required = (
        report.config_sha256,
        report.policy_sha256,
        report.manifest_sha256,
        report.source_commit,
        report.worker_image,
        report.broker_executable_sha256,
        report.broker_module_sha256,
    )
    if not report.allowed or not all(required):
        raise SecurityBoundaryError("an allowed complete remote audit is required")
    return AdmissionReceipt(
        schema="hermes.secure-worker.admission-receipt.v1",
        config_sha256=str(report.config_sha256),
        policy_sha256=str(report.policy_sha256),
        manifest_sha256=str(report.manifest_sha256),
        source_commit=str(report.source_commit),
        worker_image=str(report.worker_image),
        broker_executable_sha256=str(report.broker_executable_sha256),
        broker_module_sha256=str(report.broker_module_sha256),
    )


def write_admission_receipt(
    report: BoundaryReport,
    receipt_path: str | os.PathLike[str],
    *,
    pack_root: str | os.PathLike[str],
) -> AdmissionReceipt:
    path = Path(receipt_path).resolve()
    pack = Path(pack_root).resolve()
    _assert_external_file(pack, path, "admission receipt")
    if path.exists():
        raise SecurityBoundaryError("admission receipt destination already exists")
    receipt = admission_receipt_from_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(asdict(receipt)))
    return receipt


def verify_admission_receipt(
    report: BoundaryReport,
    receipt_path: str | os.PathLike[str],
    *,
    pack_root: str | os.PathLike[str],
) -> AdmissionReceipt:
    path = Path(receipt_path).resolve()
    _assert_external_file(Path(pack_root).resolve(), path, "admission receipt")
    if path.is_symlink() or not path.is_file():
        raise SecurityBoundaryError("admission receipt is missing or unsafe")
    try:
        receipt = AdmissionReceipt(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError) as exc:
        raise SecurityBoundaryError("admission receipt is invalid") from exc
    expected = admission_receipt_from_report(report)
    if receipt != expected:
        raise SecurityBoundaryError("admission receipt does not match effective runtime")
    return receipt
