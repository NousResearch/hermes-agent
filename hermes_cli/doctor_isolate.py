"""Non-mutating clean-room profile isolation for ``hermes doctor --isolate``."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from time import monotonic
from typing import Any, Callable

import yaml

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override


_SCHEMA_VERSION = 1
_EXCLUDED_NAMES = {
    "auth.json", ".env", "state.db", "state.db-wal", "state.db-shm", "gateway.pid",
    "gateway_state.json", "processes.json", "node_modules", "__pycache__",
}
_EXCLUDED_SUFFIXES = (".sock", ".pid", ".pyc", ".pyo", ".tmp")
_CONCURRENT_ROOTS = frozenset({"logs", "cache", "tmp"})
_DEFAULT_SHARED_ROOTS = {
    "hermes-agent", ".worktrees", "profiles", "bin", "node_modules",
    "local-models", "llama.cpp", "managed-node",
}
_SLICE_ROOTS: dict[str, tuple[str, ...]] = {
    "plugins": ("plugins",),
    "persona_skills_memory": (
        "SOUL.md", "AGENTS.md", "CLAUDE.md", ".cursorrules", "skills", "memories", "memory",
    ),
    "session_state": ("sessions", "checkpoints"),
    "cron_platform_gateway": ("cron", "platforms",),
}
_CONFIG_KEYS: dict[str, frozenset[str] | None] = {
    "config": None,
    "plugins": frozenset({"plugins", "hooks"}),
    "mcp": frozenset({"mcp", "tools"}),
    "persona_skills_memory": frozenset({"memory", "skills", "curator"}),
    "session_state": frozenset({"sessions", "checkpoints"}),
    "cron_platform_gateway": frozenset({"cron", "gateway"}),
}
_SLICE_ORDER = tuple(_CONFIG_KEYS)
_RESERVED_CONFIG_KEYS = frozenset().union(
    *[keys for keys in _CONFIG_KEYS.values() if keys is not None]
) | {"model", "providers", "_config_version"}


@dataclass
class SliceResult:
    id: str
    status: str
    error_class: str | None = None
    runtime: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"id": self.id, "status": self.status}
        if self.error_class:
            result["error_class"] = self.error_class
        if self.runtime is not None:
            result["runtime"] = self.runtime
        return result


@dataclass
class IsolationReport:
    source_profile: str
    source_generation: str
    source_manifest: list[dict[str, Any]]
    candidate_location: str
    control: dict[str, Any] = field(default_factory=dict)
    slices: list[SliceResult] = field(default_factory=list)
    classification: str = "inconclusive"
    culprit: dict[str, Any] | None = None
    cleanup_status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "classification": self.classification,
            "source_profile": self.source_profile,
            "source_generation": self.source_generation,
            "source_manifest": self.source_manifest,
            "candidate_location": self.candidate_location,
            "control": self.control,
            "slices": [item.to_dict() for item in self.slices],
            "culprit": self.culprit,
            "cleanup_status": self.cleanup_status,
        }


def _profile_name(source: Path) -> str:
    return source.name if source.parent.name == "profiles" else "default"


def _source_generation() -> str:
    try:
        from hermes_cli.version import get_version
        return str(get_version())
    except Exception:
        return "unknown"


def _excluded(relative: Path) -> bool:
    return (
        bool(relative.parts and relative.parts[0] in _CONCURRENT_ROOTS)
        or any(part in _EXCLUDED_NAMES for part in relative.parts)
        or relative.name.endswith(_EXCLUDED_SUFFIXES)
    )


def _iter_profile_files(source: Path):
    default_profile = source.parent.name != "profiles"
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if default_profile and relative.parts and relative.parts[0] in _DEFAULT_SHARED_ROOTS:
            continue
        if path.is_file() and not path.is_symlink():
            yield path, relative


def _manifest(source: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path, relative in _iter_profile_files(source):
        if _excluded(relative) or path.is_symlink() or not path.is_file():
            continue
        try:
            stat = path.stat()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        entries.append({"path": relative.as_posix(), "size": stat.st_size, "sha256": digest})
    return entries


def _guarded_fingerprint(source: Path) -> dict[str, tuple[int, str]]:
    """Fingerprint immutable inspected inputs, excluding separately snapshotted live state."""
    result: dict[str, tuple[int, str]] = {}
    for path, relative in _iter_profile_files(source):
        if _excluded(relative):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        result[relative.as_posix()] = (len(data), hashlib.sha256(data).hexdigest())
    return result


def _minimal_config(source_config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("_config_version", "model", "providers"):
        if key in source_config:
            result[key] = copy.deepcopy(source_config[key])
    return result


def _slice_config(source_config: dict[str, Any], slice_id: str) -> dict[str, Any]:
    keys = _CONFIG_KEYS[slice_id]
    if keys is None:
        keys = frozenset(source_config) - _RESERVED_CONFIG_KEYS
    return {key: copy.deepcopy(source_config[key]) for key in keys if key in source_config}


def _write_config(candidate: Path, config: dict[str, Any]) -> None:
    (candidate / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )


def _config_without_secrets(config: dict[str, Any], *, keep_references: bool) -> dict[str, Any]:
    """Copy config without credentials; unresolved ``${VAR}`` references remain safe on disk."""
    from hermes_cli.config import _ENV_PLACEHOLDER_RE, _is_secret_config_key

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, child in value.items():
                if isinstance(key, str) and _is_secret_config_key(key):
                    if keep_references and isinstance(child, str) and _ENV_PLACEHOLDER_RE.match(child):
                        result[key] = child
                    continue
                result[key] = clean(child)
            return result
        if isinstance(value, list):
            return [clean(child) for child in value]
        return copy.deepcopy(value)

    return clean(config)


def _snapshot_state_db(source: Path, candidate: Path, *, timeout_seconds: float = 1.0) -> bool:
    """Take a bounded online backup, including committed WAL state, without writing source."""
    source_db = source / "state.db"
    if not source_db.exists():
        return True
    target_db = candidate / "state.db"
    deadline = monotonic() + timeout_seconds

    def progress(_status: int, _remaining: int, _total: int) -> None:
        if monotonic() >= deadline:
            raise TimeoutError("state snapshot deadline exceeded")

    source_uri = f"file:{source_db.as_posix()}?mode=ro"
    try:
        with sqlite3.connect(source_uri, uri=True, timeout=0.1) as source_conn:
            with sqlite3.connect(target_db, timeout=0.1) as target_conn:
                source_conn.backup(target_conn, pages=256, progress=progress, sleep=0.01)
        return True
    except (OSError, sqlite3.Error, TimeoutError):
        target_db.unlink(missing_ok=True)
        return False


def _copy_entry(source: Path, candidate: Path, relative: str) -> None:
    src = source / relative
    if not src.exists() or src.is_symlink():
        return
    dst = candidate / relative
    if src.is_dir():
        shutil.copytree(
            src,
            dst,
            dirs_exist_ok=True,
            symlinks=True,
            ignore=lambda directory, names: {
                name for name in names
                if _excluded((Path(directory) / name).relative_to(source))
            },
        )
    elif src.is_file() and not _excluded(Path(relative)):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst, follow_symlinks=False)


def _prepare_candidate(
    parent: Path,
    source: Path,
    label: str,
    disk_config: dict[str, Any],
    included_slices: list[str],
) -> tuple[Path, bool]:
    candidate = parent / label
    candidate.mkdir()
    _write_config(candidate, disk_config)
    for slice_id in included_slices:
        for relative in _SLICE_ROOTS.get(slice_id, ()):
            _copy_entry(source, candidate, relative)
    state_ready = "session_state" not in included_slices or _snapshot_state_db(source, candidate)
    return candidate, state_ready


def _default_probe(config: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    from hermes_cli.doctor_runtime import run_runtime_diagnostic
    return run_runtime_diagnostic(config_override=config, runtime_override=runtime).to_dict()


def _probe_candidate(
    candidate: Path,
    config: dict[str, Any],
    runtime: dict[str, Any],
    probe: Callable[[Path, dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    token = set_hermes_home_override(candidate)
    try:
        return probe(candidate, config, runtime)
    finally:
        reset_hermes_home_override(token)


def _probe_status(payload: dict[str, Any], control: dict[str, Any] | None = None) -> str:
    if payload.get("status") != "ok":
        return "fail"
    if control:
        current = (payload.get("timings") or {}).get("hermes_first_chunk_ms")
        baseline = (control.get("timings") or {}).get("hermes_first_chunk_ms")
        if isinstance(current, (int, float)) and isinstance(baseline, (int, float)):
            if current > baseline * 3 and current - baseline > 500:
                return "slow"
    return "pass"


def _resolve_source_runtime(source_config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = source_config.get("model") if isinstance(source_config.get("model"), dict) else {}
    from hermes_cli.runtime_provider import resolve_runtime_provider
    return resolve_runtime_provider(
        requested=str(model_cfg.get("provider") or "auto"),
        target_model=str(model_cfg.get("default") or ""),
    )


def run_isolation_diagnostic(
    *,
    source: Path | None = None,
    probe: Callable[[Path, dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    runtime_override: dict[str, Any] | None = None,
) -> IsolationReport:
    """Run a cumulative clean-room differential without writing under *source*."""
    source = (source or get_hermes_home()).resolve()
    probe = probe or (lambda _candidate, config, runtime: _default_probe(config, runtime))
    before = _manifest(source)
    source_fingerprint = _guarded_fingerprint(source)

    # Use the raw diagnostic reader plus the normal effective-config transform. The
    # general loader seeds profile files and writes last-known-good backups, which would
    # violate isolate mode's source-byte invariant merely by observing the profile.
    from hermes_cli.config import read_user_config_raw
    from hermes_cli.config_effective import _effective
    source_config_raw = read_user_config_raw(source / "config.yaml")
    source_config = _effective(source_config_raw)
    runtime = runtime_override if runtime_override is not None else _resolve_source_runtime(source_config)
    runtime = dict(runtime)
    # A diagnostic request must not bench/rotate/persist a source credential pool.
    runtime["credential_pool"] = None
    safe_effective = _config_without_secrets(source_config, keep_references=False)
    safe_raw = _config_without_secrets(source_config_raw, keep_references=True)
    minimal = _minimal_config(safe_effective)
    disk_minimal = _minimal_config(safe_raw)
    candidate_parent = Path(tempfile.mkdtemp(prefix=f".hermes-isolate-{source.name}-", dir=source.parent))
    report = IsolationReport(
        source_profile=_profile_name(source),
        source_generation=_source_generation(),
        source_manifest=before,
        candidate_location=str(candidate_parent),
    )
    effective = copy.deepcopy(minimal)
    disk_effective = copy.deepcopy(disk_minimal)
    included_slices: list[str] = []
    unresolved = False
    try:
        candidate, _ = _prepare_candidate(
            candidate_parent, source, "00-control", disk_effective, included_slices
        )
        control = _probe_candidate(candidate, effective, runtime, probe)
        control_status = _probe_status(control)
        report.control = {"status": control_status, "runtime": control}
        if control_status != "pass":
            report.classification = "below_profile_layer"
            return report

        for index, slice_id in enumerate(_SLICE_ORDER, start=1):
            included_slices.append(slice_id)
            effective.update(_slice_config(safe_effective, slice_id))
            disk_effective.update(_slice_config(safe_raw, slice_id))
            candidate, state_ready = _prepare_candidate(
                candidate_parent,
                source,
                f"{index:02d}-{slice_id}",
                disk_effective,
                included_slices,
            )
            if not state_ready:
                report.slices.append(SliceResult(slice_id, "needs_quiescence"))
                unresolved = True
                continue
            payload = _probe_candidate(candidate, effective, runtime, probe)
            status = _probe_status(payload, control)
            error_class = payload.get("error_class") if status == "fail" else None
            report.slices.append(SliceResult(slice_id, status, error_class, payload))
            if status in {"fail", "slow"}:
                report.classification = "profile_state_regression"
                report.culprit = {
                    "slice": slice_id,
                    "component": payload.get("failed_phase"),
                    "evidence": [f"{slice_id} changed the acceptance probe from pass to {status}"],
                }
                break
        else:
            report.classification = "needs_quiescence" if unresolved else "healthy"
        return report
    finally:
        try:
            shutil.rmtree(candidate_parent)
            report.cleanup_status = "removed"
        except OSError:
            report.cleanup_status = "failed"
        if _guarded_fingerprint(source) != source_fingerprint:
            report.classification = "needs_quiescence"


def render_isolation_report(report: IsolationReport) -> None:
    print("\nClean-room profile isolation")
    print(f"  Source profile: {report.source_profile}")
    print(f"  Candidate: {report.candidate_location} ({report.cleanup_status})")
    print(f"  Sterile control: {str(report.control.get('status', 'unknown')).upper()}")
    for item in report.slices:
        print(f"  {item.id}: {item.status.upper()}")
    print(f"  Classification: {report.classification}")
    if report.culprit:
        print(f"  Culprit slice: {report.culprit.get('slice') or '(unknown)'}")
