"""Canonical filesystem-backed Project State Engine for Hermes CLI.

The authoritative project state lives only under:

    ~/.hermes/project_state/

``session.json`` is an index/current pointer.  Durable project facts live in
neighboring canonical state files (project.json, versions.json, pipelines.json,
artifacts.json, checkpoints.json, graph.json).  Historical ``SESSION_STATE.json``
files are imported once for backwards compatibility, but are never considered
authoritative after migration.
"""

from __future__ import annotations

import json
import os
import pwd
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

LEGACY_SESSION_STATE_NAME = "SESSION_STATE.json"
PROJECT_STATE_SCHEMA_VERSION = "2.0"

def _real_user_home() -> Path:
    """Return the OS account home, not Hermes' profile-scoped HOME shim."""
    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir).expanduser().resolve()
    except Exception:
        return Path(os.path.expanduser("~")).resolve()

PROJECT_STATE_DIR = _real_user_home() / ".hermes" / "project_state"
STATE_FILE_NAMES = {
    "project": "project.json",
    "versions": "versions.json",
    "pipelines": "pipelines.json",
    "artifacts": "artifacts.json",
    "checkpoints": "checkpoints.json",
    "graph": "graph.json",
    "session": "session.json",
}

# Legacy API shape retained for callers/tests that still generate or validate a
# standalone SESSION_STATE.json.  CLI commands no longer use this shape as an
# authority; they migrate it into ~/.hermes/project_state/session.json.
_REQUIRED_STATE_FIELDS = {
    "project",
    "pipeline",
    "current_version",
    "current_stage",
    "latest_design_directory",
    "latest_checkpoint_directory",
    "latest_manifest",
    "latest_verified_hashes",
    "latest_validation_report",
    "latest_result",
    "immutable_baselines_count",
    "canonical_artifacts",
    "next_recommended_prompt",
    "schema_version",
    "generated_at",
}

_PROTECTED_NAMES = {
    LEGACY_SESSION_STATE_NAME,
    *STATE_FILE_NAMES.values(),
    "manifest.json",
    "verified_hashes.txt",
    "validation_report.json",
    "canonical_artifacts.tsv",
}


class SessionStateError(RuntimeError):
    """Raised when canonical project state cannot be found or validated."""


@dataclass(frozen=True)
class ValidationResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class LocatedState:
    path: Path
    state: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SessionStateError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SessionStateError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SessionStateError(f"JSON root must be an object: {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _abs(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve())


def _state_dir() -> Path:
    return PROJECT_STATE_DIR.expanduser().resolve()


def _state_path(name: str) -> Path:
    return _state_dir() / STATE_FILE_NAMES[name]


def _version_key(version: str) -> tuple[int, ...]:
    nums = [int(part) for part in re.findall(r"\d+", str(version))]
    return tuple(nums or [0])


def _parse_hash_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    match = re.match(r"^([0-9a-fA-F]{64})\s+[ *]?(.*)$", stripped)
    if not match:
        return None
    return match.group(1).lower(), match.group(2)


def _validate_manifest(path: Path) -> ValidationResult:
    try:
        manifest = _read_json(path)
    except SessionStateError as exc:
        return ValidationResult("manifest", False, str(exc))
    result = manifest.get("result") or manifest.get("status")
    token = manifest.get("result_token")
    if result not in {"PASS", "READONLY_DESIGN"}:
        return ValidationResult("manifest", False, f"unexpected result/status: {result!r}")
    if not token and manifest.get("result") == "PASS":
        return ValidationResult("manifest", False, "missing result_token")
    return ValidationResult("manifest", True, str(path))


def _validate_verified_hashes(path: Path, base_dir: Path | None = None) -> ValidationResult:
    if not path.is_file():
        return ValidationResult("verified_hashes", False, f"missing: {path}")
    base = base_dir or path.parent
    mismatches: list[str] = []
    parsed_any = False
    import hashlib

    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parsed = _parse_hash_line(line)
        if parsed is None:
            if line.strip():
                mismatches.append(f"line {idx}: invalid hash line")
            continue
        parsed_any = True
        expected, rel = parsed
        target = (base / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
        try:
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as exc:
            mismatches.append(f"{rel}: {exc}")
            continue
        if actual != expected:
            mismatches.append(f"{rel}: expected {expected}, got {actual}")
    if not parsed_any:
        return ValidationResult("verified_hashes", False, "no hash entries")
    if mismatches:
        return ValidationResult("verified_hashes", False, "; ".join(mismatches[:5]))
    return ValidationResult("verified_hashes", True, str(path))


def _validation_booleans(data: dict[str, Any]) -> list[bool]:
    values: list[bool] = []
    validations = data.get("validations")
    if isinstance(validations, dict):
        values.extend(v is True for v in validations.values())
    checks = data.get("checks") or data.get("validation_details")
    if isinstance(checks, list):
        for check in checks:
            if isinstance(check, dict) and "passed" in check:
                values.append(check.get("passed") is True)
    return values


def _validate_validation_report(path: Path) -> ValidationResult:
    try:
        report = _read_json(path)
    except SessionStateError as exc:
        return ValidationResult("validation_report", False, str(exc))
    booleans = _validation_booleans(report)
    if booleans and not all(booleans):
        return ValidationResult("validation_report", False, "one or more checks failed")
    if not booleans and report.get("result") not in {"PASS", None}:
        return ValidationResult("validation_report", False, f"unexpected result: {report.get('result')!r}")
    return ValidationResult("validation_report", True, str(path))


def _is_hidden_relative(root: Path, path: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return any(part.startswith(".") for part in rel.parts[:-1])


def find_legacy_session_state(start: Path | None = None) -> Path | None:
    """Find the newest legacy SESSION_STATE.json for one-time migration only."""
    root = (start or Path.cwd()).expanduser().resolve()
    candidates: list[Path] = []
    for base in [root, *root.parents]:
        candidate = base / LEGACY_SESSION_STATE_NAME
        if candidate.is_file():
            candidates.append(candidate)
    if root.is_dir():
        try:
            for candidate in root.rglob(LEGACY_SESSION_STATE_NAME):
                if not root.name.startswith(".") and _is_hidden_relative(root, candidate):
                    continue
                if candidate.is_file():
                    candidates.append(candidate)
        except OSError:
            pass
    unique = {candidate.resolve() for candidate in candidates}
    return max(unique, key=lambda p: p.stat().st_mtime_ns) if unique else None


def find_session_state(start: Path | None = None) -> Path:
    """Return the single canonical session index path, migrating if needed."""
    ensure_project_state(start)
    path = _state_path("session")
    if not path.is_file():
        raise SessionStateError(f"canonical session missing: {path}")
    return path


def _read_canonical_artifact_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split("\t")
        rows.append({header[i]: values[i] if i < len(values) else "" for i in range(len(header))})
    return rows


def _extract_next_prompt(path: Path) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("recommended:"):
            return stripped.split(":", 1)[1].strip()
    for line in text.splitlines():
        stripped = line.strip(" -")
        if "HRP" in stripped and "DESIGN" in stripped.upper():
            return stripped
    return ""


def _parse_next_step(prompt: str, fallback_version: str, fallback_stage: str) -> dict[str, str]:
    text = (prompt or "").strip().rstrip(".")
    family = "HRP"
    version = fallback_version
    module = fallback_stage.replace("_", " ").title()
    mode = "DESIGN_READONLY" if "DESIGN" in fallback_stage.upper() else ""
    kind = "design" if "DESIGN" in fallback_stage.upper() else "unknown"
    match = re.search(r"\b([A-Z]+)\s+v([0-9.]+)\s*[—-]\s*(.*)$", text)
    if match:
        family, version, rest = match.group(1), match.group(2), match.group(3).strip()
        upper_rest = rest.upper()
        if "DESIGN READONLY" in upper_rest:
            mode = "DESIGN_READONLY"
            module = re.sub(r"\s+DESIGN\s+READONLY\s*$", "", rest, flags=re.IGNORECASE).strip()
            kind = "design"
        elif "READONLY" in upper_rest:
            mode = "READONLY"
            module = re.sub(r"\s+READONLY\s*$", "", rest, flags=re.IGNORECASE).strip()
        else:
            module = rest
    return {
        "family": family,
        "series": "Hermes Runtime Platform",
        "version": version,
        "module": module,
        "mode": mode,
        "kind": kind,
    }


def _next_step_prompt(next_step: dict[str, Any]) -> str:
    family = str(next_step.get("family") or "HRP")
    version = str(next_step.get("version") or "").strip()
    module = str(next_step.get("module") or "").strip()
    mode = str(next_step.get("mode") or "").strip()
    mode_text = mode.replace("_", " ").strip()
    parts = [module, mode_text]
    suffix = " ".join(part for part in parts if part).strip()
    return f"{family} v{version} — {suffix}." if version else suffix


def _infer_project_and_stage(package: str, version: str, checkpoint_dir: Path) -> tuple[str, str]:
    if package:
        parts = package.split("_V")
        project = parts[0] if parts else package
        marker = f"V{version.replace('.', '_')}_"
        stage = package.split(marker, 1)[1] if marker in package else package
        return project, stage
    name = checkpoint_dir.name.upper()
    match = re.search(r"(.*)_V\d+_\d+_(.*?)(?:_CHECKPOINT)?$", name)
    if match:
        return match.group(1), match.group(2)
    return checkpoint_dir.name, "UNKNOWN"


def _infer_design_dir_from_checkpoint(checkpoint_dir: Path, checkpoint_manifest: dict[str, Any]) -> Path:
    source = checkpoint_manifest.get("source_dir")
    if isinstance(source, str) and source:
        return Path(source).expanduser().resolve()
    name = checkpoint_dir.name
    for suffix in ("_checkpoint", "-checkpoint", "_CHECKPOINT", "-CHECKPOINT"):
        if name.endswith(suffix):
            candidate = checkpoint_dir.with_name(name[: -len(suffix)])
            if candidate.is_dir():
                return candidate.resolve()
    raise SessionStateError(f"cannot infer design directory from {checkpoint_dir}")


def validate_state(state: dict[str, Any]) -> list[ValidationResult]:
    """Validate a legacy standalone SESSION_STATE payload.

    This function is kept for backwards-compatible callers.  Canonical CLI
    validation uses validate_located_state() against project_state/session.json.
    """
    manifest = Path(str(state.get("latest_manifest", ""))).expanduser()
    hashes = Path(str(state.get("latest_verified_hashes", ""))).expanduser()
    validation = Path(str(state.get("latest_validation_report", ""))).expanduser()
    return [
        _validate_manifest(manifest),
        _validate_verified_hashes(hashes, manifest.parent if str(manifest) else None),
        _validate_validation_report(validation),
    ]


def generate_session_state(
    *,
    checkpoint_dir: Path | None = None,
    output: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    """Generate a legacy SESSION_STATE.json and migrate canonical project_state.

    Backward-compatible API only.  The returned SESSION_STATE.json is an import
    source, not an authority.  The canonical index remains
    ~/.hermes/project_state/session.json.
    """
    root = (project_root or Path.cwd()).expanduser().resolve()
    checkpoint = (checkpoint_dir or discover_latest_checkpoint(root)).expanduser().resolve()
    legacy = _state_from_checkpoint(checkpoint)
    pipeline = [str(item) for item in legacy.get("pipeline", [])] if isinstance(legacy.get("pipeline"), list) else []
    state = {
        "project": legacy.get("project"),
        "pipeline": pipeline,
        "current_version": legacy.get("current_version"),
        "current_stage": legacy.get("current_stage"),
        "latest_design_directory": legacy.get("latest_design_directory"),
        "latest_checkpoint_directory": legacy.get("latest_checkpoint_directory"),
        "latest_manifest": legacy.get("latest_manifest"),
        "latest_verified_hashes": legacy.get("latest_verified_hashes"),
        "latest_validation_report": legacy.get("latest_validation_report"),
        "latest_result": legacy.get("latest_result"),
        "immutable_baselines_count": max(0, len(pipeline) - 2),
        "canonical_artifacts": legacy.get("canonical_artifacts"),
        "next_recommended_prompt": legacy.get("next_recommended_prompt"),
        "schema_version": "1.0",
        "generated_at": _utc_now(),
    }
    results = validate_state(state)
    if not _all_pass(results):
        details = "; ".join(f"{r.name}: {r.detail}" for r in results if not r.passed)
        raise SessionStateError(f"refusing to write invalid SESSION_STATE: {details}")
    target = (output or (root / LEGACY_SESSION_STATE_NAME)).expanduser().resolve()
    _write_json(target, state)
    return target


def discover_latest_checkpoint(start: Path | None = None) -> Path:
    root = (start or Path.cwd()).expanduser().resolve()
    candidates: list[Path] = []
    for pattern in ("*checkpoint*", "*CHECKPOINT*"):
        for path in root.glob(pattern):
            if path.is_dir() and (path / "manifest.json").is_file():
                candidates.append(path)
    if not candidates and root.is_dir():
        for path in root.rglob("manifest.json"):
            parent = path.parent
            if "checkpoint" in parent.name.lower() and (parent / "canonical_artifacts.tsv").is_file():
                candidates.append(parent)
    if not candidates:
        raise SessionStateError("no matching checkpoint artifacts found")
    return max(candidates, key=lambda p: p.stat().st_mtime_ns).resolve()


def _scan_hrp_versions(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifests = [p for p in root.glob("hermes_runtime_platform_v*/manifest.json") if p.is_file()]
    design_by_dir: dict[Path, dict[str, Any]] = {}
    checkpoints_by_design: dict[Path, dict[str, Any]] = {}

    for path in manifests:
        try:
            manifest = _read_json(path)
        except SessionStateError:
            continue
        parent = path.parent.resolve()
        if "checkpoint" in parent.name.lower():
            try:
                source = _infer_design_dir_from_checkpoint(parent, manifest)
            except SessionStateError:
                source = parent.with_name(parent.name.replace("_checkpoint", "")).resolve()
            checkpoints_by_design[source] = {"dir": parent, "manifest": manifest}
        elif manifest.get("version") and manifest.get("result") == "PASS":
            design_by_dir[parent] = {"dir": parent, "manifest": manifest}

    versions: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for design_dir, item in design_by_dir.items():
        manifest = item["manifest"]
        version = str(manifest.get("version") or "")
        checkpoint_item = checkpoints_by_design.get(design_dir)
        checkpoint_dir = checkpoint_item["dir"] if checkpoint_item else Path("")
        result = str(manifest.get("result_token") or manifest.get("result") or "")
        status = str(manifest.get("status") or manifest.get("result") or "")
        versions.append(
            {
                "version": version,
                "result": result,
                "design_dir": _abs(design_dir),
                "checkpoint_dir": _abs(checkpoint_dir) if str(checkpoint_dir) else "",
                "timestamp": str(manifest.get("created_at") or ""),
                "status": status,
            }
        )
        for artifact in manifest.get("artifacts", []) if isinstance(manifest.get("artifacts"), list) else []:
            artifacts.append({"version": version, "path": _abs(design_dir / str(artifact)), "source": "design_manifest"})
        if checkpoint_item:
            cman = checkpoint_item["manifest"]
            checkpoints.append(
                {
                    "version": version,
                    "checkpoint_dir": _abs(checkpoint_dir),
                    "manifest": _abs(checkpoint_dir / "manifest.json"),
                    "verified_hashes": _abs(checkpoint_dir / "verified_hashes.txt"),
                    "canonical_artifacts": _abs(checkpoint_dir / "canonical_artifacts.tsv"),
                    "result": str(cman.get("result_token") or cman.get("result") or ""),
                    "status": str(cman.get("status") or cman.get("result") or ""),
                    "timestamp": str(cman.get("created_at") or ""),
                }
            )
            for row in _read_canonical_artifact_rows(checkpoint_dir / "canonical_artifacts.tsv"):
                record = {"version": version, "source": "canonical_artifacts_tsv"}
                record.update(row)
                artifacts.append(record)

    versions.sort(key=lambda row: _version_key(str(row.get("version", ""))))
    checkpoints.sort(key=lambda row: _version_key(str(row.get("version", ""))))
    return versions, checkpoints, artifacts


def _legacy_state_or_latest(start: Path | None = None) -> tuple[dict[str, Any], Path | None]:
    legacy_path = find_legacy_session_state(start)
    if legacy_path:
        return _read_json(legacy_path), legacy_path
    checkpoint = discover_latest_checkpoint(start)
    return _state_from_checkpoint(checkpoint), None


def _state_from_checkpoint(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    checkpoint_manifest = _read_json(checkpoint / "manifest.json")
    design = _infer_design_dir_from_checkpoint(checkpoint, checkpoint_manifest)
    design_manifest = _read_json(design / "manifest.json")
    version = str(design_manifest.get("version") or checkpoint_manifest.get("version") or "")
    package = str(design_manifest.get("package") or checkpoint_manifest.get("checkpoint") or "")
    project, stage = _infer_project_and_stage(package, version, checkpoint)
    return {
        "project": project,
        "pipeline": [*design_manifest.get("sources_consumed_unchanged", []), _abs(design), _abs(checkpoint)],
        "current_version": version,
        "current_stage": stage,
        "latest_design_directory": _abs(design),
        "latest_checkpoint_directory": _abs(checkpoint),
        "latest_manifest": _abs(design / "manifest.json"),
        "latest_verified_hashes": _abs(design / "verified_hashes.txt"),
        "latest_validation_report": _abs(design / "validation_report.json"),
        "latest_result": design_manifest.get("result_token") or checkpoint_manifest.get("source_result_token") or design_manifest.get("result"),
        "canonical_artifacts": _abs(checkpoint / "canonical_artifacts.tsv"),
        "next_recommended_prompt": _extract_next_prompt(checkpoint / "next_step_recommendation.md"),
    }


def _canonical_file_refs() -> dict[str, str]:
    return {key: _abs(_state_path(key)) for key in ["project", "versions", "pipelines", "artifacts", "checkpoints", "graph"]}


def migrate_project_state(start: Path | None = None, *, force: bool = False) -> Path:
    """Create/update canonical project_state files from legacy state and disk artifacts."""
    state_dir = _state_dir()
    session_path = _state_path("session")
    if session_path.is_file() and not force:
        return session_path

    legacy, legacy_path = _legacy_state_or_latest(start)
    root = (start.expanduser().resolve() if force and start is not None else _real_user_home())
    versions, checkpoints, artifacts = _scan_hrp_versions(root)
    current_version = str(legacy.get("current_version") or (versions[-1]["version"] if versions else ""))
    current_stage = str(legacy.get("current_stage") or "")
    current_checkpoint = str(legacy.get("latest_checkpoint_directory") or "")
    current_result = str(legacy.get("latest_result") or "")
    next_step = _parse_next_step(str(legacy.get("next_recommended_prompt") or ""), current_version, current_stage)
    pipeline = [str(item) for item in legacy.get("pipeline", [])] if isinstance(legacy.get("pipeline"), list) else []

    _write_json(
        _state_path("project"),
        {
            "schema_version": PROJECT_STATE_SCHEMA_VERSION,
            "project": str(legacy.get("project") or "HERMES_RUNTIME_PLATFORM"),
            "state_dir": _abs(state_dir),
            "created_at": _utc_now(),
            "migration": {
                "completed": True,
                "imported_from": _abs(legacy_path) if legacy_path else None,
                "legacy_authority": False,
                "previous_artifacts_untouched": True,
            },
        },
    )
    _write_json(_state_path("versions"), {"schema_version": PROJECT_STATE_SCHEMA_VERSION, "versions": versions})
    _write_json(
        _state_path("pipelines"),
        {
            "schema_version": PROJECT_STATE_SCHEMA_VERSION,
            "pipelines": [
                {
                    "name": "HRP",
                    "current": True,
                    "versions": [row.get("version") for row in versions],
                    "paths": pipeline,
                }
            ],
        },
    )
    _write_json(_state_path("artifacts"), {"schema_version": PROJECT_STATE_SCHEMA_VERSION, "artifacts": artifacts})
    _write_json(_state_path("checkpoints"), {"schema_version": PROJECT_STATE_SCHEMA_VERSION, "checkpoints": checkpoints})
    edges = []
    for prev, cur in zip(versions, versions[1:]):
        edges.append({"from": prev.get("version"), "to": cur.get("version"), "relationship": "precedes"})
    _write_json(_state_path("graph"), {"schema_version": PROJECT_STATE_SCHEMA_VERSION, "nodes": [v.get("version") for v in versions], "edges": edges})
    session = {
        "schema_version": PROJECT_STATE_SCHEMA_VERSION,
        "state_files": _canonical_file_refs(),
        "current_version": current_version,
        "current_pipeline": "HRP",
        "current_checkpoint": current_checkpoint,
        "current_result": current_result,
        "next_step": next_step,
    }
    _write_json(session_path, session)
    return session_path


def ensure_project_state(start: Path | None = None) -> Path:
    session = _state_path("session")
    if not session.is_file():
        return migrate_project_state(start)
    # Ensure every canonical file exists; if not, repair from legacy/disk.
    missing = [_state_path(name) for name in STATE_FILE_NAMES if not _state_path(name).is_file()]
    if missing:
        return migrate_project_state(start, force=True)
    return session


def load_session_state(start: Path | None = None) -> LocatedState:
    path = find_session_state(start)
    state = _read_json(path)
    required = {"schema_version", "state_files", "current_version", "current_pipeline", "current_checkpoint", "current_result", "next_step"}
    missing = sorted(required - set(state))
    if missing:
        raise SessionStateError(f"{path} missing required fields: {', '.join(missing)}")
    return LocatedState(path=path, state=state)


def _load_state_file(session: dict[str, Any], key: str) -> dict[str, Any]:
    refs = session.get("state_files")
    if not isinstance(refs, dict) or key not in refs:
        raise SessionStateError(f"session index missing state_files.{key}")
    return _read_json(Path(str(refs[key])).expanduser())


def _current_version_entry(session: dict[str, Any]) -> dict[str, Any]:
    versions = _load_state_file(session, "versions").get("versions", [])
    if isinstance(versions, list):
        for row in versions:
            if isinstance(row, dict) and str(row.get("version")) == str(session.get("current_version")):
                return row
    return {}


def _current_checkpoint_entry(session: dict[str, Any]) -> dict[str, Any]:
    checkpoints = _load_state_file(session, "checkpoints").get("checkpoints", [])
    current = str(session.get("current_checkpoint") or "")
    if isinstance(checkpoints, list):
        for row in checkpoints:
            if isinstance(row, dict) and str(row.get("checkpoint_dir")) == current:
                return row
    return {}


def validate_located_state(located: LocatedState) -> list[ValidationResult]:
    state = located.state
    results: list[ValidationResult] = []
    canonical = _state_dir()
    results.append(ValidationResult("single_canonical_session", located.path.resolve() == _state_path("session"), str(located.path)))
    refs = state.get("state_files") if isinstance(state.get("state_files"), dict) else {}
    for key in ["project", "versions", "pipelines", "artifacts", "checkpoints", "graph"]:
        path = Path(str(refs.get(key, ""))).expanduser() if refs else _state_path(key)
        results.append(ValidationResult(f"{key}_state", path.resolve() == _state_path(key) and path.is_file(), str(path)))
    try:
        project = _load_state_file(state, "project")
        migration = project.get("migration", {})
        results.append(ValidationResult("migration_complete", isinstance(migration, dict) and migration.get("completed") is True, str(migration)))
    except SessionStateError as exc:
        results.append(ValidationResult("migration_complete", False, str(exc)))
    version_entry = _current_version_entry(state)
    checkpoint_entry = _current_checkpoint_entry(state)
    if version_entry:
        manifest = Path(str(version_entry.get("design_dir", ""))) / "manifest.json"
        validation = Path(str(version_entry.get("design_dir", ""))) / "validation_report.json"
        hashes = Path(str(version_entry.get("design_dir", ""))) / "verified_hashes.txt"
        results.extend([
            _validate_manifest(manifest),
            _validate_verified_hashes(hashes, manifest.parent if str(manifest) else None),
            _validate_validation_report(validation),
        ])
    else:
        results.append(ValidationResult("current_version_registry", False, str(state.get("current_version"))))
    checkpoint_registry_ok = bool(checkpoint_entry) and str(checkpoint_entry.get("checkpoint_dir")) == str(state.get("current_checkpoint"))
    results.append(ValidationResult("single_checkpoint_registry", checkpoint_registry_ok, str(state.get("current_checkpoint"))))
    # Prove canonical state is confined to ~/.hermes/project_state.
    for key in ["project", "versions", "pipelines", "artifacts", "checkpoints", "graph", "session"]:
        try:
            _state_path(key).resolve().relative_to(canonical)
            confined = True
        except ValueError:
            confined = False
        results.append(ValidationResult(f"{key}_canonical_path", confined, str(_state_path(key))))
    return results


def _all_pass(results: Iterable[ValidationResult]) -> bool:
    return all(result.passed for result in results)


def print_project_status(located: LocatedState, *, stream: Any = None) -> int:
    stream = stream or sys.stdout
    state = located.state
    results = validate_located_state(located)
    by_name = {result.name: result for result in results}
    ok = _all_pass(results)

    def yesno(result_name: str) -> str:
        result = by_name.get(result_name)
        return "PASS" if result and result.passed else "BLOCK"

    version_entry = _current_version_entry(state)
    checkpoint_entry = _current_checkpoint_entry(state)
    print(f"Project state: {_state_dir()}", file=stream)
    print(f"SESSION_STATE: {located.path}", file=stream)
    print(f"Current Version: {state.get('current_version')}", file=stream)
    print(f"Current Pipeline: {state.get('current_pipeline')}", file=stream)
    print(f"Result: {state.get('current_result')}", file=stream)
    print(f"Checkpoint: {state.get('current_checkpoint')}", file=stream)
    print(f"Manifest: {Path(str(version_entry.get('design_dir', ''))) / 'manifest.json' if version_entry else ''}", file=stream)
    print(f"Hash validation: {yesno('verified_hashes')}", file=stream)
    print(f"Validation report: {yesno('validation_report')}", file=stream)
    print(f"Single canonical session: {yesno('single_canonical_session')}", file=stream)
    print(f"Single project state: {'PASS' if all(yesno(k + '_state') == 'PASS' for k in ['project','versions','pipelines','artifacts','checkpoints','graph']) else 'BLOCK'}", file=stream)
    print(f"Single checkpoint registry: {yesno('single_checkpoint_registry')}", file=stream)
    print(f"No duplicated authority: {'PASS' if ok else 'BLOCK'}", file=stream)
    print(f"Next step: {json.dumps(state.get('next_step'), sort_keys=True)}", file=stream)
    return 0 if ok else 1


def _has_local_legacy_session_state(start: Path) -> bool:
    root = start.expanduser().resolve()
    if root.is_file():
        return root.name == LEGACY_SESSION_STATE_NAME
    if (root / LEGACY_SESSION_STATE_NAME).is_file():
        return True
    if root.is_dir():
        try:
            return any(path.is_file() for path in root.rglob(LEGACY_SESSION_STATE_NAME))
        except OSError:
            return False
    return False


def command_status(args: Any) -> int:
    start = Path(getattr(args, "path", None) or Path.cwd())
    if getattr(args, "path", None) and _has_local_legacy_session_state(start):
        migrate_project_state(start, force=True)
    located = load_session_state(start)
    return print_project_status(located)


def command_resume(args: Any) -> int:
    start = Path(getattr(args, "path", None) or Path.cwd())
    if getattr(args, "path", None) and _has_local_legacy_session_state(start):
        migrate_project_state(start, force=True)
    located = load_session_state(start)
    results = validate_located_state(located)
    if not _all_pass(results):
        for result in results:
            if not result.passed:
                print(f"BLOCK {result.name}: {result.detail}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(located.state, indent=2, sort_keys=True))
    else:
        print(f"SESSION_STATE: {located.path}")
        print(f"Project state: {_state_dir()}")
        print(f"Current Version: {located.state.get('current_version')}")
        print(f"Current Pipeline: {located.state.get('current_pipeline')}")
        print(f"Result: {located.state.get('current_result')}")
        print("Resume validation: PASS")
    return 0


def command_checkpoint(args: Any) -> int:
    checkpoint_arg = getattr(args, "checkpoint_dir", None)
    if checkpoint_arg:
        legacy = _state_from_checkpoint(Path(checkpoint_arg))
        # Regenerate canonical files from this explicit checkpoint without touching historical artifacts.
        temp_session = migrate_project_state(Path(getattr(args, "project_root", None) or Path.cwd()), force=True)
        state = _read_json(temp_session)
        state["current_version"] = str(legacy.get("current_version") or state.get("current_version"))
        state["current_checkpoint"] = str(legacy.get("latest_checkpoint_directory") or state.get("current_checkpoint"))
        state["current_result"] = str(legacy.get("latest_result") or state.get("current_result"))
        state["next_step"] = _parse_next_step(str(legacy.get("next_recommended_prompt") or ""), state["current_version"], str(legacy.get("current_stage") or ""))
        _write_json(temp_session, state)
        path = temp_session
    else:
        path = migrate_project_state(Path(getattr(args, "project_root", None) or Path.cwd()), force=True)
    print(f"SESSION_STATE: {path}")
    print(f"Project state: {_state_dir()}")
    print("Checkpoint session state: PASS")
    print("Checkpoint project state: PASS")
    return 0


def command_next(args: Any) -> int:
    rc = command_checkpoint(args)
    if rc:
        return rc
    located = load_session_state(Path(getattr(args, "project_root", None) or Path.cwd()))
    rc = print_project_status(located)
    if rc:
        return rc
    resume_args = type("ResumeArgs", (), {"path": str(_state_dir()), "json": False})()
    rc = command_resume(resume_args)
    if rc:
        return rc
    next_step = located.state.get("next_step")
    if not isinstance(next_step, dict):
        print("BLOCK next_step is missing", file=sys.stderr)
        return 1
    prompt = _next_step_prompt(next_step)
    if not prompt:
        print("BLOCK next_step prompt is empty", file=sys.stderr)
        return 1
    if getattr(args, "dry_run", False):
        print(f"Next execution dry-run: {prompt}")
        return 0
    hermes_exe = shutil.which("hermes") or sys.argv[0]
    completed = subprocess.run([hermes_exe, "chat", "-q", prompt], text=True)
    return int(completed.returncode or 0)


def command_gc(args: Any) -> int:
    """Archive temporary conversation metadata while preserving canonical project state."""
    from hermes_cli.config import get_hermes_home

    ensure_project_state(Path.cwd())
    hermes_home = Path(getattr(args, "hermes_home", None) or get_hermes_home()).expanduser().resolve()
    sessions_dir = hermes_home / "sessions"
    archive_dir = hermes_home / "session_state_gc_archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidates: list[Path] = []
    if sessions_dir.is_dir():
        for pattern in ("*.tmp", "*.bak", "*.old", "*.partial", "*.scratch.json", "*.scratch.jsonl"):
            candidates.extend(path for path in sessions_dir.rglob(pattern) if path.is_file())

    archived = 0
    for path in candidates:
        if path.name in _PROTECTED_NAMES:
            continue
        try:
            path.resolve().relative_to(_state_dir())
            continue
        except ValueError:
            pass
        if any(part in {"design", "checkpoint"} for part in path.parts):
            continue
        if getattr(args, "dry_run", False):
            print(f"would archive: {path}")
            archived += 1
            continue
        rel = path.relative_to(sessions_dir)
        target = archive_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        archived += 1
    print(f"GC archived: {archived}")
    print(f"Canonical state preserved: PASS ({_state_dir()})")
    print(f"Canonical project state preserved: PASS ({_state_dir()})")
    return 0
