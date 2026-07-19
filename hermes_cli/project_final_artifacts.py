"""Deterministic project finalization report, manifest, and usage artifacts.

This module is deliberately a pure snapshot boundary: callers supply one
immutable ``ProjectFinalizationSnapshot`` and rendering never consults mutable
Kanban task or event state.  Publication writes three UTF-8 artifacts
atomically, then records only their immutable identity in HOF-002.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_home
from hermes_cli.project_finalization_contract import (
    MEMBERSHIP_KINDS,
    get_project_finalization,
    record_final_artifacts,
    validate_checker_verdict,
    validate_generation,
    validate_terminal_outcome,
)

MANIFEST_SCHEMA_VERSION = 1
ARTIFACT_FILENAMES = ("final-report.md", "manifest.json", "usage-summary.json")
_BOARD_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ProjectFinalizationSnapshot:
    """Copied value object used as the sole input to artifact rendering.

    Collection fields intentionally accept ordinary JSON-like values.  They
    are deep-copied at construction so later mutation of the caller's input
    cannot change the snapshot used for publication.
    """

    board_id: str
    root_task_id: str
    generation: int
    candidate_snapshot_version: str | None = None
    goal: str = ""
    title: str = ""
    root_goal: str | None = None
    root_title: str | None = None
    terminal_outcome: str | None = None
    terminal_evaluation: Any = None
    required_tasks: Any = field(default_factory=tuple)
    support_tasks: Any = field(default_factory=tuple)
    repair_tasks: Any = field(default_factory=tuple)
    checker_task_id: str | None = None
    checker_verdict: str | None = None
    checker_identity: Any = None
    runs: Any = field(default_factory=tuple)
    terminal_runs: Any = field(default_factory=tuple)
    commits: Any = field(default_factory=tuple)
    tests: Any = field(default_factory=tuple)
    test_results: Any = field(default_factory=tuple)
    evidence: Any = field(default_factory=tuple)
    evidence_paths: Any = field(default_factory=tuple)
    usage: Any = field(default_factory=dict)
    usage_aggregate: Any = None
    what_done: Any = field(default_factory=tuple)
    what_verified: Any = field(default_factory=tuple)
    what_failed: Any = field(default_factory=tuple)
    current_state: Any = None
    current_exact_state: Any = None
    blockers: Any = field(default_factory=tuple)
    blocker_data: Any = None
    next_step: Any = None
    delivery: Any = field(default_factory=dict)
    delivery_placeholder: Any = None
    cleanup: Any = field(default_factory=dict)
    cleanup_policy: Any = None
    limitations: Any = field(default_factory=tuple)
    remaining_limitations: Any = None
    failures: Any = field(default_factory=tuple)
    failure_data: Any = None
    created_at: Any = None

    def __post_init__(self) -> None:
        for name in (
            "required_tasks", "support_tasks", "repair_tasks", "runs", "terminal_runs", "commits",
            "tests", "test_results", "evidence", "evidence_paths", "usage", "usage_aggregate",
            "what_done", "what_verified", "what_failed", "blockers", "blocker_data", "delivery",
            "delivery_placeholder", "cleanup", "cleanup_policy", "limitations", "remaining_limitations",
            "failures", "failure_data", "terminal_evaluation", "current_state", "current_exact_state",
            "next_step", "checker_identity",
        ):
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))
        if self.root_goal is not None and not self.goal:
            object.__setattr__(self, "goal", self.root_goal)
        if self.root_title is not None and not self.title:
            object.__setattr__(self, "title", self.root_title)
        for primary, alias in (
            ("runs", "terminal_runs"), ("tests", "test_results"), ("evidence", "evidence_paths"),
            ("usage", "usage_aggregate"), ("current_state", "current_exact_state"),
            ("blockers", "blocker_data"), ("delivery", "delivery_placeholder"),
            ("cleanup", "cleanup_policy"), ("limitations", "remaining_limitations"),
            ("failures", "failure_data"),
        ):
            if getattr(self, primary) in (None, "", (), [], {}):
                alias_value = getattr(self, alias)
                if alias_value not in (None, "", (), [], {}):
                    object.__setattr__(self, primary, copy.deepcopy(alias_value))
        if self.checker_task_id is None:
            checker = _first_task_id(self.repair_tasks, "checker")
            if checker:
                object.__setattr__(self, "checker_task_id", checker)


@dataclass(frozen=True)
class ProjectFinalArtifacts:
    """Published artifact identities and canonical paths."""

    root_path: str
    report_path: str
    report_sha256: str
    manifest_path: str
    manifest_sha256: str
    usage_summary_path: str
    usage_summary_sha256: str
    usage_summary_json: str


def _first_task_id(tasks: Any, membership_kind: str) -> str | None:
    if isinstance(tasks, Mapping):
        tasks = (tasks,)
    if not isinstance(tasks, (list, tuple)):
        return None
    for task in tasks:
        if isinstance(task, Mapping) and task.get("membership_kind") == membership_kind:
            value = task.get("task_id")
            if isinstance(value, str) and value:
                return value
    return None


def _stable(value: Any) -> Any:
    """Return JSON-compatible data with deterministic object and array order."""
    if isinstance(value, Mapping):
        return {str(k): _stable(value[k]) for k in sorted(value, key=lambda k: str(k))}
    if isinstance(value, (list, tuple, set, frozenset)):
        values = [_stable(item) for item in value]
        return sorted(values, key=lambda item: _json(item))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False, indent=2, allow_nan=False) + "\n"


def _unknown(value: Any) -> Any:
    return "unknown" if value is None or value == "" else value


def _sanitize_usage(value: Any) -> dict[str, Any]:
    """Keep only aggregate fields; conversational/provider payloads are dropped."""
    source = value if isinstance(value, Mapping) else {}
    allowed = {
        "usage_status", "status", "record_count", "total_api_calls", "total_input_tokens",
        "total_output_tokens", "total_cache_read_tokens", "total_cache_write_tokens",
        "total_reasoning_tokens", "total_aux_input_tokens", "total_aux_output_tokens",
        "total_aux_cache_read_tokens", "total_aux_cache_write_tokens", "total_accepted_result_tokens",
        "total_cost_usd", "cost_status", "unknown", "groups", "by_group", "by_role",
    }
    result: dict[str, Any] = {}
    for key in sorted(allowed):
        if key in source:
            result[key] = _stable(source[key])
    groups = result.get("groups") or result.get("by_group") or []
    safe_groups = []
    group_keys = {
        "role", "profile", "provider", "model", "call_kind", "record_count", "api_calls",
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
        "aux_input_tokens", "aux_output_tokens", "aux_cache_read_tokens", "aux_cache_write_tokens",
        "accepted_result_tokens",
    }
    if isinstance(groups, (list, tuple)):
        for group in groups:
            if isinstance(group, Mapping):
                safe_groups.append({key: _unknown(group[key]) for key in sorted(group_keys) if key in group})
    result["groups"] = sorted(safe_groups, key=_json)
    result.pop("by_group", None)
    result.pop("by_role", None)
    if not result:
        result = {"usage_status": "unknown", "unknown": ["usage aggregate not supplied"]}
    elif "usage_status" not in result and "status" not in result:
        result["usage_status"] = "known"
    return _stable(result)


def _validate_snapshot(snapshot: ProjectFinalizationSnapshot) -> None:
    if not isinstance(snapshot, ProjectFinalizationSnapshot):
        raise TypeError("snapshot must be a ProjectFinalizationSnapshot")
    if not snapshot.board_id or not snapshot.root_task_id:
        raise ValueError("board_id and root_task_id are required")
    if "/" in snapshot.root_task_id or "\\" in snapshot.root_task_id or snapshot.root_task_id in {".", ".."}:
        raise ValueError("root_task_id must be a single safe path component")
    validate_generation(snapshot.generation)
    if snapshot.candidate_snapshot_version is not None:
        from hermes_cli.project_finalization_contract import validate_candidate_snapshot_version
        validate_candidate_snapshot_version(snapshot.candidate_snapshot_version)
    if snapshot.terminal_outcome is None:
        raise ValueError("terminal_outcome is required")
    validate_terminal_outcome(snapshot.terminal_outcome)
    if snapshot.checker_verdict is not None:
        validate_checker_verdict(snapshot.checker_verdict)
        if not snapshot.checker_task_id:
            raise ValueError("checker_task_id is required when a verdict is recorded")
    if snapshot.terminal_outcome == "COMPLETE" and snapshot.checker_verdict != "PASS":
        raise ValueError("COMPLETE artifacts require an authoritative PASS verdict")
    if snapshot.created_at is None or snapshot.created_at == "":
        raise ValueError("created_at is required")
    evidence = snapshot.evidence
    if not isinstance(evidence, (list, tuple)) or not evidence:
        raise ValueError("required evidence is missing")
    for item in evidence:
        if not isinstance(item, Mapping) or not item.get("path"):
            raise ValueError("evidence path is required")
        digest = item.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValueError("evidence sha256 must be a lowercase SHA-256 digest")
    for collection in (snapshot.required_tasks, snapshot.support_tasks, snapshot.repair_tasks):
        if isinstance(collection, Mapping):
            collection = (collection,)
        if collection is None:
            continue
        if not isinstance(collection, (list, tuple)):
            raise ValueError("task collections must be arrays")
        for item in collection:
            if isinstance(item, Mapping):
                kind = item.get("membership_kind")
                if kind is not None and kind not in MEMBERSHIP_KINDS:
                    raise ValueError(f"invalid membership_kind: {kind}")


def _safe_board_id(board_id: str) -> str:
    sanitized = _BOARD_COMPONENT_RE.sub("_", board_id).strip("._")
    if not sanitized or sanitized in {".", ".."}:
        raise ValueError("board_id cannot produce a safe artifact directory")
    return sanitized


def artifact_root(snapshot: ProjectFinalizationSnapshot, *, hermes_home: Path | None = None) -> Path:
    _validate_snapshot(snapshot)
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    root = home / "reports" / "project-finalization" / _safe_board_id(snapshot.board_id) / snapshot.root_task_id / f"generation-{snapshot.generation}"
    if snapshot.candidate_snapshot_version is not None:
        digest = snapshot.candidate_snapshot_version.removeprefix("sha256:")
        root = root / f"candidate-{digest}"
    return root


def _section_value(value: Any) -> str:
    if value is None or value == "" or value == [] or value == ():
        return "unknown"
    if isinstance(value, str):
        return value
    return _pretty_json(_stable(value)).rstrip("\n")


def aggregate_snapshot_usage(conn: Any, snapshot: ProjectFinalizationSnapshot) -> dict[str, Any]:
    """Aggregate usage for exactly the task IDs named by an immutable snapshot."""
    _validate_snapshot(snapshot)
    task_roles: dict[str, str] = {snapshot.root_task_id: "builder"}
    task_ids = {snapshot.root_task_id}
    role_by_kind = {"required": "builder", "support": "builder", "repair": "repair", "checker": "checker"}
    for collection in (snapshot.required_tasks, snapshot.support_tasks, snapshot.repair_tasks):
        items = (collection,) if isinstance(collection, Mapping) else collection
        if not isinstance(items, (list, tuple)):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            task_id = item.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                continue
            task_ids.add(task_id)
            role = item.get("role") or role_by_kind.get(item.get("membership_kind"), "unknown")
            task_roles[task_id] = str(role)
    if snapshot.checker_task_id:
        task_ids.add(snapshot.checker_task_id)
        task_roles[snapshot.checker_task_id] = "checker"
    from hermes_cli.kanban_usage_ledger import aggregate_project_usage
    return aggregate_project_usage(conn, board=snapshot.board_id, task_ids=task_ids, task_roles=task_roles)


def build_final_report(snapshot: ProjectFinalizationSnapshot, *, root_path: Path | None = None) -> bytes:
    _validate_snapshot(snapshot)
    sections = [
        ("Goal", snapshot.goal or snapshot.title or "unknown"),
        ("Terminal outcome", {"outcome": snapshot.terminal_outcome, "evaluation": snapshot.terminal_evaluation}),
        ("What was done", snapshot.what_done or snapshot.required_tasks),
        ("What was verified", snapshot.what_verified or snapshot.tests),
        ("What failed", snapshot.what_failed or snapshot.failures),
        ("Current exact state", snapshot.current_state or {"terminal_outcome": snapshot.terminal_outcome}),
        ("Remaining blockers", snapshot.blockers),
        ("Next actionable step", snapshot.next_step),
        ("Tasks and runs", {"required": snapshot.required_tasks, "support": snapshot.support_tasks, "repair": snapshot.repair_tasks, "runs": snapshot.runs}),
        ("Commits", snapshot.commits),
        ("Tests", snapshot.tests),
        ("Evidence", snapshot.evidence),
        ("Usage", _sanitize_usage(snapshot.usage)),
        ("Telegram delivery", snapshot.delivery),
        ("Cleanup schedule", snapshot.cleanup),
        ("Limitations", snapshot.limitations),
    ]
    lines = []
    for title, value in sections:
        lines.extend((f"# {title}", _section_value(value), ""))
    return ("\n".join(lines)).encode("utf-8")


def _manifest(snapshot: ProjectFinalizationSnapshot, root_path: Path, report_path: Path, report_hash: str, usage_path: Path, usage_hash: str) -> dict[str, Any]:
    usage = _sanitize_usage(snapshot.usage)
    return {
        "board_id": snapshot.board_id,
        "root_task_id": snapshot.root_task_id,
        "generation": snapshot.generation,
        "candidate_snapshot_version": snapshot.candidate_snapshot_version,
        "terminal_outcome": snapshot.terminal_outcome,
        "checker_task_id": snapshot.checker_task_id,
        "checker_verdict": snapshot.checker_verdict,
        "checker_identity": _stable(snapshot.checker_identity),
        "required_tasks": _stable(snapshot.required_tasks),
        "support_tasks": _stable(snapshot.support_tasks),
        "repair_tasks": _stable(snapshot.repair_tasks),
        "runs": _stable(snapshot.runs),
        "commits": _stable(snapshot.commits),
        "tests": _stable(snapshot.tests),
        "evidence": _stable(snapshot.evidence),
        "usage": usage,
        "report_path": str(report_path),
        "report_sha256": report_hash,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": snapshot.created_at,
        "usage_summary_path": str(usage_path),
        "usage_summary_sha256": usage_hash,
        "root_path": str(root_path),
    }


def build_project_artifacts(snapshot: ProjectFinalizationSnapshot, *, hermes_home: Path | None = None) -> tuple[bytes, bytes, bytes, Path]:
    root = artifact_root(snapshot, hermes_home=hermes_home)
    report_path = root / "final-report.md"
    manifest_path = root / "manifest.json"
    usage_path = root / "usage-summary.json"
    usage_obj = _sanitize_usage(snapshot.usage)
    usage_bytes = _pretty_json(usage_obj).encode("utf-8")
    usage_hash = hashlib.sha256(usage_bytes).hexdigest()
    report_bytes = build_final_report(snapshot, root_path=root)
    report_hash = hashlib.sha256(report_bytes).hexdigest()
    manifest_bytes = _pretty_json(_manifest(snapshot, root, report_path, report_hash, usage_path, usage_hash)).encode("utf-8")
    return report_bytes, manifest_bytes, usage_bytes, root


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _result(root: Path, report: bytes, manifest: bytes, usage: bytes) -> ProjectFinalArtifacts:
    report_path = root / "final-report.md"
    manifest_path = root / "manifest.json"
    usage_path = root / "usage-summary.json"
    return ProjectFinalArtifacts(
        root_path=str(root), report_path=str(report_path), report_sha256=hashlib.sha256(report).hexdigest(),
        manifest_path=str(manifest_path), manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        usage_summary_path=str(usage_path), usage_summary_sha256=hashlib.sha256(usage).hexdigest(),
        usage_summary_json=usage.decode("utf-8").rstrip("\n"),
    )


def write_project_final_artifacts(snapshot: ProjectFinalizationSnapshot, *, hermes_home: Path | None = None) -> ProjectFinalArtifacts:
    """Publish the three files without touching the HOF-002 database row."""
    report, manifest, usage, root = build_project_artifacts(snapshot, hermes_home=hermes_home)
    root.mkdir(parents=True, exist_ok=True)
    paths = [root / name for name in ARTIFACT_FILENAMES]
    data_by_path = dict(zip(paths, (report, manifest, usage)))
    existing = [path for path in paths if path.exists()]
    if any(path.read_bytes() != data_by_path[path] for path in existing):
        raise ValueError("conflicting immutable artifact set already exists")
    if len(existing) == len(paths):
        return _result(root, report, manifest, usage)
    replaced: list[Path] = []
    try:
        for path in paths:
            if path in existing:
                continue
            data = data_by_path[path]
            _atomic_write(path, data)
            replaced.append(path)
    except Exception:
        for path in replaced:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    return _result(root, report, manifest, usage)


def publish_project_final_artifacts(conn: Any, snapshot: ProjectFinalizationSnapshot, *, hermes_home: Path | None = None) -> ProjectFinalArtifacts:
    """Atomically publish artifacts and persist their frozen HOF-002 identity."""
    report, manifest, usage, root = build_project_artifacts(snapshot, hermes_home=hermes_home)
    expected = _result(root, report, manifest, usage)
    artifact_paths = (
        Path(expected.report_path), Path(expected.manifest_path), Path(expected.usage_summary_path),
    )
    artifacts_existed_before = any(path.exists() for path in artifact_paths)
    existing_row = get_project_finalization(conn, board_id=snapshot.board_id, root_task_id=snapshot.root_task_id, generation=snapshot.generation)
    if existing_row is None:
        raise ValueError("project finalization does not exist")
    values = (
        existing_row.final_report_path, existing_row.final_report_sha256,
        existing_row.manifest_path, existing_row.manifest_sha256,
    )
    expected_values = (
        expected.report_path, expected.report_sha256,
        expected.manifest_path, expected.manifest_sha256,
    )
    if any(value is not None for value in values) and values != expected_values:
        raise ValueError("conflicting immutable artifact identity")
    if (
        existing_row.usage_summary_json is not None
        and existing_row.usage_summary_json != expected.usage_summary_json
    ):
        raise ValueError("conflicting usage summary")
    try:
        published = write_project_final_artifacts(snapshot, hermes_home=hermes_home)
        recorded = record_final_artifacts(
            conn, board_id=snapshot.board_id, root_task_id=snapshot.root_task_id, generation=snapshot.generation,
            report_path=published.report_path, report_sha256=published.report_sha256,
            manifest_path=published.manifest_path, manifest_sha256=published.manifest_sha256,
            usage_summary_json=published.usage_summary_json,
            candidate_snapshot_version=snapshot.candidate_snapshot_version,
        )
        if (
            recorded.final_report_path != published.report_path
            or recorded.final_report_sha256 != published.report_sha256
            or recorded.manifest_path != published.manifest_path
            or recorded.manifest_sha256 != published.manifest_sha256
            or recorded.usage_summary_json != published.usage_summary_json
            or recorded.artifact_candidate_snapshot_version
            != snapshot.candidate_snapshot_version
        ):
            raise ValueError("persisted artifact identity verification failed")
        return published
    except Exception:
        if not artifacts_existed_before:
            for path in artifact_paths:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        raise


# Explicit aliases used by callers that distinguish construction from publication.
build_final_artifacts = build_project_artifacts
persist_final_artifacts = publish_project_final_artifacts

__all__ = [
    "ARTIFACT_FILENAMES", "MANIFEST_SCHEMA_VERSION", "ProjectFinalArtifacts",
    "ProjectFinalizationSnapshot", "artifact_root", "build_final_report",
    "aggregate_snapshot_usage", "build_project_artifacts", "build_final_artifacts", "write_project_final_artifacts",
    "publish_project_final_artifacts", "persist_final_artifacts",
]
