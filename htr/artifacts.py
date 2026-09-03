"""HTR artifact manifest helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from htr import paths

def _check_artifact_path_safe(path_str: str, attempt_dir: Path) -> str:
    """Validate that *path_str* resolves inside *attempt_dir* (no traversal)."""
    resolved = (attempt_dir / path_str).resolve()
    attempt_resolved = attempt_dir.resolve()
    if not str(resolved).startswith(str(attempt_resolved) + "/"):
        raise ValueError(f"artifact path escapes attempt workspace: {path_str!r}")
    return str(resolved)

from htr.execution_lock import begin_run_write, run_mutation_boundary
from htr.io import atomic_write_json, read_json
from htr.schemas import validate as validate_schema


class ArtifactConflict(Exception):
    """Raised when an artifact entry conflicts with an existing manifest entry."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_entry_fingerprint(entry: dict[str, Any]) -> str:
    return json.dumps(
        {
            "path": entry.get("path"),
            "kind": entry.get("kind"),
            "sha256": entry.get("sha256"),
            "size_bytes": entry.get("size_bytes"),
            "metadata": entry.get("metadata", {}),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _normalize_manifest(
    manifest: dict[str, Any],
    *,
    run_id: str,
    task_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    normalized = dict(manifest)
    normalized.setdefault("schema_version", "1")
    normalized.setdefault("run_id", run_id)
    normalized.setdefault("task_id", task_id)
    normalized["attempt_id"] = attempt_id
    normalized.setdefault("artifacts", [])
    validate_schema(normalized, "artifact_manifest")
    return normalized


def read_artifact_manifest(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Read and validate the artifact manifest for *attempt_id*."""
    target = paths.artifact_manifest_path(run_id, task_id, attempt_id, base_dir)
    manifest = read_json(target)
    return _normalize_manifest(
        manifest,
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
    )


@run_mutation_boundary
def write_artifact_manifest(
    run_id: str,
    task_id: str,
    attempt_id: str,
    manifest: dict[str, Any],
    base_dir: Path | None = None,
) -> Path:
    """Atomically write *manifest* to the attempt workspace."""
    normalized = _normalize_manifest(
        manifest,
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
    )
    target = paths.artifact_manifest_path(run_id, task_id, attempt_id, base_dir)
    begin_run_write()
    atomic_write_json(target, normalized)
    return target


def list_artifacts(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Return artifact entries from the attempt manifest."""
    manifest = read_artifact_manifest(run_id, task_id, attempt_id, base_dir)
    return list(manifest["artifacts"])


@run_mutation_boundary
def add_artifact(
    run_id: str,
    task_id: str,
    attempt_id: str,
    *,
    path: str,
    kind: str,
    sha256: str | None = None,
    size_bytes: int | None = None,
    metadata: dict[str, Any] | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Append an artifact entry to the manifest when not already present."""
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict")

    manifest = read_artifact_manifest(run_id, task_id, attempt_id, base_dir)
    artifacts = list(manifest["artifacts"])

    candidate = {
        "path": path,
        "kind": kind,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "created_at": _utc_now_iso(),
        "metadata": metadata,
    }
    validate_schema(candidate, "artifact_entry")

    for index, existing in enumerate(artifacts):
        if existing.get("path") == path and existing.get("kind") == kind:
            if _artifact_entry_fingerprint(existing) == _artifact_entry_fingerprint(candidate):
                return existing
            raise ArtifactConflict(
                f"artifact already exists for path {path!r} and kind {kind!r} "
                "with conflicting metadata/checksum/size"
            )

    artifacts.append(candidate)
    manifest["artifacts"] = artifacts
    begin_run_write()
    write_artifact_manifest(run_id, task_id, attempt_id, manifest, base_dir)
    return candidate
