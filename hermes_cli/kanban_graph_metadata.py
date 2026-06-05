"""Dry-run lean graph_metadata sidecar parser and validator for Kanban.

This module is intentionally helper-only.  Nothing in the dispatcher, gateway,
or Kanban persistence path imports it automatically; callers must opt in when
they want to inspect candidate graph-shaped handoff metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import yaml

_REQUIRED_FIELDS: tuple[str, ...] = (
    "workflow_id",
    "node_id",
    "node_type",
    "owner_profile",
    "depends_on",
    "objective",
    "deliverable",
    "acceptance_criteria",
    "validation_required",
    "mutation_level",
    "source_of_truth.state",
)
_ARTIFACT_FIELDS: tuple[str, ...] = ("artifact_ref", "save_location")
_MIRROR_CONTENT_FIELDS: tuple[str, ...] = (
    "kanban_mirror_comment",
    "mirror_comment",
    "evidence_comment",
    "kanban_mirror_content",
    "mirror_content",
    "evidence_content",
    "kanban_mirror_excerpt",
    "mirror_excerpt",
    "evidence_excerpt",
)
_SOURCE_OF_TRUTH_STATES: frozenset[str] = frozenset(
    {"candidate", "validation_passed", "promoted", "rejected"}
)
_SCRATCH_PATH_RE = re.compile(
    r"/(?:\.hermes/)?kanban/(?:boards/[^/]+/)?workspaces/[^/]+(?:/|$)"
)
_FENCED_YAML_RE = re.compile(
    r"```(?:ya?ml)?\s*\n(?P<body>.*?)\n```",
    re.IGNORECASE | re.DOTALL,
)
_MARKDOWN_SECTION_RE = re.compile(
    r"^#{1,6}\s+graph_metadata\s*$\n(?P<body>.*?)(?=^#{1,6}\s+|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
_VALIDATION_READY_MIRROR_RE = re.compile(
    r"\bvalidation[- ]ready mirror\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GraphMetadataValidation:
    """Validation result for an optional Phase 1 graph_metadata sidecar."""

    valid: bool
    legacy: bool = False
    missing_fields: list[str] | None = None
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "missing_fields", list(self.missing_fields or []))
        object.__setattr__(self, "errors", list(self.errors or []))


@dataclass(frozen=True)
class DurableEvidence:
    """Durable-evidence classification for a graph_metadata artifact pointer."""

    state: str
    reason: str


def extract_graph_metadata_from_text(text: str | None) -> dict[str, Any] | None:
    """Return the first ``graph_metadata`` block found in markdown/YAML text.

    Missing or malformed metadata is treated as legacy/no-op and returns None;
    this helper is for dry-run inspection, not enforcement of legacy packets.
    """

    if not text:
        return None

    for section in _MARKDOWN_SECTION_RE.finditer(text):
        parsed = _safe_yaml(section.group("body"))
        if isinstance(parsed, Mapping):
            return dict(parsed.get("graph_metadata", parsed))

    candidates: list[str] = [match.group("body") for match in _FENCED_YAML_RE.finditer(text)]
    candidates.append(text)
    for candidate in candidates:
        parsed = _safe_yaml(candidate)
        if isinstance(parsed, Mapping):
            block = parsed.get("graph_metadata")
            if isinstance(block, Mapping):
                return dict(block)
    return None


def extract_graph_metadata_from_completion_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return ``metadata.graph_metadata`` from completion metadata, if present."""

    if not isinstance(metadata, Mapping):
        return None
    block = metadata.get("graph_metadata")
    if isinstance(block, Mapping):
        return dict(block)
    return None


def validate_graph_metadata(
    metadata: Mapping[str, Any] | None,
) -> GraphMetadataValidation:
    """Validate the lean Phase 1 required field set.

    ``None`` means a legacy task/handoff with no sidecar metadata and is valid
    no-op. Optional/advisory fields are allowed but never required here.
    """

    if metadata is None:
        return GraphMetadataValidation(valid=True, legacy=True)
    if not isinstance(metadata, Mapping):
        return GraphMetadataValidation(valid=False, errors=["graph_metadata must be an object"])

    missing = [field for field in _REQUIRED_FIELDS if _missing(metadata, field)]
    if all(_missing(metadata, field) for field in _ARTIFACT_FIELDS):
        missing.append("artifact_ref or save_location")

    errors: list[str] = []
    state = _nested_get(metadata, "source_of_truth.state")
    if state is not None and state not in _SOURCE_OF_TRUTH_STATES:
        errors.append("source_of_truth.state must be one of candidate, validation_passed, promoted, rejected")

    return GraphMetadataValidation(
        valid=not missing and not errors,
        legacy=False,
        missing_fields=missing,
        errors=errors,
    )


def classify_durable_evidence(
    metadata: Mapping[str, Any] | None,
    *,
    comments: Sequence[str] | None = None,
    workspace_kind: str | None = None,
) -> DurableEvidence:
    """Classify artifact evidence for a Phase 1 graph_metadata sidecar.

    States are exactly: ``durable``, ``mirrored``, ``insufficient``, and
    ``not_applicable``. Scratch workspace paths are never durable unless the
    caller explicitly identifies a shared ``workspace_kind='dir'`` surface.
    """

    if metadata is None:
        return DurableEvidence("not_applicable", "legacy task has no graph_metadata")

    artifact_ref = metadata.get("artifact_ref")
    save_location = metadata.get("save_location")
    artifact = _first_nonempty_string(artifact_ref, save_location)

    if not artifact:
        if _looks_non_artifact_node(metadata):
            return DurableEvidence("not_applicable", "non-artifact node has no artifact pointer")
        if _has_validation_ready_mirror(metadata, comments):
            return DurableEvidence("mirrored", "validation-ready mirror evidence is present")
        return DurableEvidence("insufficient", "artifact-producing node has no durable artifact pointer or mirror")

    if _is_scratch_workspace_path(artifact, workspace_kind=workspace_kind):
        if _has_validation_ready_mirror(metadata, comments):
            return DurableEvidence("mirrored", "scratch artifact path is paired with validation-ready mirror evidence")
        return DurableEvidence("insufficient", "scratch artifact path has no validation-ready mirror evidence")

    if _is_durable_location(artifact, workspace_kind=workspace_kind):
        return DurableEvidence("durable", "artifact pointer uses a durable location class")

    if _has_validation_ready_mirror(metadata, comments):
        return DurableEvidence("mirrored", "artifact pointer is paired with validation-ready mirror evidence")
    return DurableEvidence("insufficient", "artifact pointer is not a recognized durable location")


def _safe_yaml(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return None


def _missing(data: Mapping[str, Any], dotted_key: str) -> bool:
    value = _nested_get(data, dotted_key)
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) == 0:
        return True
    return False


def _nested_get(data: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _first_nonempty_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _is_scratch_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return bool(_SCRATCH_PATH_RE.search(normalized))


def _is_scratch_workspace_path(path: str, *, workspace_kind: str | None = None) -> bool:
    normalized = path.replace("\\", "/")
    if workspace_kind == "dir":
        return False
    if _is_scratch_path(normalized):
        return True
    if workspace_kind == "scratch" and _is_local_absolute_path(normalized):
        return "/durable-artifacts/" not in normalized
    return False


def _is_local_absolute_path(path: str) -> bool:
    return path.startswith("/") or bool(re.match(r"^[A-Za-z]:/", path))


def _is_durable_location(path: str, *, workspace_kind: str | None = None) -> bool:
    normalized = path.replace("\\", "/")
    lower = normalized.lower()
    if workspace_kind == "dir":
        return True
    if "/durable-artifacts/" in normalized:
        return True
    if lower.startswith(("gdrive://", "docs://")):
        return True
    if lower.startswith("http://") or lower.startswith("https://"):
        return any(host in lower for host in ("drive.google.com", "docs.google.com"))
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", path):
        return True
    if _is_scratch_path(normalized):
        return False
    if workspace_kind == "scratch" and _is_local_absolute_path(normalized):
        return False
    return normalized.startswith("/") and "/kanban/workspaces/" not in normalized


def _has_validation_ready_mirror(
    metadata: Mapping[str, Any],
    comments: Sequence[str] | None,
) -> bool:
    has_labeled_metadata_content = any(
        _contains_validation_ready_mirror_label(metadata.get(field)) for field in _MIRROR_CONTENT_FIELDS
    )
    has_labeled_comment = any(
        isinstance(comment, str) and _VALIDATION_READY_MIRROR_RE.search(comment)
        for comment in comments or ()
    )
    return has_labeled_metadata_content or has_labeled_comment


def _contains_validation_ready_mirror_label(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_VALIDATION_READY_MIRROR_RE.search(value))
    if isinstance(value, Mapping):
        return any(_contains_validation_ready_mirror_label(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_validation_ready_mirror_label(item) for item in value)
    return False


def _looks_non_artifact_node(metadata: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(metadata.get(key, ""))
        for key in ("node_type", "deliverable", "objective", "mutation_level")
    ).lower()
    return any(
        marker in text
        for marker in (
            "validation verdict",
            "validator",
            "routing decision",
            "operational block",
            "no artifact",
            "not_applicable",
        )
    )
