"""Pure-Python foundation for workspace documents.

Workspace documents are inert safekeeping artifacts: drafted skill templates,
memory notes, workspace instructions, prompt templates, runbooks, and generic
Markdown notes that live under ``<workspace>/.hermes/docs/``. This module only
covers schema validation and safe path resolution. It deliberately does not
render MDX, expose tools/endpoints, or hook documents into skills, memory, or
prompt construction.

Workspace identity decision
---------------------------
For this first slice, a workspace identity is the resolved absolute workspace
root path. That is deterministic, stable within a checkout/worktree, and avoids
string-prefix mistakes around ``~/.hermes``. A normal repository may itself live
under ``~/.hermes`` (Hermes local development commonly does); only the existing
``agent.file_safety`` classifiers decide whether a path is profile state or a
sandbox/container mirror.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from agent.file_safety import (
    classify_container_mirror_target,
    classify_cross_profile_target,
    classify_sandbox_mirror_target,
    get_container_mirror_warning,
    get_cross_profile_warning,
    get_sandbox_mirror_warning,
)
from agent.skill_utils import parse_frontmatter
from hermes_constants import get_hermes_home

StrPath = Union[str, "os.PathLike[str]"]

WORKSPACE_DOCS_RELPATH = Path(".hermes") / "docs"
WORKSPACE_DOC_FRONTMATTER_TYPE = "hermes-workspace-document"


class WorkspaceDocError(Exception):
    """Base class for workspace document validation/safety failures."""


class WorkspaceDocValidationError(WorkspaceDocError):
    """Frontmatter is missing or fails schema validation."""


class WorkspaceDocPathError(WorkspaceDocError):
    """Requested document path escapes the workspace/docs boundary."""


class WorkspaceDocSafetyError(WorkspaceDocError):
    """Resolved path collides with a Hermes profile/sandbox safety guard."""

    def __init__(self, message: str, *, classifier: str):
        super().__init__(message)
        self.classifier = classifier


class WorkspaceDocType(str, Enum):
    SKILL_TEMPLATE = "skill-template"
    MEMORY_NOTE = "memory-note"
    WORKSPACE_INSTRUCTIONS = "workspace-instructions"
    PROMPT_TEMPLATE = "prompt-template"
    RUNBOOK = "runbook"
    GENERIC_MD = "generic-md"


class WorkspaceDocStatus(str, Enum):
    DRAFT = "draft"
    READY = "ready"
    ARCHIVED = "archived"


class WorkspaceDocApplyState(str, Enum):
    UNAPPLIED = "unapplied"
    EXPORTED = "exported"
    IMPORTED = "imported"


@dataclass(frozen=True)
class WorkspaceDocFrontmatter:
    """Validated frontmatter for an inert workspace document."""

    doc_type: WorkspaceDocType
    title: str
    workspace_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    status: WorkspaceDocStatus = WorkspaceDocStatus.DRAFT
    apply_state: WorkspaceDocApplyState = WorkspaceDocApplyState.UNAPPLIED
    description: Optional[str] = None
    tags: tuple[str, ...] = ()


def _require_string(frontmatter: dict[str, Any], field: str) -> str:
    value = frontmatter.get(field)
    if not isinstance(value, str) or not value.strip():
        raise WorkspaceDocValidationError(f"missing required field: {field}")
    return value.strip()


def _optional_string(frontmatter: dict[str, Any], field: str) -> Optional[str]:
    value = frontmatter.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise WorkspaceDocValidationError(f"{field} must be a string")
    return value.strip() or None


def _enum_value(enum_cls: type[Enum], raw: Any, field: str, default: Any = None) -> Any:
    if raw is None:
        return default
    try:
        return enum_cls(str(raw))
    except ValueError as exc:
        valid = ", ".join(member.value for member in enum_cls)  # type: ignore[attr-defined]
        raise WorkspaceDocValidationError(
            f"invalid {field} {raw!r}; must be one of: {valid}"
        ) from exc


def validate_workspace_doc_frontmatter(frontmatter: dict[str, Any]) -> WorkspaceDocFrontmatter:
    """Validate workspace-document YAML frontmatter.

    Required fields are ``type: hermes-workspace-document``, ``doc_type``, and
    ``title``. ``status`` defaults to ``draft`` and ``apply_state`` defaults to
    ``unapplied`` so newly-created documents are explicitly inert until a later
    import/export flow changes them.
    """

    if not isinstance(frontmatter, dict):
        raise WorkspaceDocValidationError("frontmatter must be a mapping")

    document_type = frontmatter.get("type")
    if document_type != WORKSPACE_DOC_FRONTMATTER_TYPE:
        raise WorkspaceDocValidationError(
            f"type must be {WORKSPACE_DOC_FRONTMATTER_TYPE!r}"
        )

    doc_type = _enum_value(
        WorkspaceDocType,
        frontmatter.get("doc_type"),
        "doc_type",
    )
    if doc_type is None:
        raise WorkspaceDocValidationError("missing required field: doc_type")

    tags_raw = frontmatter.get("tags") or []
    if not isinstance(tags_raw, list) or not all(isinstance(tag, str) for tag in tags_raw):
        raise WorkspaceDocValidationError("tags must be a list of strings")

    return WorkspaceDocFrontmatter(
        doc_type=doc_type,
        title=_require_string(frontmatter, "title"),
        workspace_id=_optional_string(frontmatter, "workspace_id"),
        created_at=_optional_string(frontmatter, "created_at"),
        updated_at=_optional_string(frontmatter, "updated_at"),
        status=_enum_value(
            WorkspaceDocStatus,
            frontmatter.get("status"),
            "status",
            WorkspaceDocStatus.DRAFT,
        ),
        apply_state=_enum_value(
            WorkspaceDocApplyState,
            frontmatter.get("apply_state"),
            "apply_state",
            WorkspaceDocApplyState.UNAPPLIED,
        ),
        description=_optional_string(frontmatter, "description"),
        tags=tuple(tags_raw),
    )


def parse_workspace_doc(content: str) -> tuple[WorkspaceDocFrontmatter, str]:
    """Parse and validate a workspace document from Markdown text."""

    raw_frontmatter, body = parse_frontmatter(content)
    if not raw_frontmatter:
        raise WorkspaceDocValidationError("workspace document is missing YAML frontmatter")
    return validate_workspace_doc_frontmatter(raw_frontmatter), body


@dataclass(frozen=True)
class WorkspaceIdentity:
    """A deterministic identity for a workspace or profile fallback."""

    kind: str
    identity: str
    root: Path


def resolve_workspace_identity(workspace_root: Optional[StrPath] = None) -> WorkspaceIdentity:
    """Resolve the current workspace identity.

    The identity is a resolved absolute root path. When no workspace root is
    available, the active profile's resolved ``HERMES_HOME`` is used as an
    explicit ``profile-fallback`` identity.
    """

    if workspace_root is not None:
        root = Path(os.path.expanduser(str(workspace_root))).resolve()
        return WorkspaceIdentity(kind="workspace", identity=str(root), root=root)
    home = get_hermes_home().resolve()
    return WorkspaceIdentity(kind="profile-fallback", identity=str(home), root=home)


def workspace_doc_frontmatter_defaults(workspace_root: StrPath) -> dict[str, str]:
    """Return default inert frontmatter values for a new workspace document."""

    return {
        "type": WORKSPACE_DOC_FRONTMATTER_TYPE,
        "workspace_id": resolve_workspace_identity(workspace_root).identity,
        "status": WorkspaceDocStatus.DRAFT.value,
        "apply_state": WorkspaceDocApplyState.UNAPPLIED.value,
    }


def _resolve_workspace_root(workspace_root: StrPath) -> Path:
    return Path(os.path.expanduser(str(workspace_root))).resolve()


def get_workspace_docs_root(workspace_root: StrPath) -> Path:
    """Return the resolved ``<workspace_root>/.hermes/docs`` root.

    If ``.hermes`` or ``docs`` is a symlink escaping the workspace, this raises
    ``WorkspaceDocPathError`` instead of silently storing safekeeping data
    elsewhere.
    """

    root = _resolve_workspace_root(workspace_root)
    docs_root = (root / WORKSPACE_DOCS_RELPATH).resolve()
    try:
        docs_root.relative_to(root)
    except ValueError as exc:
        raise WorkspaceDocPathError(
            f"workspace docs root {docs_root} escapes workspace root {root}"
        ) from exc
    return docs_root


def _raise_if_file_safety_guarded(resolved: Path, *, mirror_prefix: Optional[str]) -> None:
    resolved_str = str(resolved)

    if classify_cross_profile_target(resolved_str) is not None:
        raise WorkspaceDocSafetyError(
            get_cross_profile_warning(resolved_str) or "cross-profile target",
            classifier="cross_profile",
        )
    if classify_sandbox_mirror_target(resolved_str) is not None:
        raise WorkspaceDocSafetyError(
            get_sandbox_mirror_warning(resolved_str) or "sandbox-mirror target",
            classifier="sandbox_mirror",
        )
    if mirror_prefix and classify_container_mirror_target(resolved_str, mirror_prefix) is not None:
        raise WorkspaceDocSafetyError(
            get_container_mirror_warning(resolved_str, mirror_prefix) or "container-mirror target",
            classifier="container_mirror",
        )


def resolve_workspace_doc_path(
    workspace_root: StrPath,
    relative_path: str = "",
    *,
    mirror_prefix: Optional[str] = None,
) -> Path:
    """Resolve a document path under ``<workspace_root>/.hermes/docs`` safely."""

    if relative_path and os.path.isabs(relative_path):
        raise WorkspaceDocPathError(f"absolute path not allowed: {relative_path!r}")

    docs_root = get_workspace_docs_root(workspace_root)
    _raise_if_file_safety_guarded(docs_root, mirror_prefix=mirror_prefix)

    resolved = (docs_root / relative_path).resolve() if relative_path else docs_root
    try:
        resolved.relative_to(docs_root)
    except ValueError as exc:
        raise WorkspaceDocPathError(
            f"{relative_path!r} escapes the workspace docs root {docs_root}"
        ) from exc

    _raise_if_file_safety_guarded(resolved, mirror_prefix=mirror_prefix)
    return resolved


def is_workspace_docs_path(path: StrPath) -> bool:
    """Return True when *path* is under a ``.hermes/docs`` segment."""

    resolved = Path(os.path.expanduser(str(path))).resolve()
    parts = resolved.parts
    return any(
        parts[index] == ".hermes" and parts[index + 1] == "docs"
        for index in range(len(parts) - 1)
    )


def workspace_docs_are_inert() -> bool:
    """Document the slice-1 invariant that workspace docs are safekeeping data."""

    return True


# ---------------------------------------------------------------------------
# Slice 2: backend CRUD (list/read/write/archive). Still no MDX rendering,
# import/export, model tool, or profile index — callers (e.g. the REST layer
# in hermes_cli.web_server) own transport concerns; this module keeps owning
# schema validation, path safety, and atomic writes.
# ---------------------------------------------------------------------------

WORKSPACE_DOC_MAX_BYTES = 2 * 1024 * 1024


def frontmatter_to_mapping(frontmatter: WorkspaceDocFrontmatter) -> dict[str, Any]:
    """Convert validated frontmatter back into a plain YAML-serializable mapping."""

    mapping: dict[str, Any] = {
        "type": WORKSPACE_DOC_FRONTMATTER_TYPE,
        "doc_type": frontmatter.doc_type.value,
        "title": frontmatter.title,
        "status": frontmatter.status.value,
        "apply_state": frontmatter.apply_state.value,
    }
    if frontmatter.workspace_id is not None:
        mapping["workspace_id"] = frontmatter.workspace_id
    if frontmatter.created_at is not None:
        mapping["created_at"] = frontmatter.created_at
    if frontmatter.updated_at is not None:
        mapping["updated_at"] = frontmatter.updated_at
    if frontmatter.description is not None:
        mapping["description"] = frontmatter.description
    if frontmatter.tags:
        mapping["tags"] = list(frontmatter.tags)
    return mapping


def render_workspace_doc(frontmatter: dict[str, Any], body: str) -> str:
    """Serialize frontmatter + Markdown body back into workspace-document text."""

    import yaml

    fm_yaml = yaml.safe_dump(dict(frontmatter), sort_keys=False, allow_unicode=True).strip()
    body_text = (body or "").strip("\n")
    if body_text:
        return f"---\n{fm_yaml}\n---\n\n{body_text}\n"
    return f"---\n{fm_yaml}\n---\n"


def _atomic_write_workspace_doc(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.hermes-tmp-{os.getpid()}")
    renamed = False
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
        renamed = True
    finally:
        if not renamed:
            tmp.unlink(missing_ok=True)


def write_workspace_doc(
    workspace_root: StrPath,
    relative_path: str,
    *,
    frontmatter: dict[str, Any],
    body: str,
    mirror_prefix: Optional[str] = None,
) -> tuple[Path, WorkspaceDocFrontmatter]:
    """Validate and atomically create/update a workspace document.

    Callers own merge-with-existing-frontmatter and timestamp policy; this
    validates the final frontmatter mapping, resolves+guards the path, and
    performs the atomic write. Directories under ``.hermes/docs`` are created
    as needed; nothing outside that root is ever touched.
    """

    if not relative_path or not relative_path.lower().endswith(".md"):
        raise WorkspaceDocPathError(f"document path must end with .md: {relative_path!r}")

    encoded_body = (body or "").encode("utf-8")
    if len(encoded_body) > WORKSPACE_DOC_MAX_BYTES:
        raise WorkspaceDocValidationError("document body exceeds the maximum allowed size")
    if b"\0" in encoded_body:
        raise WorkspaceDocValidationError("document body must be text, not binary")

    validated = validate_workspace_doc_frontmatter(frontmatter)
    target = resolve_workspace_doc_path(workspace_root, relative_path, mirror_prefix=mirror_prefix)

    text = render_workspace_doc(frontmatter_to_mapping(validated), body)
    _atomic_write_workspace_doc(target, text)
    return target, validated


def archive_workspace_doc(
    workspace_root: StrPath,
    relative_path: str,
    *,
    mirror_prefix: Optional[str] = None,
) -> tuple[Path, WorkspaceDocFrontmatter]:
    """Mark an existing workspace document archived in place, idempotently.

    Reversible and explicit: this only flips ``status`` to ``archived`` in the
    document's own frontmatter. The file stays at the same path so a later
    un-archive is just another status update.
    """

    target = resolve_workspace_doc_path(workspace_root, relative_path, mirror_prefix=mirror_prefix)
    if not target.is_file():
        raise FileNotFoundError(str(target))

    frontmatter, body = parse_workspace_doc(target.read_text(encoding="utf-8"))
    if frontmatter.status is WorkspaceDocStatus.ARCHIVED:
        return target, frontmatter

    archived = replace(frontmatter, status=WorkspaceDocStatus.ARCHIVED)
    text = render_workspace_doc(frontmatter_to_mapping(archived), body)
    _atomic_write_workspace_doc(target, text)
    return target, archived


@dataclass(frozen=True)
class WorkspaceDocSummary:
    """One entry in a workspace-docs listing."""

    relative_path: str
    valid: bool
    frontmatter: Optional[WorkspaceDocFrontmatter] = None
    error: Optional[str] = None


def list_workspace_docs(
    workspace_root: StrPath,
    *,
    mirror_prefix: Optional[str] = None,
) -> list[WorkspaceDocSummary]:
    """List Markdown documents under ``<workspace_root>/.hermes/docs``, sorted by path."""

    docs_root = resolve_workspace_doc_path(workspace_root, mirror_prefix=mirror_prefix)
    if not docs_root.is_dir():
        return []

    paths = sorted(
        (path for path in docs_root.rglob("*.md") if path.is_file()),
        key=lambda path: path.relative_to(docs_root).as_posix(),
    )

    summaries: list[WorkspaceDocSummary] = []
    for path in paths:
        relative_path = path.relative_to(docs_root).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
            frontmatter, _body = parse_workspace_doc(content)
        except (WorkspaceDocValidationError, OSError, UnicodeDecodeError) as exc:
            summaries.append(WorkspaceDocSummary(relative_path=relative_path, valid=False, error=str(exc)))
            continue
        summaries.append(WorkspaceDocSummary(relative_path=relative_path, valid=True, frontmatter=frontmatter))
    return summaries
