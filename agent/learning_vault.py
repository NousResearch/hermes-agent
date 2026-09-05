"""Bounded, read-only adapter for the shared agent-learning Vault catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from agent.skill_utils import parse_frontmatter


MAX_NOTE_BYTES = 16 * 1024
MAX_SUMMARY_CHARS = 1200
MAX_CATALOG_FILES = 5000
_AUTHORITIES = {"source-index", "narrative-only"}
_AGENTS = {"claude", "codex", "hermes", "shared"}
_KINDS = {"memory", "working-preference", "skill-reference"}


@dataclass(frozen=True)
class VaultLearningNode:
    record_id: str
    label: str
    kind: str
    origin_agent: str
    area: str
    status: str
    timestamp: Optional[int]
    related_record_ids: tuple[str, ...]
    summary: str
    execution_status: str


@dataclass(frozen=True)
class VaultLearningDiagnostic:
    reason_code: str
    source_label: str = ""


def _text(value: object) -> str:
    return str(value or "").strip()


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "on"}


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(_text(item) for item in value if _text(item))
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return tuple(
                _text(item) for item in stripped[1:-1].split(",") if _text(item)
            )
    return ()


def _timestamp(metadata: dict[str, Any], path: Path) -> Optional[int]:
    for key in ("verified_at", "observed_at", "created_at"):
        value = _text(metadata.get(key))
        if not value:
            continue
        try:
            from datetime import datetime, timezone

            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return int(parsed.timestamp())
        except (ValueError, OverflowError):
            continue
    try:
        return int(path.stat().st_mtime)
    except OSError:
        return None


def _catalog_roots(vault: Path) -> Iterable[Path]:
    for agent in ("Claude", "Codex", "Hermes", "Shared"):
        yield vault / "Memories" / agent
    for agent in ("Claude", "Codex", "Hermes"):
        yield vault / "Reference" / "Agent Skills" / agent


def _catalog_files(vault: Path) -> Iterable[Path]:
    seen = 0
    for root in _catalog_roots(vault):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.md")):
            if seen >= MAX_CATALOG_FILES:
                return
            seen += 1
            yield path


def _relative_label(path: Path, vault: Path) -> str:
    try:
        return path.resolve().relative_to(vault).as_posix()
    except (OSError, ValueError):
        return path.name


def _read_note(
    path: Path, vault: Path
) -> tuple[str | None, VaultLearningDiagnostic | None]:
    label = _relative_label(path, vault)
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(vault) or not resolved.is_file():
            return None, VaultLearningDiagnostic("note_outside_vault", label)
        if resolved.stat().st_size > MAX_NOTE_BYTES:
            return None, VaultLearningDiagnostic("note_too_large", label)
        return resolved.read_text(encoding="utf-8"), None
    except UnicodeDecodeError:
        return None, VaultLearningDiagnostic("invalid_utf8", label)
    except OSError:
        return None, VaultLearningDiagnostic("note_unreadable", label)


def _node_from_note(
    path: Path, vault: Path
) -> tuple[VaultLearningNode | None, VaultLearningDiagnostic | None]:
    raw, diagnostic = _read_note(path, vault)
    if raw is None:
        return None, diagnostic
    metadata, body = parse_frontmatter(raw)
    label = _relative_label(path, vault)
    record_id = _text(metadata.get("record_id"))
    if not record_id:
        return None, VaultLearningDiagnostic("record_id_required", label)
    if _text(metadata.get("schema_name")) != "agent_learning_record_v1":
        return None, VaultLearningDiagnostic("unsupported_schema", label)
    if not _truthy(metadata.get("sync_owned")):
        return None, VaultLearningDiagnostic("sync_owned_required", label)
    if _text(metadata.get("classification")) != "diagnostic-only":
        return None, VaultLearningDiagnostic("unsupported_classification", label)
    if _text(metadata.get("authority")) not in _AUTHORITIES:
        return None, VaultLearningDiagnostic("unsupported_authority", label)
    source_kind = _text(metadata.get("kind"))
    if source_kind not in _KINDS:
        return None, VaultLearningDiagnostic("unsupported_kind", label)
    origin_agent = _text(metadata.get("agent")).lower()
    if origin_agent not in _AGENTS:
        return None, VaultLearningDiagnostic("unsupported_agent", label)
    execution_status = _text(metadata.get("execution_status"))
    if source_kind == "skill-reference" and execution_status != "reference-only":
        return None, VaultLearningDiagnostic("skill_not_reference_only", label)
    summary = body.strip()[:MAX_SUMMARY_CHARS]
    graph_kind = (
        "skill-reference" if source_kind == "skill-reference" else "shared-memory"
    )
    return (
        VaultLearningNode(
            record_id=record_id,
            label=_text(metadata.get("title")) or path.stem,
            kind=graph_kind,
            origin_agent=origin_agent,
            area=_text(metadata.get("memory_area")) or "General",
            status=_text(metadata.get("status")) or "unknown",
            timestamp=_timestamp(metadata, path),
            related_record_ids=_strings(metadata.get("related_record_ids")),
            summary=summary,
            execution_status=execution_status or "not-applicable",
        ),
        None,
    )


def read_vault_learning(
    vault_dir: Path,
) -> tuple[list[VaultLearningNode], list[VaultLearningDiagnostic]]:
    """Read allowlisted normalized notes; individual failures stay diagnostic."""

    try:
        vault = vault_dir.expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        return [], [VaultLearningDiagnostic("vault_unavailable")]
    if not vault.is_dir():
        return [], [VaultLearningDiagnostic("vault_unavailable")]

    nodes: list[VaultLearningNode] = []
    diagnostics: list[VaultLearningDiagnostic] = []
    seen: set[str] = set()
    for path in _catalog_files(vault):
        node, diagnostic = _node_from_note(path, vault)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
            continue
        if node is None:
            continue
        if node.record_id in seen:
            diagnostics.append(
                VaultLearningDiagnostic(
                    "duplicate_record_id", _relative_label(path, vault)
                )
            )
            continue
        seen.add(node.record_id)
        nodes.append(node)
    return nodes, diagnostics
