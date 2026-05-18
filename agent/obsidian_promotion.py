"""Dry-run Obsidian promotion planning for curated Hermes artifacts.

The module is deliberately side-effect-light. Planning classifies a candidate
and returns previews only. Writing is available through a separate helper that
requires an explicit approval flag and a caller-supplied vault path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


APPROVAL_FLAG = "approved=True"


class ObsidianPromotionAction(str, Enum):
    """Promotion actions aligned with the Umbbi vault folder roles."""

    RAW_EVIDENCE = "RAW_EVIDENCE"
    DEV_SYNTHESIS = "DEV_SYNTHESIS"
    PROJECT_SUMMARY = "PROJECT_SUMMARY"
    KNOWLEDGE_NOTE = "KNOWLEDGE_NOTE"
    MEDIA_ARTIFACT = "MEDIA_ARTIFACT"
    REJECT = "REJECT"


@dataclass(frozen=True)
class ObsidianPromotionCandidate:
    """Artifact proposed for curated Obsidian promotion."""

    title: str
    content: str
    source_type: str = "unknown"
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    profile: Optional[str] = None
    project: Optional[str] = None
    tags: Sequence[str] = ()
    created_at: Optional[datetime | str] = None


@dataclass(frozen=True)
class ObsidianPromotionPlan:
    """Dry-run plan for a possible Obsidian promotion."""

    action: ObsidianPromotionAction
    target_relative_path: Optional[str]
    rationale: str
    requires_approval: bool
    approval_flag: str
    duplicate_candidates: list[str]
    frontmatter_preview: str
    markdown_preview: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "target_relative_path": self.target_relative_path,
            "rationale": self.rationale,
            "requires_approval": self.requires_approval,
            "approval_flag": self.approval_flag,
            "duplicate_candidates": list(self.duplicate_candidates),
            "frontmatter_preview": self.frontmatter_preview,
            "markdown_preview": self.markdown_preview,
        }


_TARGET_DIRS = {
    ObsidianPromotionAction.RAW_EVIDENCE: "Raw",
    ObsidianPromotionAction.DEV_SYNTHESIS: "Dev",
    ObsidianPromotionAction.PROJECT_SUMMARY: "Projects",
    ObsidianPromotionAction.KNOWLEDGE_NOTE: "Knowledge",
    ObsidianPromotionAction.MEDIA_ARTIFACT: "Media",
}

_ACTION_TYPES = {
    ObsidianPromotionAction.RAW_EVIDENCE: "raw",
    ObsidianPromotionAction.DEV_SYNTHESIS: "dev-synthesis",
    ObsidianPromotionAction.PROJECT_SUMMARY: "project-summary",
    ObsidianPromotionAction.KNOWLEDGE_NOTE: "knowledge-note",
    ObsidianPromotionAction.MEDIA_ARTIFACT: "media-artifact",
    ObsidianPromotionAction.REJECT: "rejected",
}

_SECRET_KEY_RE = re.compile(
    r"\b[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|PRIVATE_KEY|ACCESS_KEY)\b\s*[:=]\s*([^\s,;]+)",
    re.IGNORECASE,
)
_SECRET_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9_]{12,}|xox[baprs]-[A-Za-z0-9-]{12,})\b"
)
_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)

_FORBIDDEN_SOURCE_RE = re.compile(
    r"\b("
    r"full[_ -]?transcript|whole[_ -]?transcript|debug[_ -]?log|temporary[_ -]?log|"
    r"scratch|workspace[_ -]?dump|raw[_ -]?dump|active[_ -]?plan|active[_ -]?task"
    r")\b",
    re.IGNORECASE,
)
_FORBIDDEN_TEXT_RE = re.compile(
    r"\b("
    r"complete transcript|full transcript|whole transcript|every user and assistant message|"
    r"temporary debug log|debug log|active plan|active task|scratch output|"
    r"raw workspace dump|workspace dump|intermediate tool outputs|tool outputs? full copy"
    r")\b",
    re.IGNORECASE,
)
_FORBIDDEN_PATH_RE = re.compile(
    r"(^|/)(plan|tasks|task|debug-log|debug_log|latest-summary|scratch|workspace-dump)\.(md|txt|log|json)$",
    re.IGNORECASE,
)

_RAW_RE = re.compile(
    r"\b(raw evidence|source evidence|source:|evidence:|citation|clip|clipping|captured|"
    r"transcript excerpt|original source)\b",
    re.IGNORECASE,
)
_PROJECT_RE = re.compile(
    r"\b(project final summary|final summary|project summary|retrospective|decision record|"
    r"adr:|architecture decision|postmortem|outcome summary)\b",
    re.IGNORECASE,
)
_DEV_RE = re.compile(
    r"\b(hermes|mcp|cli|agent|automation|debugging|developer|dev synthesis|"
    r"system operation|runbook|implementation pattern)\b",
    re.IGNORECASE,
)
_KNOWLEDGE_RE = re.compile(
    r"\b(knowledge note|research synthesis|concept|long-term reference|explainer|"
    r"general knowledge|analysis)\b",
    re.IGNORECASE,
)
_MEDIA_RE = re.compile(
    r"\b(media artifact|slide|slides|presentation|image|video|excalidraw|creative artifact|"
    r"final artwork|thumbnail|poster)\b",
    re.IGNORECASE,
)
_DRAFT_MEDIA_RE = re.compile(
    r"\b(draft|failed|prompt experiment|iteration log|unused variant|rough cut)\b",
    re.IGNORECASE,
)


def plan_obsidian_promotion(
    candidate: ObsidianPromotionCandidate | Mapping[str, Any],
    *,
    vault_path: Optional[str | Path] = None,
) -> ObsidianPromotionPlan:
    """Return a dry-run promotion plan without writing to Obsidian."""

    normalized = _coerce_candidate(candidate)
    action, rationale = _classify_candidate(normalized)
    redacted_title = _redact_text((normalized.title or "").strip()) or "Untitled Note"
    redacted_content = _redact_text(normalized.content or "")

    if action == ObsidianPromotionAction.REJECT:
        frontmatter = _build_frontmatter(normalized, action, title=redacted_title)
        markdown = _build_markdown_preview(
            normalized,
            action,
            title=redacted_title,
            content=redacted_content,
            frontmatter=frontmatter,
        )
        return ObsidianPromotionPlan(
            action=action,
            target_relative_path=None,
            rationale=rationale,
            requires_approval=False,
            approval_flag=APPROVAL_FLAG,
            duplicate_candidates=[],
            frontmatter_preview=frontmatter,
            markdown_preview=markdown,
        )

    target_dir = _TARGET_DIRS[action]
    target_relative_path = f"{target_dir}/{_slugify_title(redacted_title)}.md"
    duplicates = _find_duplicate_candidates(vault_path, target_relative_path, redacted_title)
    frontmatter = _build_frontmatter(normalized, action, title=redacted_title)
    markdown = _build_markdown_preview(
        normalized,
        action,
        title=redacted_title,
        content=redacted_content,
        frontmatter=frontmatter,
    )

    return ObsidianPromotionPlan(
        action=action,
        target_relative_path=target_relative_path,
        rationale=rationale,
        requires_approval=True,
        approval_flag=APPROVAL_FLAG,
        duplicate_candidates=duplicates,
        frontmatter_preview=frontmatter,
        markdown_preview=markdown,
    )


def write_obsidian_promotion(
    plan: ObsidianPromotionPlan,
    *,
    vault_path: str | Path,
    approved: bool = False,
) -> Path:
    """Write an approved promotion plan under a supplied vault path.

    The helper refuses dry-run plans, rejected plans, absolute target paths,
    traversal outside the vault, and overwrites.
    """

    if not approved:
        raise PermissionError(f"Obsidian writes require {APPROVAL_FLAG}.")
    if plan.action == ObsidianPromotionAction.REJECT or not plan.target_relative_path:
        raise ValueError("Rejected promotion plans do not have a writable target.")

    target = _resolve_vault_target(vault_path, plan.target_relative_path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing Obsidian note: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(plan.markdown_preview, encoding="utf-8")
    return target


def _coerce_candidate(candidate: ObsidianPromotionCandidate | Mapping[str, Any]) -> ObsidianPromotionCandidate:
    if isinstance(candidate, ObsidianPromotionCandidate):
        return candidate
    if not isinstance(candidate, Mapping):
        raise TypeError("candidate must be an ObsidianPromotionCandidate or mapping")
    return ObsidianPromotionCandidate(
        title=str(candidate.get("title") or ""),
        content=str(candidate.get("content") or ""),
        source_type=str(candidate.get("source_type") or "unknown"),
        source_path=_optional_str(candidate.get("source_path")),
        source_url=_optional_str(candidate.get("source_url")),
        profile=_optional_str(candidate.get("profile")),
        project=_optional_str(candidate.get("project")),
        tags=tuple(str(tag) for tag in candidate.get("tags") or ()),
        created_at=candidate.get("created_at"),
    )


def _classify_candidate(
    candidate: ObsidianPromotionCandidate,
) -> tuple[ObsidianPromotionAction, str]:
    title = candidate.title or ""
    content = candidate.content or ""
    source_type = candidate.source_type or "unknown"
    source_path = candidate.source_path or ""
    profile = (candidate.profile or "").strip().lower()
    haystack = " ".join([title, content, source_type, source_path])

    if not title.strip() and not content.strip():
        return (
            ObsidianPromotionAction.REJECT,
            "Empty candidates are not eligible for Obsidian promotion.",
        )
    if _looks_like_secret(haystack):
        return (
            ObsidianPromotionAction.REJECT,
            "Secret-looking candidates are not eligible; previews are redacted for review.",
        )
    if _is_forbidden_artifact(title, content, source_type, source_path):
        return (
            ObsidianPromotionAction.REJECT,
            "Whole transcripts, temporary/debug logs, active plans/tasks, scratch outputs, and raw workspace dumps are not eligible for Obsidian promotion.",
        )
    if _MEDIA_RE.search(haystack):
        if _DRAFT_MEDIA_RE.search(haystack):
            return (
                ObsidianPromotionAction.REJECT,
                "Draft media iterations and prompt experiment logs are not eligible for Obsidian promotion.",
            )
        return (
            ObsidianPromotionAction.MEDIA_ARTIFACT,
            "Completed creative or reusable media artifacts are curated under Media/; dry-run only until approved.",
        )
    if _RAW_RE.search(haystack) or "raw" in source_type.lower():
        return (
            ObsidianPromotionAction.RAW_EVIDENCE,
            "Raw means evidence/source material and maps to Raw/; dry-run only until approved.",
        )
    if _PROJECT_RE.search(haystack) or "summary" in source_type.lower() or "decision" in source_type.lower():
        profile_note = ""
        if profile == "vault":
            profile_note = " The vault profile is investment/risk/strategy work, not Obsidian governance ownership."
        return (
            ObsidianPromotionAction.PROJECT_SUMMARY,
            "Project summaries, decisions, retrospectives, and outcome notes map to Projects/; dry-run only until approved."
            + profile_note,
        )
    if profile == "forge":
        return (
            ObsidianPromotionAction.DEV_SYNTHESIS,
            "Forge defaults reusable Hermes/system outputs to Dev/ unless they are source evidence or project summaries; dry-run only until approved.",
        )
    if _DEV_RE.search(haystack):
        return (
            ObsidianPromotionAction.DEV_SYNTHESIS,
            "Reusable development, automation, Hermes, MCP, CLI, or system-operations synthesis maps to Dev/; dry-run only until approved.",
        )
    if _KNOWLEDGE_RE.search(haystack) or profile in {"lumina", "vault"}:
        profile_note = ""
        if profile == "vault":
            profile_note = " The vault profile is investment/risk/strategy work, not Obsidian governance ownership."
        return (
            ObsidianPromotionAction.KNOWLEDGE_NOTE,
            "Curated general knowledge and research synthesis maps to Knowledge/; dry-run only until approved."
            + profile_note,
        )
    return (
        ObsidianPromotionAction.KNOWLEDGE_NOTE,
        "No narrower promotion rule matched, so the conservative curated target is Knowledge/; dry-run only until approved.",
    )


def _is_forbidden_artifact(title: str, content: str, source_type: str, source_path: str) -> bool:
    title_source = " ".join([title, source_type, source_path])
    if _FORBIDDEN_SOURCE_RE.search(title_source) or _FORBIDDEN_PATH_RE.search(source_path):
        return True
    return bool(_FORBIDDEN_TEXT_RE.search(content))


def _build_frontmatter(
    candidate: ObsidianPromotionCandidate,
    action: ObsidianPromotionAction,
    *,
    title: str,
) -> str:
    created_date = _date_for_frontmatter(candidate.created_at)
    status = "archived" if action == ObsidianPromotionAction.RAW_EVIDENCE else "draft"
    lines = [
        "---",
        f"title: {_yaml_string(title)}",
        f"created: '{created_date}'",
        f"last_modified: '{created_date}'",
        f"type: {_yaml_string(_ACTION_TYPES[action])}",
        f"status: {_yaml_string(status)}",
    ]
    tags = _clean_tags(candidate.tags, action)
    if tags:
        lines.append("tags:")
        lines.extend(f"  - {tag}" for tag in tags)
    if candidate.profile:
        lines.append(f"profile: {_yaml_string(_redact_text(candidate.profile))}")
    if candidate.project:
        lines.append(f"project: {_yaml_string(_redact_text(candidate.project))}")
    if candidate.source_type:
        lines.append(f"source_type: {_yaml_string(_redact_text(candidate.source_type))}")
    if candidate.source_path:
        lines.append(f"source_path: {_yaml_string(_redact_text(candidate.source_path))}")
    if candidate.source_url:
        lines.append(f"source_url: {_yaml_string(_redact_text(candidate.source_url))}")
    lines.append("---")
    return "\n".join(lines)


def _build_markdown_preview(
    candidate: ObsidianPromotionCandidate,
    action: ObsidianPromotionAction,
    *,
    title: str,
    content: str,
    frontmatter: str,
) -> str:
    parts = [
        frontmatter,
        "",
        f"# {title}",
        "",
        content.strip(),
    ]
    provenance = _provenance_lines(candidate, action)
    if provenance:
        parts.extend(["", "## Provenance", "", *provenance])
    return "\n".join(parts).rstrip() + "\n"


def _provenance_lines(candidate: ObsidianPromotionCandidate, action: ObsidianPromotionAction) -> list[str]:
    lines = [f"- promotion_action: {_redact_text(action.value)}"]
    if candidate.profile:
        lines.append(f"- profile: {_redact_text(candidate.profile)}")
    if candidate.project:
        lines.append(f"- project: {_redact_text(candidate.project)}")
    if candidate.source_type:
        lines.append(f"- source_type: {_redact_text(candidate.source_type)}")
    if candidate.source_path:
        lines.append(f"- source_path: {_redact_text(candidate.source_path)}")
    if candidate.source_url:
        lines.append(f"- source_url: {_redact_text(candidate.source_url)}")
    return lines


def _find_duplicate_candidates(
    vault_path: Optional[str | Path],
    target_relative_path: str,
    title: str,
) -> list[str]:
    if not vault_path:
        return []
    vault = Path(vault_path)
    target_rel = Path(target_relative_path)
    target_dir = vault / target_rel.parent
    if not target_dir.exists() or not target_dir.is_dir():
        return []

    duplicates: list[str] = []
    seen: set[str] = set()
    target_stem = target_rel.stem.casefold()
    title_key = _title_key(title)
    for path in sorted(target_dir.glob("*.md")):
        rel = path.relative_to(vault).as_posix()
        stem_matches = path.stem.casefold() == target_stem
        title_matches = _title_key(_extract_note_title(path)) == title_key
        if (stem_matches or title_matches) and rel not in seen:
            duplicates.append(rel)
            seen.add(rel)
    return duplicates


def _extract_note_title(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    for line in text.splitlines()[:40]:
        stripped = line.strip()
        if stripped.lower().startswith("title:"):
            return stripped.split(":", 1)[1].strip().strip("'\"")
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def _resolve_vault_target(vault_path: str | Path, target_relative_path: str) -> Path:
    rel = Path(target_relative_path)
    if rel.is_absolute():
        raise ValueError("Obsidian promotion target must be relative to the vault.")
    if any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError("Obsidian promotion target must not traverse outside the vault.")
    vault = Path(vault_path).resolve()
    target = (vault / rel).resolve()
    try:
        target.relative_to(vault)
    except ValueError as exc:
        raise ValueError("Obsidian promotion target escaped the supplied vault path.") from exc
    return target


def _looks_like_secret(text: str) -> bool:
    return bool(
        _SECRET_KEY_RE.search(text)
        or _SECRET_TOKEN_RE.search(text)
        or _PRIVATE_KEY_BLOCK_RE.search(text)
    )


def _redact_text(text: Optional[str]) -> str:
    if not text:
        return ""
    redacted = _PRIVATE_KEY_BLOCK_RE.sub("[REDACTED PRIVATE KEY]", str(text))
    redacted = _SECRET_KEY_RE.sub(lambda m: m.group(0).replace(m.group(1), "[REDACTED]"), redacted)
    redacted = _SECRET_TOKEN_RE.sub("[REDACTED]", redacted)
    return redacted


def _slugify_title(title: str) -> str:
    safe = re.sub(r"[\\/:*?\"<>|\r\n\t]+", " ", title)
    safe = re.sub(r"\s+", "-", safe.strip().lower())
    safe = re.sub(r"-{2,}", "-", safe).strip(".-_")
    return safe or "untitled-note"


def _title_key(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip()).casefold()


def _clean_tags(tags: Sequence[str], action: ObsidianPromotionAction) -> list[str]:
    cleaned = ["obsidian-promotion", _ACTION_TYPES[action]]
    for tag in tags:
        value = re.sub(r"\s+", "-", str(tag).strip().lstrip("#").lower())
        value = re.sub(r"[^a-z0-9._/-]+", "", value).strip("-")
        if value:
            cleaned.append(value)
    deduped: list[str] = []
    for tag in cleaned:
        if tag not in deduped:
            deduped.append(tag)
    return deduped


def _date_for_frontmatter(value: Optional[datetime | str]) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date().isoformat()
    if isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}", value):
        return value[:10]
    return datetime.now(timezone.utc).date().isoformat()


def _yaml_string(value: str) -> str:
    return json.dumps(_redact_text(value), ensure_ascii=False)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


__all__ = [
    "APPROVAL_FLAG",
    "ObsidianPromotionAction",
    "ObsidianPromotionCandidate",
    "ObsidianPromotionPlan",
    "plan_obsidian_promotion",
    "write_obsidian_promotion",
]
