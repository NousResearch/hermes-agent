"""Experimental uSwarm-style helper primitives.

This module intentionally has no import-time side effects and all runtime-facing
features are default-off.  Callers pass config explicitly (or opt in with the
HERMES_USWARM_* env vars) before attaching these payloads to prompts/results.
"""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from agent.model_metadata import estimate_tokens_rough

_CONTEXT_PACK_SCHEMA = "uswarm.context_pack.v1"
_CHILD_SIDECAR_SCHEMA = "uswarm.child_run_sidecar.v1"
_REWRITE_SMELL_ID = "gond-smell-011-whole-file-rewrite"

_TRUTHY = {"1", "true", "yes", "on", "y"}
_FALSEY = {"0", "false", "no", "off", "n"}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUTHY:
            return True
        if lowered in _FALSEY:
            return False
    return default


def _nested(mapping: Optional[Mapping[str, Any]], *keys: str) -> Any:
    current: Any = mapping or {}
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _feature_enabled(config: Optional[Mapping[str, Any]], section: str, env_name: str) -> bool:
    env_value = os.getenv(env_name)
    if env_value is not None:
        return _coerce_bool(env_value, default=False)
    return _coerce_bool(_nested(config, "uswarm_helpers", section, "enabled"), default=False)


def is_uswarm_context_pack_enabled(config: Optional[Mapping[str, Any]] = None) -> bool:
    """Return whether experimental context-pack emission is enabled."""

    return _feature_enabled(config, "context_pack", "HERMES_USWARM_CONTEXT_PACK")


def is_uswarm_child_sidecar_enabled(config: Optional[Mapping[str, Any]] = None) -> bool:
    """Return whether experimental child-run sidecar emission is enabled."""

    return _feature_enabled(config, "child_run_sidecar", "HERMES_USWARM_CHILD_SIDECAR")


def is_uswarm_rewrite_smell_enabled(config: Optional[Mapping[str, Any]] = None) -> bool:
    """Return whether experimental whole-file-rewrite smell checks are enabled."""

    return _feature_enabled(config, "rewrite_smell", "HERMES_USWARM_REWRITE_SMELL")


def _entry_content(item: Mapping[str, Any]) -> str:
    return str(item.get("content") or item.get("text") or "")


def _safe_pack_path(ref: str, *, allowed_base: Optional[str] = None) -> Optional[str]:
    """Return a context-pack reference only if it stays inside allowed_base.

    CP-1: when context packs are enabled, file-like section paths must not let
    absolute paths or `..` traversal escape the configured/base workspace.
    Non-path labels are accepted as relative references.
    """

    text = str(ref or "").strip()
    if not text:
        return None
    if "\x00" in text:
        return None
    # WSL/Linux pathlib treats Windows-style drive, UNC, and backslash paths as
    # ordinary relative strings.  Context-pack refs may originate from Windows
    # paths, so reject escape syntax before POSIX normalization.
    windows_candidate = PureWindowsPath(text)
    if windows_candidate.drive or text.startswith("\\"):
        return None
    if ".." in text.replace("\\", "/").split("/"):
        return None
    candidate = Path(text).expanduser()
    if allowed_base:
        base = Path(allowed_base).expanduser().resolve()
        if candidate.is_absolute():
            try:
                resolved = candidate.resolve(strict=False)
                resolved.relative_to(base)
            except ValueError:
                return None
            return str(resolved)
        parts = candidate.parts
        if any(part == ".." for part in parts):
            return None
        try:
            resolved = (base / candidate).resolve(strict=False)
            resolved.relative_to(base)
        except ValueError:
            return None
        return text
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        return None
    return text


def build_context_pack(
    items: Iterable[Mapping[str, Any]],
    *,
    token_budget: int = 4000,
    reserve_tokens: int = 0,
    allowed_base: Optional[str] = None,
) -> Dict[str, Any]:
    """Pack ordered context items into a small structured envelope.

    Items are accepted as dictionaries with at least ``content`` (or ``text``),
    plus optional ``path``, ``kind``, ``title`` and ``metadata``.  Oversized
    entries are omitted rather than partially copied so downstream consumers get
    deterministic references and token accounting.
    """

    available = max(0, int(token_budget) - max(0, int(reserve_tokens)))
    entries: list[Dict[str, Any]] = []
    estimated = 0
    omitted: list[Dict[str, Any]] = []

    for idx, raw_item in enumerate(items):
        item = dict(raw_item or {})
        content = _entry_content(item)
        tokens = estimate_tokens_rough(content)
        raw_ref = str(item.get("path") or item.get("title") or f"entry-{idx}")
        ref = _safe_pack_path(raw_ref, allowed_base=allowed_base)
        if ref is None:
            omitted.append({"path": raw_ref, "estimated_tokens": tokens, "reason": "unsafe_path"})
            continue
        entry = {
            "id": str(item.get("id") or f"ctx-{idx}"),
            "kind": str(item.get("kind") or "text"),
            "path": ref,
            "estimated_tokens": tokens,
            "content": content,
        }
        if isinstance(item.get("metadata"), Mapping):
            entry["metadata"] = dict(item["metadata"])

        if estimated + tokens > available:
            remaining = max(0, available - estimated)
            if remaining > 0:
                # Keep a deterministic prefix instead of dropping the first
                # oversized item entirely; rough token estimator is ~4 chars.
                clipped = content[: max(1, remaining * 4)]
                clipped_tokens = estimate_tokens_rough(clipped)
                clipped_tokens = min(clipped_tokens, remaining)
                entry["content"] = clipped
                entry["estimated_tokens"] = clipped_tokens
                entry["truncated"] = True
                entries.append(entry)
                estimated += clipped_tokens
            omitted.append({"path": ref, "estimated_tokens": tokens, "reason": "token_budget"})
            continue
        entries.append(entry)
        estimated += tokens

    return {
        "schema_version": _CONTEXT_PACK_SCHEMA,
        "token_budget": int(token_budget),
        "reserve_tokens": int(reserve_tokens),
        "estimated_tokens": estimated,
        "entries": entries,
        "omitted": omitted,
        "truncated": bool(omitted),
    }


_ALLOWED_SIDE_STATUSES = {"queued", "running", "completed", "failed", "timeout", "interrupted", "cancelled"}
_ALLOWED_EVIDENCE_KINDS = {"test", "command", "file", "url", "screenshot", "note"}


def build_child_run_sidecar(
    *,
    task_id: str,
    child_id: str,
    status: str,
    goal: str = "",
    summary: Optional[str] = None,
    error: Optional[str] = None,
    allowed_tools: Optional[Sequence[str]] = None,
    blocked_tools: Optional[Sequence[str]] = None,
    evidence: Optional[Sequence[Mapping[str, Any]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a minimal, JSON-safe child-run sidecar envelope."""

    normalized_status = str(status or "").strip().lower() or "running"
    if normalized_status not in _ALLOWED_SIDE_STATUSES:
        normalized_status = "failed" if error else "running"

    clean_evidence = []
    for item in evidence or []:
        if not isinstance(item, Mapping):
            continue
        ref = str(item.get("ref") or "").strip()
        if not ref:
            continue
        kind = str(item.get("kind") or "note").strip().lower()
        if kind not in _ALLOWED_EVIDENCE_KINDS:
            kind = "note"
        clean = {"kind": kind, "ref": ref}
        note = str(item.get("note") or "").strip()
        if note:
            clean["note"] = note
        clean_evidence.append(clean)

    return {
        "schema_version": _CHILD_SIDECAR_SCHEMA,
        "task_id": str(task_id or ""),
        "child_id": str(child_id or ""),
        "status": normalized_status,
        "goal": str(goal or ""),
        "summary": summary,
        "error": error,
        "allowed_tools": [str(t) for t in (allowed_tools or [])],
        "blocked_tools": [str(t) for t in (blocked_tools or [])],
        "evidence": clean_evidence,
        "metadata": dict(metadata or {}),
    }


@dataclass
class RewriteSmellConfig:
    min_existing_lines: int = 500
    targeted_patch_ratio: float = 0.20
    whole_rewrite_ratio: float = 0.80


@dataclass
class RewriteSmellDecision:
    level: str
    smell: bool
    smell_id: str = _REWRITE_SMELL_ID
    severity: str = "none"
    reason: str = ""
    changed_ratio: Optional[float] = None
    existing_lines: int = 0
    new_lines: int = 0
    patch_tool_available: bool = True
    remediation: str = "Use patch/replace for small changes to large existing files."
    override_conditions: list[str] = field(
        default_factory=lambda: [
            "generated or vendored file",
            "format migration intentionally rewrites most lines",
            "patch tool unavailable or cannot express the edit safely",
        ]
    )
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "smell": self.smell,
            "smell_id": self.smell_id,
            "severity": self.severity,
            "reason": self.reason,
            "changed_ratio": self.changed_ratio,
            "existing_lines": self.existing_lines,
            "new_lines": self.new_lines,
            "patch_tool_available": self.patch_tool_available,
            "remediation": self.remediation,
            "override_conditions": list(self.override_conditions),
            "error": self.error,
        }


def evaluate_rewrite_smell(
    path: str,
    old_content: str,
    new_content: str,
    *,
    patch_tool_available: bool = True,
    config: Optional[RewriteSmellConfig] = None,
) -> Dict[str, Any]:
    """Detect likely unsafe whole-file rewrites of large files.

    This is a pure advisory rule: callers decide whether to show a warning,
    block the operation, or ignore it.  New files and small files are clean.
    """

    cfg = config or RewriteSmellConfig()
    try:
        old_lines = (old_content or "").splitlines()
        new_lines = (new_content or "").splitlines()
        existing_count = len(old_lines)
        new_count = len(new_lines)
        if existing_count == 0:
            return RewriteSmellDecision(
                level="clean",
                smell=False,
                reason="New file or missing previous content.",
                existing_lines=existing_count,
                new_lines=new_count,
                patch_tool_available=patch_tool_available,
            ).to_dict()
        if existing_count < cfg.min_existing_lines:
            return RewriteSmellDecision(
                level="clean",
                smell=False,
                reason=f"Existing file has {existing_count} lines; below large-file threshold.",
                existing_lines=existing_count,
                new_lines=new_count,
                patch_tool_available=patch_tool_available,
            ).to_dict()

        matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
        similarity = matcher.ratio()
        changed_ratio = max(0.0, min(1.0, 1.0 - similarity))

        if changed_ratio <= cfg.targeted_patch_ratio and patch_tool_available:
            return RewriteSmellDecision(
                level="warn",
                smell=True,
                severity="high",
                reason=(
                    f"Whole-file rewrite of large file {path!r} changes only "
                    f"{changed_ratio:.0%} of lines; targeted patch would be safer."
                ),
                changed_ratio=changed_ratio,
                existing_lines=existing_count,
                new_lines=new_count,
                patch_tool_available=patch_tool_available,
            ).to_dict()
        if changed_ratio >= cfg.whole_rewrite_ratio:
            return RewriteSmellDecision(
                level="clean",
                smell=False,
                reason=f"Rewrite changes {changed_ratio:.0%} of lines; looks intentionally broad.",
                changed_ratio=changed_ratio,
                existing_lines=existing_count,
                new_lines=new_count,
                patch_tool_available=patch_tool_available,
            ).to_dict()
        return RewriteSmellDecision(
            level="note",
            smell=False,
            severity="low",
            reason=f"Moderate changed ratio ({changed_ratio:.0%}) in a large file.",
            changed_ratio=changed_ratio,
            existing_lines=existing_count,
            new_lines=new_count,
            patch_tool_available=patch_tool_available,
        ).to_dict()
    except Exception as exc:  # pragma: no cover - defensive degrade path
        return RewriteSmellDecision(
            level="clean",
            smell=False,
            reason="Rule evaluation failed; degrading to clean.",
            patch_tool_available=patch_tool_available,
            error=f"{type(exc).__name__}: {exc}",
        ).to_dict()
