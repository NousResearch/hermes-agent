"""Capability Discovery Runtime canary.

Pure, local, read-only search over existing Hermes capability artifacts. The
module accepts an ObjectiveContext and returns capability_report.json-shaped
data. It does not build goals, strategies, execution contracts, task boards,
workers, provider calls, or knowledge-store writes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPORT_FIELDS = (
    "matched_skills",
    "matched_workflows",
    "matched_roles",
    "matched_policies",
    "matched_templates",
    "matched_reports",
    "matched_checkpoints",
    "matched_capabilities",
    "confidence",
    "reusable_assets",
    "missing_capabilities",
)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "by",
        "de",
        "del",
        "el",
        "en",
        "for",
        "in",
        "is",
        "la",
        "las",
        "los",
        "no",
        "of",
        "on",
        "or",
        "para",
        "por",
        "que",
        "the",
        "to",
        "un",
        "una",
        "y",
    }
)

_TEXT_SUFFIXES = frozenset(
    {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".py"}
)

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROFILE_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
_DEFAULT_BASE_HOME = (
    _DEFAULT_PROFILE_HOME.parents[1]
    if _DEFAULT_PROFILE_HOME.name == "orchestrator"
    and len(_DEFAULT_PROFILE_HOME.parents) >= 2
    and _DEFAULT_PROFILE_HOME.parent.name == "profiles"
    else _DEFAULT_PROFILE_HOME
)


@dataclass(frozen=True)
class ObjectiveContext:
    objective_text: str
    user_id: str = "executive-runtime-canary"
    constraints: tuple[str, ...] = ()
    source_checkpoint: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ObjectiveContext":
        return cls(
            objective_text=str(data.get("objective_text") or data.get("objective") or ""),
            user_id=str(data.get("user_id") or "executive-runtime-canary"),
            constraints=tuple(str(v) for v in (data.get("constraints") or ())),
            source_checkpoint=(
                str(data["source_checkpoint"]) if data.get("source_checkpoint") else None
            ),
        )


@dataclass(frozen=True)
class CapabilityDiscoveryIndex:
    skill_roots: tuple[Path, ...] = (
        _DEFAULT_PROFILE_HOME / "skills",
        _DEFAULT_BASE_HOME / "skills",
    )
    workflow_roots: tuple[Path, ...] = (_DEFAULT_REPO_ROOT, _DEFAULT_BASE_HOME / "reports")
    role_roots: tuple[Path, ...] = (_DEFAULT_REPO_ROOT / "agent",)
    policy_roots: tuple[Path, ...] = (_DEFAULT_REPO_ROOT / "agent" / "executive",)
    template_roots: tuple[Path, ...] = (
        _DEFAULT_REPO_ROOT / "agent" / "executive" / "schemas",
        _DEFAULT_REPO_ROOT / "skills",
        _DEFAULT_PROFILE_HOME / "skills",
    )
    report_roots: tuple[Path, ...] = (_DEFAULT_BASE_HOME / "reports",)
    checkpoint_roots: tuple[Path, ...] = (_DEFAULT_BASE_HOME / "reports",)
    capability_roots: tuple[Path, ...] = (
        _DEFAULT_REPO_ROOT / "agent" / "executive",
        _DEFAULT_REPO_ROOT / "tools",
        _DEFAULT_PROFILE_HOME / "skills",
    )


@dataclass(frozen=True)
class CapabilityMatch:
    kind: str
    name: str
    path: str
    score: float
    reasons: tuple[str, ...]
    digest: str
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "path": self.path,
            "score": round(float(self.score), 4),
            "reasons": list(self.reasons),
            "digest": self.digest,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True)
class CapabilityReport:
    matched_skills: tuple[CapabilityMatch, ...]
    matched_workflows: tuple[CapabilityMatch, ...]
    matched_roles: tuple[CapabilityMatch, ...]
    matched_policies: tuple[CapabilityMatch, ...]
    matched_templates: tuple[CapabilityMatch, ...]
    matched_reports: tuple[CapabilityMatch, ...]
    matched_checkpoints: tuple[CapabilityMatch, ...]
    matched_capabilities: tuple[CapabilityMatch, ...]
    confidence: float
    reusable_assets: tuple[dict[str, Any], ...]
    missing_capabilities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched_skills": [m.to_dict() for m in self.matched_skills],
            "matched_workflows": [m.to_dict() for m in self.matched_workflows],
            "matched_roles": [m.to_dict() for m in self.matched_roles],
            "matched_policies": [m.to_dict() for m in self.matched_policies],
            "matched_templates": [m.to_dict() for m in self.matched_templates],
            "matched_reports": [m.to_dict() for m in self.matched_reports],
            "matched_checkpoints": [m.to_dict() for m in self.matched_checkpoints],
            "matched_capabilities": [m.to_dict() for m in self.matched_capabilities],
            "confidence": round(float(self.confidence), 4),
            "reusable_assets": list(self.reusable_assets),
            "missing_capabilities": list(self.missing_capabilities),
        }


def _tokenize(text: str) -> set[str]:
    normalized = re.sub(r"[^\w\s-]", " ", text.lower())
    return {token for token in normalized.split() if len(token) > 1 and token not in _STOPWORDS}


def _objective_tokens(context: ObjectiveContext) -> set[str]:
    joined = "\n".join((context.objective_text, "\n".join(context.constraints)))
    tokens = _tokenize(joined)
    if context.source_checkpoint:
        tokens.update(_tokenize(Path(context.source_checkpoint).name))
    return tokens


def _safe_iter_files(roots: Iterable[Path]) -> tuple[Path, ...]:
    seen: dict[str, Path] = {}
    for root in roots:
        try:
            resolved = Path(root).expanduser()
        except (OSError, RuntimeError):
            continue
        if not resolved.exists():
            continue
        if resolved.is_file():
            paths = (resolved,)
        else:
            try:
                paths = tuple(p for p in resolved.rglob("*") if p.is_file())
            except (OSError, RuntimeError):
                continue
        for path in paths:
            if path.suffix.lower() in _TEXT_SUFFIXES:
                seen.setdefault(str(path), path)
    return tuple(seen.values())


def _read_text_prefix(path: Path, limit: int = 12_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except (OSError, UnicodeError):
        return ""


def _score(tokens: set[str], path: Path, text: str, hints: tuple[str, ...]) -> tuple[float, tuple[str, ...]]:
    haystack = f"{path.name}\n{path.parent.name}\n{text}"
    hay_tokens = _tokenize(haystack)
    if not tokens or not hay_tokens:
        overlap_score = 0.0
    else:
        overlap_score = len(tokens & hay_tokens) / max(1, min(len(tokens), 18))
    lower_haystack = haystack.lower()
    reasons: list[str] = []
    hint_hits = [hint for hint in hints if hint in lower_haystack]
    if hint_hits:
        reasons.extend(f"hint:{hint}" for hint in hint_hits[:4])
    overlap_hits = sorted(tokens & hay_tokens)[:8]
    if overlap_hits:
        reasons.append("token_overlap:" + ",".join(overlap_hits))
    hint_bonus = min(0.35, 0.07 * len(hint_hits))
    score = max(0.0, min(1.0, overlap_score + hint_bonus))
    return score, tuple(reasons)


def _digest(path: Path, text: str) -> str:
    payload = f"{path}\n{text[:4096]}"
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def _excerpt(text: str) -> str:
    return " ".join(text.strip().split())[:280]


def _match_files(
    *,
    kind: str,
    roots: Iterable[Path],
    tokens: set[str],
    hints: tuple[str, ...],
    threshold: float = 0.18,
    limit: int = 20,
) -> tuple[CapabilityMatch, ...]:
    matches: list[CapabilityMatch] = []
    for path in _safe_iter_files(roots):
        text = _read_text_prefix(path)
        score, reasons = _score(tokens, path, text, hints)
        if score < threshold and not any(h in str(path).lower() for h in hints):
            continue
        if not reasons:
            reasons = ("path_hint",)
        matches.append(
            CapabilityMatch(
                kind=kind,
                name=path.stem if path.name != "SKILL.md" else path.parent.name,
                path=str(path),
                score=score,
                reasons=reasons,
                digest=_digest(path, text),
                excerpt=_excerpt(text),
            )
        )
    matches.sort(key=lambda m: (m.score, m.path), reverse=True)
    return tuple(matches[:limit])


def _checkpoint_roots(index: CapabilityDiscoveryIndex, context: ObjectiveContext) -> tuple[Path, ...]:
    roots = tuple(index.checkpoint_roots)
    if context.source_checkpoint:
        roots = (Path(context.source_checkpoint),) + roots
    return roots


def _dedupe(matches: Iterable[CapabilityMatch]) -> tuple[CapabilityMatch, ...]:
    by_path: dict[str, CapabilityMatch] = {}
    for match in matches:
        current = by_path.get(match.path)
        if current is None or match.score > current.score:
            by_path[match.path] = match
    return tuple(sorted(by_path.values(), key=lambda m: (m.score, m.path), reverse=True))


def _assets(matches: tuple[CapabilityMatch, ...]) -> tuple[dict[str, Any], ...]:
    assets = []
    for match in matches[:25]:
        assets.append(
            {
                "kind": match.kind,
                "name": match.name,
                "path": match.path,
                "score": round(float(match.score), 4),
                "reuse_reason": "; ".join(match.reasons[:3]),
            }
        )
    return tuple(assets)


def _missing(**groups: tuple[CapabilityMatch, ...]) -> tuple[str, ...]:
    return tuple(name for name, matches in groups.items() if not matches)


def _confidence(groups: tuple[tuple[CapabilityMatch, ...], ...]) -> float:
    populated = [g for g in groups if g]
    if not populated:
        return 0.0
    coverage = len(populated) / len(groups)
    best_scores = [max(m.score for m in group) for group in populated]
    quality = sum(best_scores) / len(best_scores)
    return max(0.0, min(1.0, (coverage * 0.55) + (quality * 0.45)))


def discover_capabilities(
    context: ObjectiveContext | dict[str, Any],
    *,
    index: CapabilityDiscoveryIndex | None = None,
) -> CapabilityReport:
    """Search existing capability artifacts and return a serializable report.

    This canary is search-only. It reads text artifacts under the configured
    roots and computes deterministic matches. It has no writer function by
    design; callers that need a file artifact must serialize ``to_dict()`` at
    the boundary they control.
    """
    objective_context = (
        ObjectiveContext.from_mapping(context) if isinstance(context, dict) else context
    )
    if not objective_context.objective_text.strip():
        raise ValueError("objective_text must be non-empty")
    if not objective_context.user_id.strip():
        raise ValueError("user_id must be non-empty")

    discovery_index = index or CapabilityDiscoveryIndex()
    tokens = _objective_tokens(objective_context)

    matched_skills = _match_files(
        kind="skill",
        roots=discovery_index.skill_roots,
        tokens=tokens,
        hints=("skill", "description", "workflow", "canary", "hermes"),
        threshold=0.07,
    )
    matched_workflows = _match_files(
        kind="workflow",
        roots=discovery_index.workflow_roots,
        tokens=tokens,
        hints=("workflow", "roadmap", "validation", "rollback", "runbook"),
    )
    matched_roles = _match_files(
        kind="role",
        roots=discovery_index.role_roots,
        tokens=tokens,
        hints=("role", "orchestrator", "reviewer", "operator", "agent"),
        threshold=0.22,
    )
    matched_policies = _match_files(
        kind="policy",
        roots=discovery_index.policy_roots,
        tokens=tokens,
        hints=("policy", "approval", "forbidden", "risk", "constraint"),
        threshold=0.16,
    )
    matched_templates = _match_files(
        kind="template",
        roots=discovery_index.template_roots,
        tokens=tokens,
        hints=("schema", "template", "$schema", "manifest"),
        threshold=0.12,
    )
    matched_reports = _match_files(
        kind="report",
        roots=discovery_index.report_roots,
        tokens=tokens,
        hints=("report", "validation", "manifest", "hash", "rollback"),
        threshold=0.14,
    )
    matched_checkpoints = _match_files(
        kind="checkpoint",
        roots=_checkpoint_roots(discovery_index, objective_context),
        tokens=tokens,
        hints=("checkpoint", "checkpoint_pass", "official", "frozen"),
        threshold=0.12,
    )
    explicit_capabilities = _match_files(
        kind="capability",
        roots=discovery_index.capability_roots,
        tokens=tokens,
        hints=("capability", "discovery", "analyzer", "runtime", "canary"),
        threshold=0.14,
    )

    matched_capabilities = _dedupe(
        tuple(explicit_capabilities)
        + tuple(matched_skills)
        + tuple(matched_workflows)
        + tuple(matched_policies)
        + tuple(matched_templates)
        + tuple(matched_reports)
        + tuple(matched_checkpoints)
    )[:40]
    groups = (
        matched_skills,
        matched_workflows,
        matched_roles,
        matched_policies,
        matched_templates,
        matched_reports,
        matched_checkpoints,
        matched_capabilities,
    )

    return CapabilityReport(
        matched_skills=matched_skills,
        matched_workflows=matched_workflows,
        matched_roles=matched_roles,
        matched_policies=matched_policies,
        matched_templates=matched_templates,
        matched_reports=matched_reports,
        matched_checkpoints=matched_checkpoints,
        matched_capabilities=matched_capabilities,
        confidence=_confidence(groups),
        reusable_assets=_assets(matched_capabilities),
        missing_capabilities=_missing(
            skills=matched_skills,
            workflows=matched_workflows,
            roles=matched_roles,
            policies=matched_policies,
            templates=matched_templates,
            reports=matched_reports,
            checkpoints=matched_checkpoints,
            capabilities=matched_capabilities,
        ),
    )


__all__ = [
    "CapabilityDiscoveryIndex",
    "CapabilityMatch",
    "CapabilityReport",
    "ObjectiveContext",
    "REPORT_FIELDS",
    "discover_capabilities",
]
