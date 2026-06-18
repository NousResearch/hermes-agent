from __future__ import annotations

"""Kanban route watchdog.

A lightweight, non-invasive routing overlay that sits *outside* the core
planner/decomposer logic and evaluates ready tasks just before dispatch.

Goals:
- catch execution cards stranded on the orchestrator before they burn a spawn
  cycle
- catch obvious brand/role mismatches when a different profile is a much
  better fit
- hold ambiguous / low-signal routes for explicit human review instead of
  silently guessing

The watchdog is intentionally heuristic and fail-open. If roster metadata is
missing or we cannot compute a meaningful comparison, we return no decision and
let the normal dispatcher proceed.
"""

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Optional


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "if", "in", "into", "is", "it", "its", "need", "of",
    "on", "or", "our", "please", "should", "so", "that", "the", "their",
    "this", "to", "we", "with", "you", "your",
}

_CONTROL_TOKENS = {
    "coordinate", "coordinating", "coordination", "decompose", "decomposer",
    "decomposition", "judge", "orchestrate", "orchestration",
    "orchestrator", "plan", "planning", "review", "reviewer", "route",
    "router", "routing", "spec", "specify", "specification", "triage",
}

_EXECUTION_TOKENS = {
    "analyze", "audit", "build", "code", "create", "debug", "deploy",
    "design", "draft", "fix", "implement", "implementation", "install",
    "investigate", "patch", "port", "refactor", "repair", "research",
    "run", "ship", "test", "testing", "write",
}

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_\-/]{1,}")


@dataclass
class RouteCandidate:
    name: str
    score: float
    shared_terms: list[str]
    description: str = ""
    description_auto: bool = False


@dataclass
class RouteWatchdogDecision:
    kind: str
    reason: str
    detail: str
    suggested_assignee: Optional[str]
    current_assignee_score: float
    suggested_score: float
    top_candidates: list[RouteCandidate]

    def review_reason(self) -> str:
        if self.kind == "stranded_orchestrator":
            target = self.suggested_assignee or "another specialist"
            return (
                "review-required: route-watchdog stranded_orchestrator — "
                f"task looks like execution work and matches {target} better "
                "than the orchestrator; review before dispatch"
            )
        if self.kind == "brand_mismatch":
            target = self.suggested_assignee or "another profile"
            return (
                "review-required: route-watchdog brand_mismatch — "
                f"task signals match {target} materially better than the "
                "current assignee; review before dispatch"
            )
        return (
            "review-required: route-watchdog low_confidence — task routing "
            "signal is ambiguous or too weak; review before dispatch"
        )

    def comment_body(self) -> str:
        lines = [
            "route-watchdog hold",
            f"kind: {self.kind}",
            f"reason: {self.reason}",
        ]
        if self.detail:
            lines.append("")
            lines.append(self.detail)
        if self.top_candidates:
            lines.append("")
            lines.append("top candidates:")
            for cand in self.top_candidates[:3]:
                shared = ", ".join(cand.shared_terms[:6]) or "none"
                auto = " auto-description" if cand.description_auto else ""
                lines.append(
                    f"- {cand.name}: score={cand.score:.2f}; shared={shared}{auto}"
                )
        return "\n".join(lines)


@dataclass
class RouteWatchdogConfig:
    mode: str = "report"
    orchestrator_profile: str = ""
    default_assignee: str = ""
    min_score: float = 0.38
    min_margin: float = 0.14

    @property
    def enabled(self) -> bool:
        return self.mode in {"report", "hold"}


class _ProfileLike:
    name: str
    description: str
    description_auto: bool


def load_watchdog_config(
    raw: Optional[dict],
    *,
    orchestrator_profile: Optional[str] = None,
    default_assignee: Optional[str] = None,
) -> RouteWatchdogConfig:
    raw = raw or {}
    mode = str(raw.get("mode") or "report").strip().lower()
    if mode not in {"off", "report", "hold"}:
        mode = "report"
    try:
        min_score = float(raw.get("min_score", 0.38))
    except Exception:
        min_score = 0.38
    try:
        min_margin = float(raw.get("min_margin", 0.14))
    except Exception:
        min_margin = 0.14
    return RouteWatchdogConfig(
        mode=mode,
        orchestrator_profile=(orchestrator_profile or raw.get("orchestrator_profile") or "").strip(),
        default_assignee=(default_assignee or raw.get("default_assignee") or "").strip(),
        min_score=max(0.05, min(0.95, min_score)),
        min_margin=max(0.02, min(0.5, min_margin)),
    )


def _profile_name(profile: Any) -> str:
    if isinstance(profile, dict):
        return str(profile.get("name") or "").strip()
    return str(getattr(profile, "name", "") or "").strip()


def _profile_description(profile: Any) -> str:
    if isinstance(profile, dict):
        return str(profile.get("description") or "").strip()
    return str(getattr(profile, "description", "") or "").strip()


def _profile_description_auto(profile: Any) -> bool:
    if isinstance(profile, dict):
        return bool(profile.get("description_auto", False))
    return bool(getattr(profile, "description_auto", False))


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens: list[str] = []
    for raw in _WORD_RE.findall(text.lower()):
        parts = re.split(r"[_\-/]", raw)
        for token in parts:
            token = token.strip()
            if len(token) < 3 or token in _STOPWORDS:
                continue
            tokens.append(token)
    return tokens


def _task_text(task: Any, board_meta: Optional[dict]) -> str:
    title = str(getattr(task, "title", None) if not isinstance(task, dict) else task.get("title") or "")
    body = str(getattr(task, "body", None) if not isinstance(task, dict) else task.get("body") or "")
    parts = [title, title, title, body]
    if isinstance(board_meta, dict):
        name = str(board_meta.get("name") or "")
        slug = str(board_meta.get("slug") or "")
        desc = str(board_meta.get("description") or "")
        parts.extend([name, slug, desc])
    return "\n".join(p for p in parts if p)


def _is_execution_like(tokens: set[str]) -> bool:
    if tokens & _EXECUTION_TOKENS:
        return True
    if tokens & _CONTROL_TOKENS:
        return False
    return True


def _score_candidate(task_counts: Counter[str], profile_name: str, profile_description: str, *, description_auto: bool) -> RouteCandidate:
    profile_tokens = _tokenize(f"{profile_name} {profile_name} {profile_description}")
    if not profile_tokens:
        return RouteCandidate(name=profile_name, score=0.0, shared_terms=[])
    profile_counts = Counter(profile_tokens)
    shared = sorted(set(task_counts.keys()) & set(profile_counts.keys()))
    if not shared:
        return RouteCandidate(
            name=profile_name,
            score=0.0,
            shared_terms=[],
            description=profile_description,
            description_auto=description_auto,
        )
    overlap_weight = float(sum(min(task_counts[t], profile_counts[t]) for t in shared))
    task_mass = float(sum(task_counts.values()) or 1.0)
    profile_mass = float(sum(profile_counts.values()) or 1.0)
    coverage = overlap_weight / task_mass
    specificity = overlap_weight / profile_mass
    score = coverage * 0.72 + specificity * 0.28
    if description_auto:
        score *= 0.92
    return RouteCandidate(
        name=profile_name,
        score=round(score, 4),
        shared_terms=shared,
        description=profile_description,
        description_auto=description_auto,
    )


def _load_roster() -> list[Any]:
    from hermes_cli import profiles as profiles_mod

    try:
        roster = profiles_mod.list_profiles()
    except Exception:
        return []
    return [p for p in roster if _profile_name(p)]


def evaluate_route(
    task: Any,
    *,
    config: RouteWatchdogConfig,
    board_meta: Optional[dict] = None,
    roster: Optional[list[Any]] = None,
    child_ids: Optional[list[str]] = None,
) -> Optional[RouteWatchdogDecision]:
    """Return a review-required routing decision, or ``None``.

    This is intentionally fail-open. Missing roster metadata or a no-signal task
    simply yields ``None``.
    """
    if not config.enabled:
        return None

    assignee = str((task.get("assignee") if isinstance(task, dict) else getattr(task, "assignee", None)) or "").strip()
    if not assignee:
        return None

    task_tokens_list = _tokenize(_task_text(task, board_meta))
    if not task_tokens_list:
        return RouteWatchdogDecision(
            kind="low_confidence",
            reason="task has almost no routing signal in title/body/board metadata",
            detail="No meaningful lexical signal was available to verify the current route. Hold for explicit review rather than guessing.",
            suggested_assignee=None,
            current_assignee_score=0.0,
            suggested_score=0.0,
            top_candidates=[],
        )
    task_counts = Counter(task_tokens_list)
    task_token_set = set(task_counts.keys())
    execution_like = _is_execution_like(task_token_set)

    roster = roster if roster is not None else _load_roster()
    if not roster:
        return None

    candidates: list[RouteCandidate] = []
    for profile in roster:
        name = _profile_name(profile)
        if not name:
            continue
        candidates.append(
            _score_candidate(
                task_counts,
                name,
                _profile_description(profile),
                description_auto=_profile_description_auto(profile),
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c.score, c.name))
    current = next((c for c in candidates if c.name == assignee), RouteCandidate(name=assignee, score=0.0, shared_terms=[]))
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    best_non_orch = next((c for c in candidates if c.name != config.orchestrator_profile), None)

    if (
        config.orchestrator_profile
        and assignee == config.orchestrator_profile
        and execution_like
        and not (child_ids or [])
        and best_non_orch is not None
        and best_non_orch.score >= config.min_score
        and (best_non_orch.score - current.score) >= config.min_margin
    ):
        return RouteWatchdogDecision(
            kind="stranded_orchestrator",
            reason=(
                f"execution-shaped task is parked on orchestrator {assignee} even though "
                f"{best_non_orch.name} is a materially better lexical match"
            ),
            detail=(
                f"Current assignee {assignee} scored {current.score:.2f}; best specialist "
                f"match {best_non_orch.name} scored {best_non_orch.score:.2f}. The task has "
                "no child dependencies yet, so leaving it on the orchestrator would likely "
                "waste a dispatch cycle instead of decomposing further."
            ),
            suggested_assignee=best_non_orch.name,
            current_assignee_score=current.score,
            suggested_score=best_non_orch.score,
            top_candidates=candidates[:3],
        )

    if (
        best.name != assignee
        and best.score >= config.min_score
        and (best.score - current.score) >= config.min_margin
    ):
        return RouteWatchdogDecision(
            kind="brand_mismatch",
            reason=(
                f"task signals align with {best.name} much more strongly than the current "
                f"assignee {assignee}"
            ),
            detail=(
                f"Current assignee {assignee} scored {current.score:.2f}; best match {best.name} "
                f"scored {best.score:.2f}. Shared task/profile terms indicate a likely routing "
                "mismatch. Hold for review instead of silently reassigning."
            ),
            suggested_assignee=best.name,
            current_assignee_score=current.score,
            suggested_score=best.score,
            top_candidates=candidates[:3],
        )

    top_margin = best.score - (second.score if second is not None else 0.0)
    if best.score < config.min_score or (second is not None and top_margin < config.min_margin):
        ambiguity = (
            f"best score {best.score:.2f} is below threshold {config.min_score:.2f}"
            if best.score < config.min_score
            else f"top-two margin {top_margin:.2f} is below threshold {config.min_margin:.2f}"
        )
        detail = (
            f"Current assignee {assignee} scored {current.score:.2f}. Best candidate "
            f"{best.name} scored {best.score:.2f}"
        )
        if second is not None:
            detail += f" and runner-up {second.name} scored {second.score:.2f}"
        detail += ". The route signal is ambiguous/weak, so the safer action is a review hold."
        return RouteWatchdogDecision(
            kind="low_confidence",
            reason=ambiguity,
            detail=detail,
            suggested_assignee=(best.name if best.score >= current.score else None),
            current_assignee_score=current.score,
            suggested_score=best.score,
            top_candidates=candidates[:3],
        )

    return None
