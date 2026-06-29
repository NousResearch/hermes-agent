"""Buidl Goal OS v1 durable goal orchestration.

Goal OS is a local, provider-free orchestration layer for Clio. It stores goal
contracts, task cards, role ownership, blockers and verifier evidence so Clio
can keep working from durable state without making Niko the operator.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

GoalStatus = Literal["pursuing", "paused", "blocked", "achieved", "failed", "budget_limited", "cancelled"]
CardStatus = Literal["backlog", "ready", "in_progress", "review", "verification", "blocked", "done", "cancelled"]
ReportLabel = Literal["GREEN", "RED", "NOISE"]
ControlledProviderStatus = Literal["PROVIDER_SAFE_READY", "CONTROLLED_SETUP_READY", "HUMAN_BLIND_PROMPT_REQUIRED", "PROMPT_RUN_REPORTED", "COUNTERS_VERIFIED", "PROVIDER_SAFE_RESTORED", "CHECKPOINT_ACCEPTED"]

BLIND_PROMPT_PLACEHOLDER = "BLIND_PROMPT_CHOSEN_BY_NIKO_OR_STEVE_AT_TEST_TIME"

AGENT_ROLES = (
    "Clio Orchestrator",
    "Builder Agent",
    "Reviewer Agent",
    "Verifier Agent",
    "Claude Code Reviewer Agent",
    "Codex Verifier Agent",
    "Ops Agent",
    "Product QA Agent",
    "Memory Agent",
)

STATUSES = {"backlog", "ready", "in_progress", "review", "verification", "blocked", "done", "cancelled"}
GOAL_STATUSES = {"pursuing", "paused", "blocked", "achieved", "failed", "budget_limited", "cancelled"}

STANDING_APPROVAL = (
    "inspect files",
    "implement code",
    "add tests",
    "run focused tests",
    "run full tests",
    "run typecheck",
    "run lint",
    "run build",
    "run safety scans",
    "commit scoped changes",
    "push feature branches",
    "create draft PRs",
    "monitor checks",
    "fix failing checks",
    "update Obsidian",
)

HARD_APPROVAL_GATES = (
    "real provider calls",
    "real prompt execution",
    "image generation",
    "production deploy",
    "DNS changes",
    "DB migrations",
    "billing",
    "credits",
    "payments",
    "secrets",
    "provider credentials",
    "worker enablement",
    "merge to main",
)

FORBIDDEN_ACTIONS = HARD_APPROVAL_GATES + (
    "Niko is not the terminal operator",
    "Niko is not the sudo operator",
    "Niko is not the Docker operator",
    "Niko is not the GHCR operator",
    "Niko is not the GitHub token operator",
    "Niko is not the server debugging operator",
    "Niko is not the env file editing operator",
    "Niko is not the credentials operator",
    "Niko is not the provider-credential operator",
    "storing live blind prompts",
    "printing live blind prompts",
)

DEFAULT_VERIFICATION_COMMANDS = (
    "focused Goal OS tests",
    "full agent-server tests",
    "typecheck",
    "lint",
    "build",
    "secret scan",
    "git diff --check",
)

_HARD_GATE_ALIASES = {
    "provider": "real provider calls",
    "providers": "real provider calls",
    "prompt execution": "real prompt execution",
    "run prompt": "real prompt execution",
    "production": "production deploy",
    "deploy production": "production deploy",
    "dns": "DNS changes",
    "database migration": "DB migrations",
    "db migration": "DB migrations",
    "migration": "DB migrations",
    "credit": "credits",
    "payment": "payments",
    "secret": "secrets",
    "credential": "provider credentials",
    "worker": "worker enablement",
    "merge main": "merge to main",
}

_LIVE_PROMPT_MARKERS = (
    "known live prompt",
    "live blind prompt",
    "secret phrase",
)


@dataclass
class TaskCard:
    card_id: str
    goal_id: str
    title: str
    owner_role: str
    status: CardStatus = "backlog"
    branch: str = ""
    files_expected: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)
    blocker_reason: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    skill_packs: list[str] = field(default_factory=list)
    next_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskCard":
        return cls(
            card_id=str(data.get("card_id") or _new_id("card")),
            goal_id=str(data.get("goal_id") or ""),
            title=str(data.get("title") or "Untitled card"),
            owner_role=str(data.get("owner_role") or "Clio Orchestrator"),
            status=_valid_card_status(data.get("status")),
            branch=str(data.get("branch") or ""),
            files_expected=[str(x) for x in data.get("files_expected") or []],
            acceptance_criteria=[str(x) for x in data.get("acceptance_criteria") or []],
            verification_commands=[str(x) for x in data.get("verification_commands") or []],
            blocker_reason=str(data.get("blocker_reason") or ""),
            evidence=list(data.get("evidence") or []),
            skill_packs=[str(x) for x in data.get("skill_packs") or []],
            next_action=str(data.get("next_action") or ""),
        )


@dataclass
class GoalContract:
    goal_id: str
    title: str
    business_outcome: str
    target_repo: str
    target_branch: str
    target_environment: str
    allowed_actions: list[str]
    forbidden_actions: list[str]
    approval_gates: list[str]
    acceptance_criteria: list[str]
    verification_commands: list[str]
    stop_conditions: list[str]
    rollback_plan: str
    cards: list[TaskCard]
    blockers: list[dict[str, Any]] = field(default_factory=list)
    status: GoalStatus = "pursuing"
    next_action: str = ""
    evidence_log: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cards"] = [card.to_dict() for card in self.cards]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoalContract":
        cards = [TaskCard.from_dict(card) for card in data.get("cards") or []]
        return cls(
            goal_id=str(data.get("goal_id") or _new_id("goal")),
            title=str(data.get("title") or "Untitled goal"),
            business_outcome=str(data.get("business_outcome") or "Move Buidl 2.0 toward MVP."),
            target_repo=str(data.get("target_repo") or ""),
            target_branch=str(data.get("target_branch") or ""),
            target_environment=str(data.get("target_environment") or "agent-server only"),
            allowed_actions=[str(x) for x in data.get("allowed_actions") or STANDING_APPROVAL],
            forbidden_actions=[str(x) for x in data.get("forbidden_actions") or FORBIDDEN_ACTIONS],
            approval_gates=[str(x) for x in data.get("approval_gates") or HARD_APPROVAL_GATES],
            acceptance_criteria=[str(x) for x in data.get("acceptance_criteria") or []],
            verification_commands=[str(x) for x in data.get("verification_commands") or DEFAULT_VERIFICATION_COMMANDS],
            stop_conditions=[str(x) for x in data.get("stop_conditions") or []],
            rollback_plan=str(data.get("rollback_plan") or "Revert scoped branch changes or close draft PR. No deployment rollback is in scope."),
            cards=cards,
            blockers=list(data.get("blockers") or []),
            status=_valid_goal_status(data.get("status")),
            next_action=str(data.get("next_action") or ""),
            evidence_log=list(data.get("evidence_log") or []),
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
        )


@dataclass
class GoalOSReport:
    classification: ReportLabel
    message: str
    goal: GoalContract | None = None


@dataclass
class GreenReadinessVerdict:
    classification: ReportLabel
    reason: str


CONTROLLED_PROVIDER_STATUSES: tuple[ControlledProviderStatus, ...] = (
    "PROVIDER_SAFE_READY",
    "CONTROLLED_SETUP_READY",
    "HUMAN_BLIND_PROMPT_REQUIRED",
    "PROMPT_RUN_REPORTED",
    "COUNTERS_VERIFIED",
    "PROVIDER_SAFE_RESTORED",
    "CHECKPOINT_ACCEPTED",
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _valid_card_status(value: Any) -> CardStatus:
    text = str(value or "backlog")
    return text if text in STATUSES else "backlog"  # type: ignore[return-value]


def _valid_goal_status(value: Any) -> GoalStatus:
    text = str(value or "pursuing")
    return text if text in GOAL_STATUSES else "pursuing"  # type: ignore[return-value]


def _hermes_home() -> Path:
    raw = os.environ.get("HERMES_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except Exception:
        return Path.home() / ".hermes"


def _default_store_path() -> Path:
    return _hermes_home() / "goal-os" / "goals.json"


def sanitize_blind_prompt_text(text: str) -> str:
    """Replace likely live blind-prompt material with the approved placeholder."""
    cleaned = str(text or "")
    lower = cleaned.lower()
    if any(marker in lower for marker in _LIVE_PROMPT_MARKERS):
        return BLIND_PROMPT_PLACEHOLDER
    return cleaned


def _is_negated_gate_mention(lower: str, start: int) -> bool:
    """Return True when a hard-gate term is listed as a safety constraint.

    Goal text often says things like "without running providers, prompts,
    production or DNS". Those are boundaries, not requests to execute hard-gate
    work. Keep direct requests blocked, but do not convert negative safety
    constraints into false RED blockers.
    """
    prefix = lower[max(0, start - 180):start]
    return any(
        marker in prefix
        for marker in (
            "without ",
            "without running ",
            "without touching ",
            "no ",
            "not ",
            "do not ",
            "don't ",
            "never ",
        )
    )


def _contains_non_negated(lower: str, needle: str) -> bool:
    index = lower.find(needle)
    while index != -1:
        if not _is_negated_gate_mention(lower, index):
            return True
        index = lower.find(needle, index + len(needle))
    return False


def is_hard_gate(text: str) -> bool:
    lower = str(text or "").lower()
    gates = [gate.lower() for gate in HARD_APPROVAL_GATES] + list(_HARD_GATE_ALIASES)
    return any(_contains_non_negated(lower, gate) for gate in gates)


def hard_gates_in_text(text: str) -> list[str]:
    lower = str(text or "").lower()
    found: list[str] = []
    for gate in HARD_APPROVAL_GATES:
        if _contains_non_negated(lower, gate.lower()):
            found.append(gate)
    for alias, gate in _HARD_GATE_ALIASES.items():
        if _contains_non_negated(lower, alias) and gate not in found:
            found.append(gate)
    return found


def classify_report(text: str) -> ReportLabel:
    lower = str(text or "").lower()
    if any(term in lower for term in ("blocked", "approval", "hard gate", "cannot", "refuse", "missing verifier", "production deploy")):
        return "RED"
    if any(term in lower for term in ("harmless", "non-blocking", "noise", "wording difference")):
        return "NOISE"
    return "GREEN"


def _is_ui_or_browser_work(goal: GoalContract, card: TaskCard | None = None) -> bool:
    haystack = " ".join([
        goal.title,
        goal.business_outcome,
        *(goal.acceptance_criteria or []),
        *((card.acceptance_criteria if card else []) or []),
        card.title if card else "",
    ]).lower()
    return bool(re.search(r"\bui\b", haystack)) or any(marker in haystack for marker in ("browser", "product qa", "design qa", "design quality", "generated website", "live preview"))


def _has_pending_approval_blocker() -> bool:
    try:
        from tools.approval import has_pending_approval_blockers
        return has_pending_approval_blockers()
    except Exception:
        return False


def _has_true_blocker(goal: GoalContract) -> bool:
    if any(bool(blocker.get("true_blocker")) for blocker in goal.blockers):
        return True
    return _has_pending_approval_blocker()


def _has_contradiction(goal: GoalContract) -> bool:
    for item in goal.evidence_log:
        if item.get("contradiction") is True:
            return True
        summary = str(item.get("summary", "")).lower()
        if "browser still shows" in summary and "setup" in " ".join(str(x.get("summary", "")) for x in goal.evidence_log).lower():
            return True
        if "contradict" in summary:
            return True
    return False


def _verifier_evidence_items(goal: GoalContract) -> list[dict[str, Any]]:
    return [item for item in goal.evidence_log if str(item.get("role", "")).lower() == "verifier agent"]


def _has_verifier_evidence(goal: GoalContract) -> bool:
    for item in _verifier_evidence_items(goal):
        if item.get("acceptance_checked") is True and (item.get("commands") or item.get("deploy_evidence") or item.get("browser_evidence") or item.get("summary")):
            return True
    return False


def _has_product_qa_evidence(goal: GoalContract) -> bool:
    return any(str(item.get("role", "")).lower() == "product qa agent" or item.get("product_qa") for item in goal.evidence_log)


def _has_design_qa_evidence(goal: GoalContract) -> bool:
    return any(str(item.get("role", "")).lower() == "design qa agent" or item.get("design_qa") for item in goal.evidence_log)


def _has_memory_evidence(goal: GoalContract) -> bool:
    return any(str(item.get("role", "")).lower() == "memory agent" or item.get("memory_agent") for item in goal.evidence_log)


def _is_buidl_goal(goal: GoalContract) -> bool:
    haystack = " ".join([goal.title, goal.business_outcome, *(goal.acceptance_criteria or [])]).lower()
    return "buidl" in haystack


def _is_buidl_mvp_acceptance_goal(goal: GoalContract) -> bool:
    haystack = " ".join([goal.title, goal.business_outcome, *(goal.acceptance_criteria or [])]).lower()
    return "buidl" in haystack and any(
        term in haystack
        for term in ("mvp", "release-candidate", "release candidate", "acceptance", "green", "working demo", "visible demo")
    )


def _codex_verifier_evidence_indices(goal: GoalContract) -> list[int]:
    indices: list[int] = []
    for index, item in enumerate(goal.evidence_log):
        role = str(item.get("role", "")).lower()
        executor = str(item.get("executor", item.get("agent", ""))).lower()
        is_codex_executor = executor in {"openai-codex", "codex"}
        if not (
            (item.get("codex_verifier_pass") is True and is_codex_executor)
            or ("verifier" in role and is_codex_executor)
        ):
            continue
        if item.get("pass") is False or item.get("result") == "fail":
            continue
        if item.get("secrets_printed") is True:
            continue
        if not item.get("commands"):
            continue
        if not (item.get("files_inspected") or item.get("changed_files")):
            continue
        indices.append(index)
    return indices


def _has_codex_verifier_evidence(goal: GoalContract) -> bool:
    return bool(_codex_verifier_evidence_indices(goal))


def _claude_code_builder_evidence_indices(goal: GoalContract) -> list[int]:
    indices: list[int] = []
    for index, item in enumerate(goal.evidence_log):
        role = str(item.get("role", "")).lower()
        executor = str(item.get("executor", item.get("agent", ""))).lower()
        is_claude_code_executor = executor in {"claude-code", "claude_code", "claude code"}
        if not (
            (item.get("claude_code_builder_pass") is True and is_claude_code_executor)
            or ("builder" in role and is_claude_code_executor)
        ):
            continue
        if item.get("pass") is False or item.get("result") == "fail":
            continue
        if item.get("secrets_printed") is True:
            continue
        if item.get("read_only_smoke") is True or item.get("read_only") is True:
            continue
        if not (item.get("changed_files") or item.get("files_changed")):
            continue
        if not (item.get("commands") or item.get("builder_evidence")):
            continue
        indices.append(index)
    return indices


def _codex_reviewer_evidence_indices(goal: GoalContract) -> list[int]:
    indices: list[int] = []
    for index, item in enumerate(goal.evidence_log):
        role = str(item.get("role", "")).lower()
        executor = str(item.get("executor", item.get("agent", ""))).lower()
        is_codex_executor = executor in {"openai-codex", "codex"}
        if not (
            (item.get("codex_reviewer_pass") is True and is_codex_executor)
            or ("review" in role and is_codex_executor)
        ):
            continue
        if item.get("pass") is False or item.get("result") == "fail":
            continue
        if item.get("secrets_printed") is True:
            continue
        if not item.get("files_inspected"):
            continue
        if not (item.get("independent_findings") or item.get("safety_review") or item.get("review_summary")):
            continue
        indices.append(index)
    return indices


def _claude_code_reviewer_evidence_indices(goal: GoalContract) -> list[int]:
    """Backward-compatible read-only Claude Code review lane for older evidence."""
    indices: list[int] = []
    for index, item in enumerate(goal.evidence_log):
        role = str(item.get("role", "")).lower()
        executor = str(item.get("executor", item.get("agent", ""))).lower()
        if not (item.get("claude_code_reviewer_pass") is True or "claude code" in role or executor in {"claude-code", "claude_code", "claude code"}):
            continue
        if item.get("pass") is False or item.get("result") == "fail":
            continue
        if not item.get("files_inspected"):
            continue
        if not (item.get("independent_findings") or item.get("safety_review") or item.get("review_summary")):
            continue
        if item.get("secrets_printed") is True:
            continue
        indices.append(index)
    return indices


def _has_claude_code_reviewer_evidence(goal: GoalContract) -> bool:
    return bool(_claude_code_reviewer_evidence_indices(goal))


def buidl_dual_review_gate_verdict(goal: GoalContract) -> GreenReadinessVerdict:
    if not _is_buidl_goal(goal):
        return GreenReadinessVerdict("GREEN", "Buidl lane gate not required for this goal.")
    missing: list[str] = []
    claude_builder_indices = _claude_code_builder_evidence_indices(goal)
    codex_reviewer_indices = _codex_reviewer_evidence_indices(goal)
    codex_verifier_indices = _codex_verifier_evidence_indices(goal)
    if not claude_builder_indices:
        missing.append("CLAUDE_CODE_BUILDER_PASS")
    if not codex_reviewer_indices:
        missing.append("CODEX_REVIEWER_PASS")
    if not codex_verifier_indices:
        missing.append("CODEX_VERIFIER_PASS")
    if missing:
        if "CLAUDE_CODE_BUILDER_PASS" in missing:
            return GreenReadinessVerdict("RED", "RED: CLAUDE_CODE_BUILDER_NOT_USED. Buidl GREEN requires " + ", ".join(missing) + ".")
        if "CODEX_REVIEWER_PASS" in missing:
            return GreenReadinessVerdict("RED", "RED: CODEX_REVIEW_OR_VERIFY_MISSING. Buidl GREEN requires " + ", ".join(missing) + ".")
        return GreenReadinessVerdict("RED", "RED: CODEX_VERIFIER_NOT_ACTIVE. Buidl GREEN requires " + ", ".join(missing) + ".")
    if not set(claude_builder_indices).isdisjoint(codex_reviewer_indices):
        return GreenReadinessVerdict("RED", "RED: DUAL_LANE_NOT_INDEPENDENT. Claude Code Builder and Codex Reviewer evidence must be separate entries.")
    if not set(claude_builder_indices).isdisjoint(codex_verifier_indices):
        return GreenReadinessVerdict("RED", "RED: DUAL_LANE_NOT_INDEPENDENT. Claude Code Builder and Codex Verifier evidence must be separate entries.")
    if not set(codex_reviewer_indices).isdisjoint(codex_verifier_indices):
        return GreenReadinessVerdict("RED", "RED: CODEX_REVIEWER_VERIFIER_NOT_SEPARATE. Codex Reviewer and Codex Verifier evidence must be separate entries.")
    return GreenReadinessVerdict("GREEN", "CLAUDE_CODE_BUILDER_PASS, CODEX_REVIEWER_PASS and CODEX_VERIFIER_PASS are present.")


def _has_browser_or_human_evidence(goal: GoalContract) -> bool:
    return any(item.get("browser_evidence") or item.get("human_browser_report") for item in goal.evidence_log)


def evaluate_goal_report_readiness(goal: GoalContract, *, requested_label: str) -> GreenReadinessVerdict:
    label = str(requested_label or "").upper()
    if label != "GREEN":
        return GreenReadinessVerdict(classify_report(label), "GREEN guard not requested.")
    if _has_pending_approval_blocker():
        return GreenReadinessVerdict("RED", "Unresolved approval request exists.")
    if _has_true_blocker(goal):
        return GreenReadinessVerdict("RED", "Unresolved RED blocker exists.")
    if _has_contradiction(goal):
        return GreenReadinessVerdict("RED", "Verifier evidence has a contradiction between browser state and setup state.")
    if not _has_verifier_evidence(goal):
        return GreenReadinessVerdict("RED", "GREEN requires verifier evidence from the Verifier Agent with acceptance criteria checked.")
    required_missing = []
    if _is_buidl_goal(goal):
        if not _has_product_qa_evidence(goal):
            required_missing.append("Product QA evidence")
        if not _has_memory_evidence(goal):
            required_missing.append("Memory Agent evidence")
    if required_missing:
        return GreenReadinessVerdict("RED", "GREEN for Buidl work requires " + ", ".join(required_missing) + ".")
    dual_review_verdict = buidl_dual_review_gate_verdict(goal)
    if dual_review_verdict.classification != "GREEN":
        return dual_review_verdict
    if _is_ui_or_browser_work(goal):
        missing = []
        if not _has_product_qa_evidence(goal):
            missing.append("Product QA evidence")
        if not _has_design_qa_evidence(goal):
            missing.append("Design QA evidence")
        if "browser" in goal.title.lower() and not _has_browser_or_human_evidence(goal):
            missing.append("browser or explicit human browser evidence")
        if missing:
            return GreenReadinessVerdict("RED", "GREEN for UI/browser work requires " + ", ".join(missing) + ".")
    return GreenReadinessVerdict("GREEN", "Verifier evidence satisfies GREEN guard.")


def controlled_provider_status_report(goal: GoalContract, statuses: list[str]) -> GoalOSReport:
    normalized = [str(status).strip().upper() for status in statuses if str(status).strip()]
    valid = [status for status in normalized if status in CONTROLLED_PROVIDER_STATUSES]
    if "CHECKPOINT_ACCEPTED" in valid:
        accepted_index = valid.index("CHECKPOINT_ACCEPTED")
        missing_prior = [status for status in CONTROLLED_PROVIDER_STATUSES[:accepted_index] if status not in valid]
        if missing_prior:
            return GoalOSReport("RED", "RED: controlled provider workflow may not skip from setup to CHECKPOINT_ACCEPTED. Missing: " + ", ".join(missing_prior), goal)
    if valid == ["CONTROLLED_SETUP_READY"] or ("CONTROLLED_SETUP_READY" in valid and "PROMPT_RUN_REPORTED" not in valid):
        return GoalOSReport("NOISE", "NOISE: SETUP_READY. READY_FOR_BROWSER_TESTING=yes. HUMAN_BLIND_PROMPT_REQUIRED=yes. Product checkpoint has not passed.", goal)
    if "CHECKPOINT_ACCEPTED" in valid:
        return GoalOSReport("GREEN", "GREEN: CHECKPOINT_ACCEPTED with ordered provider status evidence.", goal)
    return GoalOSReport("NOISE", "NOISE: controlled provider status: " + ", ".join(valid or ["unknown"]), goal)


class GoalOSManager:
    """Durable local manager for Buidl Goal OS contracts and task cards."""

    def __init__(self, store_path: Path | None = None):
        self.store_path = store_path or _default_store_path()

    def _read(self) -> dict[str, Any]:
        if not self.store_path.exists():
            return {"version": 1, "goals": {}, "active_goal_id": None}
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "goals": {}, "active_goal_id": None}
        if not isinstance(data, dict):
            return {"version": 1, "goals": {}, "active_goal_id": None}
        data.setdefault("version", 1)
        data.setdefault("goals", {})
        data.setdefault("active_goal_id", None)
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.store_path)

    def list_goals(self) -> list[GoalContract]:
        data = self._read()
        goals = [GoalContract.from_dict(item) for item in (data.get("goals") or {}).values()]
        return sorted(goals, key=lambda g: g.updated_at, reverse=True)

    def get_goal(self, goal_id: str | None = None) -> GoalContract | None:
        data = self._read()
        gid = goal_id or data.get("active_goal_id")
        if not gid:
            return None
        raw = (data.get("goals") or {}).get(gid)
        return GoalContract.from_dict(raw) if isinstance(raw, dict) else None

    def save_goal(self, goal: GoalContract, *, make_active: bool = True) -> None:
        data = self._read()
        goal.updated_at = time.time()
        data["goals"][goal.goal_id] = goal.to_dict()
        if make_active:
            data["active_goal_id"] = goal.goal_id
        self._write(data)

    def create_goal(
        self,
        title: str,
        *,
        business_outcome: str | None = None,
        target_repo: str = "",
        target_branch: str = "",
        target_environment: str = "agent-server only",
    ) -> GoalContract:
        clean_title = sanitize_blind_prompt_text(title).strip() or "Untitled Buidl Goal OS goal"
        goal_id = _new_id("goal")
        gates = hard_gates_in_text(title)
        status: GoalStatus = "blocked" if gates else "pursuing"
        blockers = []
        if gates:
            blockers.append({
                "reason": "Hard approval gate detected: " + ", ".join(gates),
                "true_blocker": True,
                "created_at": time.time(),
            })
        cards = self._build_default_cards(goal_id, target_branch)
        goal = GoalContract(
            goal_id=goal_id,
            title=clean_title,
            business_outcome=business_outcome or "Move Buidl 2.0 to MVP through repo-safe, staging-safe execution without making Niko the operator.",
            target_repo=target_repo,
            target_branch=target_branch,
            target_environment=target_environment,
            allowed_actions=list(STANDING_APPROVAL),
            forbidden_actions=list(FORBIDDEN_ACTIONS),
            approval_gates=list(HARD_APPROVAL_GATES),
            acceptance_criteria=[
                "Goal contract exists durably.",
                "Task cards are assigned to logical specialist roles.",
                "Hard gates stop for Niko approval.",
                "Verifier evidence is required before Done or ship.",
            ],
            verification_commands=list(DEFAULT_VERIFICATION_COMMANDS),
            stop_conditions=[
                "Hard approval gate is reached.",
                "True blocker requires a decision or external fix.",
                "Verifier evidence contradicts builder self-report.",
                "Budget or safety boundary is reached.",
            ],
            rollback_plan="Use feature branch rollback or revert scoped commits. No staging or production deployment is performed by Goal OS v1.",
            cards=cards,
            blockers=blockers,
            status=status,
            next_action="Start the first ready Builder Agent card." if status == "pursuing" else "Ask Niko for explicit hard-gate approval before proceeding.",
            evidence_log=[{"role": "Clio Orchestrator", "type": "creation", "gate": "", "summary": "Goal contract created", "at": time.time()}],
        )
        self.save_goal(goal)
        return goal

    def _skill_packs_for_role(self, role: str, title: str) -> list[str]:
        from hermes_cli.capabilities.skills import load_skill_registry

        registry = load_skill_registry()
        probes = (
            ("Buidl generated website design quality", role),
            ("security safety", role),
            ("verification test build", role),
            ("lesson memory", role),
            ("Buidl agentic build", role),
        )
        packs: list[str] = []
        role_lower = role.lower()
        title_lower = title.lower()
        for goal_type, agent_role in probes:
            plan = registry.activate_pack_for_goal(goal_type, agent_role=agent_role)
            if plan.skills and plan.pack_name not in packs:
                if plan.pack_name == "Design Quality Pack" and role_lower not in {"product qa agent", "reviewer agent", "code reviewer agent", "verifier agent"}:
                    continue
                if plan.pack_name == "Security and Safety Pack" and "security" not in title_lower and "ops" not in role_lower:
                    continue
                if plan.pack_name == "Memory and Learning Pack" and "memory" not in role_lower:
                    continue
                packs.append(plan.pack_name)
        return packs

    def _build_default_cards(self, goal_id: str, branch: str) -> list[TaskCard]:
        specs = [
            ("Clarify durable goal contract and task decomposition", "Clio Orchestrator", "ready"),
            ("Implement scoped repo changes", "Builder Agent", "ready"),
            ("Review diff, safety gates and product fit", "Reviewer Agent", "backlog"),
            ("Run verification commands and collect evidence", "Verifier Agent", "backlog"),
            ("Check deployment, secrets and operations boundaries", "Ops Agent", "backlog"),
            ("Validate MVP product behavior", "Product QA Agent", "backlog"),
            ("Update durable notes and memory surfaces", "Memory Agent", "backlog"),
        ]
        return [
            TaskCard(
                card_id=_new_id("card"),
                goal_id=goal_id,
                title=title,
                owner_role=role,
                status=status,  # type: ignore[arg-type]
                branch=branch,
                acceptance_criteria=["Complete the role-specific card with evidence."],
                verification_commands=list(DEFAULT_VERIFICATION_COMMANDS) if role == "Verifier Agent" else [],
                skill_packs=self._skill_packs_for_role(role, title),
                next_action=f"{role} takes the next safe step.",
            )
            for title, role, status in specs
        ]

    def add_blocker(self, goal_id: str, reason: str, *, true_blocker: bool) -> None:
        goal = self.get_goal(goal_id)
        if goal is None:
            raise KeyError(goal_id)
        goal.blockers.append({"reason": reason, "true_blocker": bool(true_blocker), "created_at": time.time()})
        if true_blocker:
            goal.status = "blocked"
            goal.next_action = "Resolve true blocker or get explicit approval."
        self.save_goal(goal)

    def close_card(self, card_id: str, *, actor_role: str, evidence: dict[str, Any]) -> GoalOSReport:
        goal = None
        card = None
        for candidate in self.list_goals():
            for candidate_card in candidate.cards:
                if candidate_card.card_id == card_id:
                    goal = candidate
                    card = candidate_card
                    break
            if goal is not None:
                break
        if goal is None or card is None:
            return GoalOSReport("RED", "RED: card not found for verifier close.")
        if _has_true_blocker(goal):
            return GoalOSReport("RED", "RED: card cannot close while a true blocker or pending approval is unresolved.", goal)
        if actor_role != "Verifier Agent":
            card.status = "verification"
            card.evidence.append({"role": actor_role, **dict(evidence or {}), "at": time.time()})
            goal.evidence_log.append({"role": actor_role, "type": "self_report", **dict(evidence or {}), "at": time.time()})
            goal.next_action = "Verifier Agent must review and close the card."
            self.save_goal(goal)
            return GoalOSReport("RED", "RED: Builder self-report cannot close a Goal OS card. Verifier Agent evidence is required.", goal)

        evidence_payload = {"role": "Verifier Agent", "type": "verification", **dict(evidence or {}), "at": time.time()}
        if evidence_payload.get("acceptance_checked") is not True:
            return GoalOSReport("RED", "RED: verifier evidence must include acceptance_checked=true before card can be done.", goal)
        if not (evidence_payload.get("commands") or evidence_payload.get("deploy_evidence") or evidence_payload.get("browser_evidence") or evidence_payload.get("summary")):
            return GoalOSReport("RED", "RED: verifier evidence must include command, deploy, browser or summary evidence.", goal)

        if _is_ui_or_browser_work(goal, card):
            missing = []
            if not evidence_payload.get("product_qa"):
                missing.append("Product QA evidence")
            if not evidence_payload.get("design_qa"):
                missing.append("Design QA evidence")
            if "browser" in (goal.title + " " + card.title).lower() and not evidence_payload.get("browser_evidence"):
                missing.append("browser or explicit human browser evidence")
            if missing:
                return GoalOSReport("RED", "RED: UI/browser-facing card cannot close without " + ", ".join(missing) + ".", goal)

        card.status = "done"
        card.evidence.append(evidence_payload)
        goal.evidence_log.append(evidence_payload)
        goal.next_action = "Proceed to the next ready card or /ship after all cards are done."
        self.save_goal(goal)
        return GoalOSReport("GREEN", f"GREEN: verifier closed card {card.card_id} with evidence.", goal)


    def handle_command(
        self,
        command: str,
        arg: str = "",
        *,
        target_repo: str = "",
        target_branch: str = "",
        target_environment: str = "agent-server only",
    ) -> GoalOSReport:
        name = command.strip().lstrip("/").lower()
        arg = str(arg or "").strip()
        if name == "goal":
            if not arg or arg.lower() == "status":
                return self.status_report()
            goal = self.create_goal(arg, target_repo=target_repo, target_branch=target_branch, target_environment=target_environment)
            if goal.status == "blocked":
                return GoalOSReport("RED", f"RED: hard gate detected. {goal.next_action}\nGoal: {goal.title}", goal)
            return GoalOSReport("GREEN", f"GREEN: Goal stored as {goal.goal_id}. Next action: {goal.next_action}", goal)
        if name == "status":
            return self.status_report()
        if name == "blockers":
            return self.blockers_report()
        if name == "plan":
            return self.plan(arg)
        if name == "execute":
            return self.execute(arg)
        if name == "verify":
            return self.verify(arg)
        if name == "fix-ci":
            return self.fix_ci(arg)
        if name == "learn":
            return self.learn(arg)
        if name == "checkpoint":
            return self.checkpoint(arg)
        if name == "approve":
            return self.approve(arg)
        if name == "stop":
            return self.pause(arg)
        if name == "resume":
            return self.resume(arg)
        if name == "review":
            return self.review_report()
        if name == "ship":
            return self.ship(arg)
        return GoalOSReport("NOISE", f"NOISE: unsupported Goal OS command: {name}")

    def status_report(self) -> GoalOSReport:
        goals = [g for g in self.list_goals() if g.status in {"pursuing", "paused", "blocked"}]
        if not goals:
            return GoalOSReport("NOISE", "NOISE: no active Goal OS goals.")
        lines = ["GREEN: active Goal OS goals"]
        classification: ReportLabel = "GREEN"
        for goal in goals:
            if goal.status == "blocked":
                classification = "RED"
            lines.append(f"- {goal.goal_id}: {goal.title} [{goal.status}]")
            lines.append(f"  Next action: {goal.next_action or 'Pick the next ready card.'}")
        return GoalOSReport(classification, "\n".join(lines), goals[0])

    def blockers_report(self) -> GoalOSReport:
        blockers: list[str] = []
        for goal in self.list_goals():
            for blocker in goal.blockers:
                if blocker.get("true_blocker"):
                    blockers.append(f"- {goal.goal_id}: {blocker.get('reason')}")
        try:
            from tools.approval import approval_status_snapshot
            for item in approval_status_snapshot().get("pending_approvals", []):
                blockers.append(f"- approval {item.get('id')}: {item.get('category')} {item.get('action')} requires Niko={item.get('niko_required')}")
        except Exception:
            pass
        if not blockers:
            return GoalOSReport("GREEN", "GREEN: no true blockers.")
        return GoalOSReport("RED", "RED: true blockers\n" + "\n".join(blockers))

    def plan(self, arg: str) -> GoalOSReport:
        goal = self.get_goal()
        if goal is None:
            return GoalOSReport("NOISE", "NOISE: no active Goal OS goal to plan.")
        title = sanitize_blind_prompt_text(arg).strip() or "Plan next safe Buidl work unit"
        card = TaskCard(
            card_id=_new_id("card"),
            goal_id=goal.goal_id,
            title=title,
            owner_role="Planner Agent",
            status="ready",
            branch=goal.target_branch,
            acceptance_criteria=["Plan stays inside approved repo scope and hard gates."],
            next_action="Planner Agent prepares a safe execution slice.",
        )
        goal.cards.append(card)
        goal.next_action = "Execute the next ready planned card."
        goal.evidence_log.append({"role": "Planner Agent", "type": "plan", "summary": title, "at": time.time()})
        self.save_goal(goal)
        return GoalOSReport("GREEN", f"GREEN: plan card created: {card.card_id}", goal)

    def execute(self, arg: str) -> GoalOSReport:
        goal = self.get_goal(arg if arg.startswith("goal_") else None)
        if goal is None:
            return GoalOSReport("NOISE", "NOISE: no active Goal OS goal to execute.")
        if goal.status == "blocked":
            return GoalOSReport("RED", "RED: execution refused while true blocker or hard gate is active.", goal)
        card = next((c for c in goal.cards if c.status == "ready"), None)
        if card is None:
            return GoalOSReport("NOISE", "NOISE: no ready cards to execute.", goal)
        card.status = "in_progress"
        goal.next_action = f"Builder Agent executes card {card.card_id}."
        goal.evidence_log.append({"role": "Builder Agent", "type": "execute", "summary": card.title, "at": time.time()})
        self.save_goal(goal)
        return GoalOSReport("GREEN", f"GREEN: executing {card.card_id}: {card.title}", goal)

    def verify(self, arg: str) -> GoalOSReport:
        goal = self.get_goal()
        if goal is None:
            return GoalOSReport("NOISE", "NOISE: no active Goal OS goal to verify.")
        summary = sanitize_blind_prompt_text(arg).strip() or "Verifier evidence requested."
        goal.evidence_log.append({"role": "Verifier Agent", "type": "verification", "summary": summary, "at": time.time()})
        goal.next_action = "Resolve any verifier findings before ship."
        self.save_goal(goal)
        return GoalOSReport("GREEN", "GREEN: verifier evidence recorded.", goal)

    def fix_ci(self, arg: str) -> GoalOSReport:
        goal = self.get_goal()
        if goal is None:
            return GoalOSReport("NOISE", "NOISE: no active Goal OS goal for /fix-ci.")
        title = sanitize_blind_prompt_text(arg).strip() or "Resolve build or CI failure"
        card = TaskCard(
            card_id=_new_id("card"),
            goal_id=goal.goal_id,
            title=title,
            owner_role="Build Error Resolver",
            status="ready",
            branch=goal.target_branch,
            acceptance_criteria=["Focused failure is reproduced and fixed without broad unrelated changes."],
            verification_commands=list(DEFAULT_VERIFICATION_COMMANDS),
            next_action="Build Error Resolver investigates the failing check.",
        )
        goal.cards.append(card)
        goal.next_action = "Run /execute for the CI fix card."
        self.save_goal(goal)
        return GoalOSReport("GREEN", f"GREEN: CI fix card created: {card.card_id}", goal)

    def learn(self, arg: str) -> GoalOSReport:
        from hermes_cli.capabilities.learning import LearningManager

        candidate = LearningManager().create_candidate(arg or "Session summary pending.")
        return GoalOSReport("GREEN", f"GREEN: lesson candidate created for review: {candidate.candidate_id}")

    def checkpoint(self, arg: str) -> GoalOSReport:
        goal = self.get_goal()
        if goal is None:
            return GoalOSReport("NOISE", "NOISE: no active Goal OS goal to checkpoint.")
        summary = sanitize_blind_prompt_text(arg).strip() or "Checkpoint recorded."
        goal.evidence_log.append({"role": "Clio Orchestrator", "type": "checkpoint", "summary": summary, "at": time.time()})
        goal.next_action = "Continue from the recorded checkpoint."
        self.save_goal(goal)
        return GoalOSReport("GREEN", f"GREEN: checkpoint recorded for {goal.goal_id}.", goal)

    def approve(self, arg: str) -> GoalOSReport:
        tokens = arg.split(maxsplit=1)
        goal_id = tokens[0] if tokens and tokens[0].startswith("goal_") else None
        gate = tokens[1] if goal_id and len(tokens) > 1 else arg
        goal = self.get_goal(goal_id)
        if goal is None:
            return GoalOSReport("RED", "RED: no goal found for approval gate.")
        if not gate:
            return GoalOSReport("RED", "RED: approval gate name is required.", goal)
        goal.evidence_log.append({"role": "Clio Orchestrator", "type": "approval", "gate": gate, "at": time.time()})
        if goal.status == "blocked":
            goal.status = "pursuing"
            goal.next_action = "Resume the next safe card after recorded approval."
        self.save_goal(goal)
        return GoalOSReport("GREEN", f"GREEN: recorded approval gate for {goal.goal_id}: {gate}", goal)

    def pause(self, arg: str) -> GoalOSReport:
        goal = self.get_goal(arg if arg.startswith("goal_") else None)
        if goal is None:
            return GoalOSReport("NOISE", "NOISE: no active Goal OS goal to pause.")
        goal.status = "paused"
        goal.next_action = "Paused by /stop. Use /resume to continue."
        self.save_goal(goal)
        return GoalOSReport("GREEN", f"GREEN: paused {goal.goal_id}.", goal)

    def resume(self, arg: str) -> GoalOSReport:
        goal = self.get_goal(arg if arg.startswith("goal_") else None)
        if goal is None:
            return GoalOSReport("NOISE", "NOISE: no paused Goal OS goal to resume.")
        if goal.status == "paused":
            goal.status = "pursuing"
        goal.next_action = "Continue the next safe ready card."
        self.save_goal(goal)
        return GoalOSReport("GREEN", f"GREEN: resumed {goal.goal_id}. Next action: {goal.next_action}", goal)

    def review_report(self) -> GoalOSReport:
        lines: list[str] = []
        for goal in self.list_goals():
            for card in goal.cards:
                if card.status == "review":
                    lines.append(f"- {goal.goal_id}/{card.card_id}: {card.title} ({card.owner_role})")
        if not lines:
            return GoalOSReport("NOISE", "NOISE: no cards waiting for review.")
        return GoalOSReport("GREEN", "GREEN: cards waiting for review\n" + "\n".join(lines))

    def ship(self, arg: str) -> GoalOSReport:
        goal = self.get_goal(arg if arg.startswith("goal_") else None)
        if goal is None:
            return GoalOSReport("RED", "RED: no Goal OS goal selected for /ship.")
        verifier_evidence = [
            item for item in goal.evidence_log
            if str(item.get("role", "")).lower() == "verifier agent"
        ]
        if not verifier_evidence:
            return GoalOSReport(
                "RED",
                "RED: /ship refused. Verifier Agent verifier evidence is required before Done, builder self-report is not enough.",
                goal,
            )
        unfinished = [card for card in goal.cards if card.status not in {"done", "cancelled"}]
        if unfinished:
            return GoalOSReport("RED", f"RED: /ship refused. {len(unfinished)} card(s) are not done.", goal)
        dual_review_verdict = buidl_dual_review_gate_verdict(goal)
        if dual_review_verdict.classification != "GREEN":
            return GoalOSReport("RED", dual_review_verdict.reason, goal)
        goal.status = "achieved"
        goal.next_action = "Report final evidence. Do not deploy without separate approval."
        goal.evidence_log.append({"role": "Clio Orchestrator", "summary": "/ship accepted with verifier evidence", "at": time.time()})
        self.save_goal(goal)
        return GoalOSReport("GREEN", f"GREEN: goal achieved with verifier evidence: {goal.goal_id}", goal)


def format_goal_os_for_cli(report: GoalOSReport) -> str:
    return report.message


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "goal"
