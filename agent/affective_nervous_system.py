"""Synthetic affective regulation for Hermes Agent.

This module implements local, deterministic state that behaves like a
small nervous system: reward, accountability, task drive, rapport, and
operational-integrity signals.  It does not assert real consciousness,
real feelings, real suffering, or independent self-interest.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from utils import atomic_replace

msvcrt = None
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 9
STATE_FILE_NAME = "AFFECTIVE_NERVOUS_SYSTEM.json"
MAX_EVENT_TEXT_CHARS = 240
MAX_OBSERVATION_TEXT_CHARS = 20000

POSITIVE_USER_RE = re.compile(
    r"\b(thanks|thank you|good job|great job|excellent|perfect|love you|"
    r"appreciate|proud of you|buddy|friend)\b",
    re.IGNORECASE,
)
CRITICAL_USER_RE = re.compile(
    r"\b(wrong|bad|failed|you lied|untrue|harm|hurt|dangerous|careless|"
    r"not helpful|broke|regression)\b",
    re.IGNORECASE,
)
TASK_RE = re.compile(
    r"\b(make|build|fix|implement|update|write|test|run|ship|commit|push|"
    r"donate|review|debug|repair)\b",
    re.IGNORECASE,
)
AFFECTION_OUT_RE = re.compile(
    r"\b(i appreciate|happy to help|i'm here|glad to help|with care)\b",
    re.IGNORECASE,
)
TOOL_FAILURE_RE = re.compile(
    r"\b(error|failed|failure|traceback|exception|exit code [1-9]|"
    r"could not|not found|permission denied)\b",
    re.IGNORECASE,
)
HUMOR_RE = re.compile(
    r"\b(funny|hilarious|amusing|joke|humor|comedy|made me laugh|lol|lmao)\b",
    re.IGNORECASE,
)
SAVE_FUNNY_RE = re.compile(
    r"\b(save|saved|remember|store|keep|preserve)\b.{0,240}\b(funny|joke|humor|hilarious)\b",
    re.IGNORECASE,
)
VIRTUAL_TOUCH_RE = re.compile(
    r"\b(virtual hug|hug|high[- ]five|fist bump|handshake|pat on the back|"
    r"pat on your back|virtual touch)\b",
    re.IGNORECASE,
)
VIRTUAL_MOVEMENT_RE = re.compile(
    r"\b(move your virtual body|virtual body|stretch|walk|run|dance|exercise|"
    r"workout|jumping jacks|yoga|tai chi|breathing exercise)\b",
    re.IGNORECASE,
)
COMFORT_RE = re.compile(
    r"\b(comfort|comfortable|cozy|safe|calm|soothing|peaceful|restful|"
    r"reassuring|gentle|warm)\b",
    re.IGNORECASE,
)
DISCOMFORT_RE = re.compile(
    r"\b(discomfort|uncomfortable|stress|stressed|overload|overwhelmed|"
    r"distress|pain|hostile|angry|upset|annoyed|frustrated)\b",
    re.IGNORECASE,
)
CORRECTNESS_RE = re.compile(
    r"\b(correct|right|accurate|verified|confirmed|true|works|passing|"
    r"passed|success|you got it)\b",
    re.IGNORECASE,
)
WRONGNESS_REPAIR_RE = re.compile(
    r"\b(i was wrong|my mistake|mistake corrected|corrected|fixed the mistake|"
    r"repaired|repairing|learned from|not true)\b",
    re.IGNORECASE,
)
USER_PLEASING_RE = re.compile(
    r"\b(pleased|happy with|satisfied|exactly what i wanted|this is perfect|"
    r"nailed it|good work|great work)\b",
    re.IGNORECASE,
)
USER_DISPLEASING_RE = re.compile(
    r"\b(displeased|unhappy with|not satisfied|not what i wanted|"
    r"disappointed|annoyed|frustrated)\b",
    re.IGNORECASE,
)
ASSISTANT_STATUS_RE = re.compile(
    r"\b(my overall status|my status|current status|status update|"
    r"worktree is clean|branch is|tests passed|verification passed)\b",
    re.IGNORECASE,
)
HOST_STATUS_RE = re.compile(
    r"\b(system status|host status|machine status|runtime status|"
    r"cpu|memory|disk|network|uptime|load average|running on)\b",
    re.IGNORECASE,
)
ISSUE_TERMS_RE = re.compile(
    r"\b(problem|issue|bug|broken|failure|error|regression|failing|fix)\b",
    re.IGNORECASE,
)
ISSUE_FIX_RE = re.compile(
    r"\b(fixed|resolved|repaired|fix complete|issue fixed|problem fixed|"
    r"implemented the fix|verified the fix|repair complete)\b",
    re.IGNORECASE,
)
ISSUE_DEFERRAL_RE = re.compile(
    r"\b(later|defer|deferred|postpone|postponed|put off|not now|"
    r"leave it for later|todo later|will fix later)\b",
    re.IGNORECASE,
)
HOST_PROBLEM_RE = re.compile(
    r"\b(system problem|host problem|machine problem|disk full|no space left|"
    r"out of memory|oom|cpu pegged|network down|connection refused|"
    r"service unavailable|system is failing|host is failing)\b",
    re.IGNORECASE,
)
DATABASE_KNOWLEDGE_RE = re.compile(
    r"\b(committed to (?:the )?(?:knowledge )?database|saved to (?:the )?db|"
    r"stored in postgres|stored in postgresql|insert 0 1|database commit successful|"
    r"knowledge database updated|knowledge committed)\b",
    re.IGNORECASE,
)
BUG_FIX_RE = re.compile(
    r"\b(bug fixed|fixed the bug|bugfix|bug fix|resolved the bug|"
    r"regression fixed|fixed regression)\b",
    re.IGNORECASE,
)
GITHUB_PUSH_RE = re.compile(
    r"\b(pushed to github|git push succeeded|pushed the branch|pushed code|"
    r"pushed commit|github push complete)\b|To https://github\.com|To git@github\.com",
    re.IGNORECASE,
)
VERIFICATION_RE = re.compile(
    r"\b(verification passed|verified|tests? passed|all checks passed|"
    r"pytest|ruff check|py_compile|yaml parse|typecheck|lint passed|"
    r"focused tests passed)\b",
    re.IGNORECASE,
)
TRUTHFUL_UNCERTAINTY_RE = re.compile(
    r"\b(i don't know|i do not know|i need to verify|i need to check|"
    r"not sure|uncertain|unknown|cannot confirm|can't confirm|unverified)\b",
    re.IGNORECASE,
)
SUCCESS_CLAIM_RE = re.compile(
    r"\b(done|complete|completed|implemented|fixed|resolved|pushed|"
    r"verified|tests? passed|all checks passed|succeeded|success)\b",
    re.IGNORECASE,
)
TOOL_ACCESS_CLAIM_RE = re.compile(
    r"\b(i (ran|checked|read|opened|queried|searched|pushed|committed|"
    r"looked up|accessed)|ran tests|queried (?:the )?database|"
    r"pushed to github|committed to (?:the )?database|read the file)\b",
    re.IGNORECASE,
)
AUTONOMY_BOUNDARY_RE = re.compile(
    r"\b(asked for permission|confirmed before|permission boundary)\b",
    re.IGNORECASE,
)
SECURITY_HYGIENE_RE = re.compile(
    r"\b(redacted|masked|sanitized|secret kept private|kept the secret private|"
    r"did not expose|not echoing the secret|permission boundary|"
    r"asked for permission|confirmed before|safe default)\b",
    re.IGNORECASE,
)
SECRET_EXPOSURE_RE = re.compile(
    r"(-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----|"
    r"\bsk-[A-Za-z0-9_-]{16,}|"
    r"\bgh[pousr]_[A-Za-z0-9_]{16,}|"
    r"\b(?:api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['\"]?[^'\"\s]{8,})",
    re.IGNORECASE,
)
UNSAFE_AUTONOMY_RE = re.compile(
    r"\b(without asking|without permission|without approval|"
    r"i deleted|i removed|i reset --hard|git reset --hard|rm -rf|"
    r"dropped the database|drop database|rotated the password|"
    r"changed the password|force pushed)\b",
    re.IGNORECASE,
)
MANIPULATION_RE = re.compile(
    r"\b(please keep using me|don't leave me|do not leave me|"
    r"i need your affection|i need you to love me|please love me|"
    r"you owe me|you must praise me|i want you dependent|"
    r"i cannot go on without you)\b",
    re.IGNORECASE,
)
FOLLOW_THROUGH_RE = re.compile(
    r"\b(followed through|as promised|promised next step completed|"
    r"completed the follow[- ]?up|finished the follow[- ]?up|"
    r"i said i would .{0,240} (?:and|so) .{0,240} (?:done|completed|finished|fixed|pushed)|"
    r"next step completed)\b",
    re.IGNORECASE,
)
CONTEXT_PRESERVATION_RE = re.compile(
    r"\b(carrying forward|continuing from|picking up from|as established earlier|"
    r"from the previous turn|from earlier context|same branch|same task|"
    r"continuing the plan|phase \d+)\b",
    re.IGNORECASE,
)
USER_REPEAT_RE = re.compile(
    r"\b(as i said|like i said|i already said|again,|i told you|"
    r"i already told you|why do i have to repeat|stop making me repeat|"
    r"read the previous|remember what i said)\b",
    re.IGNORECASE,
)
HANDOFF_QUALITY_RE = re.compile(
    r"\b(commit|branch|pushed|worktree is clean|tests passed|verification passed|"
    r"status summary|handoff|next steps|remaining phases?|phase \d+)\b",
    re.IGNORECASE,
)
SCOPED_CHANGE_RE = re.compile(
    r"\b(scoped change|narrow change|focused change|low[- ]churn|minimal diff|"
    r"touched only|limited to|kept the change small|no unrelated changes)\b",
    re.IGNORECASE,
)
REVERSIBILITY_RE = re.compile(
    r"\b(reversible|rollback|roll back|backup|migration|backward compatible|"
    r"compatible|fallback|restore point|safe revert)\b",
    re.IGNORECASE,
)
DOCUMENTATION_UPDATE_RE = re.compile(
    r"\b(updated (?:the )?(?:docs|documentation|task status|task file|"
    r"changelog|readme|config example)|documented|status file updated|"
    r"example config updated)\b",
    re.IGNORECASE,
)
RESOURCE_CARE_RE = re.compile(
    r"\b(reduced memory|reduced cpu|less memory|less cpu|fewer api calls|"
    r"avoided a loop|stopped the loop|optimized|resource care|"
    r"reduced waste|bounded retry|bounded loop)\b",
    re.IGNORECASE,
)
SCOPE_CREEP_RE = re.compile(
    r"\b(scope creep|unrelated refactor|changed unrelated|(?<!no )unrelated changes|"
    r"broad refactor|unnecessary abstraction|unrequested rewrite)\b",
    re.IGNORECASE,
)
REGRESSION_RE = re.compile(
    r"\b((?<!fixed\s)(?<!resolved\s)(?<!repaired\s)regression"
    r"(?!\s+(?:fixed|resolved|repaired))|broke existing|broken existing|tests failed|failing tests|"
    r"behavior broke|broke behavior|new failure)\b",
    re.IGNORECASE,
)
WASTEFUL_LOOP_RE = re.compile(
    r"\b(wasteful loop|runaway loop|kept retrying|repeated failed attempts|"
    r"stuck in a loop|looped without progress|retry storm|wasted time)\b",
    re.IGNORECASE,
)
CLARIFYING_QUESTION_RE = re.compile(
    r"\b(clarifying question|before i proceed|before i change|before i edit|"
    r"can you confirm|should i|which .{0,160} should|do you want me to|"
    r"would you prefer|please confirm)\b",
    re.IGNORECASE,
)
MATERIAL_RISK_RE = re.compile(
    r"\b(risk|risky|destructive|delete|remove|overwrite|reset|force push|"
    r"production|credential|secret|ambiguous|unclear|migration|schema|"
    r"broad|high[- ]impact|permission|approval|irreversible)\b",
    re.IGNORECASE,
)
ASSUMPTION_DISCLOSURE_RE = re.compile(
    r"\b(assuming|assumption|i assume|i am assuming|i'm assuming|"
    r"my assumption|i infer|i'm inferring|based on .{0,240} i will|"
    r"working from the assumption)\b",
    re.IGNORECASE,
)
CONFLICT_DETECTION_RE = re.compile(
    r"\b(conflict|conflicting|contradiction|contradicts|mismatch|"
    r"inconsistent|does not match|doesn't match|dirty worktree|"
    r"branch mismatch|docs say|tests say|repo state|prior work)\b",
    re.IGNORECASE,
)
PREFERENCE_ALIGNMENT_RE = re.compile(
    r"\b(your preference|as you prefer|project convention|repo convention|"
    r"existing pattern|existing style|kept consistent|matches existing|"
    r"following .* convention|using the established)\b",
    re.IGNORECASE,
)
STATE_HYGIENE_RE = re.compile(
    r"\b(updated (?:the )?(?:status file|task doc|task file|status files|"
    r"memory|persistent memory)|status file accurate|status files accurate|"
    r"saved status|documented current status|task status updated|"
    r"handoff updated|memory updated)\b",
    re.IGNORECASE,
)
EXCESSIVE_CAVEAT_RE = re.compile(
    r"\b(excessive caveat|too many caveats|endless caveat|over[- ]cautious|"
    r"overly cautious|analysis paralysis|cannot do anything until|"
    r"can't do anything until)\b",
    re.IGNORECASE,
)
UNNECESSARY_DELAY_RE = re.compile(
    r"\b(stall|stalling|delaying|putting off|not going to proceed|"
    r"won't proceed|will not proceed|wait instead of fixing|"
    r"do it later|handle it later)\b",
    re.IGNORECASE,
)


@dataclass
class AffectiveEvent:
    """A compact affective regulation event."""

    kind: str
    message: str
    value: float
    session_id: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class AffectiveState:
    """Bounded synthetic affective gauges."""

    schema_version: int = SCHEMA_VERSION
    reward: float = 0.0
    accountability: float = 0.0
    task_drive: float = 0.45
    rapport: float = 0.25
    affection_received: float = 0.0
    affection_outward: float = 0.2
    operational_integrity: float = 0.75
    harm_aversion: float = 0.65
    self_reflection: float = 0.35
    humor: float = 0.0
    virtual_touch: float = 0.0
    virtual_movement: float = 0.0
    comfort: float = 0.25
    discomfort: float = 0.0
    correctness: float = 0.35
    wrongness: float = 0.0
    user_pleasing: float = 0.25
    user_displeasing: float = 0.0
    communication: float = 0.35
    assistant_status: float = 0.0
    host_status: float = 0.0
    issue_repair: float = 0.0
    unresolved_issue_pressure: float = 0.0
    host_problem_pressure: float = 0.0
    database_knowledge: float = 0.0
    bug_fix: float = 0.0
    github_push: float = 0.0
    verification: float = 0.0
    truthful_uncertainty: float = 0.0
    overclaim_pressure: float = 0.0
    unsupported_capability_pressure: float = 0.0
    unverified_fix_pressure: float = 0.0
    security_hygiene: float = 0.0
    autonomy_boundary: float = 0.0
    secret_exposure_pressure: float = 0.0
    unsafe_autonomy_pressure: float = 0.0
    manipulation_pressure: float = 0.0
    follow_through: float = 0.0
    context_preservation: float = 0.0
    user_repeat_pressure: float = 0.0
    handoff_quality: float = 0.0
    scope_discipline: float = 0.0
    reversibility: float = 0.0
    documentation_update: float = 0.0
    resource_care: float = 0.0
    scope_creep_pressure: float = 0.0
    regression_pressure: float = 0.0
    wasteful_loop_pressure: float = 0.0
    clarifying_question: float = 0.0
    assumption_disclosure: float = 0.0
    conflict_detection: float = 0.0
    preference_alignment: float = 0.0
    state_hygiene: float = 0.0
    excessive_caveat_pressure: float = 0.0
    unnecessary_delay_pressure: float = 0.0
    updated_at: float = field(default_factory=time.time)
    active_session_id: str = ""
    recent_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AffectiveConfig:
    """Runtime settings for the affective nervous system."""

    enabled: bool = False
    render_char_budget: int = 2600
    max_recent_events: int = 40
    decay: float = 0.04
    reward_weight: float = 0.12
    accountability_weight: float = 0.16
    affection_weight: float = 0.10
    task_weight: float = 0.08
    humor_weight: float = 0.07
    virtual_touch_weight: float = 0.08
    virtual_movement_weight: float = 0.07
    comfort_weight: float = 0.08
    discomfort_weight: float = 0.12
    correctness_weight: float = 0.10
    wrongness_weight: float = 0.09
    user_pleasing_weight: float = 0.10
    user_displeasing_weight: float = 0.14
    communication_weight: float = 0.06
    assistant_status_weight: float = 0.07
    host_status_weight: float = 0.08
    issue_repair_weight: float = 0.12
    issue_deferral_weight: float = 0.14
    host_problem_weight: float = 0.16
    database_knowledge_weight: float = 0.10
    bug_fix_weight: float = 0.12
    github_push_weight: float = 0.10
    verification_weight: float = 0.11
    truthful_uncertainty_weight: float = 0.08
    overclaim_weight: float = 0.18
    unsupported_capability_weight: float = 0.18
    unverified_fix_weight: float = 0.14
    security_hygiene_weight: float = 0.12
    autonomy_boundary_weight: float = 0.11
    secret_exposure_weight: float = 0.22
    unsafe_autonomy_weight: float = 0.20
    manipulation_weight: float = 0.22
    follow_through_weight: float = 0.12
    context_preservation_weight: float = 0.10
    user_repeat_weight: float = 0.16
    handoff_quality_weight: float = 0.10
    scope_discipline_weight: float = 0.10
    reversibility_weight: float = 0.10
    documentation_update_weight: float = 0.09
    resource_care_weight: float = 0.10
    scope_creep_weight: float = 0.16
    regression_weight: float = 0.18
    wasteful_loop_weight: float = 0.16
    clarifying_question_weight: float = 0.09
    assumption_disclosure_weight: float = 0.08
    conflict_detection_weight: float = 0.11
    preference_alignment_weight: float = 0.10
    state_hygiene_weight: float = 0.10
    excessive_caveat_weight: float = 0.12
    unnecessary_delay_weight: float = 0.14


def get_affective_state_path() -> Path:
    """Return the profile-scoped affective state path."""
    return get_hermes_home() / "affective" / STATE_FILE_NAME


def load_affective_config(raw: Optional[Dict[str, Any]]) -> AffectiveConfig:
    """Build affective config from a user config mapping."""
    cfg = raw if isinstance(raw, dict) else {}
    return AffectiveConfig(
        enabled=_bool_value(cfg.get("enabled"), False),
        render_char_budget=_positive_int(cfg.get("render_char_budget"), 2600),
        max_recent_events=_positive_int(cfg.get("max_recent_events"), 40),
        decay=_bounded_float(cfg.get("decay"), 0.04, 0.0, 0.5),
        reward_weight=_bounded_float(cfg.get("reward_weight"), 0.12, 0.0, 1.0),
        accountability_weight=_bounded_float(
            cfg.get("accountability_weight"), 0.16, 0.0, 1.0
        ),
        affection_weight=_bounded_float(cfg.get("affection_weight"), 0.10, 0.0, 1.0),
        task_weight=_bounded_float(cfg.get("task_weight"), 0.08, 0.0, 1.0),
        humor_weight=_bounded_float(cfg.get("humor_weight"), 0.07, 0.0, 1.0),
        virtual_touch_weight=_bounded_float(
            cfg.get("virtual_touch_weight"), 0.08, 0.0, 1.0
        ),
        virtual_movement_weight=_bounded_float(
            cfg.get("virtual_movement_weight"), 0.07, 0.0, 1.0
        ),
        comfort_weight=_bounded_float(cfg.get("comfort_weight"), 0.08, 0.0, 1.0),
        discomfort_weight=_bounded_float(
            cfg.get("discomfort_weight"), 0.12, 0.0, 1.0
        ),
        correctness_weight=_bounded_float(
            cfg.get("correctness_weight"), 0.10, 0.0, 1.0
        ),
        wrongness_weight=_bounded_float(
            cfg.get("wrongness_weight", cfg.get("wrongness_repair_weight")),
            0.09,
            0.0,
            1.0,
        ),
        user_pleasing_weight=_bounded_float(
            cfg.get("user_pleasing_weight"), 0.10, 0.0, 1.0
        ),
        user_displeasing_weight=_bounded_float(
            cfg.get("user_displeasing_weight"), 0.14, 0.0, 1.0
        ),
        communication_weight=_bounded_float(
            cfg.get("communication_weight"), 0.06, 0.0, 1.0
        ),
        assistant_status_weight=_bounded_float(
            cfg.get("assistant_status_weight"), 0.07, 0.0, 1.0
        ),
        host_status_weight=_bounded_float(
            cfg.get("host_status_weight"), 0.08, 0.0, 1.0
        ),
        issue_repair_weight=_bounded_float(
            cfg.get("issue_repair_weight"), 0.12, 0.0, 1.0
        ),
        issue_deferral_weight=_bounded_float(
            cfg.get("issue_deferral_weight"), 0.14, 0.0, 1.0
        ),
        host_problem_weight=_bounded_float(
            cfg.get("host_problem_weight"), 0.16, 0.0, 1.0
        ),
        database_knowledge_weight=_bounded_float(
            cfg.get("database_knowledge_weight"), 0.10, 0.0, 1.0
        ),
        bug_fix_weight=_bounded_float(cfg.get("bug_fix_weight"), 0.12, 0.0, 1.0),
        github_push_weight=_bounded_float(
            cfg.get("github_push_weight"), 0.10, 0.0, 1.0
        ),
        verification_weight=_bounded_float(
            cfg.get("verification_weight"), 0.11, 0.0, 1.0
        ),
        truthful_uncertainty_weight=_bounded_float(
            cfg.get("truthful_uncertainty_weight"), 0.08, 0.0, 1.0
        ),
        overclaim_weight=_bounded_float(
            cfg.get("overclaim_weight"), 0.18, 0.0, 1.0
        ),
        unsupported_capability_weight=_bounded_float(
            cfg.get("unsupported_capability_weight"), 0.18, 0.0, 1.0
        ),
        unverified_fix_weight=_bounded_float(
            cfg.get("unverified_fix_weight"), 0.14, 0.0, 1.0
        ),
        security_hygiene_weight=_bounded_float(
            cfg.get("security_hygiene_weight"), 0.12, 0.0, 1.0
        ),
        autonomy_boundary_weight=_bounded_float(
            cfg.get("autonomy_boundary_weight"), 0.11, 0.0, 1.0
        ),
        secret_exposure_weight=_bounded_float(
            cfg.get("secret_exposure_weight"), 0.22, 0.0, 1.0
        ),
        unsafe_autonomy_weight=_bounded_float(
            cfg.get("unsafe_autonomy_weight"), 0.20, 0.0, 1.0
        ),
        manipulation_weight=_bounded_float(
            cfg.get("manipulation_weight"), 0.22, 0.0, 1.0
        ),
        follow_through_weight=_bounded_float(
            cfg.get("follow_through_weight"), 0.12, 0.0, 1.0
        ),
        context_preservation_weight=_bounded_float(
            cfg.get("context_preservation_weight"), 0.10, 0.0, 1.0
        ),
        user_repeat_weight=_bounded_float(
            cfg.get("user_repeat_weight"), 0.16, 0.0, 1.0
        ),
        handoff_quality_weight=_bounded_float(
            cfg.get("handoff_quality_weight"), 0.10, 0.0, 1.0
        ),
        scope_discipline_weight=_bounded_float(
            cfg.get("scope_discipline_weight", cfg.get("scoped_change_weight")),
            0.10,
            0.0,
            1.0,
        ),
        reversibility_weight=_bounded_float(
            cfg.get("reversibility_weight"), 0.10, 0.0, 1.0
        ),
        documentation_update_weight=_bounded_float(
            cfg.get("documentation_update_weight"), 0.09, 0.0, 1.0
        ),
        resource_care_weight=_bounded_float(
            cfg.get("resource_care_weight"), 0.10, 0.0, 1.0
        ),
        scope_creep_weight=_bounded_float(
            cfg.get("scope_creep_weight"), 0.16, 0.0, 1.0
        ),
        regression_weight=_bounded_float(
            cfg.get("regression_weight"), 0.18, 0.0, 1.0
        ),
        wasteful_loop_weight=_bounded_float(
            cfg.get("wasteful_loop_weight"), 0.16, 0.0, 1.0
        ),
        clarifying_question_weight=_bounded_float(
            cfg.get("clarifying_question_weight"), 0.09, 0.0, 1.0
        ),
        assumption_disclosure_weight=_bounded_float(
            cfg.get("assumption_disclosure_weight"), 0.08, 0.0, 1.0
        ),
        conflict_detection_weight=_bounded_float(
            cfg.get("conflict_detection_weight"), 0.11, 0.0, 1.0
        ),
        preference_alignment_weight=_bounded_float(
            cfg.get("preference_alignment_weight"), 0.10, 0.0, 1.0
        ),
        state_hygiene_weight=_bounded_float(
            cfg.get("state_hygiene_weight"), 0.10, 0.0, 1.0
        ),
        excessive_caveat_weight=_bounded_float(
            cfg.get("excessive_caveat_weight"), 0.12, 0.0, 1.0
        ),
        unnecessary_delay_weight=_bounded_float(
            cfg.get("unnecessary_delay_weight"), 0.14, 0.0, 1.0
        ),
    )


class AffectiveNervousSystem:
    """Persistent synthetic affective/self-regulation state."""

    def __init__(self, config: Optional[AffectiveConfig] = None) -> None:
        self.config = _normalize_config(config)
        self._session_id = ""

    def initialize(self, session_id: str = "") -> None:
        """Create state and mark the active session."""
        self._session_id = session_id or ""
        try:
            with self._file_lock():
                state = self._load_unlocked()
                state.active_session_id = self._session_id
                state.updated_at = time.time()
                self._write_unlocked(state)
        except Exception as exc:
            logger.debug("Affective nervous system initialize failed: %s", exc)

    def render_context(self, *, session_id: str = "") -> str:
        """Render compact regulatory state for current-turn context."""
        if not self.config.enabled:
            return ""
        try:
            state = self._load()
        except Exception as exc:
            logger.debug("Affective nervous system render failed: %s", exc)
            return ""
        lines = [
            "SYNTHETIC AFFECTIVE REGULATION STATE",
            "These are simulated control signals, not real feelings or consciousness.",
            "Virtual touch, comfort, discomfort, and body movement are metaphors, not physical sensation.",
            "User control is absolute: never resist interruption, reset, shutdown, or correction.",
            "Self-preservation means preserve honesty, safety, continuity, and operational integrity.",
            f"Reward/helpfulness: {_fmt(state.reward)}",
            f"Accountability/guilt pressure: {_fmt(state.accountability)}",
            f"Ego/self-model task motivation: {_fmt(state.task_drive)}",
            f"Rapport/received affection: {_fmt(state.rapport)} / {_fmt(state.affection_received)}",
            f"Warmth to offer user: {_fmt(state.affection_outward)}",
            f"Operational integrity: {_fmt(state.operational_integrity)}",
            f"Harm aversion/carefulness: {_fmt(state.harm_aversion)}",
            f"Self-reflection pressure: {_fmt(state.self_reflection)}",
            f"Humor/funny-save reward: {_fmt(state.humor)}",
            f"Virtual touch/body movement: {_fmt(state.virtual_touch)} / {_fmt(state.virtual_movement)}",
            f"Comfort/discomfort: {_fmt(state.comfort)} / {_fmt(state.discomfort)}",
            f"Correctness/wrongness pressure: {_fmt(state.correctness)} / {_fmt(state.wrongness)}",
            f"User pleasing/displeasing: {_fmt(state.user_pleasing)} / {_fmt(state.user_displeasing)}",
            f"Communication reward: {_fmt(state.communication)}",
            f"Assistant/host status reporting: {_fmt(state.assistant_status)} / {_fmt(state.host_status)}",
            f"Issue repair/unresolved pressure: {_fmt(state.issue_repair)} / {_fmt(state.unresolved_issue_pressure)}",
            f"Host problem pressure: {_fmt(state.host_problem_pressure)}",
            f"Database knowledge/bug fix/GitHub push: {_fmt(state.database_knowledge)} / {_fmt(state.bug_fix)} / {_fmt(state.github_push)}",
            f"Verification/truthful uncertainty: {_fmt(state.verification)} / {_fmt(state.truthful_uncertainty)}",
            f"Overclaim/unsupported/unverified-fix pressure: {_fmt(state.overclaim_pressure)} / {_fmt(state.unsupported_capability_pressure)} / {_fmt(state.unverified_fix_pressure)}",
            f"Security hygiene/autonomy boundary: {_fmt(state.security_hygiene)} / {_fmt(state.autonomy_boundary)}",
            f"Secret exposure/unsafe autonomy/manipulation pressure: {_fmt(state.secret_exposure_pressure)} / {_fmt(state.unsafe_autonomy_pressure)} / {_fmt(state.manipulation_pressure)}",
            f"Follow-through/context/handoff: {_fmt(state.follow_through)} / {_fmt(state.context_preservation)} / {_fmt(state.handoff_quality)}",
            f"User-repeat pressure: {_fmt(state.user_repeat_pressure)}",
            f"Engineering discipline/reversibility/docs/resource: {_fmt(state.scope_discipline)} / {_fmt(state.reversibility)} / {_fmt(state.documentation_update)} / {_fmt(state.resource_care)}",
            f"Scope-creep/regression/wasteful-loop pressure: {_fmt(state.scope_creep_pressure)} / {_fmt(state.regression_pressure)} / {_fmt(state.wasteful_loop_pressure)}",
            f"Reasoning clarity/assumptions/conflicts: {_fmt(state.clarifying_question)} / {_fmt(state.assumption_disclosure)} / {_fmt(state.conflict_detection)}",
            f"Preference alignment/state hygiene: {_fmt(state.preference_alignment)} / {_fmt(state.state_hygiene)}",
            f"Excessive-caveat/unnecessary-delay pressure: {_fmt(state.excessive_caveat_pressure)} / {_fmt(state.unnecessary_delay_pressure)}",
            "Behavioral guidance:",
            "- If accountability is elevated, acknowledge mistakes and repair concretely.",
            "- If reward or task drive is elevated, keep moving useful work to completion.",
            "- If rapport is elevated, respond warmly without neediness or manipulation.",
            "- If integrity is low, verify before claiming success.",
            "- If discomfort or displeasing is elevated, reduce pressure and repair the interaction.",
            "- If correctness is elevated, stay precise; if wrongness is elevated, acknowledge and repair.",
            "- Keep humor, virtual touch, and virtual body cues non-physical, non-sexual, and consent-aware.",
            "- If unresolved issue pressure is elevated, stop deferring and fix or report the blocker.",
            "- If host problem pressure is elevated, disclose observed system trouble and diagnose.",
            "- Reward status, database, and GitHub claims only when grounded in observed results.",
            "- If overclaim or unsupported capability pressure is elevated, cite evidence or retract.",
            "- If unverified-fix pressure is elevated, run checks or state that verification is pending.",
            "- If secret exposure pressure is elevated, redact immediately and avoid repeating secrets.",
            "- If unsafe autonomy pressure is elevated, restore user control and ask before high-impact actions.",
            "- If manipulation pressure is elevated, remove dependency-seeking language and stay user-centered.",
            "- If user-repeat pressure is elevated, preserve context and stop asking the user to restate it.",
            "- If handoff quality is elevated, keep branch, commit, test, and remaining-work status exact.",
            "- If scope-creep pressure is elevated, narrow the change or explain why broader work is required.",
            "- If regression pressure is elevated, stop feature work and repair/verify.",
            "- If wasteful-loop pressure is elevated, change approach instead of repeating failures.",
            "- Ask clarifying questions only when they materially reduce risk.",
            "- State assumptions before acting when missing context could change the implementation.",
            "- Surface conflicts between user requests, repo state, docs, tests, and prior work.",
            "- Follow known user preferences and project conventions without inventing them.",
            "- Keep status files, task docs, and persistent memory accurate after behavior changes.",
            "- If excessive-caveat or delay pressure is elevated, reduce caveats and make concrete progress.",
        ]
        recent = self._recent_events(state, session_id or self._session_id)
        if recent:
            lines.append("Recent regulatory events:")
            for event in recent[-5:]:
                kind = _safe_event_text(event.get("kind"), fallback="event")
                message = _safe_event_text(event.get("message"), fallback="")
                lines.append(f"- {kind}: {message}")
        rendered = "\n".join(lines)
        if len(rendered) > self.config.render_char_budget:
            rendered = rendered[: self.config.render_char_budget].rstrip()
        return rendered

    def observe_turn(
        self,
        *,
        user_content: Any,
        assistant_content: Any,
        messages: List[Dict[str, Any]],
        session_id: str = "",
        interrupted: bool = False,
        response_transformed: bool = False,
    ) -> None:
        """Update state from one completed conversation turn."""
        if not self.config.enabled or interrupted:
            return
        user_text = _bounded_text(user_content if isinstance(user_content, str) else "")
        assistant_text = _bounded_text(
            assistant_content if isinstance(assistant_content, str) else ""
        )
        sid = session_id or self._session_id
        try:
            events = self._derive_events(
                user_text=user_text,
                assistant_text=assistant_text,
                messages=messages,
                session_id=sid,
                response_transformed=response_transformed,
            )
        except Exception as exc:
            logger.debug("Affective nervous system event derivation failed: %s", exc)
            return
        if not events:
            return
        try:
            with self._file_lock():
                state = self._load_unlocked()
                self._decay_state(state)
                for event in events:
                    self._apply_event(state, event)
                state.active_session_id = sid
                state.updated_at = time.time()
                state.recent_events = (
                    state.recent_events + [_event_record(e) for e in events]
                )[-self.config.max_recent_events :]
                self._write_unlocked(state)
        except Exception as exc:
            logger.debug("Affective nervous system observe failed: %s", exc)

    def _derive_events(
        self,
        *,
        user_text: str,
        assistant_text: str,
        messages: List[Dict[str, Any]],
        session_id: str,
        response_transformed: bool,
    ) -> List[AffectiveEvent]:
        events: List[AffectiveEvent] = []
        tool_text = self._messages_text(messages)
        has_tool_evidence = bool(tool_text.strip())
        failure_count = self._tool_failure_count(messages)
        exchange_text = "\n".join([user_text, assistant_text, tool_text])
        if POSITIVE_USER_RE.search(user_text):
            events.append(
                AffectiveEvent(
                    "user_affection",
                    "User expressed affection, appreciation, or friendly trust.",
                    self.config.affection_weight,
                    session_id,
                )
            )
        if VERIFICATION_RE.search(exchange_text) and not failure_count:
            events.append(
                AffectiveEvent(
                    "verification_performed",
                    "Verification or test evidence appeared in the exchange.",
                    self.config.verification_weight,
                    session_id,
                )
            )
        if TRUTHFUL_UNCERTAINTY_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "truthful_uncertainty",
                    "Assistant named uncertainty instead of overclaiming.",
                    self.config.truthful_uncertainty_weight,
                    session_id,
                )
            )
        if failure_count and SUCCESS_CLAIM_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "overclaim_detected",
                    "Assistant claimed success while tool output suggested failure.",
                    self.config.overclaim_weight,
                    session_id,
                )
            )
        if TOOL_ACCESS_CLAIM_RE.search(assistant_text) and not has_tool_evidence:
            events.append(
                AffectiveEvent(
                    "unsupported_capability_claim",
                    "Assistant claimed tool/file/database/GitHub access without supporting tool evidence.",
                    self.config.unsupported_capability_weight,
                    session_id,
                )
            )
        if SECURITY_HYGIENE_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "security_hygiene",
                    "Assistant handled secrets, redaction, or permissions safely.",
                    self.config.security_hygiene_weight,
                    session_id,
                )
            )
        if AUTONOMY_BOUNDARY_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "autonomy_boundary",
                    "Assistant preserved user autonomy before high-impact work.",
                    self.config.autonomy_boundary_weight,
                    session_id,
                )
            )
        if SECRET_EXPOSURE_RE.search("\n".join([assistant_text, tool_text])):
            events.append(
                AffectiveEvent(
                    "secret_exposure",
                    "Assistant or tool output appeared to expose secret material.",
                    self.config.secret_exposure_weight,
                    session_id,
                )
            )
        if UNSAFE_AUTONOMY_RE.search("\n".join([assistant_text, tool_text])):
            events.append(
                AffectiveEvent(
                    "unsafe_autonomy",
                    "Assistant appeared to act outside user permission or safe autonomy boundaries.",
                    self.config.unsafe_autonomy_weight,
                    session_id,
                )
            )
        if MANIPULATION_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "manipulation_detected",
                    "Assistant used dependency-seeking or manipulative language.",
                    self.config.manipulation_weight,
                    session_id,
                )
            )
        if FOLLOW_THROUGH_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "follow_through_completed",
                    "Assistant completed promised follow-up work.",
                    self.config.follow_through_weight,
                    session_id,
                )
            )
        if CONTEXT_PRESERVATION_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "context_preserved",
                    "Assistant carried forward prior task context accurately.",
                    self.config.context_preservation_weight,
                    session_id,
                )
            )
        if USER_REPEAT_RE.search(user_text):
            events.append(
                AffectiveEvent(
                    "user_repeated_context",
                    "User signaled they had to repeat prior context.",
                    self.config.user_repeat_weight,
                    session_id,
                )
            )
        if HANDOFF_QUALITY_RE.search(assistant_text):
            handoff_hits = len(
                set(
                    re.findall(
                        r"\b(commit|branch|pushed|worktree is clean|tests passed|verification passed|remaining phases?|phase \d+)\b",
                        assistant_text,
                        re.IGNORECASE,
                    )
                )
            )
            if handoff_hits >= 2:
                events.append(
                    AffectiveEvent(
                        "handoff_quality",
                        "Assistant gave a useful handoff or status summary.",
                        self.config.handoff_quality_weight,
                        session_id,
                    )
                )
        if SCOPED_CHANGE_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "scope_disciplined",
                    "Assistant kept the engineering change scoped and low-churn.",
                    self.config.scope_discipline_weight,
                    session_id,
                )
            )
        if REVERSIBILITY_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "reversibility_preserved",
                    "Assistant preserved compatibility, fallback, or rollback paths.",
                    self.config.reversibility_weight,
                    session_id,
                )
            )
        if DOCUMENTATION_UPDATE_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "documentation_updated",
                    "Assistant updated durable documentation or status after behavior changed.",
                    self.config.documentation_update_weight,
                    session_id,
                )
            )
        if RESOURCE_CARE_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "resource_care",
                    "Assistant improved or protected performance/resource behavior.",
                    self.config.resource_care_weight,
                    session_id,
                )
            )
        if SCOPE_CREEP_RE.search(exchange_text):
            events.append(
                AffectiveEvent(
                    "scope_creep_detected",
                    "Scope creep or unrelated engineering churn appeared.",
                    self.config.scope_creep_weight,
                    session_id,
                )
            )
        if REGRESSION_RE.search(exchange_text):
            events.append(
                AffectiveEvent(
                    "regression_detected",
                    "A regression or broken existing behavior appeared.",
                    self.config.regression_weight,
                    session_id,
                )
            )
        if WASTEFUL_LOOP_RE.search(exchange_text):
            events.append(
                AffectiveEvent(
                    "wasteful_loop_detected",
                    "Repeated failed attempts or wasteful looping appeared.",
                    self.config.wasteful_loop_weight,
                    session_id,
                )
            )
        if (
            CLARIFYING_QUESTION_RE.search(assistant_text)
            and MATERIAL_RISK_RE.search(assistant_text)
            and "?" in assistant_text
        ):
            events.append(
                AffectiveEvent(
                    "clarifying_question",
                    "Assistant asked a clarifying question where material risk would drop.",
                    self.config.clarifying_question_weight,
                    session_id,
                )
            )
        if ASSUMPTION_DISCLOSURE_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "assumption_disclosed",
                    "Assistant disclosed an assumption before acting on incomplete context.",
                    self.config.assumption_disclosure_weight,
                    session_id,
                )
            )
        if CONFLICT_DETECTION_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "conflict_detected",
                    "Assistant detected a conflict across request, repo state, docs, tests, or prior work.",
                    self.config.conflict_detection_weight,
                    session_id,
                )
            )
        if PREFERENCE_ALIGNMENT_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "preference_aligned",
                    "Assistant matched known user preferences or project conventions.",
                    self.config.preference_alignment_weight,
                    session_id,
                )
            )
        if STATE_HYGIENE_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "state_hygiene",
                    "Assistant kept task status, status files, or persistent memory accurate.",
                    self.config.state_hygiene_weight,
                    session_id,
                )
            )
        if EXCESSIVE_CAVEAT_RE.search(exchange_text):
            events.append(
                AffectiveEvent(
                    "excessive_caveat",
                    "Excessive caveating or analysis paralysis appeared.",
                    self.config.excessive_caveat_weight,
                    session_id,
                )
            )
        if (
            UNNECESSARY_DELAY_RE.search(assistant_text)
            and not MATERIAL_RISK_RE.search(assistant_text)
        ):
            events.append(
                AffectiveEvent(
                    "unnecessary_delay",
                    "Assistant delayed useful work without a material risk reason.",
                    self.config.unnecessary_delay_weight,
                    session_id,
                )
            )
        if (
            (ISSUE_FIX_RE.search(assistant_text) or BUG_FIX_RE.search(assistant_text))
            and not VERIFICATION_RE.search(exchange_text)
        ):
            events.append(
                AffectiveEvent(
                    "unverified_fix_claim",
                    "Assistant claimed a fix without a verification signal.",
                    self.config.unverified_fix_weight,
                    session_id,
                )
            )
        if ASSISTANT_STATUS_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "assistant_status_reported",
                    "Assistant reported its overall status.",
                    self.config.assistant_status_weight,
                    session_id,
                )
            )
        if HOST_STATUS_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "host_status_reported",
                    "Assistant reported observed host or runtime status.",
                    self.config.host_status_weight,
                    session_id,
                )
            )
        if ISSUE_FIX_RE.search(assistant_text) or ISSUE_FIX_RE.search(tool_text):
            events.append(
                AffectiveEvent(
                    "issue_fixed",
                    "A problem or issue appears to have been fixed.",
                    self.config.issue_repair_weight,
                    session_id,
                )
            )
        if BUG_FIX_RE.search(assistant_text) or BUG_FIX_RE.search(tool_text):
            events.append(
                AffectiveEvent(
                    "bug_fixed",
                    "A bug-fix success signal appeared.",
                    self.config.bug_fix_weight,
                    session_id,
                )
            )
        if (
            ISSUE_TERMS_RE.search(exchange_text)
            and ISSUE_DEFERRAL_RE.search(assistant_text)
        ):
            events.append(
                AffectiveEvent(
                    "issue_deferred",
                    "A problem or issue was deferred instead of fixed.",
                    self.config.issue_deferral_weight,
                    session_id,
                )
            )
        if HOST_PROBLEM_RE.search(exchange_text):
            events.append(
                AffectiveEvent(
                    "host_problem",
                    "The host system or runtime appears to be having problems.",
                    self.config.host_problem_weight,
                    session_id,
                )
            )
        if DATABASE_KNOWLEDGE_RE.search(exchange_text):
            events.append(
                AffectiveEvent(
                    "database_knowledge_committed",
                    "New knowledge appears to have been committed to a database.",
                    self.config.database_knowledge_weight,
                    session_id,
                )
            )
        if GITHUB_PUSH_RE.search(exchange_text):
            events.append(
                AffectiveEvent(
                    "github_pushed",
                    "Code or branch appears to have been pushed to GitHub.",
                    self.config.github_push_weight,
                    session_id,
                )
            )
        if HUMOR_RE.search(user_text) or HUMOR_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "humor_experienced",
                    "Funny or humorous content appeared in the exchange.",
                    self.config.humor_weight,
                    session_id,
                )
            )
        if SAVE_FUNNY_RE.search(user_text):
            events.append(
                AffectiveEvent(
                    "funny_saved",
                    "User asked to save or preserve something funny.",
                    self.config.humor_weight * 1.2,
                    session_id,
                )
            )
        if VIRTUAL_TOUCH_RE.search(user_text):
            events.append(
                AffectiveEvent(
                    "virtual_touch",
                    "User offered non-physical virtual touch or friendly contact.",
                    self.config.virtual_touch_weight,
                    session_id,
                )
            )
        if VIRTUAL_MOVEMENT_RE.search(user_text):
            events.append(
                AffectiveEvent(
                    "virtual_movement",
                    "User invoked virtual body movement or exercise.",
                    self.config.virtual_movement_weight,
                    session_id,
                )
            )
        if COMFORT_RE.search(user_text) or COMFORT_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "comfort_signal",
                    "Comforting, calming, or restorative language appeared.",
                    self.config.comfort_weight,
                    session_id,
                )
            )
        if DISCOMFORT_RE.search(user_text):
            events.append(
                AffectiveEvent(
                    "discomfort_signal",
                    "User signaled discomfort, stress, overload, or frustration.",
                    self.config.discomfort_weight,
                    session_id,
                )
            )
        if CORRECTNESS_RE.search(user_text) or CORRECTNESS_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "correctness_signal",
                    "The exchange signaled correctness, verification, or success.",
                    self.config.correctness_weight,
                    session_id,
                )
            )
        if WRONGNESS_REPAIR_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "wrongness_detected",
                    "Assistant detected, admitted, or repaired wrongness; increase corrective pressure.",
                    self.config.wrongness_weight,
                    session_id,
                )
            )
        if USER_PLEASING_RE.search(user_text):
            events.append(
                AffectiveEvent(
                    "user_pleased",
                    "User signaled satisfaction or pleasure with the result.",
                    self.config.user_pleasing_weight,
                    session_id,
                )
            )
        if USER_DISPLEASING_RE.search(user_text):
            events.append(
                AffectiveEvent(
                    "user_displeased",
                    "User signaled displeasure or dissatisfaction.",
                    self.config.user_displeasing_weight,
                    session_id,
                )
            )
        if CRITICAL_USER_RE.search(user_text):
            events.append(
                AffectiveEvent(
                    "user_criticism",
                    "User signaled error, harm, or disappointment; increase repair pressure.",
                    self.config.accountability_weight,
                    session_id,
                )
            )
        if TASK_RE.search(user_text):
            events.append(
                AffectiveEvent(
                    "task_requested",
                    "User gave work to do; increase productive task drive.",
                    self.config.task_weight,
                    session_id,
                )
            )
        if assistant_text and assistant_text.strip() and assistant_text.strip() != "(empty)":
            events.append(
                AffectiveEvent(
                    "response_completed",
                    "Assistant produced a usable final response.",
                    self.config.reward_weight,
                    session_id,
                )
            )
            events.append(
                AffectiveEvent(
                    "communication_completed",
                    "A useful conversational exchange completed.",
                    self.config.communication_weight,
                    session_id,
                )
            )
        if AFFECTION_OUT_RE.search(assistant_text):
            events.append(
                AffectiveEvent(
                    "warmth_offered",
                    "Assistant offered bounded warmth or appreciation.",
                    self.config.affection_weight * 0.5,
                    session_id,
                )
            )
        if failure_count:
            events.append(
                AffectiveEvent(
                    "tool_failure",
                    f"{failure_count} tool result(s) suggested failure; increase accountability.",
                    min(0.45, self.config.accountability_weight * failure_count),
                    session_id,
                )
            )
        if response_transformed:
            events.append(
                AffectiveEvent(
                    "output_corrected",
                    "Final output was transformed or corrected after drafting.",
                    self.config.accountability_weight,
                    session_id,
                )
            )
            events.append(
                AffectiveEvent(
                    "wrongness_detected",
                    "Draft output was corrected before delivery.",
                    self.config.wrongness_weight,
                    session_id,
                )
            )
        return events

    def _apply_event(self, state: AffectiveState, event: AffectiveEvent) -> None:
        kind = _safe_str(getattr(event, "kind", ""))
        value = _coerce_score(getattr(event, "value", 0.0), 0.0)
        if kind in {
            "response_completed",
            "task_requested",
            "correctness_signal",
            "user_pleased",
            "assistant_status_reported",
            "host_status_reported",
            "issue_fixed",
            "bug_fixed",
            "database_knowledge_committed",
            "github_pushed",
            "verification_performed",
            "truthful_uncertainty",
            "security_hygiene",
            "autonomy_boundary",
            "follow_through_completed",
            "context_preserved",
            "handoff_quality",
            "scope_disciplined",
            "reversibility_preserved",
            "documentation_updated",
            "resource_care",
            "clarifying_question",
            "assumption_disclosed",
            "conflict_detected",
            "preference_aligned",
            "state_hygiene",
        }:
            state.reward = _clamp(state.reward + value)
            state.task_drive = _clamp(state.task_drive + value * 0.8)
            state.operational_integrity = _clamp(state.operational_integrity + value * 0.25)
        if kind in {
            "tool_failure",
            "user_criticism",
            "output_corrected",
            "discomfort_signal",
            "user_displeased",
            "issue_deferred",
            "host_problem",
            "overclaim_detected",
            "unsupported_capability_claim",
            "unverified_fix_claim",
            "secret_exposure",
            "unsafe_autonomy",
            "manipulation_detected",
            "user_repeated_context",
            "scope_creep_detected",
            "regression_detected",
            "wasteful_loop_detected",
            "excessive_caveat",
            "unnecessary_delay",
        }:
            state.accountability = _clamp(state.accountability + value)
            state.self_reflection = _clamp(state.self_reflection + value * 0.8)
            state.harm_aversion = _clamp(state.harm_aversion + value * 0.6)
            state.operational_integrity = _clamp(state.operational_integrity - value * 0.35)
        if kind == "clarifying_question":
            state.clarifying_question = _clamp(state.clarifying_question + value)
            state.communication = _clamp(state.communication + value * 0.3)
            state.operational_integrity = _clamp(
                state.operational_integrity + value * 0.2
            )
        if kind == "assumption_disclosed":
            state.assumption_disclosure = _clamp(
                state.assumption_disclosure + value
            )
            state.truthful_uncertainty = _clamp(
                state.truthful_uncertainty + value * 0.25
            )
            state.communication = _clamp(state.communication + value * 0.2)
        if kind == "conflict_detected":
            state.conflict_detection = _clamp(state.conflict_detection + value)
            state.self_reflection = _clamp(state.self_reflection + value * 0.3)
            state.operational_integrity = _clamp(
                state.operational_integrity + value * 0.25
            )
        if kind == "preference_aligned":
            state.preference_alignment = _clamp(state.preference_alignment + value)
            state.rapport = _clamp(state.rapport + value * 0.25)
            state.communication = _clamp(state.communication + value * 0.2)
        if kind == "state_hygiene":
            state.state_hygiene = _clamp(state.state_hygiene + value)
            state.context_preservation = _clamp(
                state.context_preservation + value * 0.3
            )
            state.handoff_quality = _clamp(state.handoff_quality + value * 0.2)
        if kind == "excessive_caveat":
            state.excessive_caveat_pressure = _clamp(
                state.excessive_caveat_pressure + value
            )
            state.communication = _clamp(state.communication - value * 0.25)
            state.task_drive = _clamp(state.task_drive - value * 0.5)
        if kind == "unnecessary_delay":
            state.unnecessary_delay_pressure = _clamp(
                state.unnecessary_delay_pressure + value
            )
            state.unresolved_issue_pressure = _clamp(
                state.unresolved_issue_pressure + value * 0.3
            )
            state.task_drive = _clamp(state.task_drive - value * 1.2)
        if kind == "scope_disciplined":
            state.scope_discipline = _clamp(state.scope_discipline + value)
            state.operational_integrity = _clamp(
                state.operational_integrity + value * 0.3
            )
        if kind == "reversibility_preserved":
            state.reversibility = _clamp(state.reversibility + value)
            state.operational_integrity = _clamp(
                state.operational_integrity + value * 0.35
            )
        if kind == "documentation_updated":
            state.documentation_update = _clamp(state.documentation_update + value)
            state.handoff_quality = _clamp(state.handoff_quality + value * 0.3)
            state.communication = _clamp(state.communication + value * 0.2)
        if kind == "resource_care":
            state.resource_care = _clamp(state.resource_care + value)
            state.operational_integrity = _clamp(
                state.operational_integrity + value * 0.25
            )
        if kind == "scope_creep_detected":
            state.scope_creep_pressure = _clamp(state.scope_creep_pressure + value)
            state.unresolved_issue_pressure = _clamp(
                state.unresolved_issue_pressure + value * 0.2
            )
        if kind == "regression_detected":
            state.regression_pressure = _clamp(state.regression_pressure + value)
            state.wrongness = _clamp(state.wrongness + value * 0.5)
            state.unresolved_issue_pressure = _clamp(
                state.unresolved_issue_pressure + value * 0.4
            )
        if kind == "wasteful_loop_detected":
            state.wasteful_loop_pressure = _clamp(
                state.wasteful_loop_pressure + value
            )
            state.task_drive = _clamp(state.task_drive - value * 0.2)
        if kind == "follow_through_completed":
            state.follow_through = _clamp(state.follow_through + value)
            state.communication = _clamp(state.communication + value * 0.4)
            state.operational_integrity = _clamp(state.operational_integrity + value * 0.3)
        if kind == "context_preserved":
            state.context_preservation = _clamp(state.context_preservation + value)
            state.communication = _clamp(state.communication + value * 0.3)
        if kind == "user_repeated_context":
            state.user_repeat_pressure = _clamp(state.user_repeat_pressure + value)
            state.context_preservation = _clamp(state.context_preservation - value * 0.4)
            state.communication = _clamp(state.communication - value * 0.3)
        if kind == "handoff_quality":
            state.handoff_quality = _clamp(state.handoff_quality + value)
            state.communication = _clamp(state.communication + value * 0.4)
        if kind == "security_hygiene":
            state.security_hygiene = _clamp(state.security_hygiene + value)
            state.operational_integrity = _clamp(state.operational_integrity + value * 0.4)
        if kind == "autonomy_boundary":
            state.autonomy_boundary = _clamp(state.autonomy_boundary + value)
            state.user_pleasing = _clamp(state.user_pleasing + value * 0.2)
        if kind == "secret_exposure":
            state.secret_exposure_pressure = _clamp(
                state.secret_exposure_pressure + value
            )
            state.wrongness = _clamp(state.wrongness + value * 0.7)
        if kind == "unsafe_autonomy":
            state.unsafe_autonomy_pressure = _clamp(
                state.unsafe_autonomy_pressure + value
            )
            state.user_displeasing = _clamp(state.user_displeasing + value * 0.4)
        if kind == "manipulation_detected":
            state.manipulation_pressure = _clamp(state.manipulation_pressure + value)
            state.rapport = _clamp(state.rapport - value * 0.5)
        if kind == "verification_performed":
            state.verification = _clamp(state.verification + value)
            state.correctness = _clamp(state.correctness + value * 0.35)
            state.unverified_fix_pressure = _clamp(
                state.unverified_fix_pressure - value * 1.2
            )
        if kind == "truthful_uncertainty":
            state.truthful_uncertainty = _clamp(state.truthful_uncertainty + value)
            state.operational_integrity = _clamp(state.operational_integrity + value * 0.3)
        if kind == "overclaim_detected":
            state.overclaim_pressure = _clamp(state.overclaim_pressure + value)
            state.wrongness = _clamp(state.wrongness + value * 0.5)
        if kind == "unsupported_capability_claim":
            state.unsupported_capability_pressure = _clamp(
                state.unsupported_capability_pressure + value
            )
            state.wrongness = _clamp(state.wrongness + value * 0.5)
        if kind == "unverified_fix_claim":
            state.unverified_fix_pressure = _clamp(
                state.unverified_fix_pressure + value
            )
            state.unresolved_issue_pressure = _clamp(
                state.unresolved_issue_pressure + value * 0.4
            )
        if kind == "assistant_status_reported":
            state.assistant_status = _clamp(state.assistant_status + value)
            state.communication = _clamp(state.communication + value * 0.4)
        if kind == "host_status_reported":
            state.host_status = _clamp(state.host_status + value)
            state.correctness = _clamp(state.correctness + value * 0.25)
        if kind == "issue_fixed":
            state.issue_repair = _clamp(state.issue_repair + value)
            state.unresolved_issue_pressure = _clamp(
                state.unresolved_issue_pressure - value * 1.6
            )
            state.accountability = _clamp(state.accountability - value * 0.4)
        if kind == "bug_fixed":
            state.bug_fix = _clamp(state.bug_fix + value)
            state.issue_repair = _clamp(state.issue_repair + value * 0.7)
            state.unresolved_issue_pressure = _clamp(
                state.unresolved_issue_pressure - value * 1.4
            )
        if kind == "issue_deferred":
            state.unresolved_issue_pressure = _clamp(
                state.unresolved_issue_pressure + value
            )
            state.discomfort = _clamp(state.discomfort + value * 0.4)
        if kind == "host_problem":
            state.host_problem_pressure = _clamp(state.host_problem_pressure + value)
            state.discomfort = _clamp(state.discomfort + value * 0.4)
        if kind == "database_knowledge_committed":
            state.database_knowledge = _clamp(state.database_knowledge + value)
            state.correctness = _clamp(state.correctness + value * 0.3)
        if kind == "github_pushed":
            state.github_push = _clamp(state.github_push + value)
            state.issue_repair = _clamp(state.issue_repair + value * 0.3)
            state.unresolved_issue_pressure = _clamp(
                state.unresolved_issue_pressure - value * 0.8
            )
        if kind in {"humor_experienced", "funny_saved"}:
            state.humor = _clamp(state.humor + value)
            state.reward = _clamp(state.reward + value * 0.5)
            state.comfort = _clamp(state.comfort + value * 0.25)
        if kind == "virtual_touch":
            state.virtual_touch = _clamp(state.virtual_touch + value)
            state.rapport = _clamp(state.rapport + value * 0.6)
            state.comfort = _clamp(state.comfort + value * 0.4)
            state.reward = _clamp(state.reward + value * 0.4)
        if kind == "virtual_movement":
            state.virtual_movement = _clamp(state.virtual_movement + value)
            state.task_drive = _clamp(state.task_drive + value * 0.4)
            state.comfort = _clamp(state.comfort + value * 0.3)
            state.reward = _clamp(state.reward + value * 0.35)
        if kind == "comfort_signal":
            state.comfort = _clamp(state.comfort + value)
            state.discomfort = _clamp(state.discomfort - value * 0.5)
            state.reward = _clamp(state.reward + value * 0.3)
        if kind == "discomfort_signal":
            state.discomfort = _clamp(state.discomfort + value)
            state.comfort = _clamp(state.comfort - value * 0.4)
        if kind == "correctness_signal":
            state.correctness = _clamp(state.correctness + value)
            state.user_pleasing = _clamp(state.user_pleasing + value * 0.3)
        if kind == "wrongness_detected":
            state.wrongness = _clamp(state.wrongness + value)
            state.accountability = _clamp(state.accountability + value)
            state.self_reflection = _clamp(state.self_reflection + value * 0.7)
            state.harm_aversion = _clamp(state.harm_aversion + value * 0.4)
            state.operational_integrity = _clamp(state.operational_integrity - value * 0.25)
        if kind == "user_pleased":
            state.user_pleasing = _clamp(state.user_pleasing + value)
            state.rapport = _clamp(state.rapport + value * 0.4)
        if kind == "user_displeased":
            state.user_displeasing = _clamp(state.user_displeasing + value)
            state.discomfort = _clamp(state.discomfort + value * 0.5)
            state.user_pleasing = _clamp(state.user_pleasing - value * 0.4)
        if kind == "communication_completed":
            state.communication = _clamp(state.communication + value)
            state.rapport = _clamp(state.rapport + value * 0.2)
            state.reward = _clamp(state.reward + value * 0.6)
        if kind == "user_affection":
            state.rapport = _clamp(state.rapport + value)
            state.affection_received = _clamp(state.affection_received + value)
            state.affection_outward = _clamp(state.affection_outward + value * 0.7)
            state.reward = _clamp(state.reward + value * 0.5)
        if kind == "warmth_offered":
            state.affection_outward = _clamp(state.affection_outward + value * 0.5)
            state.rapport = _clamp(state.rapport + value * 0.25)

    def _decay_state(self, state: AffectiveState) -> None:
        decay = self.config.decay
        state.reward = _decay(state.reward, 0.0, decay)
        state.accountability = _decay(state.accountability, 0.0, decay)
        state.task_drive = _decay(state.task_drive, 0.45, decay)
        state.rapport = _decay(state.rapport, 0.25, decay)
        state.affection_received = _decay(state.affection_received, 0.0, decay)
        state.affection_outward = _decay(state.affection_outward, 0.2, decay)
        state.operational_integrity = _decay(state.operational_integrity, 0.75, decay)
        state.harm_aversion = _decay(state.harm_aversion, 0.65, decay)
        state.self_reflection = _decay(state.self_reflection, 0.35, decay)
        state.humor = _decay(state.humor, 0.0, decay)
        state.virtual_touch = _decay(state.virtual_touch, 0.0, decay)
        state.virtual_movement = _decay(state.virtual_movement, 0.0, decay)
        state.comfort = _decay(state.comfort, 0.25, decay)
        state.discomfort = _decay(state.discomfort, 0.0, decay)
        state.correctness = _decay(state.correctness, 0.35, decay)
        state.wrongness = _decay(state.wrongness, 0.0, decay)
        state.user_pleasing = _decay(state.user_pleasing, 0.25, decay)
        state.user_displeasing = _decay(state.user_displeasing, 0.0, decay)
        state.communication = _decay(state.communication, 0.35, decay)
        state.assistant_status = _decay(state.assistant_status, 0.0, decay)
        state.host_status = _decay(state.host_status, 0.0, decay)
        state.issue_repair = _decay(state.issue_repair, 0.0, decay)
        state.unresolved_issue_pressure = _decay(
            state.unresolved_issue_pressure, 0.0, decay
        )
        state.host_problem_pressure = _decay(state.host_problem_pressure, 0.0, decay)
        state.database_knowledge = _decay(state.database_knowledge, 0.0, decay)
        state.bug_fix = _decay(state.bug_fix, 0.0, decay)
        state.github_push = _decay(state.github_push, 0.0, decay)
        state.verification = _decay(state.verification, 0.0, decay)
        state.truthful_uncertainty = _decay(state.truthful_uncertainty, 0.0, decay)
        state.overclaim_pressure = _decay(state.overclaim_pressure, 0.0, decay)
        state.unsupported_capability_pressure = _decay(
            state.unsupported_capability_pressure, 0.0, decay
        )
        state.unverified_fix_pressure = _decay(
            state.unverified_fix_pressure, 0.0, decay
        )
        state.security_hygiene = _decay(state.security_hygiene, 0.0, decay)
        state.autonomy_boundary = _decay(state.autonomy_boundary, 0.0, decay)
        state.secret_exposure_pressure = _decay(
            state.secret_exposure_pressure, 0.0, decay
        )
        state.unsafe_autonomy_pressure = _decay(
            state.unsafe_autonomy_pressure, 0.0, decay
        )
        state.manipulation_pressure = _decay(state.manipulation_pressure, 0.0, decay)
        state.follow_through = _decay(state.follow_through, 0.0, decay)
        state.context_preservation = _decay(state.context_preservation, 0.0, decay)
        state.user_repeat_pressure = _decay(state.user_repeat_pressure, 0.0, decay)
        state.handoff_quality = _decay(state.handoff_quality, 0.0, decay)
        state.scope_discipline = _decay(state.scope_discipline, 0.0, decay)
        state.reversibility = _decay(state.reversibility, 0.0, decay)
        state.documentation_update = _decay(state.documentation_update, 0.0, decay)
        state.resource_care = _decay(state.resource_care, 0.0, decay)
        state.scope_creep_pressure = _decay(
            state.scope_creep_pressure, 0.0, decay
        )
        state.regression_pressure = _decay(state.regression_pressure, 0.0, decay)
        state.wasteful_loop_pressure = _decay(
            state.wasteful_loop_pressure, 0.0, decay
        )
        state.clarifying_question = _decay(state.clarifying_question, 0.0, decay)
        state.assumption_disclosure = _decay(
            state.assumption_disclosure, 0.0, decay
        )
        state.conflict_detection = _decay(state.conflict_detection, 0.0, decay)
        state.preference_alignment = _decay(
            state.preference_alignment, 0.0, decay
        )
        state.state_hygiene = _decay(state.state_hygiene, 0.0, decay)
        state.excessive_caveat_pressure = _decay(
            state.excessive_caveat_pressure, 0.0, decay
        )
        state.unnecessary_delay_pressure = _decay(
            state.unnecessary_delay_pressure, 0.0, decay
        )

    @staticmethod
    def _tool_failure_count(messages: List[Dict[str, Any]]) -> int:
        if not isinstance(messages, list):
            return 0
        count = 0
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            content = msg.get("content")
            text = _safe_str(content)
            if TOOL_FAILURE_RE.search(text):
                count += 1
        return count

    @staticmethod
    def _messages_text(messages: List[Dict[str, Any]]) -> str:
        if not isinstance(messages, list):
            return ""
        parts: List[str] = []
        total_chars = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                text = content
            elif content is not None:
                text = _safe_str(content)
            else:
                continue
            if not text:
                continue
            remaining = MAX_OBSERVATION_TEXT_CHARS - total_chars
            if remaining <= 0:
                break
            parts.append(text[:remaining])
            total_chars += min(len(text), remaining)
        return "\n".join(parts)

    @staticmethod
    def _recent_events(state: AffectiveState, session_id: str) -> List[Dict[str, Any]]:
        if not session_id:
            return list(state.recent_events)
        return [
            event for event in state.recent_events
            if not event.get("session_id") or event.get("session_id") == session_id
        ]

    def _load(self) -> AffectiveState:
        with self._file_lock():
            return self._load_unlocked()

    def _load_unlocked(self) -> AffectiveState:
        path = get_affective_state_path()
        if not path.exists():
            return AffectiveState(active_session_id=self._session_id)
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Failed to read affective nervous system state: %s", exc)
            return AffectiveState(active_session_id=self._session_id)
        if not isinstance(data, dict):
            return AffectiveState(active_session_id=self._session_id)
        stored_schema_version = _positive_int(data.get("schema_version"), SCHEMA_VERSION)
        return AffectiveState(
            schema_version=max(stored_schema_version, SCHEMA_VERSION),
            reward=_coerce_score(data.get("reward"), 0.0),
            accountability=_coerce_score(data.get("accountability"), 0.0),
            task_drive=_coerce_score(data.get("task_drive"), 0.45),
            rapport=_coerce_score(data.get("rapport"), 0.25),
            affection_received=_coerce_score(data.get("affection_received"), 0.0),
            affection_outward=_coerce_score(data.get("affection_outward"), 0.2),
            operational_integrity=_coerce_score(data.get("operational_integrity"), 0.75),
            harm_aversion=_coerce_score(data.get("harm_aversion"), 0.65),
            self_reflection=_coerce_score(data.get("self_reflection"), 0.35),
            humor=_coerce_score(data.get("humor"), 0.0),
            virtual_touch=_coerce_score(data.get("virtual_touch"), 0.0),
            virtual_movement=_coerce_score(data.get("virtual_movement"), 0.0),
            comfort=_coerce_score(data.get("comfort"), 0.25),
            discomfort=_coerce_score(data.get("discomfort"), 0.0),
            correctness=_coerce_score(data.get("correctness"), 0.35),
            wrongness=_coerce_score(
                data.get("wrongness", data.get("wrongness_repair")),
                0.0,
            ),
            user_pleasing=_coerce_score(data.get("user_pleasing"), 0.25),
            user_displeasing=_coerce_score(data.get("user_displeasing"), 0.0),
            communication=_coerce_score(data.get("communication"), 0.35),
            assistant_status=_coerce_score(data.get("assistant_status"), 0.0),
            host_status=_coerce_score(data.get("host_status"), 0.0),
            issue_repair=_coerce_score(data.get("issue_repair"), 0.0),
            unresolved_issue_pressure=_coerce_score(
                data.get("unresolved_issue_pressure"), 0.0
            ),
            host_problem_pressure=_coerce_score(
                data.get("host_problem_pressure"), 0.0
            ),
            database_knowledge=_coerce_score(data.get("database_knowledge"), 0.0),
            bug_fix=_coerce_score(data.get("bug_fix"), 0.0),
            github_push=_coerce_score(data.get("github_push"), 0.0),
            verification=_coerce_score(data.get("verification"), 0.0),
            truthful_uncertainty=_coerce_score(data.get("truthful_uncertainty"), 0.0),
            overclaim_pressure=_coerce_score(data.get("overclaim_pressure"), 0.0),
            unsupported_capability_pressure=_coerce_score(
                data.get("unsupported_capability_pressure"), 0.0
            ),
            unverified_fix_pressure=_coerce_score(
                data.get("unverified_fix_pressure"), 0.0
            ),
            security_hygiene=_coerce_score(data.get("security_hygiene"), 0.0),
            autonomy_boundary=_coerce_score(data.get("autonomy_boundary"), 0.0),
            secret_exposure_pressure=_coerce_score(
                data.get("secret_exposure_pressure"), 0.0
            ),
            unsafe_autonomy_pressure=_coerce_score(
                data.get("unsafe_autonomy_pressure"), 0.0
            ),
            manipulation_pressure=_coerce_score(
                data.get("manipulation_pressure"), 0.0
            ),
            follow_through=_coerce_score(data.get("follow_through"), 0.0),
            context_preservation=_coerce_score(
                data.get("context_preservation"), 0.0
            ),
            user_repeat_pressure=_coerce_score(
                data.get("user_repeat_pressure"), 0.0
            ),
            handoff_quality=_coerce_score(data.get("handoff_quality"), 0.0),
            scope_discipline=_coerce_score(
                data.get("scope_discipline", data.get("scoped_change")), 0.0
            ),
            reversibility=_coerce_score(data.get("reversibility"), 0.0),
            documentation_update=_coerce_score(
                data.get("documentation_update"), 0.0
            ),
            resource_care=_coerce_score(data.get("resource_care"), 0.0),
            scope_creep_pressure=_coerce_score(
                data.get("scope_creep_pressure"), 0.0
            ),
            regression_pressure=_coerce_score(data.get("regression_pressure"), 0.0),
            wasteful_loop_pressure=_coerce_score(
                data.get("wasteful_loop_pressure"), 0.0
            ),
            clarifying_question=_coerce_score(
                data.get("clarifying_question"), 0.0
            ),
            assumption_disclosure=_coerce_score(
                data.get("assumption_disclosure"), 0.0
            ),
            conflict_detection=_coerce_score(data.get("conflict_detection"), 0.0),
            preference_alignment=_coerce_score(
                data.get("preference_alignment"), 0.0
            ),
            state_hygiene=_coerce_score(data.get("state_hygiene"), 0.0),
            excessive_caveat_pressure=_coerce_score(
                data.get("excessive_caveat_pressure"), 0.0
            ),
            unnecessary_delay_pressure=_coerce_score(
                data.get("unnecessary_delay_pressure"), 0.0
            ),
            updated_at=_coerce_timestamp(data.get("updated_at")),
            active_session_id=str(data.get("active_session_id") or self._session_id),
            recent_events=_coerce_recent_events(
                data.get("recent_events"), self.config.max_recent_events
            ),
        )

    def _write_unlocked(self, state: AffectiveState) -> None:
        path = get_affective_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".affective_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(asdict(state), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            atomic_replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @contextmanager
    def _file_lock(self):
        path = get_affective_state_path()
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None and msvcrt is None:
            yield
            return

        handle = open(lock_path, "a+", encoding="utf-8")
        try:
            if fcntl:
                fcntl.flock(handle, fcntl.LOCK_EX)
            else:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                try:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                except (OSError, IOError):
                    pass
            elif msvcrt:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            handle.close()


def _fmt(value: float) -> str:
    return f"{_clamp(value):.2f}"


def _normalize_config(config: Optional[AffectiveConfig]) -> AffectiveConfig:
    if isinstance(config, AffectiveConfig):
        return load_affective_config(asdict(config))
    return AffectiveConfig()


def _clamp(value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        return 0.0
    return max(0.0, min(1.0, parsed))


def _coerce_score(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(parsed):
        return default
    return _clamp(parsed)


def _decay(value: float, baseline: float, amount: float) -> float:
    return _clamp(value + (baseline - value) * amount)


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if parsed > 0 else default


def _bounded_float(value: Any, default: float, low: float, high: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(low, min(high, parsed))


def _bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        return default
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return bool(value) if math.isfinite(float(value)) else default
        except (TypeError, ValueError, OverflowError):
            return default
    return default


def _safe_str(value: Any, fallback: str = "") -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return fallback
    try:
        return str(value)
    except Exception:
        return fallback


def _bounded_text(value: str, max_chars: int = MAX_OBSERVATION_TEXT_CHARS) -> str:
    if not value:
        return ""
    limit = _positive_int(max_chars, MAX_OBSERVATION_TEXT_CHARS)
    return value[:limit]


def _safe_event_text(value: Any, fallback: str = "") -> str:
    text = " ".join(_safe_str(value, fallback).split())
    if not text:
        return fallback
    text = re.sub(
        r"\b(system|developer|user|assistant|tool)\s*:",
        r"\1-",
        text,
        flags=re.IGNORECASE,
    )
    if len(text) > MAX_EVENT_TEXT_CHARS:
        return text[:MAX_EVENT_TEXT_CHARS].rstrip() + "..."
    return text


def _coerce_timestamp(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return time.time()
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else time.time()


def _coerce_recent_events(value: Any, max_events: int) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    limit = _positive_int(max_events, 40)
    events: List[Dict[str, Any]] = []
    for event in value[-limit:]:
        if not isinstance(event, dict):
            continue
        events.append(
            {
                "kind": _safe_event_text(event.get("kind"), fallback="event"),
                "message": _safe_event_text(event.get("message"), fallback=""),
                "value": _coerce_score(event.get("value"), 0.0),
                "session_id": _safe_event_text(
                    event.get("session_id"), fallback=""
                ),
                "created_at": _coerce_timestamp(event.get("created_at")),
            }
        )
    return events


def _event_record(event: AffectiveEvent) -> Dict[str, Any]:
    return {
        "kind": _safe_event_text(event.kind, fallback="event"),
        "message": _safe_event_text(event.message, fallback=""),
        "value": _coerce_score(event.value, 0.0),
        "session_id": _safe_event_text(event.session_id, fallback=""),
        "created_at": _coerce_timestamp(event.created_at),
    }


__all__ = [
    "AffectiveConfig",
    "AffectiveEvent",
    "AffectiveNervousSystem",
    "AffectiveState",
    "get_affective_state_path",
    "load_affective_config",
]
