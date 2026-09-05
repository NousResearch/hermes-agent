"""Opt-in adaptive reasoning-effort adjustment (config: agent.adaptive_reasoning).

When enabled, each eligible user turn is classified by a deterministic,
bounded heuristic — no extra LLM call, no new model tool — and the agent's
``reasoning_config`` is temporarily adjusted away from the configured
baseline (``agent.reasoning_effort``) for turns that materially warrant it:

- ``high``  — substantial multi-step analysis, coding/debugging,
  consequential infrastructure changes, or in-depth research/synthesis.
- ``xhigh`` — genuinely difficult architecture/security/high-stakes
  diagnosis or complex cross-component work, and only with corroborating
  signals (never on a lone keyword).
- ``low``   — only when ``min_effort`` opts a floor below the baseline in,
  and only for positively simple turns (casual chatter, short factual
  questions, single mechanical retrieval steps) carrying no complexity,
  error, code, multi-question, or multi-step signal. Ambiguity stays at
  the baseline; without ``min_effort`` the feature is escalation-only.

Adjustment requires an explicit reasoning baseline: when ``reasoning_config``
is None the provider decides whether thinking exists at all, and installing a
config for one turn would toggle thinking presence on/off mid-conversation —
which fragments the Anthropic prompt-cache namespace. With a baseline set,
thinking stays enabled at every selected level (``none`` is never selected
and a disabled baseline is never re-enabled), so current adaptive-thinking
models keep one cache namespace; legacy budget-token models may still re-key
their message cache when the budget changes.

The adjusted effort applies for the full tool-calling loop of that turn
(``AIAgent.run_conversation`` wraps the turn in begin/end below) and the
baseline is restored afterwards, so one adjusted task never rewrites the
session's configured level. Explicit user choices always win: surfaces set
``agent.reasoning_user_override`` when a session-scoped ``/reasoning`` pick
(or the Desktop effort menu) is active, which disables adjustment entirely.
Delegate subagents receive the parent's parsed policy and effective level:
they reclassify their own delegated goal within the same bounds, unless an
explicit ``delegation.reasoning_effort`` pins them (marked as a user
override by the delegate tool).

A single ``AgentNotice`` (kind="ttl") is emitted on the existing notice rail
when the level changes, deduplicated across consecutive turns at the same
level — model context is never touched, so role alternation is unaffected and
current adaptive-thinking models keep one prompt-cache namespace. Subagents
adjust silently. All continuation and
dedup state lives on the individual ``AIAgent`` object, so concurrent
sessions never share effective efforts.
"""

import functools
import inspect
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from hermes_constants import VALID_REASONING_EFFORTS

logger = logging.getLogger(__name__)

EFFORT_RANK: Dict[str, int] = {
    level: rank for rank, level in enumerate(VALID_REASONING_EFFORTS)
}

_DEFAULT_MAX_EFFORT = "xhigh"

# Regex work is bounded: signals are matched against at most this many
# characters (length-based signals still use the full text length).
_SCAN_LIMIT = 8000

NOTICE_KEY = "adaptive-reasoning"
_NOTICE_TTL_MS = 12000

_EFFORT_LABELS = {"low": "Low", "high": "High", "xhigh": "XHigh"}


@dataclass
class AdaptiveTurnToken:
    """Restore token for one escalated turn.

    ``applied`` is the exact dict object installed on the agent; restore is
    identity-guarded so a mid-turn rewrite (fallback activation re-resolving
    reasoning for a new model) is never clobbered.
    """

    saved: Optional[Dict[str, Any]]
    applied: Dict[str, Any]
    effort: str
    reason: str


# ── Deterministic signal table ───────────────────────────────────────────────
# Each category matches at most once; the score is the sum of matched weights.
# xhigh-class categories mark the turn as *potentially* xhigh, but xhigh still
# requires corroboration (total score >= _XHIGH_THRESHOLD).

_CATEGORIES: List[Tuple[str, str, int, bool, "re.Pattern[str]"]] = [
    (
        "security",
        "security-critical work",
        3,
        True,
        re.compile(
            r"security\s+(?:audit|review)|vulnerabilit|\bexploit|\bcve-\d"
            r"|threat\s+model|privilege\s+escalation|auth(?:entication)?\s+bypass"
            r"|cryptograph|penetration\s+test|\bhardening\b|incident\s+response",
            re.IGNORECASE,
        ),
    ),
    (
        "architecture",
        "architecture/design work",
        3,
        True,
        re.compile(
            r"\barchitect(?:ure|ural|ing)?\b|system\s+design|\bredesign"
            r"|cross-?component|\bdesign\b.{0,60}\b(?:system|architecture|protocol|schema|pipeline)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "high_stakes",
        "high-stakes diagnosis",
        3,
        True,
        re.compile(
            r"data\s+(?:loss|corruption)|\bcorrupt(?:ed|ion|s)?\b|\boutage\b"
            r"|\bincident\b|production\s+(?:down|outage|is\s+down)"
            r"|disaster\s+recovery|irreversibl",
            re.IGNORECASE,
        ),
    ),
    (
        "debugging",
        "debugging/diagnosis",
        3,
        False,
        re.compile(
            r"\bdebug|\bdiagnos|root[\s-]?cause|race\s+condition|\bdeadlock"
            r"|memory\s+leak|\bsegfault|\bcrash(?:es|ing|ed)?\b|\bregression\b"
            r"|\bflaky\b|\bintermittent"
            r"|why\s+(?:is|does|did|are|do|isn't|doesn't|won't)\b.{0,80}\b(?:fail|break|crash|hang|error|not\s+work)"
            r"|stops?\s+working|keeps?\s+failing",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "implementation",
        "multi-step implementation work",
        3,
        False,
        re.compile(
            r"\bimplement|\brefactor|\brewrit(?:e|ing)\b|\bintegrat(?:e|ing|ion)"
            r"|\bmigrat(?:e|ing|ion)|\boptimi[sz](?:e|ing)|add\s+support\s+for"
            r"|build\s+(?:a|an|the|out)\b"
            r"|write\s+(?:a|an|the)\b.{0,50}\b(?:module|service|feature|parser|library|tool|test\s+suite|script)\b"
            r"|fix\s+(?:the|this|that)\b.{0,40}\b(?:bug|issue|error)\b"
            r"|set\s+up\b.{0,50}\b(?:pipeline|cluster|server|environment)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "research",
        "in-depth research",
        3,
        False,
        re.compile(
            r"\bresearch\b|\binvestigat(?:e|ion)\b|deep[\s-]dive"
            r"|literature\s+review|comprehensive\s+(?:analysis|review|survey)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "analysis",
        "comparative analysis",
        2,
        False,
        re.compile(
            r"\banaly[sz]e\b|\bevaluate\b|\bcompar(?:e|ison)\b|\bassess\b"
            r"|\btrade-?offs?\b|pros\s+and\s+cons",
            re.IGNORECASE,
        ),
    ),
    (
        "infrastructure",
        "consequential infrastructure work",
        2,
        False,
        re.compile(
            r"\bdeploy(?:ment)?\b|\bprovision|\bkubernetes\b|\bk8s\b|\bterraform\b"
            r"|\bansible\b|\bsystemd\b|\bnginx\b|\bdns\b|\btls\b|certificates?\b"
            r"|\bfirewall\b|load\s+balancer|database\s+migration|\bbackup\b"
            r"|\bfailover\b|\bcluster\b|\bupgrade\b",
            re.IGNORECASE,
        ),
    ),
    (
        "error_evidence",
        "an error trace",
        2,
        False,
        re.compile(
            r"Traceback \(most recent call last\)|\n\s*File \"|\bException\b"
            r"|\bstack\s?trace\b|\berror:|\berrno\b|exit\s+code\s+\d+"
            r"|\bpanic(?:ked)?\b|segmentation\s+fault",
            re.IGNORECASE,
        ),
    ),
    (
        "consequence",
        "production impact",
        1,
        False,
        re.compile(
            r"\bproduction\b|\bprod\b|live\s+(?:system|site|server)|\bcustomers?\b",
            re.IGNORECASE,
        ),
    ),
]

_MULTI_STEP_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+\S", re.MULTILINE)

_CASUAL_RE = re.compile(
    r"(?:hi|hiya|hello|hey|yo|good\s+(?:morning|afternoon|evening|night)"
    r"|thanks?|thank\s+you(?:\s+so\s+much)?|thx|ty|cool|nice|great|awesome"
    r"|ok(?:ay)?|lol|haha|got\s+it|sounds\s+good|good\s+job|well\s+done"
    r"|no|yes|yep|nope|bye|goodbye|see\s+you|what'?s\s+up|how\s+are\s+you"
    r"|test|ping)[\s.!?~]*",
    re.IGNORECASE,
)

_CONTINUATION_RE = re.compile(
    r"(?:yes|yeah|yep|ya|sure|ok(?:ay)?|sounds\s+good|go\s+ahead|go\s+for\s+it"
    r"|proceed|continue|keep\s+going|do\s+it|please\s+do|do\s+that|lgtm"
    r"|looks\s+good(?:\s+to\s+me)?|approved|yes\s+please|carry\s+on)[\s.!?~]*",
    re.IGNORECASE,
)

_HIGH_THRESHOLD = 3
_XHIGH_THRESHOLD = 6

# ── Positive-simplicity patterns (downshift candidates) ─────────────────────
# ``low`` requires positive evidence of simplicity, never mere absence of
# complexity signals. Each pattern is a fullmatch on the normalized text and
# is only consulted when the turn carries zero category/structural signals.

# Fact lookups: interrogatives that ask for a single concrete answer. "how do
# I ..." / "why ..." are deliberately excluded — those ask for procedure or
# explanation, which is ordinary (medium) work.
_SIMPLE_FACTUAL_RE = re.compile(
    r"(?:what|what's|who|who's|whom|whose|when|where|which"
    r"|how\s+(?:many|much|old|long|far|big|tall|often))\b"
    r"[^?.!;\n]{0,80}\??",
    re.IGNORECASE,
)
_SIMPLE_FACTUAL_MAX_WORDS = 12

# Single mechanical retrieval steps: read-only imperatives with one clause.
# State-changing verbs (restart/delete/deploy/...) are deliberately excluded.
_MECHANICAL_RE = re.compile(
    r"(?:please\s+)?(?:show|list|open|display|print|read|cat)\b[^,;.!?\n]{0,60}[.!]?",
    re.IGNORECASE,
)
_MECHANICAL_MAX_WORDS = 8


def parse_adaptive_reasoning_config(raw: Any) -> Optional[Dict[str, Any]]:
    """Sanitize the ``agent.adaptive_reasoning`` config section.

    Returns ``{"enabled": True, "max_effort": <level>}`` — plus
    ``"min_effort"`` when a valid downshift floor is opted in — when the
    feature is enabled, else ``None`` (absent section, non-dict, or enabled
    falsy). An unrecognized ``max_effort`` falls back to ``xhigh`` rather
    than disabling the feature the user explicitly turned on. An invalid
    ``min_effort`` (unknown level, ``none``/``false`` — thinking presence is
    never toggled — or a floor above the ceiling) is dropped, keeping the
    escalation-only behavior. The result is a fixed point of this function,
    so a parent's parsed policy can be re-parsed for a delegate child.
    """
    if not isinstance(raw, dict) or not raw.get("enabled"):
        return None
    max_effort = str(raw.get("max_effort") or _DEFAULT_MAX_EFFORT).strip().lower()
    if max_effort not in EFFORT_RANK:
        logger.warning(
            "Unknown adaptive_reasoning.max_effort '%s', using '%s'",
            raw.get("max_effort"),
            _DEFAULT_MAX_EFFORT,
        )
        max_effort = _DEFAULT_MAX_EFFORT
    cfg: Dict[str, Any] = {"enabled": True, "max_effort": max_effort}
    min_raw = raw.get("min_effort")
    if min_raw not in (None, ""):
        min_effort = str(min_raw).strip().lower()
        if min_effort not in EFFORT_RANK:
            logger.warning(
                "Unknown adaptive_reasoning.min_effort '%s', keeping "
                "escalation-only behavior",
                min_raw,
            )
        elif EFFORT_RANK[min_effort] > EFFORT_RANK[max_effort]:
            logger.warning(
                "adaptive_reasoning.min_effort '%s' is above max_effort '%s', "
                "ignoring the floor",
                min_effort,
                max_effort,
            )
        else:
            cfg["min_effort"] = min_effort
    return cfg


def extract_message_text(user_message: Any) -> str:
    """Best-effort text of a user message (str or multimodal content blocks)."""
    if isinstance(user_message, str):
        return user_message
    if isinstance(user_message, list):
        parts = []
        for block in user_message:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    if isinstance(user_message, dict) and isinstance(user_message.get("text"), str):
        return user_message["text"]
    return ""


def _positively_simple_reason(stripped: str, norm: str) -> str:
    """Reason string when the turn is positively simple, else "".

    Caller guarantees the turn carries no category or structural signals —
    this only checks for affirmative evidence of simplicity.
    """
    if _CASUAL_RE.fullmatch(norm):
        return "casual message"
    words = len(stripped.split())
    if words <= _SIMPLE_FACTUAL_MAX_WORDS and _SIMPLE_FACTUAL_RE.fullmatch(norm):
        return "simple factual request"
    if (
        words <= _MECHANICAL_MAX_WORDS
        and " and " not in norm
        and " then " not in norm
        and _MECHANICAL_RE.fullmatch(norm)
    ):
        return "single mechanical step"
    return ""


def classify_reasoning_effort(
    text: str, prior_effort: Optional[str] = None
) -> Tuple[str, str]:
    """Classify one user turn into (effort, reason) — deterministic, no I/O.

    ``prior_effort`` is the adaptive level of the previous turn (or None):
    short affirmative follow-ups inherit it so "go ahead" doesn't drop a
    complex task back to the baseline mid-flight. Unrelated trivial turns
    re-classify from scratch, so nothing stays stuck at xhigh.

    ``low`` is only returned on positive evidence of simplicity (casual
    chatter, a short fact lookup, a single read-only mechanical step) with
    zero complexity signals; whether it takes effect is the caller's floor
    decision. Ambiguous turns — including empty text, e.g. an image-only
    message — stay ``medium``.
    """
    stripped = (text or "").strip()
    if not stripped:
        return "medium", ""
    norm = re.sub(r"\s+", " ", stripped.lower())

    if prior_effort in EFFORT_RANK and _CONTINUATION_RE.fullmatch(norm):
        return prior_effort, "continuing the escalated task"

    # Terse turns (≤4 words, no code) never escalate — but they still get the
    # full signal scan below: a signal-bearing terse turn must not downshift.
    terse = len(stripped.split()) <= 4 and "```" not in stripped

    scan = stripped[:_SCAN_LIMIT]
    score = 0
    xhigh_class = False
    matched: List[Tuple[int, str]] = []
    for _name, label, weight, is_xhigh, pattern in _CATEGORIES:
        if pattern.search(scan):
            score += weight
            xhigh_class = xhigh_class or is_xhigh
            matched.append((weight, label))

    if len(_MULTI_STEP_RE.findall(scan)) >= 3:
        score += 2
        matched.append((2, "a multi-step request"))
    if len(stripped) > 500:
        score += 1 if len(stripped) <= 1500 else 2
        matched.append((1, "a detailed request"))
    if "```" in scan:
        score += 1
        matched.append((1, "included code"))
    if scan.count("?") >= 2:
        score += 1
        matched.append((1, "multiple questions"))

    if not terse:
        if xhigh_class and score >= _XHIGH_THRESHOLD:
            matched.sort(key=lambda item: -item[0])
            return "xhigh", " and ".join(label for _w, label in matched[:2])
        if score >= _HIGH_THRESHOLD:
            matched.sort(key=lambda item: -item[0])
            return "high", " and ".join(label for _w, label in matched[:2])

    if not matched:
        simple_reason = _positively_simple_reason(stripped, norm)
        if simple_reason:
            return "low", simple_reason
    return "medium", ""


def _notify_adjustment(agent: Any, effort: str, reason: str, lowered: bool) -> None:
    """Fire one AgentNotice on the existing notice rail (CLI/TUI/Desktop/
    messaging all relay it); fall back to a forced console line when no
    notice callback is bound. Subagents adjust silently — a child toast on
    the parent's rail would be noise. Never raises."""
    if getattr(agent, "platform", "") == "subagent":
        return
    label = _EFFORT_LABELS.get(effort, effort.capitalize())
    verb = "lowered" if lowered else "raised"
    suffix = f" — {reason}" if reason else ""
    text = f"🧠 Reasoning {verb} to {label} for this task{suffix}."
    try:
        callback = getattr(agent, "notice_callback", None)
        if callback:
            from agent.credits_tracker import AgentNotice

            callback(
                AgentNotice(
                    text=text,
                    level="info",
                    kind="ttl",
                    ttl_ms=_NOTICE_TTL_MS,
                    key=NOTICE_KEY,
                )
            )
            return
        vprint = getattr(agent, "_vprint", None)
        if callable(vprint):
            vprint(text, force=True)
    except Exception:
        logger.debug("adaptive reasoning notice emission failed", exc_info=True)


def begin_adaptive_reasoning_turn(
    agent: Any, user_message: Any, moa_config: Optional[Dict[str, Any]] = None
) -> Optional[AdaptiveTurnToken]:
    """Classify this turn and, if warranted, install an adjusted
    ``reasoning_config`` on the agent for the duration of the turn.

    Returns a restore token for :func:`end_adaptive_reasoning_turn`, or None
    when the feature is off, a user override is active, the turn is a MoA
    turn (per-slot reasoning owns effort there), no explicit reasoning
    baseline is set (adjusting would toggle thinking presence and fragment
    the prompt-cache namespace), or no adjustment applies.
    Never raises — a classification failure must not break the turn.
    """
    try:
        cfg = getattr(agent, "adaptive_reasoning", None)
        if not isinstance(cfg, dict) or not cfg.get("enabled"):
            return None
        if getattr(agent, "reasoning_user_override", False):
            agent._adaptive_prev_effort = None
            agent._adaptive_last_notified_effort = None
            return None
        if moa_config:
            return None
        reasoning_config = getattr(agent, "reasoning_config", None)
        if not isinstance(reasoning_config, dict):
            # No explicit baseline: the provider decides whether thinking
            # exists at all. Installing one for a single turn would toggle
            # thinking presence mid-conversation — inert instead.
            return None
        if reasoning_config.get("enabled") is False:
            # Thinking explicitly disabled — never silently re-enable it.
            return None
        baseline = str(reasoning_config.get("effort") or "medium").strip().lower()
        if baseline not in EFFORT_RANK:
            return None
        if isinstance(user_message, str):
            try:
                from hermes_cli.moa_config import decode_moa_turn

                if decode_moa_turn(user_message)[1] is not None:
                    return None
            except Exception:
                pass
        text = extract_message_text(user_message)
        effort, reason = classify_reasoning_effort(
            text, prior_effort=getattr(agent, "_adaptive_prev_effort", None)
        )
        max_effort = cfg.get("max_effort") or _DEFAULT_MAX_EFFORT
        if EFFORT_RANK.get(effort, 0) > EFFORT_RANK.get(max_effort, 0):
            effort = max_effort
        # The floor is the baseline unless a lower min_effort opted a
        # downshift range in; a below-floor classification clamps up to it.
        floor = cfg.get("min_effort")
        if floor not in EFFORT_RANK or EFFORT_RANK[floor] > EFFORT_RANK[baseline]:
            floor = baseline
        if EFFORT_RANK.get(effort, 0) < EFFORT_RANK[floor]:
            effort = floor
        if EFFORT_RANK.get(effort, 0) == EFFORT_RANK.get(baseline, 0):
            # Unchanged this turn: clear continuation + notice-dedup state
            # so the next adjustment re-notifies.
            agent._adaptive_prev_effort = None
            agent._adaptive_last_notified_effort = None
            return None
        lowered = EFFORT_RANK.get(effort, 0) < EFFORT_RANK.get(baseline, 0)

        applied = {"enabled": True, "effort": effort}
        token = AdaptiveTurnToken(
            saved=reasoning_config, applied=applied, effort=effort, reason=reason
        )
        agent.reasoning_config = applied
        # A lowered turn is never a continuation target: "go ahead" after a
        # casual aside refers to earlier discussed work and must run at the
        # baseline, so only escalated levels carry forward.
        agent._adaptive_prev_effort = None if lowered else effort
        if getattr(agent, "_adaptive_last_notified_effort", None) != effort:
            _notify_adjustment(agent, effort, reason, lowered)
            agent._adaptive_last_notified_effort = effort
        return token
    except Exception:
        logger.debug("begin_adaptive_reasoning_turn failed", exc_info=True)
        return None


def end_adaptive_reasoning_turn(
    agent: Any, token: Optional[AdaptiveTurnToken]
) -> None:
    """Restore the pre-turn reasoning baseline after an escalated turn.

    Identity-guarded: if something else replaced ``reasoning_config`` mid-turn
    (e.g. fallback-model activation re-resolving effort for the new model),
    that newer value is kept. Never raises.
    """
    if token is None:
        return
    try:
        if getattr(agent, "reasoning_config", None) is token.applied:
            agent.reasoning_config = token.saved
    except Exception:
        logger.debug("end_adaptive_reasoning_turn failed", exc_info=True)


_T = TypeVar("_T")


def adaptive_reasoning_turn(fn: Callable[..., _T]) -> Callable[..., _T]:
    """Bracket a turn entry point ``fn(agent, user_message, ..., moa_config=None)``
    with :func:`begin_adaptive_reasoning_turn` / :func:`end_adaptive_reasoning_turn`.

    Applied to ``AIAgent.run_conversation`` at class-definition time so the
    adjustment covers the whole call and the baseline is restored on every
    exit path (return, exception, interrupt) without touching the turn body:
    upstream can reshape the forwarder's prologue and ``finally`` freely and
    this seam never conflicts. Nothing ahead of the forwarder's ``try`` reads
    ``reasoning_config`` (the background-review fence only touches review
    state), so covering the prologue is safe.

    ``moa_config`` is resolved from the bound call so positional and keyword
    callers are treated alike. Both helpers never raise, so the wrapper adds
    no failure mode of its own.
    """
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(agent: Any, user_message: Any, *args: Any, **kwargs: Any) -> _T:
        moa_config = kwargs.get("moa_config")
        if moa_config is None and args:
            try:
                moa_config = signature.bind_partial(
                    agent, user_message, *args, **kwargs
                ).arguments.get("moa_config")
            except TypeError:
                moa_config = None
        token = begin_adaptive_reasoning_turn(
            agent, user_message, moa_config=moa_config
        )
        try:
            return fn(agent, user_message, *args, **kwargs)
        finally:
            end_adaptive_reasoning_turn(agent, token)

    return wrapper
