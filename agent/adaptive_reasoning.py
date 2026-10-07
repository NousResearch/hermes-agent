"""Opt-in adaptive reasoning-effort adjustment (config: agent.adaptive_reasoning).

When enabled, each eligible user turn is classified by a deterministic,
bounded heuristic — no extra LLM call, no new model tool — and the agent's
``reasoning_config`` is temporarily adjusted away from the configured
baseline (``agent.reasoning_effort``) for turns that materially warrant it:

- ``high``  — substantial multi-step analysis, coding/debugging,
  consequential infrastructure changes, or in-depth research/synthesis.
- ``xhigh`` — never on a lone keyword. Either an architecture/security/
  high-stakes category whose total score reaches ``_XHIGH_THRESHOLD``, or
  code work (a debugging or implementation category) with at least one
  corroborating signal: a second code category, error evidence,
  infrastructure, production impact, a multi-step list, a code block, or a
  detailed (>500 character) request. A lone code keyword stays ``high``.
- ``low``   — only when ``min_effort`` opts a floor below the baseline in,
  and only for positively simple turns (casual chatter, short factual
  questions, single mechanical retrieval steps) carrying no complexity,
  error, code, or multi-step signal. A second question mark is not
  complexity on its own; clause joins, sequencing or judgment words, ``;``,
  ``&``, an inner line break and code still rule ``low`` out. Ambiguity
  stays at the baseline; without ``min_effort`` the feature is
  escalation-only.

Short turns are judged with context, not by length alone. After a turn
that ran at an escalated level, an affirmative ("go ahead") inherits that
level, and so does a short follow-up from a closed set ("fix it", "try
again", "no, do it properly", "why?", "still broken", "redo it") when the
previous turn ran at ``high`` or above. The inherited level is never raised
further (there is no clean "the last tool loop failed" signal at turn
start) and stays within ``max_effort``. Without an escalated previous turn
those phrases are classified as before; any other turn, including casual
chatter, is classified from scratch, so nothing stays stuck at ``xhigh``.

Adjustment requires an explicit reasoning baseline: when ``reasoning_config``
is None the provider decides whether thinking exists at all, and installing a
config for one turn would toggle thinking presence on/off mid-conversation.
With a baseline set, thinking stays enabled at every selected level (``none``
is never selected and a disabled baseline is never re-enabled), which avoids
the larger invalidation a thinking on/off toggle causes (it can reach the
tools and system blocks).

A level change is still not cache-free. Hermes's direct Anthropic adapter
sets ``output_config.effort`` per request, and changing it invalidates the
cached message blocks: the first API call of a turn whose level differs from
the previous call (an adjusted turn, or the first turn back at the baseline)
re-keys the message cache and pays cache-write price. The level is held for
the whole tool loop, so that is one miss per level change, never one per
tool call. Content-keyed prefix caches are unaffected. Carrying the effort
in a per-message ``role: "system"`` entry, on models that support it, would
avoid the miss; the adapter does not do that yet.

Only the human-authored text of a turn is classified. Surfaces that enrich
the model-facing payload pass their pre-injection text (the gateway's
``title_user_message``); a ``/skill`` or bundle scaffold is reduced to the
user's instruction; and synthetic system turns (background-process and
async-delegation completion notices) get no adjustment and no notice.

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
level — model context is never touched, so role alternation is unaffected and the
notice itself never alters the cached prefix. On an Anthropic-adapter route
(``api_mode == "anthropic_messages"``) a second, once-per-session warning
notice (``CACHE_NOTICE_KEY``) says the effort change may cost a cache write;
it never shares a turn with the level-change notice, which keeps priority.
Subagents adjust silently. All continuation and
dedup state lives on the individual ``AIAgent`` object, so concurrent
sessions never share effective efforts.
"""

from contextlib import contextmanager
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from hermes_constants import VALID_REASONING_EFFORTS

logger = logging.getLogger(__name__)

EFFORT_RANK: dict[str, int] = {
    level: rank for rank, level in enumerate(VALID_REASONING_EFFORTS)
}

_DEFAULT_MAX_EFFORT = "xhigh"

# Regex work is bounded: signals are matched against at most this many
# characters (length-based signals still use the full text length).
_SCAN_LIMIT = 8000

NOTICE_KEY = "adaptive-reasoning"
CACHE_NOTICE_KEY = f"{NOTICE_KEY}.cache"
_NOTICE_TTL_MS = 12000
_CACHE_NOTICE_TEXT = (
    "⚠️ Changing reasoning effort can invalidate Anthropic's prompt cache for that turn and increase cost."
)

_EFFORT_LABELS = {"low": "Low", "high": "High", "xhigh": "XHigh"}


@dataclass
class AdaptiveTurnToken:
    """Restore token for one escalated turn.

    ``applied`` is the exact dict object installed on the agent; restore is
    identity-guarded so a mid-turn rewrite (fallback activation re-resolving
    reasoning for a new model) is never clobbered.
    """

    saved: Optional[dict[str, Any]]
    applied: dict[str, Any]
    effort: str
    reason: str
    notified: bool = False  # this turn already used the notice rail


# ── Deterministic signal table ───────────────────────────────────────────────
# Each category matches at most once; the score is the sum of matched weights.
# xhigh-class categories mark the turn as *potentially* xhigh, but xhigh still
# requires corroboration (total score >= _XHIGH_THRESHOLD). Code-class
# categories (_CODE_CATEGORIES) reach xhigh with one corroborating signal
# from _CODE_CORROBORATION instead; see classify_reasoning_effort.

_CATEGORIES: list[tuple[str, str, int, bool, "re.Pattern[str]"]] = [
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

# Debugging and implementation: the work that most benefits from thoroughness.
_CODE_CATEGORIES = frozenset({"debugging", "implementation"})
# Signals that corroborate code work: a second code category is matched
# separately, the structural ones are named below in classify.
_CODE_CORROBORATION = frozenset({"error_evidence", "infrastructure", "consequence"})

_CASUAL_RE = re.compile(
    r"(?:(?:hi|hiya|hello|hey|yo)(?:\s+there)?[\s!.?]+)?"
    r"(?:how\s+are\s+you(?:\s+doing)?|how'?s\s+it\s+going|how\s+have\s+you\s+been)[\s.!?~]*"
    r"|(?:hi|hiya|hello|hey|yo|good\s+(?:morning|afternoon|evening|night)"
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

# Short follow-ups that only make sense about the previous task. A closed
# fullmatch set; used only when the previous adaptive turn ran at high or
# above, otherwise these phrases are classified like any other turn.
_FOLLOWUP_RE = re.compile(
    r"(?:(?:no|nope|ugh|hmm)[\s,.!]+)?(?:please\s+|let'?s\s+)?"
    r"(?:fix\s+(?:it|this|that|them)(?:\s+(?:properly|again|now))?"
    r"|try\s+(?:again|harder|that\s+again)|retry|again"
    r"|redo(?:\s+(?:it|this|that))?|do\s+it\s+(?:again|properly|right)"
    r"|why(?:\s+not)?|how\s+come|what\s+happened|what\s+went\s+wrong"
    r"|(?:it'?s\s+|that'?s\s+|it\s+is\s+)?still\s+(?:broken|failing|wrong|crashing|not\s+working)"
    r"|(?:it\s+|that\s+)?(?:still\s+)?(?:didn'?t|doesn'?t|did\s+not|does\s+not)\s+work"
    r"|(?:it'?s\s+)?not\s+working|same\s+(?:error|issue|problem))"
    r"(?:[\s,]+(?:please|pls|plz))?[\s.!?~]*",
    re.IGNORECASE,
)

_HIGH_THRESHOLD = 3
_XHIGH_THRESHOLD = 6

# ── Positive-simplicity patterns (downshift candidates) ─────────────────────
# ``low`` requires positive evidence of simplicity, never mere absence of
# complexity signals. Each pattern is a fullmatch on the normalized text and
# is only consulted when the turn carries zero category/structural signals.

# Recognize complete lookup shapes, not an interrogative plus arbitrary text.
# A short unknown tail can contain another task even without a conjunction or
# punctuation ("Which tests failed fix them?"). A denylist cannot establish
# simplicity. Keep subjects closed and variable operands to a single token;
# unrecognized wording deliberately falls back to the baseline. These are
# useful heuristics, not a proof of natural-language task complexity.
_SIMPLE_FACTUAL_RE = re.compile(
    r"(?:what(?: is|'s) the (?:current )?(?:time|date|day|version|working directory)"
    r"|what(?: is|'s) the capital of [a-z][a-z'-]{0,39}"
    r"|what time is it(?: in [a-z][a-z'-]{0,39})?"
    r"|which tests (?:failed|passed)"
    r"|where is the (?:config file|readme|log file)"
    r")\?*",
    re.IGNORECASE,
)

# Clause joins, sequencing, conditionals and judgment questions are not
# positive evidence of a single simple fact or retrieval. Reject before every
# simplicity branch; false negatives are safer than lowering ambiguous work.
# Matched on the whitespace-stripped text, so a trailing line break is not
# a veto; an inner one still is. The comma veto stays (defense in depth for
# compound requests), so "hi, how are you" keeps the baseline.
_NOT_SIMPLE_RE = re.compile(
    r"\b(?:and|or|then|plus|also|before|after|while|but|if|unless|until"
    r"|should|could|would|best|better|optimal)\b|[,;\n&]|\s/\s",
    re.IGNORECASE,
)

# The same closed-shape rule applies to retrievals: a read-only verb alone
# does not make its unrestricted tail mechanical ("show me how to fix this").
# File operands admit one path token, never additional prose or shell syntax.
_MECHANICAL_RE = re.compile(
    r"(?:please )?(?:show|list|open|display|print|read|cat) (?:me )?"
    r"(?:(?:the )?(?:readme|(?:current )?config|files(?: in /[a-z0-9_./-]{1,60})?)"
    r"|[a-z0-9_/-]{1,40}\.[a-z0-9]{1,10}"
    r"|/[a-z0-9_./-]{1,60})[.!]?",
    re.IGNORECASE,
)


def parse_adaptive_reasoning_config(raw: Any) -> Optional[dict[str, Any]]:
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
    cfg: dict[str, Any] = {"enabled": True, "max_effort": max_effort}
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


def init_adaptive_reasoning_state(agent: Any, raw: Any) -> None:
    """Construction-time adaptive state for ``init_agent``.

    Interactive surfaces (CLI, gateway, TUI/Desktop) pass the raw config section; the delegate tool
    passes the parent's parsed policy so subagents inherit the parent's effective level and
    reclassify their own goal (parse is a fixed point, so re-parsing is safe). Cron and batch runs
    pass nothing and stay on their configured effort. ``reasoning_user_override`` marks an explicit
    session-scoped user pick (/reasoning, Desktop effort menu) or a delegation.reasoning_effort pin,
    which disables adaptive adjustment.
    """
    agent.adaptive_reasoning = parse_adaptive_reasoning_config(raw)
    agent.reasoning_user_override = False
    agent._adaptive_prev_effort = None
    agent._adaptive_last_notified_effort = None
    agent._adaptive_cache_warned = False


def refresh_adaptive_reasoning(agent: Any, user_config: dict[str, Any], user_override: bool) -> None:
    """Per-turn refresh for a cached gateway agent so it tracks config edits; a session /reasoning
    override is an explicit user pick and always wins over the classifier."""
    agent.adaptive_reasoning = parse_adaptive_reasoning_config(
        (user_config.get("agent") or {}).get("adaptive_reasoning"))
    agent.reasoning_user_override = user_override


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


# Synthetic system turns: completion notices the agent queues for itself
# (tools/process_registry_notifications.py). The first four shapes are
# hermes_state_timeline._SYNTHETIC_PROMPT; the batched-completion header,
# heartbeat and per-task failure notice come from the same producer. A local
# copy, because importing the timeline module pulls the session-DB stack in.
_SYNTHETIC_TURN_RE = re.compile(
    r"\s*(?:\[IMPORTANT: Background process |\[ASYNC (?:DELEGATION )?(?:BATCH )?COMPLETE\b|"
    r"A background fan-out of \d+ subagent\(s\) you dispatched earlier has finished\.|"
    r"A background subagent you dispatched earlier has finished\.|"
    r"\[IMPORTANT: \d+ background processes completed\.|"
    r"\[Background process \S+ heartbeat #|\[ASYNC DELEGATION TASK FAILED\b)",
    re.IGNORECASE,
)


def classification_text(user_message: Any, human_text: Optional[str] = None) -> Optional[str]:
    """The human-authored text to classify, or None for a synthetic system turn.

    ``human_text`` is the surface's pre-injection text when it has one (the
    gateway's ``title_user_message``, which excludes auto-loaded skill dumps,
    speaker prefixes, reply quotes and channel context); without it the
    model-facing payload is used, as before. A ``/skill`` or bundle scaffold is
    reduced to the user's instruction ("" for a bare invocation, which keeps
    the empty-text no-adjustment path).
    """
    text = human_text if isinstance(human_text, str) else extract_message_text(user_message)
    if _SYNTHETIC_TURN_RE.match(text):
        return None
    stripped = text.lstrip()
    if stripped.startswith("[IMPORTANT:"):
        from agent.skill_commands import extract_user_instruction_from_skill_message

        instruction = extract_user_instruction_from_skill_message(stripped)
        if instruction is not stripped:  # scaffolding: keep only what the user typed
            return instruction or ""
    return text


def _positively_simple_reason(stripped: str, norm: str) -> str:
    """Reason string when the turn is positively simple, else "".

    Caller guarantees the turn carries no category, code or multi-step
    signal — this only checks for affirmative evidence of simplicity.
    """
    if _NOT_SIMPLE_RE.search(stripped):
        return ""
    if _CASUAL_RE.fullmatch(norm):
        return "casual message"
    if _SIMPLE_FACTUAL_RE.fullmatch(norm):
        return "simple factual request"
    if _MECHANICAL_RE.fullmatch(norm):
        return "single mechanical step"
    return ""


def _continues_escalation(text: str, prior_effort: Optional[str]) -> bool:
    """True when this turn inherits the previous turn's escalated level: an
    affirmative after any escalated turn, or a short follow-up ("fix it",
    "try again", "why?") after a turn that ran at high or above."""
    if prior_effort not in EFFORT_RANK:
        return False
    norm = re.sub(r"\s+", " ", text.strip().lower())
    if _CONTINUATION_RE.fullmatch(norm):
        return True
    return EFFORT_RANK[prior_effort] >= EFFORT_RANK["high"] and bool(
        _FOLLOWUP_RE.fullmatch(norm)
    )


def classify_reasoning_effort(
    text: str, prior_effort: Optional[str] = None
) -> tuple[Optional[str], str]:
    """Classify one user turn into (effort, reason) — deterministic, no I/O.

    ``prior_effort`` is the adaptive level of the previous turn (or None):
    short affirmatives inherit it so "go ahead" doesn't drop a complex task
    back to the baseline mid-flight, and so do short follow-ups such as
    "fix it" or "still broken" when it was high or above. Unrelated trivial
    turns re-classify from scratch, so nothing stays stuck at xhigh.

    ``xhigh`` needs an xhigh-class category with a total score of at least
    ``_XHIGH_THRESHOLD``, or a code-class category (debugging,
    implementation) with one corroborating signal (see the module docstring).

    Apart from inherited continuations, ``low`` requires positive simplicity (casual
    chatter, a short fact lookup, a single read-only mechanical step) with
    zero complexity signals; whether it takes effect is the caller's floor
    decision. Ambiguous turns — including empty text, e.g. an image-only
    message — return ``None`` (no adjustment), not a concrete effort level.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None, ""
    norm = re.sub(r"\s+", " ", stripped.lower())

    if _continues_escalation(norm, prior_effort):
        return prior_effort, "continuing the escalated task"

    # Terse turns (≤4 words, no code) never escalate — but they still get the
    # full signal scan below: a signal-bearing terse turn must not downshift.
    terse = len(stripped.split()) <= 4 and "```" not in stripped

    scan = stripped[:_SCAN_LIMIT]
    score = 0
    xhigh_class = False
    names: set[str] = set()
    matched: list[tuple[int, str]] = []
    for name, label, weight, is_xhigh, pattern in _CATEGORIES:
        if pattern.search(scan):
            score += weight
            xhigh_class = xhigh_class or is_xhigh
            names.add(name)
            matched.append((weight, label))

    multi_step = len(_MULTI_STEP_RE.findall(scan)) >= 3
    detailed = len(stripped) > 500
    has_code = "```" in scan
    if multi_step:
        score += 2
        matched.append((2, "a multi-step request"))
    if detailed:
        score += 1 if len(stripped) <= 1500 else 2
        matched.append((1, "a detailed request"))
    if has_code:
        score += 1
        matched.append((1, "included code"))
    if scan.count("?") >= 2:
        score += 1
        matched.append((1, "multiple questions"))

    if not terse:
        code_work = names & _CODE_CATEGORIES
        corroborated_code = bool(code_work) and (
            len(code_work) >= 2 or bool(names & _CODE_CORROBORATION)
            or multi_step or has_code or detailed
        )
        if (xhigh_class and score >= _XHIGH_THRESHOLD) or corroborated_code:
            matched.sort(key=lambda item: -item[0])
            return "xhigh", " and ".join(label for _w, label in matched[:2])
        if score >= _HIGH_THRESHOLD:
            matched.sort(key=lambda item: -item[0])
            return "high", " and ".join(label for _w, label in matched[:2])

    # A second question mark alone is not complexity; every other signal
    # (category, multi-step, length, code) still rules a downshift out.
    if not names and not (multi_step or detailed or has_code):
        simple_reason = _positively_simple_reason(stripped, norm)
        if simple_reason:
            return "low", simple_reason
    return None, ""


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


def _warn_anthropic_cache_once(agent: Any) -> bool:
    """Once per session, warn on the notice rail that an adjusted effort can
    invalidate Anthropic's prompt cache. Only for the anthropic-adapter route
    (``api_mode == "anthropic_messages"``, the same test the request path uses
    to select ``build_anthropic_kwargs`` and its ``output_config.effort``), only
    when a notice callback is bound (no console fallback), never for subagents.
    Returns True when the notice was emitted. Never raises."""
    try:
        if getattr(agent, "_adaptive_cache_warned", False):
            return False
        if getattr(agent, "api_mode", None) != "anthropic_messages":
            return False
        if getattr(agent, "platform", "") == "subagent":
            return False
        callback = getattr(agent, "notice_callback", None)
        if not callback:
            return False
        from agent.credits_tracker import AgentNotice

        agent._adaptive_cache_warned = True
        callback(
            AgentNotice(
                text=_CACHE_NOTICE_TEXT,
                level="warn",
                kind="ttl",
                ttl_ms=_NOTICE_TTL_MS,
                key=CACHE_NOTICE_KEY,
            )
        )
        return True
    except Exception:
        logger.debug("adaptive reasoning cache notice emission failed", exc_info=True)
        return False


def begin_adaptive_reasoning_turn(
    agent: Any, user_message: Any, moa_config: Optional[dict[str, Any]] = None,
    *, rail_busy: bool = False, human_text: Optional[str] = None,
) -> Optional[AdaptiveTurnToken]:
    """Classify this turn and, if warranted, install an adjusted
    ``reasoning_config`` on the agent for the duration of the turn.

    Returns a restore token for :func:`end_adaptive_reasoning_turn`, or None
    when the feature is off, a user override is active, the turn is a MoA
    turn (per-slot reasoning owns effort there), no explicit reasoning
    baseline is set (adjusting would toggle thinking presence on/off), the
    turn is a synthetic system notice, or no adjustment applies.
    Never raises — a classification failure must not break the turn.

    ``rail_busy`` marks that this turn already emitted a notice (a rebase
    re-run), so the cache warning waits for a later adjusted turn.
    ``human_text`` is the surface's pre-injection text (see
    :func:`classification_text`); None classifies the payload.
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
                # Same fallback as the turn loop: an undecodable marker is plain text.
                logger.debug("adaptive reasoning MoA marker decode failed", exc_info=True)
        text = classification_text(user_message, human_text)
        if text is None:
            # Synthetic system turn: no adjustment, no notice. Continuation and
            # dedup state are left as they were, so the notice is transparent to
            # the human turns around it.
            return None
        prior_effort = getattr(agent, "_adaptive_prev_effort", None)
        effort, reason = classify_reasoning_effort(
            text, prior_effort=prior_effort
        )
        # No adjustment is distinct from a concrete level: ambiguous work
        # keeps even a low/minimal baseline, including inherited child effort.
        # A continuation capped at low is still an escalation from minimal.
        simple = effort == "low" and not _continues_escalation(text, prior_effort)
        if effort is None:
            effort = baseline
        elif simple:
            # Positive simplicity permits a downshift, never an escalation.
            effort = min(effort, baseline, key=EFFORT_RANK.__getitem__)
        max_effort = cfg.get("max_effort") or _DEFAULT_MAX_EFFORT
        if EFFORT_RANK.get(effort, 0) > EFFORT_RANK.get(max_effort, 0):
            effort = max_effort
        floor = cfg.get("min_effort") if simple else baseline
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
            token.notified = True
        elif not rail_busy:
            # One notice per turn: the level change keeps priority, so the
            # cache warning lands on the first adjusted turn without one.
            token.notified = _warn_anthropic_cache_once(agent)
        token.notified = token.notified or rail_busy
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


@dataclass
class _ActiveAdaptiveTurn:
    user_message: Any
    moa_config: Any
    prior_effort: Optional[str]
    token: Optional[AdaptiveTurnToken] = None
    installed: Any = None  # the reasoning_config this turn left in place
    human_text: Optional[str] = None


@contextmanager
def adaptive_reasoning_turn(
    agent: Any, user_message: Any, moa_config=None, *, human_text: Optional[str] = None,
):
    """Adjust only an admitted turn, restoring even when its tool loop raises.

    Enter after the review fence and durable lease admission: rejected/waiting
    turns must not alter the live agent's effort or emit an adjustment notice.
    ``human_text`` is the surface's pre-injection text, when it has one.
    """
    turn = _ActiveAdaptiveTurn(
        user_message, moa_config, getattr(agent, "_adaptive_prev_effort", None),
        human_text=human_text,
    )
    turn.token = begin_adaptive_reasoning_turn(
        agent, user_message, moa_config=moa_config, human_text=human_text,
    )
    turn.installed = getattr(agent, "reasoning_config", None)
    try:
        agent._adaptive_active_turn = turn
        yield
    finally:
        if getattr(agent, "_adaptive_active_turn", None) is turn:
            agent._adaptive_active_turn = None
        end_adaptive_reasoning_turn(agent, turn.token)


def rebase_adaptive_turn(agent: Any) -> None:
    """Re-apply this turn's adjustment against a baseline replaced at turn start.

    ``build_turn_context`` restores the primary runtime inside the turn and may install the primary
    snapshot's ``reasoning_config``; without a rebase that silently discards the adjustment for the
    whole turn. The adjustment is re-derived from the restored baseline (classification sees the
    pre-turn continuation state), and that baseline is what the turn's end restores. No-op outside
    an adaptive turn or while the config this turn installed is still in place. Never raises.
    """
    turn = getattr(agent, "_adaptive_active_turn", None)
    if not isinstance(turn, _ActiveAdaptiveTurn) or getattr(agent, "reasoning_config", None) is turn.installed:
        return
    try:
        agent._adaptive_prev_effort = turn.prior_effort
        rail_busy = bool(turn.token is not None and turn.token.notified)
        turn.token = begin_adaptive_reasoning_turn(
            agent, turn.user_message, moa_config=turn.moa_config, rail_busy=rail_busy,
            human_text=turn.human_text,
        )
        turn.installed = getattr(agent, "reasoning_config", None)
    except Exception:
        logger.debug("rebase_adaptive_turn failed", exc_info=True)
