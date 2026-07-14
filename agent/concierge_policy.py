"""Concierge routing policy — pure deterministic classifier (Phase 2 substrate).

This module is the *classifier* half of the concierge control plane.  Given a
single user-input fragment it returns a :class:`ConciergePolicyDecision` whose
``recommendation`` is one of ``MAIN`` / ``WORKER_LANE`` / ``STEER`` /
``CONTROL`` and whose ``signals`` enumerate the contributing observations.

What this module is — and isn't:

* **Pure.**  No I/O, no side effects, no logging, no clocks, no randomness.
  Every call with the same ``(text, lang_hint)`` returns the same value.  This
  is what PRD §5.2 INV-6 ("Replayable, deterministic classification") demands.
* **Stdlib-only.**  No third-party imports.  The classifier is a leaf module
  that ``agent.control_plane`` and ``agent.orchestration_runtime`` may import
  without dragging in a heavyweight dependency graph.
* **Decision schema lives here.**  The dataclass and the four enums
  (``ConciergeRecommendation`` / ``ConciergeConfidence`` / ``ConciergeSignal``)
  are *the* source of truth for the vocabulary the rest of the control plane
  uses.  ``agent/control_plane.py`` re-exports them as ``Recommendation`` /
  ``Confidence`` / ``Signal`` for the shorter name (design review §3.1).
* **Not a dispatcher.**  Producing ``Recommendation.WORKER_LANE`` does not
  start a worker; producing ``Recommendation.STEER`` does not inject into a
  running turn; producing ``Recommendation.CONTROL`` does not purge a queue.
  Those are surface-adapter responsibilities (CLI / TUI / Gateway each act on
  the decision in their own idiom).  This module is read-only.
* **Not a runtime-state inspector.**  It does not consult the task registry,
  the worker registry, the pending queue, or the running agent.  The
  recommendation is derived solely from the input text — which means a
  ``STEER`` recommendation is only a *candidate* (the surface must verify a
  turn is actually in flight before invoking ``running_agent.steer``).  PRD
  §6.5 and design review §4.5 call this out explicitly.
* **Not a model classifier.**  Only deterministic Korean/English vocabulary
  anchors and shape-based heuristics.  Design review §8.1 DR-1 locks Phase 2
  to whole-body equality for STOP/ACK; model-based intent classification is a
  future phase.

Hard boundaries (PRD §3.1 Non-goals + design review §9.2):

* No persona changes here.
* No worker-lane registration.
* No exposure of ``/mode concierge`` (legacy removed).
* No mutation of any caller-supplied object.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Iterable, Sequence

__all__ = [
    "ConciergeRecommendation",
    "ConciergeConfidence",
    "ConciergeSignal",
    "ConciergePolicyDecision",
    # temporary aliases during rename
    "ConciergeRecommendation",
    "ConciergeConfidence",
    "ConciergeSignal",
    "ConciergePolicyDecision",
    "classify_request",
    "fingerprint",
    "STOP_TOKENS_EN",
    "STOP_TOKENS_KO",
    "ACK_TOKENS_EN",
    "ACK_TOKENS_KO",
]


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------
class ConciergeRecommendation(Enum):
    """Where the dispatcher should send the fragment.

    The enum is closed: every value drives a distinct downstream branch in the
    surface adapter (see design review §4.1).  A future category must be
    added explicitly here and in :class:`agent.control_plane.Recommendation`.
    """

    MAIN = "main"             # foreground main turn (default conservative)
    WORKER_LANE = "worker_lane"  # delegate to a background worker (Phase 4+)
    STEER = "steer"           # in-flight inject candidate (surface must verify)
    CONTROL = "control"       # purely informational (STOP / ACK / NOISE)


class ConciergeConfidence(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConciergeSignal(Enum):
    """Contributing observations the classifier made about the fragment.

    Multiple signals may fire per fragment.  Signals are advisory — it is the
    ``recommendation`` field that drives the downstream branch.  Vocabulary is
    deliberately open-ended in spirit (a new signal is additive; existing tests
    stay valid) but expressed as an ``Enum`` for static-checker friendliness.
    """

    RESEARCH = "research"               # "investigate", "조사", "search", "research"
    ARTIFACT = "artifact"               # "report.md", "리포트", "draft", "정리해줘"
    CODE_EDIT = "code_edit"             # "고쳐", "수정", "implement", "refactor"
    LONG = "long"                       # heuristic >= 2 minutes of work
    MANY_TOOLS = "many_tools"           # heuristic >= 3 tool calls
    STATUS = "status"                   # "status?", "어디까지", "뭐 했어"
    STOP = "stop"                       # whole-body stop token
    STEER = "steer"                     # short addition / "근데", "그리고"
    ACK = "ack"                         # whole-body thanks/감사
    DUPLICATE = "duplicate"             # caller-supplied; the classifier itself
                                        # does not detect duplicates (no history)
    NOISE = "noise"                     # empty / unparseable
    KOREAN = "korean"                   # Hangul detected or lang_hint='ko'
    EXPLICIT_WORKER_REQ = "explicit_worker_req"  # "백그라운드", "워커에 맡겨"
    EXPLICIT_MAIN_REQ = "explicit_main_req"      # "지금 해", "직접 해"


# --------------------------------------------------------------------------
# Vocabularies — whole-body match for STOP/ACK (DR-1 lock-in), substring for
# anchors.  Korean entries match as plain substrings (Hangul has no useful word
# boundary); English entries match against tokenised lowercase text.
# --------------------------------------------------------------------------

# STOP — whole-body equality (after trim + lowercase + trailing punctuation
# strip).  PRD §7.1 + design review §8.1 DR-1.  PAUSE is intentionally folded
# into STOP-graceful (design review §8.2 OQ-4).
STOP_TOKENS_EN: tuple[str, ...] = (
    "stop",
    "cancel",
    "abort",
    "halt",
    "nevermind",
    "never mind",
    "forget it",
    "/stop",
    "/cancel",
    "/abort",
    "/kill",
)

STOP_TOKENS_KO: tuple[str, ...] = (
    "그만",
    "그만해",
    "그만둬",
    "중단",
    "취소",
    "됐어",
    "멈춰",
    "잠깐만",  # soft pause folded into STOP-graceful per design review §8.2
    "잠깐",
    "기다려",
)

# ACK — whole-body equality only (the TUI integrated module's whole-body
# heuristic, faithfully mirrored).  "ok" / "응" are intentionally NOT here:
# both lead steering in their respective languages.
ACK_TOKENS_EN: tuple[str, ...] = (
    "thanks",
    "thanks!",
    "thank you",
    "thank you!",
    "thx",
    "ty",
)

ACK_TOKENS_KO: tuple[str, ...] = (
    "고마워",
    "고맙습니다",
    "감사",
    "감사합니다",
    "수고",
    "수고했어",
)

# Status query anchors.  These are *substring* matches — a status fragment is
# usually short and contains one of these tokens whole-cloth.
_STATUS_ANCHORS_EN: tuple[str, ...] = (
    "status",
    "what are you doing",
    "what's running",
    "show me",
    "list",
    "/tasks",
    "/agents",
    "/mode",
)

_STATUS_ANCHORS_KO: tuple[str, ...] = (
    "상태",
    "어디까지",
    "뭐 했어",
    "뭐해",
    "뭐 해",
    # Bare "진행" is NOT a status query — "이거 진행해" / "작업을 진행해" mean GO.
    # Only status-shaped collocations:
    "진행상황",
    "진행 상황",
    "진행 중",
    "진행중",
    "어디까지 진행",
    "어떤 lane",
    "어떤 워커",
    "지금 뭐",
)

# Artifact-creation anchors.  Single-anchor → strong WORKER recommendation
# (PRD §8.1 final paragraph).
_ARTIFACT_ANCHORS_EN: tuple[str, ...] = (
    ".md",
    "report",
    "report.md",
    "summary",
    "summary.md",
    "draft",
    "write up",
    "writeup",
    "write a",
    "write the",
    "compose",
    "produce a",
    ".csv",
    ".svg",
    ".png",
    ".pdf",
    ".tsv",
    ".json",
)

_ARTIFACT_ANCHORS_KO: tuple[str, ...] = (
    "리포트",
    "보고서",
    "정리해",
    "정리해줘",
    "작성해",
    "작성해줘",
    "초안",
    "문서로",
)

# External-research anchors.  Single-anchor → strong WORKER recommendation.
_RESEARCH_ANCHORS_EN: tuple[str, ...] = (
    "investigate",
    "research",
    "search for",
    "look up",
    "find out",
    "crawl",
    "scrape",
    "audit",
    "deep dive",
    "deep-dive",
    "explore the",
    "study the",
)

_RESEARCH_ANCHORS_KO: tuple[str, ...] = (
    "조사",
    "조사해",
    "조사해줘",
    "찾아봐",       # "look it up" — investigative
    # NOTE: "찾아줘" alone is intentionally NOT a research anchor.  "이 파일에서
    # X 함수 찾아줘" (PRD §8.2) is a localised lookup and should stay on MAIN;
    # "찾아봐" / "조사" / "리서치" are the investigative-flavour variants.
    "리서치",
    "크롤",
    "크롤링",
    "탐색",
    "분석해",
)

# Code-edit anchors.  Single-anchor → worker candidate; combined with artifact
# or research → strong worker.
_CODE_EDIT_ANCHORS_EN: tuple[str, ...] = (
    "implement",
    "refactor",
    "fix the",
    "patch",
    "rewrite",
    "port the",
    "migrate the",
    "build the",
    "add a tool",
)

_CODE_EDIT_ANCHORS_KO: tuple[str, ...] = (
    "구현해",
    "구현해줘",
    "고쳐",
    "고쳐줘",
    "수정해",
    "수정해줘",
    "리팩",
    "리팩터",
    "재작성",
)

# Explicit overrides — the user is telling the policy where to route.
_EXPLICIT_WORKER_ANCHORS_EN: tuple[str, ...] = (
    "in the background",
    "background",
    "delegate this",
    "delegate it",
    "delegate to",
    "give it to a worker",
    "worker lane",
    "kick off a worker",
    "spawn a worker",
)

_EXPLICIT_WORKER_ANCHORS_KO: tuple[str, ...] = (
    "백그라운드",
    "워커에 맡겨",
    "워커에게 맡겨",
    "워커한테 맡겨",
    "클로드한테 맡겨",
    "위임해",
    "위임해줘",
    "맡겨줘",
)

_EXPLICIT_MAIN_ANCHORS_EN: tuple[str, ...] = (
    "do it yourself",
    "do it now",
    "right now",
    "yourself",
    "main thread",
    "in line",
    "inline",
    "directly",
    "no worker",
    "no workers",
)

_EXPLICIT_MAIN_ANCHORS_KO: tuple[str, ...] = (
    "직접 해",
    "직접해",
    "지금 해",
    "지금해",
    "지금 보여줘",
    "바로 해",
    "바로해",
    "여기서 해",
    "메인에서",
)

# Steer prefixes / additions.  Substring at the start (after trim).  A fragment
# that begins with a steer prefix is a candidate for in-flight injection.
_STEER_PREFIXES_EN: tuple[str, ...] = (
    "also ",
    "and also ",
    "additionally ",
    "btw ",
    "by the way ",
    "actually ",
    "wait ",
    "oh, ",
    "oh ",
)

_STEER_PREFIXES_KO: tuple[str, ...] = (
    "근데 ",
    "그리고 ",
    "추가로 ",
    "참, ",
    "참 ",
    "아 그리고 ",
    "아, 그리고 ",
)


# --------------------------------------------------------------------------
# ConciergePolicyDecision
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ConciergePolicyDecision:
    """Read-only verdict from :func:`classify_request`.

    Hashable (frozen + ``frozenset`` for ``signals``).  Designed to be passed
    around / logged / stored without defensive copies.  The ``debug_label`` is
    a stable, one-line summary safe to render to a transcript ``control:`` line.
    """

    recommendation: ConciergeRecommendation
    signals: FrozenSet[ConciergeSignal]
    confidence: ConciergeConfidence
    debug_label: str
    raw_text: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    # -- convenience -----------------------------------------------------
    @property
    def should_delegate(self) -> bool:
        """``True`` iff the recommendation is to dispatch to a worker lane."""
        return self.recommendation is ConciergeRecommendation.WORKER_LANE

    @property
    def is_control(self) -> bool:
        """``True`` for STOP / ACK / NOISE — no model call needed."""
        return self.recommendation is ConciergeRecommendation.CONTROL

    @property
    def is_stop(self) -> bool:
        return ConciergeSignal.STOP in self.signals

    @property
    def is_ack(self) -> bool:
        return ConciergeSignal.ACK in self.signals

    @property
    def has_korean(self) -> bool:
        return ConciergeSignal.KOREAN in self.signals

    # -- serialization --------------------------------------------------
    def to_dict(self) -> dict:
        """Return a JSON-safe view (sorted signal values for stable logs)."""
        return {
            "recommendation": self.recommendation.value,
            "confidence": self.confidence.value,
            "signals": sorted(s.value for s in self.signals),
            "debug_label": self.debug_label,
            "raw_text": self.raw_text,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# Language detection
# --------------------------------------------------------------------------
_HANGUL_SYLLABLE_RE = re.compile(r"[가-힯ᄀ-ᇿ㄰-㆏]")


def _looks_korean(text: str) -> bool:
    """``True`` iff *text* contains at least one Hangul character."""
    return bool(_HANGUL_SYLLABLE_RE.search(text))


# --------------------------------------------------------------------------
# Whole-body equality helpers
# --------------------------------------------------------------------------
# Trailing punctuation we tolerate without breaking whole-body equality.  Kept
# narrow on purpose: "stop, do X" must NOT be whole-body equal to "stop".
_WHOLE_BODY_TRAILING_RE = re.compile(r"[\s.!?…~^\"'`)\-:,;]+$", re.UNICODE)
# Smiley/emote suffix that often follows an ack ("thanks :)", "고마워 ㅎㅎ").
_WHOLE_BODY_EMOTE_RE = re.compile(r"(?:[:;][-]?[)dpo](?:[)dpo])*|ㅎ+|ㅋ+)$", re.IGNORECASE | re.UNICODE)


def _strip_trailing_noise(body: str) -> str:
    """Strip trailing punctuation / whitespace / smileys for whole-body match.

    Repeats up to three times so "thanks!!! :)" reduces to "thanks".
    """
    trimmed = body.strip()
    for _ in range(3):
        before = trimmed
        trimmed = _WHOLE_BODY_TRAILING_RE.sub("", trimmed).strip()
        trimmed = _WHOLE_BODY_EMOTE_RE.sub("", trimmed).strip()
        if trimmed == before:
            break
    return trimmed


def _is_whole_body_match(low_body: str, vocab: Sequence[str]) -> bool:
    """``True`` iff the stripped lowercase body equals one of *vocab*."""
    core = _strip_trailing_noise(low_body)
    if not core:
        return False
    return core in {v.lower() for v in vocab}


# --------------------------------------------------------------------------
# Substring helpers
# --------------------------------------------------------------------------
def _contains_any(haystack: str, needles: Iterable[str]) -> bool:
    return any(n in haystack for n in needles)


def _starts_with_any(low_body: str, prefixes: Iterable[str]) -> bool:
    return any(low_body.startswith(p) for p in prefixes)


# --------------------------------------------------------------------------
# Heuristics for shape-based signals
# --------------------------------------------------------------------------
# Tool-call estimator.  A fragment that asks for multiple distinct actions
# ("read X and grep Y and write Z") generally needs >= 3 tool calls.
_TOOL_ACTION_KEYWORDS_EN: tuple[str, ...] = (
    "read",
    "grep",
    "search",
    "find",
    "write",
    "edit",
    "patch",
    "fetch",
    "crawl",
    "build",
    "run",
    "test",
    "deploy",
    "list",
    "show",
)

_TOOL_ACTION_KEYWORDS_KO: tuple[str, ...] = (
    "읽어",
    "찾아",
    "써",
    "수정",
    "빌드",
    "테스트",
    "실행",
    "보여",
)

# Long-task heuristic words — these are duration hints not vocabulary anchors.
_LONG_WORDS_EN: tuple[str, ...] = (
    "all the",
    "every",
    "across the codebase",
    "across the repo",
    "the whole",
    "comprehensive",
    "thorough",
    "deep",
    "end-to-end",
    "end to end",
)

_LONG_WORDS_KO: tuple[str, ...] = (
    "전체",
    "모든",
    "꼼꼼",
    "철저",
    "전반",
)


def _estimate_tool_calls(low_body: str, body: str) -> int:
    """Rough lower-bound on tool-call count.

    Counts distinct action verbs (English keyword hits plus Korean substring
    hits) and adds one per ``and`` / ``and then`` conjunction.  Pure heuristic;
    used only to fire :data:`ConciergeSignal.MANY_TOOLS`.
    """
    count = 0
    seen = set()
    for kw in _TOOL_ACTION_KEYWORDS_EN:
        if kw in low_body and kw not in seen:
            seen.add(kw)
            count += 1
    for kw in _TOOL_ACTION_KEYWORDS_KO:
        if kw in body and kw not in seen:
            seen.add(kw)
            count += 1
    # Conjunctions multiply effective tool calls.
    count += low_body.count(" and then ")
    count += low_body.count(" and ") // 2  # only every other "and" counts
    return count


def _looks_long(body: str, low_body: str) -> bool:
    """``True`` if the request shape suggests >= 2 minutes of work."""
    if _contains_any(low_body, _LONG_WORDS_EN):
        return True
    if _contains_any(body, _LONG_WORDS_KO):
        return True
    # Long prose body (> 240 chars) — likely a multi-step request.
    if len(body) >= 240:
        return True
    return False


def _looks_many_tools(low_body: str, body: str) -> bool:
    return _estimate_tool_calls(low_body, body) >= 3


def _has_artifact_anchor(low_body: str, body: str) -> bool:
    return _contains_any(low_body, _ARTIFACT_ANCHORS_EN) or _contains_any(body, _ARTIFACT_ANCHORS_KO)


def _has_research_anchor(low_body: str, body: str) -> bool:
    return _contains_any(low_body, _RESEARCH_ANCHORS_EN) or _contains_any(body, _RESEARCH_ANCHORS_KO)


def _has_code_edit_anchor(low_body: str, body: str) -> bool:
    return _contains_any(low_body, _CODE_EDIT_ANCHORS_EN) or _contains_any(body, _CODE_EDIT_ANCHORS_KO)


def _is_status_query(body: str, low_body: str) -> bool:
    if _contains_any(low_body, _STATUS_ANCHORS_EN) or _contains_any(body, _STATUS_ANCHORS_KO):
        return True
    # Only a pure "?" (or whitespace) is treated as a status ping.
    # Short questions like "블로그 돼?" are normal MAIN traffic, not status.
    if body.strip() in {"?", "？"}:
        return True
    return False


def _looks_like_steer(low_body: str, body: str) -> bool:
    """Detect a steering candidate.

    Two patterns:

    * Direct prefix — the body starts with a steer marker (``also ``, ``근데 ``).
    * Post-comma prefix — the body begins with an ack/stop-shaped token but a
      steer marker follows the first comma (``thanks, also do X`` /
      ``그만, 근데 Y``).  PRD §7.1 calls this out as STEER rather than STOP/ACK
      so the steering instruction never gets silently dropped.
    """
    if _starts_with_any(low_body, _STEER_PREFIXES_EN):
        return True
    if _starts_with_any(body, _STEER_PREFIXES_KO):
        return True
    # Post-comma scan.  We check ASCII comma variants and the full-width
    # comma variants common from Korean/IME input.
    for sep in (", ", ",\n", ",  ", "， ", "，\n", "，  "):
        if sep in low_body:
            tail_low = low_body.split(sep, 1)[1].lstrip()
            if any(tail_low.startswith(p) for p in _STEER_PREFIXES_EN):
                return True
        if sep in body:
            tail = body.split(sep, 1)[1].lstrip()
            if any(tail.startswith(p) for p in _STEER_PREFIXES_KO):
                return True
    return False


def _is_explicit_worker_request(low_body: str, body: str) -> bool:
    return _contains_any(low_body, _EXPLICIT_WORKER_ANCHORS_EN) or _contains_any(
        body, _EXPLICIT_WORKER_ANCHORS_KO
    )


def _is_explicit_main_request(low_body: str, body: str) -> bool:
    return _contains_any(low_body, _EXPLICIT_MAIN_ANCHORS_EN) or _contains_any(
        body, _EXPLICIT_MAIN_ANCHORS_KO
    )


# --------------------------------------------------------------------------
# Fingerprint helper — INV-6 replay anchor (design review §3.1)
# --------------------------------------------------------------------------
def fingerprint(
    text: str,
    *,
    concierge_mode_active: bool | None = None,
    
) -> str:
    """Return a stable sha1 fingerprint of (text, mode) for replay tests.

    Lives here so both ``concierge_policy`` and ``control_plane`` can hash the
    same way without importing each other.  The mode bit is included so a
    transcript-replay test can distinguish a fragment classified with concierge
    on vs. off.

    """
    mode = bool(concierge_mode_active) if concierge_mode_active is not None else False
    h = hashlib.sha1()
    h.update(text.encode("utf-8"))
    h.update(b"\x1f")
    h.update(b"1" if mode else b"0")
    return h.hexdigest()


# --------------------------------------------------------------------------
# Public entrypoint
# --------------------------------------------------------------------------
def classify_request(
    text: str, *, lang_hint: str | None = None
) -> ConciergePolicyDecision:
    """Classify a single user-input fragment into a concierge routing verdict.

    Pure function: same ``(text, lang_hint)`` always returns the same value.
    Empty or non-string input is treated as :data:`ConciergeSignal.NOISE` with
    :data:`ConciergeRecommendation.CONTROL`.

    Parameters
    ----------
    text:
        The raw user input.  Leading/trailing whitespace is stripped before
        analysis.  Empty or whitespace-only input becomes NOISE/CONTROL.
    lang_hint:
        Optional caller-supplied language code.  ``"ko"`` short-circuits the
        Korean detection (useful for surfaces that already know the user's
        primary language).  Any other value is treated as "no hint".

    Returns
    -------
    ConciergePolicyDecision
        A frozen dataclass with the routing verdict, signals, confidence and a
        stable ``debug_label``.  Never raises for any string input.
    """
    raw = text if isinstance(text, str) else ""
    body = raw.strip()
    # Normalise to NFC so visually-identical Hangul matches consistently.
    body = unicodedata.normalize("NFC", body)
    low_body = body.lower()

    signals: set[ConciergeSignal] = set()

    # ------------------------------------------------------------------
    # 1. Empty body → NOISE / CONTROL (no model call).
    # ------------------------------------------------------------------
    if not body:
        signals.add(ConciergeSignal.NOISE)
        return ConciergePolicyDecision(
            recommendation=ConciergeRecommendation.CONTROL,
            signals=frozenset(signals),
            confidence=ConciergeConfidence.HIGH,
            debug_label="noise:empty",
            raw_text=raw,
        )

    # ------------------------------------------------------------------
    # 2. Language hint.
    # ------------------------------------------------------------------
    is_korean = lang_hint == "ko" or _looks_korean(body)
    if is_korean:
        signals.add(ConciergeSignal.KOREAN)

    # ------------------------------------------------------------------
    # 3. STOP — whole-body equality (PRD §7.1, DR-1).
    # ------------------------------------------------------------------
    if _is_whole_body_match(low_body, STOP_TOKENS_EN) or _is_whole_body_match(
        body, STOP_TOKENS_KO
    ):
        signals.add(ConciergeSignal.STOP)
        return ConciergePolicyDecision(
            recommendation=ConciergeRecommendation.CONTROL,
            signals=frozenset(signals),
            confidence=ConciergeConfidence.HIGH,
            debug_label="stop",
            raw_text=raw,
        )

    # ------------------------------------------------------------------
    # 4. ACK — whole-body equality (mirrors busyIntegrated.ts ack heuristic).
    # ------------------------------------------------------------------
    if _is_whole_body_match(low_body, ACK_TOKENS_EN) or _is_whole_body_match(
        body, ACK_TOKENS_KO
    ):
        signals.add(ConciergeSignal.ACK)
        return ConciergePolicyDecision(
            recommendation=ConciergeRecommendation.CONTROL,
            signals=frozenset(signals),
            confidence=ConciergeConfidence.HIGH,
            debug_label="ack",
            raw_text=raw,
        )

    # ------------------------------------------------------------------
    # 5. Explicit overrides (read but applied after worker/main signals).
    # ------------------------------------------------------------------
    if _is_explicit_worker_request(low_body, body):
        signals.add(ConciergeSignal.EXPLICIT_WORKER_REQ)
    if _is_explicit_main_request(low_body, body):
        signals.add(ConciergeSignal.EXPLICIT_MAIN_REQ)

    # ------------------------------------------------------------------
    # 6. Status query — short MAIN hit (returns immediately, beats worker
    #    anchors so "status of the report.md task?" stays on MAIN).
    # ------------------------------------------------------------------
    if _is_status_query(body, low_body):
        signals.add(ConciergeSignal.STATUS)
        return ConciergePolicyDecision(
            recommendation=ConciergeRecommendation.MAIN,
            signals=frozenset(signals),
            confidence=ConciergeConfidence.MEDIUM,
            debug_label="status",
            raw_text=raw,
        )

    # ------------------------------------------------------------------
    # 7. STEER — prefix-only candidate; surface verifies in-flight.
    # ------------------------------------------------------------------
    steer_candidate = _looks_like_steer(low_body, body)
    if steer_candidate:
        signals.add(ConciergeSignal.STEER)

    # ------------------------------------------------------------------
    # 8. Worker-candidate signals.
    # ------------------------------------------------------------------
    if _has_artifact_anchor(low_body, body):
        signals.add(ConciergeSignal.ARTIFACT)
    if _has_research_anchor(low_body, body):
        signals.add(ConciergeSignal.RESEARCH)
    if _has_code_edit_anchor(low_body, body):
        signals.add(ConciergeSignal.CODE_EDIT)
    if _looks_long(body, low_body):
        signals.add(ConciergeSignal.LONG)
    if _looks_many_tools(low_body, body):
        signals.add(ConciergeSignal.MANY_TOOLS)

    # ------------------------------------------------------------------
    # 9. Tie-breakers (PRD §8.3).
    # ------------------------------------------------------------------
    if ConciergeSignal.EXPLICIT_MAIN_REQ in signals:
        return ConciergePolicyDecision(
            recommendation=ConciergeRecommendation.MAIN,
            signals=frozenset(signals),
            confidence=ConciergeConfidence.HIGH,
            debug_label="main:explicit",
            raw_text=raw,
        )

    if ConciergeSignal.EXPLICIT_WORKER_REQ in signals:
        return ConciergePolicyDecision(
            recommendation=ConciergeRecommendation.WORKER_LANE,
            signals=frozenset(signals),
            confidence=ConciergeConfidence.HIGH,
            debug_label="worker:explicit",
            raw_text=raw,
        )

    # ------------------------------------------------------------------
    # 10. Strong worker signals (artifact / research / code_edit).
    #     PRD §8.1 final paragraph: single anchor is enough.
    # ------------------------------------------------------------------
    strong = signals & {
        ConciergeSignal.ARTIFACT,
        ConciergeSignal.RESEARCH,
        ConciergeSignal.CODE_EDIT,
    }
    if strong:
        # Multiple strong anchors -> HIGH; one strong anchor -> MEDIUM unless
        # accompanied by a shape-based anchor (LONG / MANY_TOOLS) which pushes
        # to HIGH (the audit transcript's report-writing case fires LONG too).
        has_shape = bool(signals & {ConciergeSignal.LONG, ConciergeSignal.MANY_TOOLS})
        if len(strong) >= 2 or has_shape:
            conf = ConciergeConfidence.HIGH
        else:
            conf = ConciergeConfidence.MEDIUM
        label = "worker:" + "+".join(sorted(s.value for s in strong))
        return ConciergePolicyDecision(
            recommendation=ConciergeRecommendation.WORKER_LANE,
            signals=frozenset(signals),
            confidence=conf,
            debug_label=label,
            raw_text=raw,
        )

    # ------------------------------------------------------------------
    # 11. Weak worker shape only (LONG + MANY_TOOLS together).
    # ------------------------------------------------------------------
    if (
        ConciergeSignal.LONG in signals
        and ConciergeSignal.MANY_TOOLS in signals
    ):
        return ConciergePolicyDecision(
            recommendation=ConciergeRecommendation.WORKER_LANE,
            signals=frozenset(signals),
            confidence=ConciergeConfidence.MEDIUM,
            debug_label="worker:shape",
            raw_text=raw,
        )

    # ------------------------------------------------------------------
    # 12. STEER candidate without strong worker signal → STEER recommendation.
    #     The surface adapter MUST verify a turn is in flight before invoking
    #     ``running_agent.steer``; otherwise it downgrades to MAIN.  Design
    #     review §4.5 explains why this verification cannot live in the
    #     classifier.
    # ------------------------------------------------------------------
    if steer_candidate:
        return ConciergePolicyDecision(
            recommendation=ConciergeRecommendation.STEER,
            signals=frozenset(signals),
            confidence=ConciergeConfidence.LOW,
            debug_label="steer:candidate",
            raw_text=raw,
            notes=("surface must verify main is in flight",),
        )

    # ------------------------------------------------------------------
    # 13. Default → MAIN (conservative bias, PRD §8.3).
    # ------------------------------------------------------------------
    return ConciergePolicyDecision(
        recommendation=ConciergeRecommendation.MAIN,
        signals=frozenset(signals),
        confidence=ConciergeConfidence.MEDIUM if signals else ConciergeConfidence.LOW,
        debug_label="main:default",
        raw_text=raw,
    )
