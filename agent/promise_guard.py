"""Dangling-promise guard: end-of-turn check (bounded continuation tracking).

A Hermes turn that ends sits idle until the next inbound message: a user reply, a Kanban
notification, an async-delegation completion or a watched background process. A final
reply that says "Doing now: I'll finish X myself" therefore promises work nothing will run.
``agent.landing_guard`` catches the opposite failure (many calls, nothing landed); this one
catches the turn that ends early on a promise.

The check runs on the final text reply (``agent.turn_stop_gates``):

* Find promise statements: a "Doing now:"-style heading, or a first-person present or future
  work claim ("I'm consolidating", "I'll finish", "I own #11 directly", "default is taking over").
  Questions, offers, negations and work that waits on the user are not promises.
* Split each into clauses and decide whether something will actually do or wake that work:
  - a clause that names a card (``t_xxxxxxxx``) is covered only if that card is active
    (todo, ready, running, scheduled, review);
  - a self-owned clause naming no card is covered only by this session's own live async
    delegation or watched background process; another profile's running card does not do
    *this* session's work;
  - a "report back / let you know" clause, or a delegated clause naming no card, is covered by
    any wake source, including an active card this session is subscribed to.
* Uncovered clauses: continue the turn ``max_nudges`` times with a note (do it now or say you
  stopped). Past that, on platforms a human reads, rewrite the claim into an honest stop.

The kanban worker stop guard (``agent.kanban_stop``) owns worker processes; this guard skips them.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

ACTIVE_CARD_STATUSES = frozenset({"todo", "ready", "running", "scheduled", "review"})
# Sessions nobody reads live: nudge, never rewrite.
_UNATTENDED_PLATFORMS = frozenset({"cron", "batch", "webhook", "kanban", "scheduler", "eval", "rl"})

_CARD_ID = re.compile(r"\bt_[0-9a-f]{6,12}\b")
_APOS = "['’]"

_BASE_VERBS = (
    r"do|finish|complete|handle|take\s+(?:over|care|it\s+from\s+here|a\s+look|on)|continue|run|start|fix|build|write|consolidate|compare|review|"
    "check|verify|update|implement|land|ship|push|merge|close|work|dig|investigate|look|get|report|"
    r"sort|clean|process|finalize|finalise|draft|prepare|apply|deploy|migrate|test|monitor|keep\s+(?:working|going|at\s+it)|"
    "own|pick|resume|rerun|re-run|retry|proceed|tackle|address|wrap|knock|pull|file|post|send|restart|"
    "reconcile|audit|scan|sweep|patch|edit|rewrite|refactor|record|document|circle|dive|redo|go ahead|"
    "chase|track|crunch|grind|batch|summari[sz]e|collect|gather|compile|assemble|rebuild|recheck|re-check"
)
_ING_STEMS = (
    "do|finish|complet|handl|continu|run|start|fix|build|writ|consolidat|compar|review|check|verif|"
    "updat|implement|land|ship|push|merg|clos|work|dig|investigat|look|get|report|sort|clean|"
    "process|finaliz|finalis|draft|prepar|apply|deploy|migrat|test|monitor|own|pick|resum|rerun|"
    "re-run|retry|proceed|tackl|address|wrap|knock|pull|fil|post|send|restart|reconcil|audit|scan|sweep|"
    "patch|edit|rewrit|refactor|record|document|circl|div|redo|chas|track|crunch|grind|batch|summari|"
    "collect|gather|compil|assembl|rebuild|recheck|re-check"
)
_ADVERBS = r"(?:(?:now|also|then|immediately|directly|personally|currently|still|actively|just|already|going to|about to|right now|next|myself|first)\s+)*"
_ING = rf"(?:{_ING_STEMS})\w*ing\b"
_BASE = rf"(?:{_BASE_VERBS})\b"

# First-person (or own-profile) subject + present-progressive or future work verb.
_SELF_CLAIM_TMPL = (
    rf"(?:\bI{_APOS}m\s+{_ADVERBS}{_ING}"
    rf"|\bI\s+am\s+{_ADVERBS}{_ING}"
    rf"|\bwe{_APOS}re\s+{_ADVERBS}{_ING}"
    rf"|\b(?:I{_APOS}ll|I\s+will|we{_APOS}ll|we\s+will|I\s+am\s+going\s+to|I{_APOS}m\s+going\s+to)\s+{_ADVERBS}{_BASE}"
    rf"|\bI\s+(?:now\s+|directly\s+|also\s+)*own\b"
    rf"|\bI\s+(?:now\s+|still\s+)?(?:retain|keep|hold)\s+ownership\b|\b(?:under|in)\s+my\s+ownership\b"
    rf"|\bI{_APOS}ve\s+(?:started|begun|taken\s+over|picked\s+up)\b"
    rf"|(?:\bI{_APOS}m\s+{_ADVERBS}|^\W*|\bwhile\s+)taking\s+(?:it\s+|this\s+|that\s+)?over\b"
    rf"|\bI{_APOS}m\s+{_ADVERBS}taking\s+(?:care\s+of|on|a\s+look|it\s+from\s+here)\b"
    rf"|\b{{profile}}\s+(?:is|will)\s+{_ADVERBS}(?:{_ING}|{_BASE}|taking\s+(?:it\s+)?over)"
    rf"|\bI{_APOS}m\s+on\s+it\b)"
)
_HEADING = re.compile(
    r"(?im)^[ \t>*_#\-]*(doing\s+now|now\s+doing|working\s+on(?:\s+(?:it|this|that))?|in\s+progress|"
    r"on\s+it|taking\s+over|currently)\s*[:*_—–-]+\s*"
)
# Promise to tell the user later. Fine while something will wake the session.
_REPORT_LATER = re.compile(
    rf"\b(report\w*|follow\s+up|circle\s+back|get\s+back\s+to\s+you|let\s+you\s+know|update\s+you|ping\s+you|"
    rf"post\s+(?:an?\s+)?update|next\s+update|keep\s+you\s+(?:posted|updated)|check\s+back|check\s+in)\b", re.I)
# Work that waits on the user (their reply wakes the session) or is only offered.
_WAITS_ON_USER = re.compile(
    rf"(\b(once|after|when|if|until)\s+you\b|\byour\s+(approval|decision|go-ahead|ok|okay|confirmation|answer|reply|call|input|choice)\b"
    rf"|\b(waiting|wait)\s+(for|on)\s+(you|your)\b|\bawait(ing)?\s+your\b|\bwant\s+me\s+to\b|\bshould\s+I\b|\bshall\s+I\b"
    rf"|\bif\s+(you\s+want|approved|that\s+works|needed|you{_APOS}d\s+like)\b|\bsay\s+(the\s+word|go)\b|\bsend\s+`?continue`?"
    rf"|\bwhen\s+you{_APOS}re\s+ready\b|\bwith\s+your\s+ok\b|\btell\s+me\b|\blet\s+me\s+know\b|\bsend\s+me\b"
    rf"|\b(reply|say|answer|confirm|approve)\b[^.;]{{0,20}}\band\s+I{_APOS}ll\b|^\W*[-*]?\s*\w+\s+it:"
    rf"|\b(when|once|after|as\s+soon\s+as)\b[^.;]{{0,60}}\b(arriv\w*|repl(y|ies|ied)|respond\w*|upload\w*|sign\w*|"
    rf"I\s+can|I\s+have|I\s+get\s+access)\b)", re.I)
# A drafted letter or post for the user ("I'll work harder than any employee") is their text, not a claim.
_DRAFT = re.compile(r"(?im)^\s*(dear\s+\w+|to\s+whom|subject:)|^\s*(sincerely|best\s+regards|regards|warmly),?\s*$")
_NEGATED = re.compile(rf"\b(won{_APOS}t|will\s+not|wouldn{_APOS}t|never|not\s+going\s+to|not\s+yet|(?:is|are|was)\s+not|isn{_APOS}t|aren{_APOS}t|no\s+longer|stopped|I\s+did\s+not|I\s+didn{_APOS}t)\b", re.I)
_SAME_REPLY = re.compile(
    rf"\b(below|in\s+this\s+(reply|message)|here{_APOS}s|as\s+follows|keep\s+(that|this|it)\s+in\s+mind|remember\s+(that|this))\b",
    re.I)
# Hypotheticals and standing policy ("going forward I'll ...") are not claims of work in flight.
_HYPOTHETICAL = re.compile(
    r"\b(if|unless|in\s+case|whether|going\s+forward|from\s+now\s+on|in\s+(?:the\s+)?future|future|next\s+time|"
    r"each\s+time|every\s+time|whenever|later|when\s+we\s+get\s+to|phase\s+\d+|^\W*because|^\W*since)\b", re.I)
# Work that starts when something else lands ("when coding finishes, I'll check ..."): it is woken
# by that event, so it is covered whenever any wake source exists.
_ON_WAKE = re.compile(
    r"\b(when|once|after|as\s+soon\s+as|by\s+the\s+time)\b[^.;]{0,80}?\b(finish\w*|done|complete\w*|land\w*|push\w*|"
    r"opens?|opened|comment\w*|block\w*|runs?|return\w*|report\w*|merge\w*|start\w*|picked\s+up|arrive\w*|pass\w*|"
    r"fail\w*|ready|back|replies|responds?|lands?|exits?|exited)\b|\bfrom\s+there\b", re.I)
_STRONG_SELF = re.compile(rf"\b(myself|directly|personally|taking\s+over|taken\s+over|I{_APOS}m\s+on\s+it)\b", re.I)
# A heading clause with no subject that opens on a work verb ("Doing now: consolidating ...") is the agent's own.
_IMPLICIT_SELF = re.compile(rf"^\W*{_ADVERBS}{_ING}", re.I)
# A clause about another profile's card claims ongoing work only with an ongoing verb or ownership.
_DELEGATED_ONGOING = re.compile(
    rf"\b(?:is|are|now|will)\s+{_ADVERBS}(?:{_ING}|{_BASE})|\b(?:owns|retains?|handles|holds|continues)\b"
    rf"|\bownership\b|\b(?:working|running)\s+on\b", re.I)
# ...and a parked, blocked or waiting card is a status report, not a claim of work.
_PARKED = re.compile(r"\b(parked|on\s+hold|held|paused|blocked|waits?|waiting|stays|remains|until|no\s+action)\b", re.I)
_CLAUSE_SPLIT = re.compile(r"\s*(?:;|\n|(?<=[.!])\s+|\s+while\s+|,\s+and\s+(?=I\b|we\b)|\s+[—–]\s+)\s*")
_CODE_BLOCK = re.compile(r"```.*?```", re.S)


@dataclass(frozen=True)
class DanglingPromiseConfig:
    """``tool_loop_guardrails.dangling_promise``. ``max_nudges`` 0 skips the nudge and goes
    straight to the rewrite (when the platform is attended)."""

    enabled: bool = True
    max_nudges: int = 1
    rewrite_when_attended: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "DanglingPromiseConfig":
        if not isinstance(data, Mapping):
            return cls()
        out: dict[str, Any] = {}
        for name, default in cls().__dict__.items():
            value = data.get(name, default)
            if isinstance(default, bool):
                out[name] = value if isinstance(value, bool) else default
            else:
                try:
                    v = int(value)
                except (TypeError, ValueError):
                    v = default
                out[name] = v if v >= 0 else default
        return cls(**out)


@dataclass(frozen=True)
class WakeState:
    """What can still run or wake this session after the turn ends."""

    card_status: Mapping[str, Optional[str]] = field(default_factory=dict)  # named cards -> status (None = unknown)
    session_cards_active: bool = False  # an active card is subscribed to this session
    live_delegation: bool = False       # this session's async delegation is still running
    live_process: bool = False          # this session's watched background process is still running
    # Cards that finished or blocked but whose notification to this session is not delivered
    # yet: that notification will wake the session, so the card still counts.
    pending_notice: frozenset = frozenset()

    def card_active(self, card_id: str) -> bool:
        return (self.card_status.get(card_id) or "") in ACTIVE_CARD_STATUSES or card_id in self.pending_notice

    @property
    def own_work_live(self) -> bool:
        return self.live_delegation or self.live_process

    @property
    def any_wake(self) -> bool:
        return self.session_cards_active or self.live_delegation or self.live_process or bool(self.pending_notice)


@dataclass(frozen=True)
class DanglingClaim:
    clause: str
    kind: str  # "self" | "report" | "delegated"
    cards: tuple[str, ...]
    span: tuple[int, int]  # statement span in the original reply (for the rewrite)
    reason: str


@dataclass(frozen=True)
class PromiseVerdict:
    action: str  # "nudge" | "rewrite"
    claims: tuple[DanglingClaim, ...]
    nudge: str = ""
    rewritten: str = ""


def _profile_name(profile: Optional[str]) -> str:
    return re.escape(profile) if profile and re.fullmatch(r"[\w-]{2,40}", profile) else "default"


def _self_claim_re(profile: Optional[str]) -> re.Pattern:
    return re.compile(_SELF_CLAIM_TMPL.format(profile=_profile_name(profile)), re.I)


def _profile_subject_re(profile: Optional[str]) -> re.Pattern:
    return re.compile(
        rf"\b{_profile_name(profile)}\s+(?:is|will)\s+{_ADVERBS}(?:{_ING}|{_BASE}|taking\s+(?:it\s+)?over)", re.I)


_FIRST_PERSON = re.compile(_SELF_CLAIM_TMPL.format(profile="(?!)"), re.I)


_QUOTED = re.compile(r"\"[^\"\n]{3,300}\"|“[^”\n]{3,300}”")


def _blank_code(text: str) -> str:
    """Blank fenced code, blockquotes and quoted speech ("I said 'I'll check'"), keeping offsets."""
    text = _CODE_BLOCK.sub(lambda m: " " * len(m.group(0)), text)
    text = _QUOTED.sub(lambda m: " " * len(m.group(0)), text)
    return re.sub(r"(?m)^[ \t]*>.*$", lambda m: " " * len(m.group(0)), text)


def _statements(text: str) -> list[tuple[int, int, bool]]:
    """(start, end, is_heading) spans: each paragraph after a promise heading, plus every sentence."""
    spans: list[tuple[int, int, bool]] = []
    heading_ranges: list[tuple[int, int]] = []
    for m in _HEADING.finditer(text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        para_end = text.find("\n\n", m.end())
        para_end = len(text) if para_end == -1 else para_end
        spans.append((line_start, para_end, True))
        heading_ranges.append((line_start, para_end))
    for m in re.finditer(r"[^\n.!?]+(?:[.!?]+|$)", text):
        s, e = m.start(), m.end()
        if not text[s:e].strip() or any(hs <= s < he for hs, he in heading_ranges):
            continue
        spans.append((s, e, False))
    return sorted(spans)


def _clean(clause: str) -> str:
    return " ".join(re.sub(r"[*`#>]+|(?<!\w)_+|_+(?!\w)", "", clause).split())


def find_dangling_claims(text: Any, wake: Optional[WakeState], *, profile: Optional[str] = None) -> list[DanglingClaim]:
    """Uncovered promise clauses in a final reply. Pure: ``wake`` carries every lookup.
    ``wake=None`` returns every candidate claim without checking coverage (the cheap pre-check)."""
    if not isinstance(text, str) or not text.strip() or _DRAFT.search(text):
        return []
    self_re = _self_claim_re(profile)
    prof_re = _profile_subject_re(profile)
    work = _blank_code(text)
    out: list[DanglingClaim] = []
    for start, end, is_heading in _statements(work):
        stmt = work[start:end]
        body = _HEADING.sub("", stmt, count=1) if is_heading else stmt
        if not is_heading and not self_re.search(body):
            continue
        if stmt.rstrip().endswith("?"):
            continue
        for n, clause in enumerate(_CLAUSE_SPLIT.split(body)):
            c = clause.strip()
            if len(c) < 4 or c.endswith("?"):
                continue
            if _WAITS_ON_USER.search(c) or _SAME_REPLY.search(c) or _HYPOTHETICAL.search(c):
                continue
            strong = bool(_STRONG_SELF.search(c))
            # "default is checking ... on t_x" names the profile's worker card, not this session,
            # unless it says the session does it itself ("default is taking over ... directly").
            prof_subject = bool(prof_re.search(c))
            # The implicit subject ("Doing now: consolidating ...") only on the heading's opening clause.
            has_self = (bool(_FIRST_PERSON.search(c)) or (prof_subject and strong)
                        or bool(is_heading and n == 0 and _IMPLICIT_SELF.match(c)))
            if not (has_self or is_heading or prof_subject):
                continue
            if _NEGATED.search(c) and not strong:
                continue
            cards = tuple(dict.fromkeys(_CARD_ID.findall(c)))
            if not has_self:
                # A clause about a card's worker: checkable only with a card id, and a parked,
                # blocked or waiting card is a status report, not a claim of work.
                if not cards or not _DELEGATED_ONGOING.search(c) or _PARKED.search(c):
                    continue
                kind = "delegated"
            elif _ON_WAKE.search(c) or (not strong and _REPORT_LATER.search(c)):
                kind = "report"  # waits on a wake event: fine while one exists
            else:
                kind = "self"
            reason = "candidate" if wake is None else _uncovered_reason(kind, cards, wake)
            if reason:
                out.append(DanglingClaim(_clean(c), kind, cards, (start, end), reason))
    return out


def _uncovered_reason(kind: str, cards: tuple[str, ...], wake: WakeState) -> Optional[str]:
    if kind != "delegated" and wake.own_work_live:
        return None  # this session's own delegation/background job is running and will report back
    if cards:
        if any(wake.card_active(cid) or wake.card_status.get(cid) is None for cid in cards):
            return None  # active, or not on this board: never accuse what cannot be checked
        states = ", ".join(f"{cid} is {wake.card_status.get(cid) or 'unknown'}" for cid in cards)
        return f"no active card behind it ({states})"
    if kind == "self":
        return None if wake.own_work_live else "no delegation or background job of this session is running"
    return None if wake.any_wake else "no card, delegation or background job will wake this session"


def build_nudge(claims: Iterable[DanglingClaim]) -> str:
    claims = list(claims)
    quoted = "; ".join(f"\"{c.clause[:200]}\" ({c.reason})" for c in claims[:3])
    return (
        "[hermes note — dangling promise: your reply ends this turn but claims work that nothing "
        f"will run: {quoted}. A turn that ends does nothing until the next message, and no active "
        "Kanban card, async delegation or background job covers this. Either do the work now in this "
        "turn with tool calls (dispatching a card or delegation counts), or replace the claim with a plain "
        "statement that you stopped, what is done, and what the next step is and who does it. Do not end "
        "the turn on a 'doing now' that will not happen.]"
    )


def rewrite_reply(text: str, claims: Iterable[DanglingClaim]) -> str:
    """Drop the claiming statements and append an honest stop."""
    claims = list(claims)
    spans = sorted({c.span for c in claims}, reverse=True)
    out = text
    for s, e in spans:
        out = out[:s] + out[e:]
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    listed = "; ".join(c.clause[:160] for c in claims[:3])
    note = (
        f"**Stopped here.** This turn has ended and nothing is running for: {listed}. "
        "It will not happen until you reply. Send `continue` to have it done now, or say to drop it."
    )
    return f"{out}\n\n{note}" if out else note


# ---- runtime wiring (lookups); everything above is pure ------------------------------------------

_NOTIFY_KINDS = ("completed", "blocked", "gave_up", "crashed", "timed_out", "review_requested")


def _other_board_status(conn: Any, ids: list[str]) -> dict[str, Optional[str]]:
    """Cards not on the default board: read the named boards' databases read-only."""
    import glob
    import sqlite3
    found: dict[str, Optional[str]] = {}
    try:
        main = next((r[2] for r in conn.execute("PRAGMA database_list") if r[1] == "main"), "")
    except Exception:
        return found
    if not main:
        return found
    q = ",".join("?" * len(ids))
    for path in sorted(glob.glob(os.path.join(os.path.dirname(main), "kanban", "boards", "*", "kanban.db"))):
        if "/_archived/" in path:
            continue
        try:
            other = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1)
            try:
                for cid, st in other.execute(f"SELECT id, status FROM tasks WHERE id IN ({q})", ids):
                    found[cid] = st
            finally:
                other.close()
        except sqlite3.Error:
            continue
    return found


def _kanban_wake(session_id: str, cards: Iterable[str], connect: Optional[Callable] = None) -> tuple[dict, bool, frozenset]:
    if connect is None:
        from hermes_cli.kanban_db_connect import connect as connect  # noqa: PLW0127
    conn = connect()
    try:
        status: dict[str, Optional[str]] = {}
        ids = list(dict.fromkeys(cards))
        if ids:
            q = ",".join("?" * len(ids))
            rows = conn.execute(f"SELECT id, status FROM tasks WHERE id IN ({q})", ids).fetchall()
            status = {cid: None for cid in ids} | {r[0]: r[1] for r in rows}
            missing = [cid for cid, st in status.items() if st is None]
            if missing:
                status.update(_other_board_status(conn, missing))
        active, pending = False, frozenset()
        if session_id:
            q = ",".join("?" * len(ACTIVE_CARD_STATUSES))
            row = conn.execute(
                f"SELECT 1 FROM kanban_notify_subs s JOIN tasks t ON t.id = s.task_id "
                f"WHERE s.chat_id = ? AND t.status IN ({q}) LIMIT 1",
                [session_id, *sorted(ACTIVE_CARD_STATUSES)]).fetchone()
            active = row is not None
            k = ",".join("?" * len(_NOTIFY_KINDS))
            pending = frozenset(r[0] for r in conn.execute(
                f"SELECT s.task_id FROM kanban_notify_subs s WHERE s.chat_id = ? AND EXISTS ("
                f"SELECT 1 FROM task_events e WHERE e.task_id = s.task_id AND e.id > s.last_event_id "
                f"AND e.kind IN ({k}))", [session_id, *_NOTIFY_KINDS]))
        return status, active, pending
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _live_delegation(agent: Any) -> bool:
    from tools.async_delegation import has_live_for_session
    return has_live_for_session(
        session_key=str(getattr(agent, "_gateway_session_key", "") or ""),
        parent_session_id=str(getattr(agent, "session_id", "") or ""))


def _live_process(agent: Any) -> bool:
    from tools.process_registry import process_registry
    sid = str(getattr(agent, "session_id", "") or "")
    key = str(getattr(agent, "_gateway_session_key", "") or "")
    with process_registry._lock:
        running = list(process_registry._running.values())
    for p in running:
        if p.exited or not (p.notify_on_complete or p.watch_patterns):
            continue
        if (sid and p.parent_session_id == sid) or (key and p.session_key == key):
            return True
    return False


def collect_wake_state(agent: Any, text: str) -> WakeState:
    cards = _CARD_ID.findall(text or "")
    status, session_active, pending = _kanban_wake(str(getattr(agent, "session_id", "") or ""), cards)
    deleg = proc = False
    try:
        deleg = _live_delegation(agent)
    except Exception:
        logger.debug("dangling-promise: delegation lookup failed", exc_info=True)
        deleg = True  # unknown: do not accuse a session that may have work in flight
    try:
        proc = _live_process(agent)
    except Exception:
        logger.debug("dangling-promise: process lookup failed", exc_info=True)
    return WakeState(card_status=status, session_cards_active=session_active,
                     live_delegation=deleg, live_process=proc, pending_notice=pending)


def _config_for(agent: Any) -> DanglingPromiseConfig:
    guard = getattr(agent, "_tool_guardrails", None)
    cfg = getattr(getattr(guard, "config", None), "dangling_promise", None)
    return cfg if isinstance(cfg, DanglingPromiseConfig) else DanglingPromiseConfig()


def _is_kanban_worker() -> bool:
    """The dispatcher-owned worker (agent.kanban_stop owns its stop); delegated children only inherit the env."""
    try:
        from agent.delegation_context import owned_kanban_task
        return bool(owned_kanban_task())
    except Exception:
        return bool(os.environ.get("HERMES_KANBAN_TASK"))


def _profile_for(agent: Any) -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return os.environ.get("HERMES_PROFILE") or "default"


def _attended(agent: Any) -> bool:
    platform = str(getattr(agent, "platform", "") or "").strip().lower()
    return platform not in _UNATTENDED_PLATFORMS


def evaluate_final_reply(agent: Any, final_response: Any, *,
                         wake_fn: Optional[Callable[[Any, str], WakeState]] = None) -> Optional[PromiseVerdict]:
    """Stop-gate entry point. None means let the reply through unchanged."""
    wake_fn = wake_fn or collect_wake_state
    cfg = _config_for(agent)
    if not cfg.enabled or not isinstance(final_response, str) or not final_response.strip():
        return None
    if _is_kanban_worker() or getattr(agent, "_interrupt_requested", False):
        return None
    profile = _profile_for(agent)
    # Cheap pre-check before any DB lookup: no candidate claim, no work.
    if not find_dangling_claims(final_response, None, profile=profile):
        return None
    try:
        wake = wake_fn(agent, final_response)
    except Exception:
        logger.debug("dangling-promise: wake lookup failed; letting reply through", exc_info=True)
        return None
    claims = tuple(find_dangling_claims(final_response, wake, profile=profile))
    if not claims:
        return None
    nudges = getattr(agent, "_dangling_promise_nudges", 0)
    if not isinstance(nudges, int):
        nudges = 0
    if nudges < cfg.max_nudges:
        return PromiseVerdict("nudge", claims, nudge=build_nudge(claims))
    if cfg.rewrite_when_attended and _attended(agent):
        return PromiseVerdict("rewrite", claims, rewritten=rewrite_reply(final_response, claims))
    return None


__all__ = [
    "ACTIVE_CARD_STATUSES", "DanglingClaim", "DanglingPromiseConfig", "PromiseVerdict", "WakeState",
    "build_nudge", "collect_wake_state", "evaluate_final_reply", "find_dangling_claims", "rewrite_reply",
]
