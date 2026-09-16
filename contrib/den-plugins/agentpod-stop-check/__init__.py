"""agentpod-stop-check — runtime stop gate for one opted-in supervisory project.

What it enforces: in a supervision turn on an opted-in project, the agent may
not end the turn while the board still holds unattended unfinished work.

How, using only supported runtime lifecycle:

1. ``pre_llm_call`` (observer, returns ``None``) — records this turn's **user
   message** for the session. That is what establishes supervision context. The
   gate never infers supervision from the model's own final answer, so an
   unrelated question is untouched no matter how it is phrased, and a
   same-session "stop / forget it / different topic" message wins immediately.

2. ``pre_verify`` — the REAL enforcement. Returns
   ``{"action": "continue", "message": ..., "final_verdict": ...}`` so the
   agent keeps working the board in the same turn (it can call tools and
   actually act), bounded by the runtime's ``agent.max_verify_nudges`` AND a
   cross-process ledger under ``$HERMES_HOME`` — and, in the same directive,
   states the verdict the turn may not end without. The runtime applies that
   verdict to the delivered answer AFTER every output transform, so the
   outcome is ordering-independent and survives the model ignoring the last
   instruction or the iteration budget running out. When the continuation
   budget is spent the directive becomes verdict-only (``action: final``).
   On turns that edited no files this needs
   ``agent.pre_verify_on_no_edit_turns: true`` (a general, default-off core
   setting) — the supervision sweeps this exists for rarely edit files.

3. ``transform_llm_output`` — legacy fallback, kept so the plugin still
   degrades usefully on a runtime without the enforced-verdict contract: the
   quiet answer is **replaced** with a short factual blocker sized to the
   platform budget.

Honest limits (tested, stated in the README, not papered over):

* The gate **dispatches nothing**. It does not spawn, claim, write to the
  board, or kill anything. "Requested" and "executed" are never conflated.
* Enforcement is ordering-independent: the ``pre_verify`` directive carries
  both the continuation and the verdict, and the runtime stamps the verdict
  onto the delivered answer after every ``transform_llm_output`` hook has run.
  An earlier-sorting transform plugin can still replace the model's own text,
  but it can no longer make the turn end quiet.
* Ownership requires a structural binding (the child's own kanban pin recorded
  at spawn, or the card's own recorded workspace). A worker launched without
  one is ``owner_unknown``: actionable, never quiet.
* Verified owners prove **liveness**, not progress — and only buy silence when
  a completion handle or a recorded, verified wake covers their exit.

Scope: inert unless ``agentpod_stop_check.enabled`` is true AND the turn's
``session_id`` is listed in ``agentpod_stop_check.session_ids``. Only the
configured board/project is read; no other project, profile or board.

config.yaml (default profile):

    agentpod_stop_check:
      enabled: true
      board: agentpod
      # project_id/tenant: omit unless the cards really carry them — a scope
      # matching zero of N cards is a hard error, not a clean board.
      session_ids: ["<supervisor session id>"]
      gate_authorities: ["den"]     # who may record a human gate
      heartbeat_stale_seconds: 900
      turn_context_ttl_seconds: 900
      max_findings: 5
      max_report_chars: 700         # platform budget for the fallback text
      max_continuations: 2

Activation is staged, reversible and core-first — see ``README.md`` and the
read-only ``activation_preflight.py``, which refuses an installed core that
cannot run the gate and a config scope that selects nothing.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

try:  # loaded as a package by PluginManager
    from . import stopcheck
except ImportError:  # direct-file load in tests
    import stopcheck  # type: ignore

logger = logging.getLogger(__name__)

PLUGIN_ID = "agentpod-stop-check"

# A supervision turn: the user asked about the state of the project/board.
DEFAULT_SUPERVISION_PATTERNS = (
    r"\bboard\b",
    r"\bkanban\b",
    r"\bsweep\b",
    r"\bsupervis",
    r"\bbacklog\b",
    r"\b(?:any )?(?:status|update|progress)\b.*\b(?:project|work|team|cards?|tasks?)\b",
    r"\bwhat(?:'s| is| are)\b.*\b(?:going on|in flight|blocked|left|outstanding)\b",
    r"\bcheck (?:on )?(?:the )?(?:work|workers|tasks|cards|project)\b",
)
# Same-session user override: stop / drop it / change of subject.
#
# These patterns locate a CANDIDATE stop token. They are deliberately broad;
# `_stop_directive()` below decides whether the candidate is actually a
# directive the user is giving *now*, because a bare occurrence of "stop"
# is not a stop command: "do not stop supervising the board" (negated),
# `you said "stop working"` (quoted history) and "tenants stop working after
# an LXD restart" (a report about the system) all contain one and none of
# them asks the agent to stop.
DEFAULT_STOP_PATTERNS = (
    r"\bstop\b",
    r"\bpause\b",
    r"\bhold off\b",
    r"\bforget it\b",
    r"\bdrop it\b",
    r"\bnever ?mind\b",
    r"\bleave it\b",
    r"\bnot now\b",
)

# The project this gate supervises. A status/progress question that names the
# project by NAME ("what is progress on AgentPod?") is a supervision turn even
# though it uses none of the board vocabulary above. Override/extend with
# `agentpod_stop_check.project_aliases`; `board` / `project_id` / `tenant`
# (when configured) are folded in automatically. An alias is REQUIRED for this
# route, so an unrelated progress question ("any update on my flight?") is
# still untouched.
DEFAULT_PROJECT_ALIASES = ("agentpod",)

_PROGRESS_PATTERN = re.compile(
    r"\b(?:status|statuses|update|updates|progress|state of|where (?:are|is) (?:we|it|things)"
    r"|how (?:is|are|'s) (?:it|things|we|that)|what'?s? left|outstanding|in flight|going on)\b"
)

# Spans the user is QUOTING (their own history, logs, an error string). A stop
# token inside one is not a directive issued on this turn — UNLESS the user
# explicitly adopts the quote as the instruction (`_ADOPTION_FRAME`).
_QUOTED_SPAN = re.compile(
    r"\"[^\"]*\"|`[^`]*`|\u201c[^\u201d]*\u201d|\u2018[^\u2019]*\u2019"
    # A single quote only delimits when it is not an intra-word apostrophe,
    # so "don't stop, it's fine" is never mangled into a quoted span.
    r"|(?<![A-Za-z0-9])'[^']*'(?![A-Za-z0-9])",
    re.DOTALL,
)
# The text that introduces a quote the user is ISSUING rather than citing:
#   Please do exactly this: "stop supervising the board"
# Quoting is not automatically historical, so an adopted span stays live text.
_ADOPTION_FRAME = re.compile(
    r"(?:do (?:exactly |precisely |just )?(?:this|that|the following)"
    r"|my (?:instruction|order|request)(?: is)?"
    r"|(?:the )?instructions?(?: is| are)?"
    r"|i(?:'m| am)? (?:telling|asking|instructing) you(?: to)?"
    r"|here(?:'s| is) (?:the|my) (?:instruction|order|request)"
    r"|repeat after me|verbatim|word for word)"
    r"\s*[:,\-\u2013\u2014]?\s*$"
)
# Clause boundaries. A directive occupies its own clause. `:` counts, so
# "URGENT: stop working" and "I'm done: stop the sweep" reach clause-initial
# position instead of dying on their preamble.
_CLAUSE_SPLIT = re.compile(r"[.!?;:\n\u2014]+|,")
# Leading list markers / bullets / numbering / emoji / whitespace. Stripped
# from a clause prefix before imperative analysis, so "- stop …", "1) stop …"
# and "\U0001F6D1 stop …" are the same imperative as a bare "stop …".
_LEAD_MARKERS = re.compile(r"^[\W\d_]+", re.UNICODE)
# Words/phrases that may precede an imperative without making it non-imperative:
# discourse markers, politeness, urgency and time adverbials, and explicit
# performative frames ("I am asking you to …").
_LEAD_UNIT = (
    r"ok|okay|alright|actually|hey|so|and|but|then|also|now|yeah|yes|no|nope|"
    r"please|just|kindly|seriously|honestly|really|finally|again|anyway|anyhow|well|"
    r"urgent|urgently|asap|immediately|right now|right away|first of all|first|"
    r"before anything else|for now|for today|for the moment|today|tonight|"
    r"at this point|from now on|going forward|temporarily|meanwhile|"
    r"i want you to|i would like you to|i'd like you to|i need you to|"
    r"i am asking you to|i'm asking you to|i ask you to|"
    r"i am telling you to|i'm telling you to|i tell you to|"
    r"i am instructing you to|i'm instructing you to|"
    r"do exactly this|do this|do the following|"
    r"you can|you should|you must|you need to|you could|you may|"
    r"can you|could you|would you|will you|let's|lets|"
    r"we should|we can|we need to"
)
_DIRECTIVE_LEAD = re.compile(
    r"^(?:(?:" + _LEAD_UNIT + r")\b[\s,:_\-\u2013\u2014]*)*\s*$"
)
# A negation immediately governing the candidate ("do not stop", "never stop",
# "no need to pause", "instead of stopping"). Apostrophes are normalised to
# ASCII first (`_normalise`), so the smart-quote "don\u2019t" negates too.
_NEGATION = re.compile(
    r"\b(?:not|never|cannot|can'?t|won'?t|wont|don'?t|dont|doesn'?t|didn'?t|shouldn'?t|"
    r"no need to|without|instead of|rather than|avoid|refrain from|keep from)\b"
    r"(?:\s+\w+){0,3}\s*$"
)
# Evidence that the clause already has a structure of its own before the stop
# token — a subject (pronoun / determiner+noun) or a subordinator. Then the
# token REPORTS behaviour or sits in a subordinate clause; it does not command:
#   "tenants stop working after an LXD restart"
#   "you are missing details and stop working"   (the user describing OUR bug)
_SUBJECTED = re.compile(
    r"\b(?:i|we|you|they|he|she|it|this|that|these|those|who|whom|whose|which|"
    r"where|when|while|if|unless|until|otherwise|because|since|after|although|"
    r"though|whether|the|a|an|my|our|your|their|its|his|her|there|"
    r"tenants?|users?|workers?|jobs?|tasks?|cards?|servers?|pods?|containers?|"
    r"services?|agents?|processes|clients?|customers?|nodes?|things?|bugs?)\b"
)
# Apostrophe normalisation (U+2019 / U+02BC -> ASCII) so every clause rule
# above sees one spelling.
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u02bc": "'"})

# Per-occurrence verdicts.
_STOP_NONE = "none"            # no live stop token at all
_STOP_REPORTED = "reported"    # negated / quoted history / has its own subject
_STOP_UNDECIDED = "undecided"  # a live stop token we cannot parse either way
_STOP_COMMAND = "command"      # the user is telling us to stop, now

_LOCK = threading.Lock()
# session_id -> (user_message, recorded_at). Per-turn supervision context.
_TURN_CONTEXT: dict[str, tuple[str, float]] = {}
# How long a recorded user message may govern. `pre_llm_call` fires once per
# turn, so a context older than this belongs to an earlier turn whose hook call
# did not repeat (hook error, adapter that skips it, subagent path) — and a
# turn the user never framed as supervision must not be gated by a stale one.
# Absence already fails closed; this makes STALENESS fail closed too.
DEFAULT_TURN_CONTEXT_TTL = 900.0


# ---------------------------------------------------------------- config ---

def _cfg() -> dict:
    try:
        from hermes_cli.config import load_config

        return (load_config() or {}).get("agentpod_stop_check", {}) or {}
    except Exception:
        return {}


def _in_scope(cfg: dict, session_id: Optional[str]) -> bool:
    """Opt-in only. No global behaviour change, ever."""
    if not cfg.get("enabled"):
        return False
    allowed = cfg.get("session_ids") or []
    if isinstance(allowed, str):
        allowed = [allowed]
    return bool(session_id) and str(session_id) in {str(s) for s in allowed}


def _matches(patterns, text: str) -> bool:
    blob = (text or "").lower()
    return any(re.search(p, blob) for p in patterns)


def _normalise(text: str) -> str:
    """One spelling for every clause rule: ASCII apostrophes, lower case."""
    return (text or "").translate(_APOSTROPHES).lower()


def _strip_quoted(text: str) -> str:
    """Blank out quoted spans the user is *citing* (their own earlier words, a
    log line, an error) — but NOT a span they explicitly adopt as this turn's
    instruction (``Please do exactly this: "stop supervising the board"``).
    Quotation marks are emphasis as often as they are citation, so adoption is
    decided by the frame that introduces the span, not by the quotes.

    Offsets need not be preserved — clauses are re-derived from the result.
    """
    blob = text or ""

    def _sub(m: "re.Match") -> str:
        lead = blob[: m.start()]
        if _ADOPTION_FRAME.search(lead):
            # Adopted: keep the span as live text, minus its delimiters.
            return " " + m.group(0)[1:-1] + " "
        return " "

    return _QUOTED_SPAN.sub(_sub, blob)


def _classify_stop(text: str, cfg: Optional[dict] = None) -> str:
    """Classify the strongest stop signal in ``text``. One of the ``_STOP_*``.

    Per candidate occurrence, in this precedence:

      * **negated** ("do not stop", "never stop", "no need to pause") or inside
        a quoted span the user is citing -> ``reported``;
      * **imperative position** — the clause prefix is nothing but list
        markers, discourse/politeness/urgency words or an explicit performative
        frame ("I am asking you to") -> ``command``;
      * **has a structure of its own** before the token — a subject or a
        subordinator -> ``reported`` ("tenants stop working after an LXD
        restart", "you are missing details and stop working": the user
        describing OUR behaviour, not commanding);
      * anything else -> ``undecided``.

    ``command`` dominates ``undecided`` dominates ``reported``. There is no
    universal fail-open: an ``undecided`` token keeps the gate QUIET (it never
    arms supervision) but is not reported as a directive either.
    """
    cfg = cfg or {}
    patterns = cfg.get("stop_patterns") or DEFAULT_STOP_PATTERNS
    blob = _normalise(_strip_quoted(_normalise(text)))
    if not blob.strip():
        return _STOP_NONE
    verdict = _STOP_NONE
    for clause in _CLAUSE_SPLIT.split(blob):
        if not clause.strip():
            continue
        for pattern in patterns:
            for m in re.finditer(pattern, clause):
                prefix = clause[: m.start()]
                if _NEGATION.search(prefix):
                    verdict = verdict if verdict != _STOP_NONE else _STOP_REPORTED
                    continue
                if _DIRECTIVE_LEAD.match(_LEAD_MARKERS.sub("", prefix)):
                    return _STOP_COMMAND
                if _SUBJECTED.search(prefix):
                    verdict = verdict if verdict != _STOP_NONE else _STOP_REPORTED
                    continue
                verdict = _STOP_UNDECIDED
    return verdict


def stop_directive(text: str, cfg: Optional[dict] = None) -> bool:
    """Is the user telling the agent, on THIS turn, to stop / drop the topic?

    Precedence for a real stop is preserved exactly: an imperative
    "stop the board sweep, forget it for now" still wins over every
    supervision signal, and so does one carrying a preamble, a bullet, a
    number, a colon or an emoji. ``True`` means *clearly commanded* — see
    ``_classify_stop`` for what a token that is negated, cited or merely
    reported means, and ``is_supervision_message`` for what an undecided one
    does.
    """
    return _classify_stop(text, cfg) == _STOP_COMMAND


def _project_aliases(cfg: dict) -> set:
    raw = cfg.get("project_aliases")
    if raw is None:
        raw = list(DEFAULT_PROJECT_ALIASES)
    if isinstance(raw, str):
        raw = [raw]
    values = [str(v) for v in (raw or [])]
    for key in ("board", "project_id", "tenant"):
        val = cfg.get(key)
        if val:
            values.append(str(val))
    return {v.strip().lower() for v in values if v and v.strip()}


def _names_the_project(text: str, cfg: dict) -> bool:
    blob = (text or "").lower()
    return any(
        re.search(r"\b" + re.escape(alias) + r"\b", blob)
        for alias in _project_aliases(cfg)
    )


def is_supervision_message(text: str, cfg: dict) -> bool:
    """Supervision context comes from the USER's message, never the answer.

    A real stop / topic-change in the same session wins over everything else
    (see ``stop_directive`` for what "real" means). A stop token we cannot
    classify either way also keeps the gate quiet — the doubt is spent on the
    user, not on the supervision. Otherwise the turn is a supervision turn
    when it uses board vocabulary, or when it asks for status/progress on the
    supervised project BY NAME.
    """
    if _classify_stop(text, cfg) in (_STOP_COMMAND, _STOP_UNDECIDED):
        return False
    if _matches(cfg.get("supervision_patterns") or DEFAULT_SUPERVISION_PATTERNS, text):
        return True
    return bool(_PROGRESS_PATTERN.search((text or "").lower())) and _names_the_project(text, cfg)


def _supervision_turn(cfg: dict, session_id: str) -> bool:
    with _LOCK:
        entry = _TURN_CONTEXT.get(str(session_id))
    if not entry:
        # No recorded user message for this turn: fail CLOSED (stay inert)
        # rather than guessing supervision from the model's own text.
        return False
    message, at = entry
    ttl = float(cfg.get("turn_context_ttl_seconds", DEFAULT_TURN_CONTEXT_TTL))
    if ttl > 0 and (time.time() - float(at or 0)) > ttl:
        # Stale context: it describes an earlier turn, not this one.
        return False
    return is_supervision_message(message, cfg)


def reset_state() -> None:
    """Test helper — clears in-process turn context (not the on-disk ledger)."""
    with _LOCK:
        _TURN_CONTEXT.clear()


# -------------------------------------------------- cross-process ledger ---

def _ledger_path(cfg: dict) -> Path:
    raw = cfg.get("ledger_path")
    if raw:
        return Path(raw)
    try:
        from hermes_constants import get_hermes_home

        base = Path(get_hermes_home())
    except Exception:
        base = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
    return base / "agentpod-stop-check" / "continuations.json"


def _grant_continuation(session_id: str, fingerprint: str, cfg: dict) -> tuple[bool, bool]:
    """Bounded continuation budget shared across processes.

    Returns ``(granted, terminal)`` where ``terminal`` marks the LAST grant for
    this ``(session, board-fingerprint)`` pair — the one that carries the
    fail-explicit demand, because the gate will not ask again.

    **Window policy (explicit).** The cap is ``max_continuations`` per
    ``(session, fingerprint)`` per rolling ``continuation_window_seconds``. A
    board whose findings never change is therefore *rate*-bounded, not exempt:
    it can buy at most ``cap`` continuations per window and no more, and each
    window's last grant is terminal. It is bounded, not one-shot-forever — a
    board that is still unattended an hour later is a genuinely new supervision
    occasion, while a loop inside one turn cannot exceed the cap (the runtime's
    own ``agent.max_verify_nudges`` bounds the turn independently).

    The cap must hold for the whole pair even when a second gateway/worker
    process runs the same session, so the ledger is a file under
    ``$HERMES_HOME`` guarded by an atomic lock directory. If the lock cannot be
    taken the answer is **no** (fail closed).
    """
    cap = int(cfg.get("max_continuations", 2))
    window = float(cfg.get("continuation_window_seconds", 900))
    now = time.time()
    key = f"{session_id}|{fingerprint}"
    path = _ledger_path(cfg)
    lock = path.with_suffix(".lock")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return (False, False)

    acquired = False
    for _ in range(40):
        try:
            os.mkdir(lock)
            acquired = True
            break
        except FileExistsError:
            try:  # break a lock abandoned by a crashed process
                if now - os.path.getmtime(lock) > 60:
                    os.rmdir(lock)
                    continue
            except OSError:
                pass
            time.sleep(0.025)
        except Exception:
            return (False, False)
    if not acquired:
        return (False, False)

    try:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
        data = {
            k: v
            for k, v in data.items()
            if isinstance(v, list) and len(v) == 2 and now - float(v[1]) <= window
        }
        count, first = data.get(key, [0, now])
        if int(count) >= cap:
            granted, terminal = False, False
        else:
            used = int(count) + 1
            data[key] = [used, float(first)]
            granted, terminal = True, used >= cap
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)
        return (granted, terminal)
    except Exception:
        return (False, False)
    finally:
        try:
            os.rmdir(lock)
        except Exception:
            pass


# -------------------------------------------------------------- evaluate ---

def _evaluate(cfg: dict) -> stopcheck.Verdict:
    return stopcheck.evaluate_board(
        board=cfg.get("board") or None,
        db_path=cfg.get("db_path") or None,
        cfg=cfg,
    )


def _already_reports(text: str, verdict: stopcheck.Verdict) -> bool:
    """The answer already names every unattended card — nothing to add.

    This is the ONLY content check, and it is a positive one: it can suppress a
    duplicate report, never establish the gate.
    """
    if not verdict.ok or not verdict.findings:
        return False
    blob = text or ""
    return all(f.task_id in blob for f in verdict.findings)


# ------------------------------------------------------------------ hooks ---

def on_pre_llm_call(
    session_id: str = "",
    user_message: str = "",
    **_: Any,
) -> None:
    """Observer: record the turn's user message. Never injects context."""
    try:
        if not session_id:
            return None
        with _LOCK:
            _TURN_CONTEXT[str(session_id)] = (str(user_message or ""), time.time())
            if len(_TURN_CONTEXT) > 64:  # bounded
                oldest = sorted(_TURN_CONTEXT.items(), key=lambda kv: kv[1][1])[:32]
                for k, _v in oldest:
                    _TURN_CONTEXT.pop(k, None)
    except Exception:
        logger.debug("[%s] pre_llm_call context capture failed", PLUGIN_ID, exc_info=True)
    return None


def on_pre_verify(
    session_id: str = "",
    platform: str = "",
    model: str = "",
    coding: bool = False,
    attempt: int = 0,
    final_response: str = "",
    changed_paths: Optional[list] = None,
    **_: Any,
) -> Optional[dict]:
    """PRIMARY enforcement: real, bounded continuation + an enforced verdict.

    Every directive carries ``final_verdict`` — the text the turn may not end
    without. The runtime applies it to the DELIVERED answer after all output
    transforms (``agent.verify_hooks.apply_pre_verify_verdict``), so the
    outcome no longer depends on the model obeying the last continuation, on
    how much budget is left, or on which transform plugin sorts first. When the
    continuation budget is spent the directive is verdict-only: stop, but stop
    LOUD.
    """
    try:
        cfg = _cfg()
        if not _in_scope(cfg, session_id) or not _supervision_turn(cfg, session_id):
            return None
        verdict = _evaluate(cfg)
        if verdict.quiet_allowed:
            return None
        if _already_reports(final_response or "", verdict):
            return None
        enforced = stopcheck.render_report(
            verdict, max_chars=int(cfg.get("max_report_chars", 700))
        )
        granted, terminal = _grant_continuation(session_id, verdict.fingerprint(), cfg)
        if not granted:
            logger.warning(
                "[%s] continuation budget spent; enforcing terminal verdict "
                "(session=%s ok=%s findings=%d)",
                PLUGIN_ID, session_id, verdict.ok, len(verdict.findings),
            )
            return {"action": "final", "message": enforced}
        logger.warning(
            "[%s] continuing supervision turn (session=%s ok=%s findings=%d terminal=%s)",
            PLUGIN_ID, session_id, verdict.ok, len(verdict.findings), terminal,
        )
        return {
            "action": "continue",
            "message": stopcheck.render_report(
                verdict,
                continuation=True,
                terminal=terminal,
                max_chars=int(cfg.get("max_continuation_chars", 2000)),
            ),
            # Same directive, enforced half: if the agent stops anyway — now or
            # after the budget runs out — this is what ships.
            "final_verdict": enforced,
        }
    except Exception:
        logger.exception("[%s] pre_verify stop-check failed", PLUGIN_ID)
        return None


def on_transform_llm_output(
    response_text: str = "",
    session_id: str = "",
    model: str = "",
    platform: str = "",
    **_: Any,
) -> Optional[str]:
    """FALLBACK: replace a still-quiet conclusion with a short blocker."""
    try:
        cfg = _cfg()
        if not _in_scope(cfg, session_id) or not _supervision_turn(cfg, session_id):
            return None
        verdict = _evaluate(cfg)
        if verdict.quiet_allowed:
            return None
        if _already_reports(response_text or "", verdict):
            return None
        report = stopcheck.render_report(
            verdict, max_chars=int(cfg.get("max_report_chars", 700))
        )
        logger.warning(
            "[%s] replaced quiet conclusion (session=%s ok=%s findings=%d chars=%d)",
            PLUGIN_ID, session_id, verdict.ok, len(verdict.findings), len(report),
        )
        # REPLACE, never append: the false "nothing changed" claim must not
        # ship at all, not even as the first paragraph.
        return report
    except Exception:
        logger.exception("[%s] stop-check failed", PLUGIN_ID)
        try:
            return (
                "STOP-CHECK ERROR — the stop-check itself failed; this turn may not "
                "be treated as 'no material change'. Re-run the board reconciliation."
            )
        except Exception:
            return None


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("pre_verify", on_pre_verify)
    ctx.register_hook("transform_llm_output", on_transform_llm_output)
    logger.info("[%s] registered (opt-in, session-scoped)", PLUGIN_ID)
