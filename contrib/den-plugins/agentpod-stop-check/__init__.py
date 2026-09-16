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


def is_supervision_message(text: str, cfg: dict) -> bool:
    """Supervision context comes from the USER's message, never the answer.

    A stop / topic-change in the same session wins over everything else.
    """
    stop = cfg.get("stop_patterns") or DEFAULT_STOP_PATTERNS
    if _matches(stop, text):
        return False
    return _matches(cfg.get("supervision_patterns") or DEFAULT_SUPERVISION_PATTERNS, text)


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
