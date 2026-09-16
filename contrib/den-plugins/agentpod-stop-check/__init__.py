"""agentpod-stop-check — runtime stop gate for the AgentPod supervisory session.

Problem this exists for (measured 2026-09-16): the default supervisor kept
concluding "no material change" while t_cefa1028 / t_fd00ad81 sat on stale
supervisor holds and t_f7fad689 needed scoped safe work. One progressing PR
worker was treated as whole-board coverage. Lifecycle wakes and completion
notifications (t_5d94b7f9) work — nothing enforced *whole-project
reconciliation before a turn is allowed to end quietly*.

Two supported runtime hooks, both scoped to one opted-in session:

1. ``transform_llm_output`` — fires once per turn in ``agent/turn_finalizer.py``
   for every non-interrupted turn, regardless of whether files were edited.
   This is the ENFORCEMENT path: when the drafted answer is a quiet
   "no material change"/"all done" conclusion and the board still has
   unattended unfinished cards, the quiet text is replaced with an explicit
   stop-check report. The quiet conclusion cannot be delivered.

2. ``pre_verify`` — fires only when the turn edited files, and can actually
   keep the agent going (``{"action": "continue", ...}``), bounded by the
   runtime's ``agent.max_verify_nudges`` plus our own per-session cap.

INTEGRATION LIMITATION (honest, not papered over): Hermes exposes no supported
hook that can continue a turn which made no file edits. For those turns the
gate is fail-EXPLICIT output only — it blocks the quiet conclusion and states
the required next step. It does NOT guarantee the agent acted, and this text
says so in the report itself. No message injection, no dispatch, no restart,
no cron, no subprocess is used to fake continuation.

Scope: inert unless ``agentpod_stop_check.enabled`` is true AND the turn's
``session_id`` is listed in ``agentpod_stop_check.session_ids``. A ``/new``
session or any other profile/session is untouched. Interrupted turns (user
``/stop``) never reach ``transform_llm_output`` — the runtime guards that.

config.yaml (default profile):

    agentpod_stop_check:
      enabled: true
      board: agentpod
      session_ids: ["<supervisor session id>"]
      heartbeat_stale_seconds: 900
      max_findings: 10
      max_continuations: 2
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Optional

try:  # loaded as a package by PluginManager
    from . import stopcheck
except ImportError:  # direct-file load in tests
    import stopcheck  # type: ignore

logger = logging.getLogger(__name__)

PLUGIN_ID = "agentpod-stop-check"

# Drafted conclusions that claim there is nothing to do / everything is done.
DEFAULT_QUIET_MARKERS = (
    r"no material change",
    r"no materially new",
    r"nothing (?:further )?to (?:do|report|act on)",
    r"no (?:new |further )?action(?:s)? (?:is |are )?(?:needed|required)",
    r"no change since",
    r"all (?:cards|tasks|work) (?:are |is )?(?:done|complete)",
    r"everything (?:is )?(?:on track|covered|done)",
    r"project (?:is )?complete",
    r"standing by",
    r"holding pattern",
)

_LOCK = threading.Lock()
# (session_id, fingerprint) -> (count, first_seen). Bounds continuation only.
_CONTINUATIONS: dict[tuple[str, str], tuple[int, float]] = {}


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


def _looks_quiet(text: str, cfg: dict) -> bool:
    markers = cfg.get("quiet_markers") or DEFAULT_QUIET_MARKERS
    blob = (text or "").lower()
    return any(re.search(m, blob) for m in markers)


def _evaluate(cfg: dict) -> stopcheck.Verdict:
    return stopcheck.evaluate_board(
        board=cfg.get("board") or None,
        db_path=cfg.get("db_path") or None,
        cfg=cfg,
    )


def _grant_continuation(session_id: str, fingerprint: str, cfg: dict) -> bool:
    """Bounded, race-safe: one continuation per (session, board state).

    Concurrent wakes/heartbeats landing on the same finding set get exactly one
    continuation between them — the rest fall through to the explicit report.
    """
    cap = int(cfg.get("max_continuations", 2))
    window = float(cfg.get("continuation_window_seconds", 900))
    now = time.time()
    key = (str(session_id), str(fingerprint))
    with _LOCK:
        for k, (_c, seen) in list(_CONTINUATIONS.items()):
            if now - seen > window:
                _CONTINUATIONS.pop(k, None)
        count, first = _CONTINUATIONS.get(key, (0, now))
        if count >= cap:
            return False
        _CONTINUATIONS[key] = (count + 1, first)
        return True


def reset_state() -> None:
    """Test helper — clears the continuation ledger."""
    with _LOCK:
        _CONTINUATIONS.clear()


# ------------------------------------------------------------------ hooks ---

def on_transform_llm_output(
    response_text: str = "",
    session_id: str = "",
    model: str = "",
    platform: str = "",
    **_: Any,
) -> Optional[str]:
    """Enforcement path: a quiet conclusion may not ship over unattended work."""
    try:
        cfg = _cfg()
        if not _in_scope(cfg, session_id):
            return None
        if not _looks_quiet(response_text, cfg):
            return None
        verdict = _evaluate(cfg)
        if verdict.quiet_allowed:
            return None  # genuinely covered — quiet is allowed
        report = stopcheck.render_report(verdict)
        logger.warning(
            "[%s] blocked quiet conclusion (session=%s ok=%s findings=%d)",
            PLUGIN_ID,
            session_id,
            verdict.ok,
            len(verdict.findings),
        )
        return f"{(response_text or '').strip()}\n\n{report}"
    except Exception:
        # Fail open on the transform itself would hide the defect; fail
        # EXPLICIT instead — but never raise into the turn.
        logger.exception("[%s] stop-check failed", PLUGIN_ID)
        try:
            return (
                f"{(response_text or '').strip()}\n\n"
                "SUPERVISOR STOP-CHECK ERROR — the stop-check itself failed; "
                "this turn may not be treated as 'no material change'."
            )
        except Exception:
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
    """Real continuation when the runtime offers one (edit turns only)."""
    try:
        cfg = _cfg()
        if not _in_scope(cfg, session_id):
            return None
        if not _looks_quiet(final_response or "", cfg):
            return None
        verdict = _evaluate(cfg)
        if verdict.quiet_allowed:
            return None
        if not _grant_continuation(session_id, verdict.fingerprint(), cfg):
            return None
        return {
            "action": "continue",
            "message": stopcheck.render_report(verdict, continuation=True),
        }
    except Exception:
        logger.exception("[%s] pre_verify stop-check failed", PLUGIN_ID)
        return None


def register(ctx) -> None:
    ctx.register_hook("transform_llm_output", on_transform_llm_output)
    ctx.register_hook("pre_verify", on_pre_verify)
    logger.info("[%s] registered (opt-in, session-scoped)", PLUGIN_ID)
