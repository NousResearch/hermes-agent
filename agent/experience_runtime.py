"""Agent-facing glue for Level 2 experience learning.

Three entry points, all best-effort and all no-ops when the feature is off:

* :func:`retrieve_experience_context` — turn prologue (``agent/turn_context.py``)
* :func:`apply_user_correction`      — turn prologue, before retrieval
* :func:`record_turn_experience`     — turn finalizer (``agent/turn_finalizer.py``)

The pure logic (extraction, scoring, rendering, redaction) lives in
:mod:`agent.experience`; the durable store is ``ExperienceStoreMixin`` on
``SessionDB``. This module owns only the config read and the agent plumbing, so
both halves stay unit-testable without an ``AIAgent``.

Level 2 boundary — this module may write experience rows, read them back, and
count outcomes. It never modifies source, skills, config, dependencies, or
runtime behaviour beyond adding advisory context to one prompt.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "retrieval_enabled": True,
    "max_results": 3,
    "min_score": 0.18,
    "max_age_days": 90.0,
    "max_context_chars": 1800,
    "prune_every": 200,
}

# Turn counter for amortized pruning, per process.
_writes_since_prune = 0


def _as_bool(value: Any) -> bool:
    """Truthiness for config values, tolerant of YAML strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in ("0", "false", "off", "no", "none", "")


def experience_config(agent: Any = None) -> Dict[str, Any]:
    """Merged ``experience:`` config block. Never raises."""
    cfg = dict(_DEFAULTS)
    try:
        from hermes_cli.config import load_config_readonly

        raw = (load_config_readonly() or {}).get("experience") or {}
        if isinstance(raw, dict):
            for key, default in _DEFAULTS.items():
                if key in raw and raw[key] is not None:
                    try:
                        # bool(str) is True for "false", so parse booleans by
                        # value rather than by constructor.
                        cfg[key] = (
                            _as_bool(raw[key])
                            if isinstance(default, bool)
                            else type(default)(raw[key])
                        )
                    except (TypeError, ValueError):
                        pass
    except Exception:
        logger.debug("experience config unavailable; using defaults", exc_info=True)
    # An env override exists for benchmarking and for a fast kill switch that
    # does not require editing config.yaml on a running gateway.
    import os

    env = os.environ.get("HERMES_EXPERIENCE")
    if env is not None:
        cfg["enabled"] = _as_bool(env)
    env_r = os.environ.get("HERMES_EXPERIENCE_RETRIEVAL")
    if env_r is not None:
        cfg["retrieval_enabled"] = _as_bool(env_r)
    return cfg


def _store(agent: Any):
    """The ``SessionDB`` for this agent, or ``None`` when unavailable.

    Persistence-isolated agents (the background review fork) must never write
    experiences: their transcript is a replay, so counting it would double-count
    every outcome.
    """
    if getattr(agent, "_persist_disabled", False):
        return None
    db = getattr(agent, "_session_db", None)
    if db is None or not hasattr(db, "record_experience"):
        return None
    return db


# ── Retrieval ───────────────────────────────────────────────────────────


def retrieve_experience_context(agent: Any, query: Any) -> str:
    """Fenced experience block for *query*, or ``""``.

    Returns empty for trivial prompts (greetings carry no task signal) and
    whenever nothing clears the relevance floor — an unrelated experience in
    context is worse than none.
    """
    cfg = experience_config(agent)
    if not (cfg["enabled"] and cfg["retrieval_enabled"]):
        return ""
    if not isinstance(query, str) or not query.strip():
        return ""

    from agent.experience import format_experience_block, rank_rows
    from agent.memory_provider import is_trivial_prompt

    if is_trivial_prompt(query):
        return ""

    db = _store(agent)
    if db is None:
        return ""

    started = time.perf_counter()
    rows = db.fetch_experience_candidates(
        workspace=workspace_key(agent), max_age_days=cfg["max_age_days"]
    )
    if not rows:
        return ""
    top = rank_rows(
        rows,
        query,
        limit=int(cfg["max_results"]),
        min_score=float(cfg["min_score"]),
        max_age_days=float(cfg["max_age_days"]),
    )
    block = format_experience_block(top, max_chars=int(cfg["max_context_chars"]))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    # Retrieval latency is a benchmark metric, so log it even on the empty path.
    logger.info(
        "experience retrieval: candidates=%d matched=%d chars=%d latency_ms=%.2f",
        len(rows), len(top), len(block), elapsed_ms,
    )
    try:
        agent._experience_last_retrieval = {
            "candidates": len(rows),
            "matched": len(top),
            "chars": len(block),
            "latency_ms": round(elapsed_ms, 3),
            "ids": [r.get("id") for r in top],
        }
    except Exception:
        pass
    return block


def _agent_cwd(agent: Any) -> str:
    """The agent's logical working directory.

    Uses the same pair every other consumer does — an explicit ``session_cwd``
    on the agent, else :func:`agent.runtime_cwd.resolve_agent_cwd` (ContextVar
    override → ``TERMINAL_CWD`` → process cwd). Reading ad-hoc attribute names
    instead silently yields the *process* cwd, which on the gateway is the home
    directory rather than the project the work is happening in — every
    experience would then be scoped to one meaningless key.
    """
    val = getattr(agent, "session_cwd", None)
    if isinstance(val, str) and val.strip():
        return val.strip()
    try:
        from agent.runtime_cwd import resolve_agent_cwd

        return str(resolve_agent_cwd())
    except Exception:
        try:
            import os

            return os.getcwd()
        except Exception:
            return ""


def _lookup_verification(agent: Any) -> Dict[str, Any]:
    """One ``verification_status`` call → project root + build/test evidence.

    ``verification_status`` already resolves the project root (git root, else
    marker root) on its way to the evidence and returns it, so a single call
    answers both questions this module has: **where** an experience belongs
    (the scoping key) and **whether** the work actually held up. Deriving the
    root separately would mean a second ``git rev-parse`` subprocess for an
    answer we are already handed.

    Falls back to ``{"root": cwd, "verification": ""}`` outside a workspace or
    when the evidence store is unavailable — the pre-P1 behaviour, where cwd is
    the scoping key and no verification signal exists.
    """
    cwd = _agent_cwd(agent)
    result: Dict[str, Any] = {"root": cwd, "verification": "", "command": ""}
    try:
        from agent.verification_evidence import verification_status

        status = verification_status(
            session_id=getattr(agent, "session_id", "") or None, cwd=cwd
        )
        if isinstance(status, dict):
            result["root"] = str(status.get("root") or cwd)
            result["verification"] = str(status.get("status") or "")
            evidence = status.get("evidence")
            if isinstance(evidence, dict):
                result["command"] = str(evidence.get("canonical_command") or "")
    except Exception:
        logger.debug("verification lookup unavailable", exc_info=True)
    try:
        agent._experience_workspace_root = result["root"]
    except Exception:
        pass
    return result


def workspace_key(agent: Any) -> str:
    """Scoping key for this agent: project root, else cwd.

    A task learned in ``repo/`` must still be found from ``repo/src``, so the
    key is the project root rather than the raw cwd.

    Cached: an agent's cwd does not move mid-session, and this is read on the
    pre-model path where a ``git rev-parse`` subprocess per turn would be pure
    waste. The verification verdict is deliberately NOT cached alongside it —
    see :func:`fresh_verification`.
    """
    cached = getattr(agent, "_experience_workspace_root", None)
    if isinstance(cached, str) and cached:
        return cached
    return _lookup_verification(agent)["root"]


def fresh_verification(agent: Any) -> Dict[str, Any]:
    """Build/test evidence as of *now*. Never cached.

    Called from the finalizer, where the answer must reflect commands this
    turn ran. A value cached at turn start would report the state before the
    work happened — precisely inverting the signal this exists to capture.
    """
    return _lookup_verification(agent)


# ── User correction ─────────────────────────────────────────────────────


def apply_user_correction(agent: Any, user_message: Any) -> Optional[str]:
    """Record a correction against this session's last experience.

    Returns the corrected experience id, or ``None``. The signal is the whole
    point of the hook: a correction is the only outcome label the *user*
    supplies directly, so it outweighs every inferred one.
    """
    cfg = experience_config(agent)
    if not cfg["enabled"]:
        return None
    if not isinstance(user_message, str) or not user_message.strip():
        return None

    from agent.experience import detect_user_correction, sanitize_stored_text

    if not detect_user_correction(user_message):
        return None

    db = _store(agent)
    if db is None:
        return None
    session_id = getattr(agent, "session_id", "") or ""
    prior = db.latest_experience_for_session(session_id)
    if not prior:
        return None
    text = sanitize_stored_text(user_message, 240)
    if db.record_experience_correction(prior["id"], text):
        logger.info(
            "experience correction recorded: id=%s session=%s",
            prior["id"], session_id or "none",
        )
        return str(prior["id"])
    return None


# ── Write ───────────────────────────────────────────────────────────────


def record_turn_experience(
    agent: Any,
    *,
    user_message: Any,
    messages: List[Dict[str, Any]],
    completed: bool,
    failed: bool,
    interrupted: bool,
    exit_reason: Any = "",
    final_response: Any = "",
    api_calls: int = 0,
    turn_id: str = "",
) -> Optional[str]:
    """Extract and persist this turn's experience. Returns its row id or ``None``."""
    global _writes_since_prune

    cfg = experience_config(agent)
    if not cfg["enabled"]:
        return None
    db = _store(agent)
    if db is None:
        return None

    from agent.experience import extract_experience

    # Read the evidence AFTER the turn's work, not before — see
    # fresh_verification. This is the ground truth that keeps a confidently
    # wrong answer from being recorded as a success.
    evidence = fresh_verification(agent)

    exp = extract_experience(
        user_message=user_message,
        messages=messages,
        completed=completed,
        failed=failed,
        interrupted=interrupted,
        exit_reason=exit_reason,
        final_response=final_response,
        api_calls=api_calls,
        duration_s=_turn_duration(agent),
        session_id=getattr(agent, "session_id", "") or "",
        turn_id=turn_id or "",
        model=getattr(agent, "model", "") or "",
        cwd=_agent_cwd(agent),
        workspace=str(evidence.get("root") or ""),
        verification=str(evidence.get("verification") or ""),
        verification_command=str(evidence.get("command") or ""),
    )
    if exp is None:
        return None

    exp_id = db.record_experience(exp.to_row())
    if exp_id:
        logger.info(
            "experience recorded: id=%s outcome=%s verification=%s tools=%d "
            "workspace=%s session=%s",
            exp_id, exp.outcome, exp.verification or "none", len(exp.tools),
            exp.workspace or "none", exp.session_id or "none",
        )
        _writes_since_prune += 1
        every = int(cfg["prune_every"])
        if every > 0 and _writes_since_prune >= every:
            _writes_since_prune = 0
            try:
                removed = db.prune_experiences()
                if removed:
                    logger.info("experience prune removed %d rows", removed)
            except Exception:
                logger.debug("experience prune failed", exc_info=True)
    return exp_id


def _turn_duration(agent: Any) -> Optional[float]:
    start = getattr(agent, "_turn_started_at", None)
    if isinstance(start, (int, float)) and start > 0:
        elapsed = time.time() - float(start)
        if 0 < elapsed < 86400:
            return elapsed
    return None
