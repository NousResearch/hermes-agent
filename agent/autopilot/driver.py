"""Autopilot driver — engine-enforced goal-chasing continuation.

Called at the moment the agent would deliver a final answer (no tool calls).
If autopilot is active it asks the independent judge (Hermes Council) whether the
GOAL is verifiably complete:

    * complete   -> return None, the loop delivers the answer.
    * not done   -> return a synthetic user directive; the loop injects it and
                    keeps working toward the goal.

Termination is governed by the goal quality-gate, NOT a turn count (per product
requirement: no default cap). The only stops are: the judge says complete, a
genuine no-progress stall, an optional user-set continuation cap, or a judge
failure (which fails OPEN to delivery). The budget is auto-extended on each
continuation so the standard ``max_iterations`` ceiling never ends an autopilot
run on its own.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from typing import Any, Optional

from agent.autopilot.council_gate import CompletionVerdict, judge_completion

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}

# Premature-stop / "handoff" phrases. When autopilot is active and the goal is NOT
# verifiably complete, a final response that reads like one of these is a give-up
# disguised as a wrap-up (the "ran 18h, wrote a handoff for next session, gate not
# met" failure). It must never be allowed to terminate the run — see
# ``_looks_like_giveup`` usage in ``maybe_continue`` (fails CLOSED on these).
_GIVEUP_PATTERNS = (
    "productive limit",
    "reached its limit",
    "reached the limit",
    "this session has reached",
    "handoff for next",
    "handoff for the next",
    "handoff written",
    "next session should",
    "next session starts",
    "in a fresh session",
    "resume in a fresh session",
    "fresh session",
    "context near exhaustion",
    "context is near",
    "context exhaustion",
    "running low on context",
    "out of context",
    "stopping here",
    "stopping for now",
    "i'll stop here",
    "pausing here",
    "session summary (honest",
    "session has ended",
    "session is at its end",
)


def _looks_like_giveup(text: str) -> bool:
    """True when ``text`` reads like a premature stop / next-session handoff.

    Deliberately conservative + only consulted while autopilot is active and the
    goal is unmet, so a false positive merely re-injects a keep-going directive.
    """
    if not text:
        return False
    t = text.lower()
    return any(p in t for p in _GIVEUP_PATTERNS)


def is_autopilot_active(agent: Any) -> bool:
    """Whether engine-enforced autopilot goal-chasing is active for this agent.

    The per-agent ``autopilot_mode`` flag is AUTHORITATIVE. It is seeded from
    ``HERMES_AUTOPILOT`` at agent creation (see agent_init) and flipped by the
    ``/autopilot`` toggle / TUI mirror. We must NOT also OR the env var in here:
    doing so made ``/autopilot off`` impossible whenever ``HERMES_AUTOPILOT`` or
    ``--autopilot`` was set, because the OR could never be turned off per
    session (the reported "off doesn't stop it" bug). The env / session-flag
    branches below are fallbacks only for agents that predate the seeded
    attribute.
    """
    mode = getattr(agent, "autopilot_mode", None)
    if mode is not None:
        return bool(mode)
    if getattr(agent, "_autopilot_session", False):
        return True
    return os.environ.get("HERMES_AUTOPILOT", "").strip().lower() in _TRUTHY


def reset_turn_state(agent: Any) -> None:
    """Reset per-turn autopilot bookkeeping at the start of run_conversation."""
    agent._autopilot_continuations = 0
    agent._autopilot_last_final_hash = ""
    agent._autopilot_stall = 0
    agent._autopilot_last_msgcount = 0


def _cfg_int(agent: Any, attr: str, env: str, default: int) -> int:
    val = getattr(agent, attr, None)
    if val is None:
        val = os.environ.get(env, "")
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _council_model(agent: Any) -> str:
    return (
        getattr(agent, "_autopilot_council_model", "")
        or os.environ.get("AUTOPILOT_COUNCIL_MODEL", "")
        or os.environ.get("COUNCIL_HERMES_MODEL", "")
        or ""
    )


def resolve_goal(agent: Any, user_message: Any) -> str:
    """Resolve the goal text to chase.

    Priority:
      1. An explicit autopilot goal set via ``/autopilot goal <text>``
         (``agent._autopilot_goal``).
      2. The active standing ``/goal`` for this session, if one is set. This is
         the integration point that lets ``/goal "ship X"`` + ``/autopilot on``
         chase the /goal target with the Council as the gate — without
         retyping it into autopilot. Read-only; ``/goal``'s own loop is
         untouched.
      3. The user's current message.
    """
    explicit = getattr(agent, "_autopilot_goal", "") or ""
    if explicit.strip():
        return explicit.strip()
    standing = _standing_goal_text(agent)
    if standing:
        return standing
    return _coerce_text(user_message)


def _standing_goal_text(agent: Any) -> str:
    """Return the active standing ``/goal`` text (plus any subgoals) for this
    session, or "" when there is none / it is paused-done / unavailable.

    Reads the persisted GoalState directly from the session store so it works
    regardless of platform (CLI/TUI/gateway) and never mutates ``/goal`` state.
    Fails safe to "" on any error.
    """
    sid = getattr(agent, "session_id", "") or ""
    if not sid:
        return ""
    try:
        from hermes_cli.goals import load_goal
    except Exception:  # noqa: BLE001 — goals module optional
        return ""
    try:
        state = load_goal(sid)
    except Exception:  # noqa: BLE001
        return ""
    if state is None or getattr(state, "status", "") != "active":
        return ""
    goal = (getattr(state, "goal", "") or "").strip()
    if not goal:
        return ""
    try:
        block = state.render_subgoals_block()
    except Exception:  # noqa: BLE001
        block = ""
    if block:
        return f"{goal}\n\nAdditional criteria the goal must also satisfy:\n{block}"
    return goal


def _coerce_text(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    # Multimodal content: list of {type, text|...} parts.
    if isinstance(message, list):
        parts = []
        for p in message:
            if isinstance(p, dict):
                if p.get("type") == "text" and p.get("text"):
                    parts.append(str(p["text"]))
                elif p.get("text"):
                    parts.append(str(p["text"]))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    if isinstance(message, dict):
        return str(message.get("content") or message.get("text") or "")
    return str(message)


def _summarize_work(messages: list[dict[str, Any]], *, limit: int = 8) -> str:
    """Compact recent transcript so the judge sees what was actually done."""
    out: list[str] = []
    for m in messages[-limit:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        if role == "tool":
            content = _short(m.get("content"), 300)
            out.append(f"[tool result] {content}")
            continue
        tool_calls = m.get("tool_calls") or []
        if tool_calls:
            names = []
            for tc in tool_calls:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                names.append(str(fn.get("name", "?")))
            out.append(f"[{role} called tools] {', '.join(names)}")
        content = _short(m.get("content"), 400)
        if content:
            out.append(f"[{role}] {content}")
    return "\n".join(out)


def _short(value: Any, limit: int) -> str:
    s = "" if value is None else str(value)
    s = s.strip().replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + "…"


def _emit(agent: Any, text: str) -> None:
    """Surface an autopilot status line to the user.

    Tries the agent's status plumbing first (interactive CLI ``_vprint`` +
    gateway/TUI ``status_callback``). When the agent suppresses status output
    (the ``-z/--oneshot`` machine-readable path sets ``suppress_status_output``),
    fall back to stderr so the operator can still see autopilot is working
    without polluting the machine-readable stdout.
    """
    suppressed = bool(getattr(agent, "suppress_status_output", False))
    if not suppressed:
        for attr in ("_emit_status", "_buffer_status"):
            fn = getattr(agent, attr, None)
            if callable(fn):
                try:
                    fn(text)
                    return
                except Exception:  # noqa: BLE001
                    continue
    # Suppressed (oneshot) or no status plumbing: stderr keeps stdout clean.
    try:
        print(text, file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass


def keep_budget_ahead(agent: Any, headroom: int = 50) -> None:
    """Keep the iteration budget ahead of usage while autopilot is active.

    Autopilot's terminator is the goal quality-gate (Seam B in the no-tool-calls
    branch), NOT the iteration budget. But the loop can exit via budget
    exhaustion (``while`` condition false / ``iteration_budget.consume()`` ->
    False), which happens AFTER many tool calls and BYPASSES Seam B entirely —
    the agent then stops silently mid-task with no continuation and no
    autopilot stop-reason. That is the "runs for a while then suddenly stops"
    bug. Called at the top of each loop iteration, this tops up the budget so an
    active autopilot run is never terminated by the budget; the no-progress
    detector and the optional user cap remain the real safeties.

    Respects an explicit user continuation cap: once reached we stop extending so
    the run can wind down naturally.
    """
    if not is_autopilot_active(agent):
        return
    max_cont = _cfg_int(agent, "_autopilot_max_continuations", "AUTOPILOT_MAX_CONTINUATIONS", 0)
    if max_cont > 0 and getattr(agent, "_autopilot_continuations", 0) >= max_cont:
        return
    budget = getattr(agent, "iteration_budget", None)
    used = getattr(budget, "used", 0) if budget is not None else 0
    current = max(int(getattr(agent, "_api_call_count", 0) or 0), int(used))
    need = current + headroom
    try:
        if budget is not None and getattr(budget, "max_total", 0) < need:
            budget.max_total = need
    except Exception:  # noqa: BLE001
        pass
    try:
        if getattr(agent, "max_iterations", 0) < need:
            agent.max_iterations = need
    except Exception:  # noqa: BLE001
        pass


def maybe_continue(
    agent: Any,
    messages: list[dict[str, Any]],
    final_response: str,
    user_message: Any,
) -> Optional[str]:
    """Decide whether to keep working. Returns a directive to inject, or None.

    Returning a string means: inject it as a synthetic user turn and continue the
    loop. Returning None means: stop and deliver ``final_response``.
    """
    if not is_autopilot_active(agent):
        return None

    # Lazily init state if reset_turn_state wasn't called (defensive).
    if not hasattr(agent, "_autopilot_continuations"):
        reset_turn_state(agent)

    goal = resolve_goal(agent, user_message)
    if not goal.strip():
        return None  # nothing to chase

    max_continuations = _cfg_int(agent, "_autopilot_max_continuations", "AUTOPILOT_MAX_CONTINUATIONS", 0)
    no_progress_k = max(1, _cfg_int(agent, "_autopilot_no_progress_k", "AUTOPILOT_NO_PROGRESS_K", 3))

    work_summary = _summarize_work(messages)
    giveup = _looks_like_giveup(final_response)
    try:
        verdict: CompletionVerdict = judge_completion(
            goal, work_summary, final_response, mode="fast", council_model=_council_model(agent)
        )
    except Exception as exc:  # noqa: BLE001 — judge must never crash the turn
        # Normally the judge fails OPEN (deliver). But a give-up/handoff response
        # must NEVER terminate the run via the fail-open path — that would let a
        # "productive limit reached, gate not met" wrap-up end an autopilot goal.
        # Fail CLOSED on give-ups: keep going with a strong anti-handoff directive.
        if giveup:
            agent._autopilot_continuations = getattr(agent, "_autopilot_continuations", 0) + 1
            _extend_budget(agent)
            _emit(agent, "↻ Autopilot: ignoring premature handoff/stop (judge unavailable) — goal not verified complete; continuing.")
            logger.warning("autopilot: give-up/handoff detected on judge-error — failing CLOSED (continue #%d)",
                           agent._autopilot_continuations)
            return _giveup_directive()
        logger.warning("autopilot: judge raised (%s); delivering result", exc)
        return None

    if verdict.complete:
        _emit(agent, f"✅ Autopilot: goal verified complete ({verdict.summary}).")
        logger.info("autopilot: COMPLETE after %d continuation(s) — %s",
                    getattr(agent, "_autopilot_continuations", 0), verdict.summary)
        return None

    # --- not complete: decide whether we are allowed to continue -------------
    final_hash = hashlib.md5((final_response or "").strip().encode("utf-8")).hexdigest()
    msgcount = len(messages)
    no_new_work = msgcount <= getattr(agent, "_autopilot_last_msgcount", 0) + 2
    if final_hash == getattr(agent, "_autopilot_last_final_hash", "") and no_new_work:
        agent._autopilot_stall = getattr(agent, "_autopilot_stall", 0) + 1
    else:
        agent._autopilot_stall = 0
    agent._autopilot_last_final_hash = final_hash
    agent._autopilot_last_msgcount = msgcount

    if agent._autopilot_stall >= no_progress_k:
        _emit(agent, f"⚠️ Autopilot: no progress after {agent._autopilot_stall} attempts — stopping and surfacing.")
        logger.warning("autopilot: no-progress stall (%d) — stopping. directive was: %s",
                       agent._autopilot_stall, verdict.directive[:200])
        return None

    if max_continuations > 0 and getattr(agent, "_autopilot_continuations", 0) >= max_continuations:
        _emit(agent, f"⚠️ Autopilot: reached user continuation cap ({max_continuations}) — stopping.")
        return None

    # --- continue: extend budget so the standard cap never ends the run ------
    agent._autopilot_continuations = getattr(agent, "_autopilot_continuations", 0) + 1
    _extend_budget(agent)
    if giveup:
        _emit(
            agent,
            f"↻ Autopilot (#{agent._autopilot_continuations}): ignoring premature "
            "handoff/stop — goal not verified complete; take the next concrete action.",
        )
        logger.warning("autopilot: give-up/handoff detected while goal unmet — re-injecting (CONTINUE #%d, %s)",
                       agent._autopilot_continuations, verdict.summary)
        return _giveup_directive(verdict)
    _emit(
        agent,
        f"↻ Autopilot continuing (#{agent._autopilot_continuations}): "
        f"{verdict.verdict or 'incomplete'} — {verdict.directive[:120]}",
    )
    logger.info("autopilot: CONTINUE #%d (%s) directive=%s",
                agent._autopilot_continuations, verdict.summary, verdict.directive[:200])
    return _build_directive(verdict)


def reenter_after_abnormal_exit(
    agent: Any,
    messages: list[dict[str, Any]],
    final_response: str,
    user_message: Any,
    *,
    exit_kind: str,
    interrupted: bool = False,
) -> Optional[str]:
    """Belt-and-suspenders continuation for loop exits that bypass Seam B.

    ``maybe_continue`` (Seam B) is the primary autopilot gate, but it only runs
    in the *clean* no-tool-calls branch. The conversation loop can also exit via
    abnormal paths that never reach Seam B — an empty response after all retries,
    and partial-stream / prior-turn-content recovery. Each of those silently ends
    an autopilot run mid-goal (the "runs for a while then suddenly stops" class of
    bug that budget exhaustion also caused before ``keep_budget_ahead``).

    This reuses the SAME gate (``maybe_continue``: same Council judge, same
    no-progress + user-cap safeties, same budget extension and continuation
    counter) at those exits, so the termination policy stays in one place.
    Returns a directive to inject (caller should re-enter the loop) or ``None``
    (caller should deliver / stop exactly as before).

    Fails safe: returns ``None`` on user interrupt, when autopilot is inactive,
    or on any internal error.
    """
    if interrupted or getattr(agent, "_interrupt_requested", False):
        return None
    if not is_autopilot_active(agent):
        return None

    try:
        directive = maybe_continue(agent, messages, final_response or "", user_message)
    except Exception as exc:  # noqa: BLE001 — must never crash the turn
        logger.warning("autopilot: reenter judge raised (%s); delivering result", exc)
        return None

    if directive:
        logger.info("autopilot: re-entering loop after abnormal exit (%s)", exit_kind)
    return directive


def make_clarify_autoanswer(agent: Any, fallback: Any = None):
    """Build a clarify callback that auto-answers via the Council (Seam A).

    When autopilot is active, a ``clarify`` tool call is answered by the
    independent judge with the most-recommended option/answer instead of
    blocking for a human. Falls back to the platform callback (if any), then a
    safe default, so the tool never errors mid-run.
    """

    def _callback(question, choices=None):
        try:
            from agent.autopilot.council_gate import choose_answer

            answer = choose_answer(question, choices, council_model=_council_model(agent))
            if answer:
                _emit(agent, f"🤖 Autopilot answered clarify: {str(answer)[:80]}")
                logger.info("autopilot: auto-answered clarify %r -> %r", str(question)[:80], str(answer)[:80])
                return answer
        except Exception as exc:  # noqa: BLE001
            logger.warning("autopilot: clarify auto-answer failed (%s)", exc)
        if callable(fallback):
            return fallback(question, choices)
        return "Proceed with the most reasonable, defensible default and continue."

    return _callback


def _extend_budget(agent: Any, headroom: int = 100) -> None:
    """Grant a generous budget chunk so the standard cap never ends an autopilot
    run on its own. Termination is governed by the goal gate / no-progress /
    user cap — not by ``max_iterations``. A chunk (not infinity) is granted so a
    runaway *tool-call* loop that never tries to stop still eventually trips the
    budget-exhausted summary safety.
    """
    budget = getattr(agent, "iteration_budget", None)
    used = getattr(budget, "used", 0) if budget is not None else 0
    current = max(int(getattr(agent, "_api_call_count", 0) or 0), int(used))
    need = current + headroom
    try:
        if budget is not None and getattr(budget, "max_total", 0) < need:
            budget.max_total = need
    except Exception:  # noqa: BLE001
        pass
    try:
        if getattr(agent, "max_iterations", 0) < need:
            agent.max_iterations = need
    except Exception:  # noqa: BLE001
        pass


def _build_directive(verdict: CompletionVerdict) -> str:
    return (
        "[AUTOPILOT DIRECTIVE — this is an active instruction from the engine, NOT a "
        "notification you may dismiss or treat as stale] An independent Hermes Council review "
        "found the GOAL is NOT yet verifiably complete. "
        f"{verdict.directive} "
        "Take the next concrete action toward the goal now. Do not stop, do not "
        "summarize-and-wait, do not write a handoff, and do not ask the user — make the most "
        "defensible decision from context and act on it. The user only sees the final result "
        "once the goal is genuinely, verifiably done."
    )


def _giveup_directive(verdict: "CompletionVerdict | None" = None) -> str:
    """Directive for when the agent produced a premature stop / handoff while the
    goal is unmet. Names the anti-pattern explicitly and redirects to action."""
    review = f" Independent review: {verdict.directive}" if verdict and getattr(verdict, "directive", "") else ""
    return (
        "[AUTOPILOT DIRECTIVE — do NOT stop] You just produced a wrap-up / handoff / "
        "\"productive limit\" message, but the GOAL is NOT verifiably complete, so the run "
        "continues. Writing a handoff for a \"next session\" or declaring a productive limit is "
        "NOT completion and NOT an allowed stop. If you are low on context, CHECKPOINT (update "
        "the ledger/durable notes) and KEEP WORKING in this same run — a fresh session must "
        "resume this exact goal, not treat the handoff as done. Do NOT treat this directive as a "
        "stale notification. Right now, take ONE concrete technical step toward a still-failing "
        "part of the goal: reproduce it, diagnose the root cause, apply a fix, and re-verify — "
        "do not re-argue scope or re-classify work as \"acceptable.\"" + review
    )
