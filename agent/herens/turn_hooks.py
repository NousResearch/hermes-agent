"""Herens turn-level pre-hook — injects reflection, lessons, and strategy context.

Called from hermes_advanced pre_llm_call when herens is enabled.
Returns an optional system-prompt addendum for the current turn only.
This lives in the volatile/transient tier — NOT the stable frozen tier.
Prompt cache safety: we do NOT touch the stable system prompt; instead we
return content that callers may inject as a separate user-visible block.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe_dmn_reflect(user_message: str, *, session_id: str = "") -> str:
    """Pull a DMN reflection snippet if brain_networks is on and DMN fires."""
    try:
        from agent.brain_networks.runtime import get_orchestrator

        orch = get_orchestrator()
        if orch is None:
            return ""
        if session_id:
            orch.bind_session(session_id)
        ctx = {"user_message": user_message, "session_id": session_id}
        if not orch.dmn.should_reflect(ctx):
            return ""
        ref = orch.dmn.generate_reflection({**ctx, "history": ""})
        content = str(ref.get("content") or "")[:500].strip()
        if not content:
            return ""
        return f"[DMN reflection] {content}"
    except Exception as exc:
        logger.debug("dmn reflect: %s", exc)
        return ""


def _safe_lessons(user_message: str, strategy: str) -> List[str]:
    """Pull experience lessons relevant to the current turn."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        exp = (cfg.get("experience") or {})
        if not exp.get("enabled", True):
            return []
        limit = int(exp.get("lesson_limit", 3))
        min_conf = float(exp.get("lesson_min_confidence", 0.6))
        from agent.experience.recall import format_lessons_for_prompt, get_relevant_lessons

        lessons = get_relevant_lessons(
            context=user_message,
            min_confidence=min_conf,
            limit=limit,
        )
        if not lessons:
            return []
        block = format_lessons_for_prompt(lessons, include_context=strategy != "react")
        return [block] if block else []
    except Exception as exc:
        logger.debug("herens lessons: %s", exc)
        return []


def _safe_ecn_focus(user_message: str, *, session_id: str = "") -> str:
    """Pull ECN focus reminder if brain_networks is on (session-persistent)."""
    try:
        from agent.brain_networks.runtime import get_orchestrator

        orch = get_orchestrator()
        if orch is None:
            return ""
        if session_id:
            orch.bind_session(session_id)
        focus = orch.ecn.evaluate_focus(
            {"user_message": user_message, "session_id": session_id}
        )
        if not focus:
            return ""
        reminder = str(focus.get("reminder") or focus.get("focus") or "")[:300].strip()
        return f"[ECN focus] {reminder}" if reminder else ""
    except Exception as exc:
        logger.debug("ecn focus: %s", exc)
        return ""


def build_turn_context_block(
    user_message: str,
    *,
    session_id: str = "",
    strategy: str = "react",
) -> str:
    """Build per-turn Herens context block (volatile — not system prompt).

    Returned string is safe to inject as a tool-result or volatile context
    addendum WITHOUT touching the stable/context prompt tiers.
    """
    from agent.herens.config import is_herens_enabled, load_herens_config

    if not is_herens_enabled():
        return ""

    cfg = load_herens_config()
    parts: List[str] = []

    # 0. Last-turn reflection (corrective nudge / quality signal).
    # This is the F1.1 wire: post-turn reflection actually influences the
    # next conversation turn as a volatile <herens-reflection> block.
    # We surface it BEFORE other context so the corrective nudge is the
    # first thing the model sees when planning this turn.
    last_reflection = None
    try:
        from agent.herens.reflect import get_last_reflection

        last_reflection = get_last_reflection(session_id)
        if last_reflection is not None:
            block = last_reflection.as_prompt_block()
            if block:
                parts.append(block)
    except Exception as exc:
        logger.debug("herens last-reflection surfacing: %s", exc)

    # 0b. Error recovery (F2.2): if the last turn failed, classify the error
    # and surface a recovery strategy (retry/pivot/delegate/clarify/abort).
    # This is the load-bearing wire that makes error_recovery actually
    # influence the next turn — without it, the classifier is just logging.
    # We only fire when reflection says the turn was NOT clean.
    if last_reflection is not None and (
        last_reflection.failure_signals
        or last_reflection.quality_score < 0.5
        or last_reflection.should_retry
    ):
        try:
            from agent.herens.error_recovery import (
                classify_error,
                get_attempt_count,
                log_recovery,
                record_attempt,
            )

            # Build a synthetic error_text from the reflection's signals +
            # critique — this is what the classifier pattern-matches on.
            err_text = " | ".join(
                list(last_reflection.failure_signals)
                + [last_reflection.critique or ""]
            )
            # P1-5 fix: use the real windowed attempt count from the tracker
            # instead of hardcoding attempt_count=1 (which made the
            # escalation logic in classify_error dead code).
            attempt_count = get_attempt_count(session_id)
            record_attempt(session_id)
            strategy_rec = classify_error(
                err_text,
                attempt_count=attempt_count + 1,
                context=last_reflection.critique or "",
            )
            note = strategy_rec.as_context_note()
            if note:
                parts.append(note)
            # Persist for analytics (best-effort, never blocks the turn).
            try:
                log_recovery(
                    strategy_rec,
                    session_id=session_id,
                    original_error=err_text[:500],
                )
            except Exception as exc:
                logger.debug("herens error-recovery log persist: %s", exc)
        except Exception as exc:
            logger.debug("herens error-recovery surfacing: %s", exc)

    # 0c. CoT prefix (F2.3): inject a Think → Plan → Act → Verify scaffold
    # for non-trivial turns. Skipped for simple queries in react mode to
    # avoid wasting model attention on "what is X?" questions. The block
    # is purely volatile context — never touches the frozen system prompt.
    try:
        from agent.herens.cot import build_cot_block

        cot_block = build_cot_block(user_message, strategy=strategy)
        if cot_block:
            parts.append(cot_block)
    except Exception as exc:
        logger.debug("herens cot surfacing: %s", exc)

    # 0d. Uncertainty detection (F3.2): if the user's message is ambiguous,
    # surface a low-confidence block that nudges the model to ask for
    # clarification rather than guessing. The block is purely volatile.
    try:
        from agent.herens.uncertainty import assess_uncertainty

        uncertainty = assess_uncertainty(user_message)
        if uncertainty.low_confidence:
            block = uncertainty.as_context_block()
            if block:
                parts.append(block)
    except Exception as exc:
        logger.debug("herens uncertainty surfacing: %s", exc)

    # 0e. Semantic transfer recall (F3.3): surface transferable knowledge from
    # other domains. E.g. if the user asks about XSS and we have rich SQLi
    # experience, surface the analogous patterns. Skipped for simple queries
    # to avoid noise. The block is purely volatile.
    try:
        from agent.herens.transfer_recall import find_transferable_knowledge, format_transfer_block

        transfer_items = find_transferable_knowledge(user_message, min_score=0.3, limit=3)
        if transfer_items:
            block = format_transfer_block(transfer_items, max_items=3)
            if block:
                parts.append(block)
    except Exception as exc:
        logger.debug("herens transfer_recall surfacing: %s", exc)

    # 1. Experience lessons
    lesson_parts = _safe_lessons(user_message, strategy)
    parts.extend(lesson_parts)

    # 2. DMN reflection (only for reflect/caduceus strategies, or spontaneous)
    if strategy in ("reflect", "caduceus") or not lesson_parts:
        dmn = _safe_dmn_reflect(user_message, session_id=session_id)
        if dmn:
            parts.append(dmn)

    # 3. ECN focus (only when brain_networks on)
    # P1-4 fix: previously this was `if (cfg.get("brain_networks") or {})`
    # which is truthy for any dict (even an empty one), so ECN focus ran
    # even when brain_networks was disabled. The correct gate is the
    # `enabled` sub-key, matching the DMN check above.
    #
    # NOTE: brain_networks is a top-level config.yaml key, NOT a herens:
    # sub-key, so we read it from load_config() (not load_herens_config()).
    try:
        from hermes_cli.config import load_config as _load_top_config

        bn_cfg = (_load_top_config().get("brain_networks") or {})
    except Exception:
        bn_cfg = {}
    if bn_cfg.get("enabled"):
        ecn = _safe_ecn_focus(user_message, session_id=session_id)
        if ecn:
            parts.append(ecn)

    # 4. Plan-Execute: remind of pending plan
    if strategy == "plan_execute":
        try:
            from agent.herens.plan_execute import get_session_plan

            plan = get_session_plan(session_id)
            phase = plan.get("phase") if isinstance(plan, dict) else None
            if phase and phase not in ("done",):
                items_total = len(plan.get("items") or [])
                parts.append(
                    f"[Herens plan] Phase: {phase} | {items_total} item(s) — "
                    "update todo before heavy work, advance phase when complete."
                )
        except Exception as exc:
            logger.debug("herens plan_execute surfacing: %s", exc)

    # 5. ToT evaluation (F3.1): for complex/ambiguous turns, run a Tree-of-
    # Thoughts evaluation and surface the selected reasoning path. Opt-in
    # via herens.tot.enabled (default: off — it's expensive). Skipped for
    # simple queries to avoid wasting model attention.
    try:
        tot_cfg = (cfg.get("tot") or {})
        if tot_cfg.get("enabled", False):
            from agent.herens.tot import evaluate_tot

            # Only fire for non-trivial turns (heuristic: long messages or
            # strategy is plan_execute/reflect/caduceus).
            is_complex = (
                len(user_message.split()) > 15
                or strategy in ("plan_execute", "reflect", "caduceus")
            )
            if is_complex:
                tot_result = evaluate_tot(
                    user_message,
                    session_id=session_id,
                    n_branches=int(tot_cfg.get("max_branches", 4)),
                )
                block = tot_result.as_prompt_block()
                if block:
                    parts.append(block)
    except Exception as exc:
        logger.debug("herens tot surfacing: %s", exc)

    # 6. InterAgent inbox (F4.1): surface unacknowledged messages from other
    # agents. Skipped for simple queries to avoid noise. Pure volatile.
    try:
        ia_cfg = (cfg.get("inter_agent") or {})
        if ia_cfg.get("enabled", True):
            from agent.herens.inter_agent import receive_messages, format_inbox_block

            # Determine this agent's ID (use session_id as a fallback).
            agent_id = session_id or "_default"
            msgs = receive_messages(
                agent_id, unacknowledged_only=True, limit=3,
            )
            if msgs:
                block = format_inbox_block(msgs, max_messages=3)
                if block:
                    parts.append(block)
    except Exception as exc:
        logger.debug("herens inter_agent surfacing: %s", exc)

    # 7. Resource optimizer (F4.2): for non-trivial turns, surface a model
    # recommendation. The agent can use this to switch models for subtasks.
    # Pure volatile — never changes the actual model the LLM uses.
    try:
        ro_cfg = (cfg.get("resource_optimizer") or {})
        if ro_cfg.get("enabled", True):
            from agent.herens.resource_optimizer import recommend_for_task

            # Only fire for non-trivial turns.
            if len(user_message.split()) > 10:
                rec = recommend_for_task(user_message)
                if rec and rec.recommended_model:
                    block = rec.as_context_block()
                    if block:
                        parts.append(block)
    except Exception as exc:
        logger.debug("herens resource_optimizer surfacing: %s", exc)

    # 8. Peer routing (F4.3): for delegation-eligible turns, surface a peer
    # recommendation if there are live peers in the session. Pure volatile.
    try:
        pr_cfg = (cfg.get("peer_router") or {})
        if pr_cfg.get("enabled", True):
            from agent.herens.peer_router import route_task, list_peers

            # Only fire if there are live peers AND the message hints at delegation.
            peers = list_peers(session_id)
            delegation_hints = ("delegate", "handoff", "route", "specialist", "swarm")
            if peers and any(h in user_message.lower() for h in delegation_hints):
                decision = route_task(user_message, session_id=session_id)
                if decision.recommended_peer:
                    block = decision.as_context_block()
                    if block:
                        parts.append(block)
    except Exception as exc:
        logger.debug("herens peer_router surfacing: %s", exc)

    # 9. Vector store recall (A1): for non-trivial turns, surface semantically
    # similar lessons/skills/experiences from the vector store. This is the
    # real-vector counterpart to the TF-based transfer_recall above — when
    # the vector store has indexed content, this block surfaces the nearest
    # neighbors across all namespaces. Pure volatile.
    try:
        vs_cfg = (cfg.get("vector_store") or {})
        if vs_cfg.get("enabled", True) and len(user_message.split()) > 4:
            from agent.herens.vector_store import get_vector_store

            store = get_vector_store()
            if store.count(None) > 0:
                # Search across all namespaces for relevant items.
                lines: List[str] = []
                for ns in ("lessons", "skills", "experiences", "default"):
                    if store.count(ns) == 0:
                        continue
                    results = store.search(
                        user_message,
                        namespace=ns,
                        limit=2,
                        min_score=float(vs_cfg.get("min_score", 0.15)),
                    )
                    for r in results:
                        title = (r.record.metadata or {}).get("name") or r.record.source
                        snippet = r.record.text[:120].replace("\n", " ")
                        lines.append(
                            f"  - [{ns}/{title}] score={r.score:.2f}: {snippet}"
                        )
                if lines:
                    parts.append(
                        "[Herens vector recall] semantically similar items:\n"
                        + "\n".join(lines[:6])
                    )
    except Exception as exc:
        logger.debug("herens vector_store surfacing: %s", exc)

    if not parts:
        return ""
    return "<herens-turn-context>\n" + "\n\n".join(parts) + "\n</herens-turn-context>"
