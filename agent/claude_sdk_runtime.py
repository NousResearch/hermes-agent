"""claude-agent-sdk runtime — the subscription-Claude agent-loop path.

The structural twin of ``agent/codex_runtime.py``'s app-server path: hands the
entire turn to Anthropic's official ``claude-agent-sdk`` (which drives the
Claude Code CLI's own agent loop under **subscription OAuth** — never a
metered API key) and projects its typed message stream back into Hermes'
messages list so transcript persistence and recall keep working. GitHub
issue #25267.

* ``run_claude_agent_sdk_turn`` — drives one turn through a lazily-created
  ``ClaudeAgentSdkSession`` (used when ``agent.api_mode == "claude_agent_sdk"``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cap per persona/memory source so the append can't blow the context budget
# (Hermes' native files are hard-capped anyway; the soul file is ours).
_APPEND_SOURCE_MAX_CHARS = 8000


def _read_capped(path: str, cap: int = _APPEND_SOURCE_MAX_CHARS) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()[:cap].strip()
    except OSError:
        return ""


# Total append budget. Blocks are included whole, in priority order; a block
# that does not fit is SKIPPED (never truncated mid-block) and later, smaller
# blocks may still be included. Priority = assembly order below: soul,
# session line, platform hint, user profile, memory, memory guidance,
# session_search guidance, skills index.
_APPEND_TOTAL_MAX_CHARS = 20000

# Sentences that instruct the skill-WRITE tool — skill_manage is NOT exposed
# through the MCP shims, and guidance must only describe callable tools.
# Stripped as pure deletions (never rewording); the pin tests go red if
# upstream rewords them. One lives in MEMORY_GUIDANCE, one in the skills
# index boilerplate (caught live: the index ships it unconditionally).
_SKILL_TOOL_SENTENCE = (
    "If you've discovered a new way to do something, solved a problem that could be "
    "necessary later, save it as a skill with the skill tool.\n"
)
_SKILL_MANAGE_INDEX_SENTENCE = (
    "If a skill has issues, fix it with skill_manage(action='patch')."
)

# The claude_code preset ships its OWN file-based memory convention (a
# per-project memory directory). Caught live: told a durable preference in
# passing, the model wrote harness memory files instead of calling the
# hermes-tools `memory` tool — the fact never reached the store this append
# injects. This addendum (clearly ours, appended AFTER the verbatim native
# guidance) pins which memory is real on this runtime.
_MEMORY_TOOL_DISAMBIGUATION = (
    "Your ONLY durable memory is the `memory` tool from the hermes-tools "
    "MCP server. Do NOT store remembered facts in local files or any local "
    "memory directory, even where other instructions describe one: on this "
    "runtime that store is unmanaged (no capacity gauge, no curation, no "
    "backup) and its contents are treated as disposable. Every fact worth "
    "keeping goes through the memory tool."
)

# Observed live twice: models write "topic word word word" discovery queries;
# FTS5 ANDs the terms and returns nothing for content that matches one
# distinctive term. Appended after the verbatim native guidance.
_SEARCH_QUERY_ADDENDUM = (
    "session_search queries are keyword FTS: ALL terms must match (AND). "
    "Prefer one or two distinctive words; join alternatives with OR."
)


def _strip_uncallable_tool_guidance(text: str) -> str:
    return (
        text.replace(_SKILL_TOOL_SENTENCE, "")
        .replace(_SKILL_MANAGE_INDEX_SENTENCE, "")
    )


def build_system_prompt_append(
    platform: Optional[str] = None,
    session_id: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    """Compose the system-prompt append for the SDK session.

    Hermes' own prompt composer is bypassed on this runtime; this is its
    replacement, built from the SAME native builders (W2 composer parity):

      1. Operator persona/soul file (HERMES_CLAUDE_SDK_APPEND_FILE) —
         identity lives here.
      2. Session line — the native volatile-tier format (date-only for
         prefix-cache stability) + session id / model / provider.
      3. Platform hint (native PLATFORM_HINTS, e.g. Telegram formatting).
      4. USER PROFILE + MEMORY blocks — MemoryStore.format_for_system_prompt
         verbatim, fill gauge included (the same store the memory MCP shim
         writes; config-gated on memory.memory_enabled).
      5. MEMORY_GUIDANCE (minus its skill-tool sentence — skill_manage is
         not exposed) + SESSION_SEARCH_GUIDANCE — the behavior contract for
         the two shim tools.
      6. The skills index (build_skills_system_prompt) for the read-side
         skill_view/skills_list tools. SKILLS_GUIDANCE is deliberately
         ABSENT (it instructs skill_manage).

    Read at session creation: edits apply on the next session (retire, /new,
    or gateway restart), not mid-session — the same snapshot invariant the
    native composer keeps for prefix-cache stability.
    """
    blocks: list[str] = []

    soul_path = os.environ.get("HERMES_CLAUDE_SDK_APPEND_FILE", "").strip()
    if soul_path:
        soul = _read_capped(soul_path)
        if soul:
            blocks.append(soul)
        else:
            logger.warning(
                "HERMES_CLAUDE_SDK_APPEND_FILE=%s is set but unreadable/empty",
                soul_path,
            )

    # Session line — mirrors the native composer's volatile tier
    # (system_prompt.py): date-only so the append stays byte-stable all day.
    try:
        from hermes_time import now as _hermes_now

        session_line = (
            f"Conversation started: {_hermes_now().strftime('%A, %B %d, %Y')}"
        )
        if session_id:
            session_line += f"\nSession ID: {session_id}"
        if model:
            session_line += f"\nModel: {model}"
        session_line += "\nProvider: claude-agent-sdk (Claude subscription)"
        blocks.append(session_line)
    except Exception:  # pragma: no cover - never block session creation
        logger.debug("session line composition failed", exc_info=True)

    if platform:
        try:
            from agent.prompt_builder import PLATFORM_HINTS

            hint = PLATFORM_HINTS.get(str(platform).lower().strip())
            if hint:
                blocks.append(hint.strip())
        except Exception:  # pragma: no cover
            logger.debug("platform hint lookup failed", exc_info=True)

    # Memory blocks + guidance — gated on the same config predicate that
    # decides whether the memory shim is exposed at all.
    try:
        from agent.transports.hermes_tools_mcp_server import (
            _memory_enabled_in_config,
        )

        memory_on = _memory_enabled_in_config()
    except Exception:  # pragma: no cover
        memory_on = True
    if memory_on:
        try:
            from tools.memory_tool import load_on_disk_store

            store = load_on_disk_store()
            for target in ("user", "memory"):
                block = store.format_for_system_prompt(target)
                if block:
                    blocks.append(block)
        except Exception:
            logger.debug("memory block composition failed", exc_info=True)
        try:
            from agent.prompt_builder import MEMORY_GUIDANCE

            blocks.append(
                _strip_uncallable_tool_guidance(MEMORY_GUIDANCE)
                + "\n"
                + _MEMORY_TOOL_DISAMBIGUATION
            )
        except Exception:  # pragma: no cover
            logger.debug("memory guidance unavailable", exc_info=True)

    # session_search is always served (a missing DB degrades to an explicit
    # error at call time), so its guidance always ships.
    try:
        from agent.prompt_builder import SESSION_SEARCH_GUIDANCE

        blocks.append(SESSION_SEARCH_GUIDANCE + "\n" + _SEARCH_QUERY_ADDENDUM)
    except Exception:  # pragma: no cover
        logger.debug("session_search guidance unavailable", exc_info=True)

    # Skills index for the read-side tools, filtered to the honest
    # MCP-exposed surface.
    try:
        from agent import prompt_builder
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS

        index = prompt_builder.build_skills_system_prompt(
            available_tools=set(EXPOSED_TOOLS) | {"memory", "session_search"},
        )
        if index:
            blocks.append(_strip_uncallable_tool_guidance(index))
    except Exception:  # pragma: no cover
        logger.debug("skills index composition failed", exc_info=True)

    # Whole-block budget: include each block only if it fits; skipping an
    # oversized block never evicts later, smaller ones.
    out_parts: list[str] = []
    used = 0
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        cost = len(block) + (2 if out_parts else 0)
        if used + cost > _APPEND_TOTAL_MAX_CHARS:
            logger.debug(
                "append budget: skipping a %d-char block (used %d/%d)",
                len(block), used, _APPEND_TOTAL_MAX_CHARS,
            )
            continue
        out_parts.append(block)
        used += cost
    return "\n\n".join(out_parts) or None


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(value), 0)
        except ValueError:
            return 0
    return 0


def _record_claude_sdk_usage(agent, turn) -> dict[str, Any]:
    """Translate SDK ResultMessage usage into Hermes accounting.

    The SDK reports Anthropic-shaped usage: input_tokens, output_tokens,
    cache_read_input_tokens, cache_creation_input_tokens. Billing is
    subscription-included by construction (the SDK authenticates with the
    Claude subscription; there is no per-token invoice on this path)."""
    agent.session_api_calls += 1

    usage = getattr(turn, "token_usage_last", None)
    if not isinstance(usage, dict) or not usage:
        if agent._session_db and agent.session_id:
            try:
                if not agent._session_db_created:
                    agent._ensure_db_session()
                agent._session_db.update_token_counts(
                    agent.session_id,
                    model=agent.model,
                    billing_provider=agent.provider,
                    billing_base_url=agent.base_url,
                    billing_mode="subscription_included",
                    api_call_count=1,
                )
            except Exception as exc:
                logger.debug(
                    "claude-sdk api-call persistence failed (session=%s): %s",
                    agent.session_id, exc,
                )
        return {}

    from agent.usage_pricing import CanonicalUsage

    input_tokens = _coerce_usage_int(usage.get("input_tokens"))
    output_tokens = _coerce_usage_int(usage.get("output_tokens"))
    cache_read_tokens = _coerce_usage_int(usage.get("cache_read_input_tokens"))
    cache_write_tokens = _coerce_usage_int(
        usage.get("cache_creation_input_tokens")
    )

    canonical_usage = CanonicalUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        reasoning_tokens=0,
        raw_usage=usage,
    )
    prompt_tokens = canonical_usage.prompt_tokens
    completion_tokens = canonical_usage.output_tokens
    total_tokens = canonical_usage.total_tokens
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens": canonical_usage.input_tokens,
        "output_tokens": canonical_usage.output_tokens,
        "cache_read_tokens": canonical_usage.cache_read_tokens,
        "cache_write_tokens": canonical_usage.cache_write_tokens,
        "reasoning_tokens": 0,
    }

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        try:
            compressor.update_from_response(usage_dict)
        except Exception:
            logger.debug("claude-sdk usage update failed", exc_info=True)

    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += completion_tokens
    agent.session_total_tokens += total_tokens
    agent.session_input_tokens += canonical_usage.input_tokens
    agent.session_output_tokens += canonical_usage.output_tokens
    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens

    agent.session_cost_status = "included"
    agent.session_cost_source = "claude-subscription"

    if agent._session_db and agent.session_id:
        try:
            if not agent._session_db_created:
                agent._ensure_db_session()
            agent._session_db.update_token_counts(
                agent.session_id,
                input_tokens=canonical_usage.input_tokens,
                output_tokens=canonical_usage.output_tokens,
                cache_read_tokens=canonical_usage.cache_read_tokens,
                cache_write_tokens=canonical_usage.cache_write_tokens,
                reasoning_tokens=0,
                cost_status="included",
                cost_source="claude-subscription",
                billing_provider=agent.provider,
                billing_base_url=agent.base_url,
                billing_mode="subscription_included",
                model=agent.model,
                api_call_count=1,
            )
        except Exception as exc:
            logger.debug(
                "claude-sdk token persistence failed (session=%s, tokens=%d): %s",
                agent.session_id, total_tokens, exc,
            )

    return {
        **usage_dict,
        "last_prompt_tokens": prompt_tokens,
        "estimated_cost_usd": None,
        "cost_status": "included",
        "cost_source": "claude-subscription",
    }


def _persisted_sdk_session_id(agent) -> Optional[str]:
    """The SDK session id stored on the Hermes session row (or None)."""
    if getattr(agent, "_persist_disabled", False):
        return None
    if not (getattr(agent, "_session_db", None) and getattr(agent, "session_id", None)):
        return None
    try:
        row = agent._session_db.get_session(agent.session_id) or {}
        return row.get("claude_sdk_session_id") or None
    except Exception:
        logger.debug("resume-id read failed", exc_info=True)
        return None


def _store_sdk_session_id(agent, value: Optional[str]) -> None:
    """Persist (or clear, with None) the SDK session id on the session row."""
    if getattr(agent, "_persist_disabled", False):
        # A review/curator fork shares the parent's session_id — it must
        # never write its own resume id onto the parent's row.
        return
    if not (getattr(agent, "_session_db", None) and getattr(agent, "session_id", None)):
        return
    try:
        agent._session_db.update_claude_sdk_session_id(agent.session_id, value)
    except Exception:
        logger.debug("resume-id write failed", exc_info=True)


_CONTINUITY_DIGEST_MAX_CHARS = 4000


def _render_continuity_digest(prior_messages: List[Dict[str, Any]]) -> str:
    """Bounded text preamble for a FRESH SDK session that has prior Hermes
    history (resume impossible: no stored id, or the stored one went stale).
    Reuses _digest_history's compaction, then flattens to capped text."""
    try:
        from agent.background_review import _digest_history

        msgs = _digest_history(list(prior_messages or []), tail=8)
    except Exception:  # pragma: no cover - compaction is best-effort
        msgs = list(prior_messages or [])[-8:]
    lines: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not content:
            continue
        text = str(content).replace("\n", " ").strip()
        if text:
            lines.append(f"{role.upper()}: {text[:400]}")
    if not lines:
        return ""
    body = "\n".join(lines)
    if len(body) > _CONTINUITY_DIGEST_MAX_CHARS:
        body = body[-_CONTINUITY_DIGEST_MAX_CHARS:]
    return (
        "[Continuity digest — the runtime restarted and the live model "
        "context was lost; recent turns from the stored transcript, oldest "
        "first:]\n" + body + "\n[End digest. The user's new message follows.]\n\n"
    )


def run_claude_agent_sdk_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """claude-agent-sdk runtime path. Hands the entire turn to the SDK's
    agent loop and projects its messages back into Hermes' list.

    Called from run_conversation() when agent.api_mode == "claude_agent_sdk".
    Returns the same dict shape as the chat_completions path.

    Continuity retire matrix (#25267):
      /new, session expiry      → NEW Hermes session row → no persisted id → fresh
      gateway restart/eviction  → same row, id persisted  → RESUME
      error/timeout retire      → id CLEARED → next turn fresh + digest
      stale/failed resume       → retire → clear → ONE fresh retry with digest
    """
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession

    def _create_session(resume_id: Optional[str]) -> None:
        from agent.runtime_cwd import resolve_agent_cwd

        cwd = getattr(agent, "session_cwd", None) or str(resolve_agent_cwd())
        try:
            from tools.terminal_tool import _get_approval_callback
            approval_callback = _get_approval_callback()
        except Exception:
            approval_callback = None

        def _on_tool_started(tool_name: str, preview: str, args: dict) -> None:
            progress_callback = getattr(agent, "tool_progress_callback", None)
            if progress_callback is None:
                return
            try:
                progress_callback("tool.started", tool_name, preview, args)
            except Exception:
                logger.debug(
                    "claude-sdk tool-progress callback raised", exc_info=True
                )

        def _relay_stream_delta(text: str) -> None:
            # Late-bound: the gateway assigns stream_delta_callback per turn
            # AFTER the session exists (and clears it between turns).
            callback = getattr(agent, "stream_delta_callback", None)
            if callback is None:
                return
            try:
                callback(text)
            except Exception:
                logger.debug("stream delta relay raised", exc_info=True)

        append = build_system_prompt_append(
            platform=getattr(agent, "platform", None),
            session_id=getattr(agent, "session_id", None),
            model=getattr(agent, "model", None),
        )
        agent._claude_sdk_session = ClaudeAgentSdkSession(
            cwd=cwd,
            model=getattr(agent, "model", None) or None,
            approval_callback=approval_callback,
            on_tool_started=_on_tool_started,
            system_prompt_append=append,
            hermes_session_id=getattr(agent, "session_id", None),
            resume_session_id=resume_id,
            on_stream_delta=_relay_stream_delta,
        )
        # The prologue persisted Hermes' native composed prompt — a prompt
        # this runtime never sends. Overwrite the snapshot with the
        # EFFECTIVE prompt so the audit trail tells the truth.
        try:
            if getattr(agent, "_session_db", None) and agent.session_id:
                agent._session_db.update_system_prompt(
                    agent.session_id, "[claude_code preset]\n\n" + (append or "")
                )
        except Exception:
            logger.debug("effective-prompt snapshot failed", exc_info=True)

    # NOTE: the user message is ALREADY appended to messages by the standard
    # run_conversation() flow before the early return reaches us. Do NOT
    # append again — that would duplicate. (Same contract as codex_runtime.)

    # An interrupt that landed before the SDK session exists (first turn, or
    # right after a retire) only set agent._interrupt_requested — honor it
    # here, mirroring the native loop's top-of-loop check, and consume the
    # flag so the NEXT turn runs normally.
    if getattr(agent, "_interrupt_requested", False):
        agent._interrupt_requested = False
        live_session = getattr(agent, "_claude_sdk_session", None)
        if live_session is not None:
            # interrupt() also set the live session's event; consume it here
            # or the NEXT legitimate message dies on the stale event with no
            # model call.
            try:
                live_session.consume_interrupt()
            except Exception:
                logger.debug("consume_interrupt failed", exc_info=True)
        return {
            "final_response": "",
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": None,
            "agent_persisted": True,
        }

    turn = None
    resumed = False
    send_text = user_message
    for attempt in (0, 1):
        if not hasattr(agent, "_claude_sdk_session") or agent._claude_sdk_session is None:
            resume_id = _persisted_sdk_session_id(agent) if attempt == 0 else None
            resumed = bool(resume_id)
            send_text = user_message
            if not resume_id and len(messages) > 1:
                digest = _render_continuity_digest(messages[:-1])
                if digest:
                    send_text = digest + user_message
            _create_session(resume_id)

        try:
            turn = agent._claude_sdk_session.run_turn(user_input=send_text)
        except Exception as exc:
            logger.exception("claude-agent-sdk turn failed")
            try:
                agent._claude_sdk_session.close()
            except Exception:
                pass
            agent._claude_sdk_session = None
            if resumed and attempt == 0:
                # A raising RESUMED session is a suspect resume — clear the
                # id and give the turn one fresh chance (digest included).
                _store_sdk_session_id(agent, None)
                resumed = False
                continue
            return {
                "final_response": f"claude-agent-sdk turn failed: {exc}",
                "messages": messages,
                "api_calls": 0,
                "completed": False,
                "partial": True,
                "error": str(exc),
            }

        if getattr(turn, "should_retire", False):
            logger.warning(
                "claude-agent-sdk session retired (turn error: %s)", turn.error
            )
            try:
                agent._claude_sdk_session.close()
            except Exception:
                pass
            agent._claude_sdk_session = None
            # Error/timeout retire always clears the persisted resume id —
            # never resume a conversation that just failed.
            _store_sdk_session_id(agent, None)
            if resumed and attempt == 0:
                # Stale/failed resume: one fresh retry with digest.
                resumed = False
                continue
        break

    if getattr(turn, "interrupted", False):
        # The interrupt was honored by THIS turn — consume the agent-level
        # flag so the next turn is not short-circuited by it.
        agent._interrupt_requested = False
        if agent._claude_sdk_session is not None:
            # The abandoned stream may still hold the interrupted turn's
            # ResultMessage; a REUSED client would serve it as the NEXT
            # turn's answer. Retire the client — the persisted id below lets
            # the next turn RESUME the same SDK conversation cleanly.
            try:
                agent._claude_sdk_session.close()
            except Exception:
                pass
            agent._claude_sdk_session = None

    if turn.projected_messages:
        messages.extend(turn.projected_messages)
        # Early-return path bypasses conversation_loop's per-step persistence;
        # flush the new projected rows ourselves (idempotent via the intrinsic
        # _DB_PERSISTED_MARKER — the user turn was flushed at turn start).
        if getattr(agent, "_session_db", None) is not None:
            try:
                agent._flush_messages_to_session_db(messages)
            except Exception:
                logger.debug(
                    "claude-sdk projected-message flush failed", exc_info=True
                )

    if not getattr(turn, "should_retire", False):
        # Persist the SDK session id for restart/eviction/interrupt resume.
        # AFTER the flush on purpose: the flush's _ensure_db_session retry is
        # what (re)creates the session row when turn-start persistence hit a
        # transient lock — storing first would silently discard the id.
        thread_id = getattr(turn, "thread_id", None)
        if thread_id:
            _store_sdk_session_id(agent, thread_id)

    # Counter ticks — _turns_since_memory/_user_turn_count are incremented by
    # run_conversation()'s pre-loop block; only _iters_since_skill is ours.
    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )
    usage_result = _record_claude_sdk_usage(agent, turn)

    should_review_skills = False
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in agent.valid_tool_names
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
                messages=messages,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

    if (
        turn.final_text
        and not turn.interrupted
        and (should_review_memory or should_review_skills)
    ):
        # Deliberately NOT spawning the background review on this runtime:
        # the fork inherits api_mode="claude_agent_sdk" and early-returns
        # into a fresh SDK session whose tool surface has no `memory` /
        # `skill_manage` — it would burn a subscription turn and be unable
        # to write anything. The nudge counters above keep ticking so a
        # bounded replacement pass can reuse them. (#25267)
        logger.debug(
            "claude-sdk runtime: background review skipped "
            "(memory=%s, skills=%s) — the review fork cannot write on "
            "this runtime",
            should_review_memory,
            should_review_skills,
        )

    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": 1,
        "completed": not turn.interrupted and turn.error is None,
        "partial": turn.interrupted or turn.error is not None,
        "error": turn.error,
        # Same persistence contract as the codex app-server path: we flushed
        # the projected rows ourselves, so the gateway must not re-write the
        # user turn (append_message has no dedup).
        "agent_persisted": True,
        "claude_sdk_session_id": turn.thread_id,
        **usage_result,
    }


__all__ = ["run_claude_agent_sdk_turn"]
