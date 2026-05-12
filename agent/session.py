"""SessionMixin -- session and state management methods extracted from AIAgent.

Encapsulates session database access, resource cleanup, session persistence,
memory commits, and dead connection cleanup.  Previously inline in AIAgent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import json
import logging
import os
import threading
import time

# Re-export utilities for mixin method access
from agent.utils import *  # noqa: F401,F403

logger = logging.getLogger(__name__)


class SessionMixin:
    """Mixin providing session management methods for AIAgent.

    All methods use `self` to access agent state and are designed to be
    mixed into the AIAgent class.
    """

    def _get_session_db_for_recall(self):
        """Return a SessionDB for recall, lazily creating it if an entrypoint forgot.

        Most frontends pass ``session_db`` into ``AIAgent`` explicitly, but recall
        is important enough that a missing constructor argument should degrade by
        opening the default state DB instead of making the advertised
        ``session_search`` tool unusable.
        """
        if self._session_db is not None:
            return self._session_db
        try:
            from hermes_state import SessionDB

            self._session_db = SessionDB()
            return self._session_db
        except Exception as exc:
            logger.debug("SessionDB unavailable for recall", exc_info=True)
            return None


    def _ensure_db_session(self) -> None:
        """Create session DB row on first use. Disables _session_db on failure."""
        if self._session_db_created or not self._session_db:
            return
        try:
            self._session_db.create_session(
                session_id=self.session_id,
                source=self.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                model=self.model,
                model_config=self._session_init_model_config,
                system_prompt=self._cached_system_prompt,
                user_id=None,
                parent_session_id=self._parent_session_id,
            )
            self._session_db_created = True
        except Exception as e:
            # Transient failure (e.g. SQLite lock). Keep _session_db alive —
            # _session_db_created stays False so next run_conversation() retries.
            logger.warning(
                "Session DB creation failed (will retry next turn): %s", e
            )


    def reset_session_state(self):
        """Reset all session-scoped token counters to 0 for a fresh session.
        
        This method encapsulates the reset logic for all session-level metrics
        including:
        - Token usage counters (input, output, total, prompt, completion)
        - Cache read/write tokens
        - API call count
        - Reasoning tokens
        - Estimated cost tracking
        - Context compressor internal counters
        
        The method safely handles optional attributes (e.g., context compressor)
        using ``hasattr`` checks.
        
        This keeps the counter reset logic DRY and maintainable in one place
        rather than scattering it across multiple methods.
        """
        # Token usage counters
        self.session_total_tokens = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_api_calls = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        
        # Turn counter (added after reset_session_state was first written — #2635)
        self._user_turn_count = 0

        # Context engine reset (works for both built-in compressor and plugins)
        if hasattr(self, "context_compressor") and self.context_compressor:
            self.context_compressor.on_session_reset()


    def _ensure_lmstudio_runtime_loaded(self, config_context_length: Optional[int] = None) -> None:
        """
        Preload the LM Studio model with at least Hermes' minimum context.
        """
        if (self.provider or "").strip().lower() != "lmstudio":
            return
        try:
            from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
            from hermes_cli.models import ensure_lmstudio_model_loaded
            if config_context_length is None:
                config_context_length = getattr(self, "_config_context_length", None)
            target_ctx = max(config_context_length or 0, MINIMUM_CONTEXT_LENGTH)
            loaded_ctx = ensure_lmstudio_model_loaded(
                self.model, self.base_url, getattr(self, "api_key", ""), target_ctx,
            )
            if loaded_ctx:
                # Push into the live compressor so the status bar reflects the
                # real loaded ctx the moment the load resolves, instead of
                # holding the previous model's value (or "ctx --") through the
                # next render tick.
                cc = getattr(self, "context_compressor", None)
                if cc is not None:
                    cc.update_model(
                        model=self.model,
                        context_length=loaded_ctx,
                        base_url=self.base_url,
                        api_key=getattr(self, "api_key", ""),
                        provider=self.provider,
                        api_mode=self.api_mode,
                    )
        except Exception as err:
            logger.debug("LM Studio preload skipped: %s", err)


    def _cleanup_task_resources(self, task_id: str) -> None:
        """Clean up VM and browser resources for a given task.

        Skips ``cleanup_vm`` when the active terminal environment is marked
        persistent (``persistent_filesystem=True``) so that long-lived sandbox
        containers survive between turns. The idle reaper in
        ``terminal_tool._cleanup_inactive_envs`` still tears them down once
        ``terminal.lifetime_seconds`` is exceeded. Non-persistent backends are
        torn down per-turn as before to prevent resource leakage (the original
        intent of this hook for the Morph backend, see commit fbd3a2fd).
        """
        try:
            if is_persistent_env(task_id):
                if self.verbose_logging:
                    logging.debug(
                        f"Skipping per-turn cleanup_vm for persistent env {task_id}; "
                        f"idle reaper will handle it."
                    )
            else:
                cleanup_vm(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup VM for task {task_id}: {e}")
        try:
            cleanup_browser(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup browser for task {task_id}: {e}")

    # ------------------------------------------------------------------
    # Background memory/skill review
    # ------------------------------------------------------------------

    _MEMORY_REVIEW_PROMPT = (
        "Review the conversation above and consider saving to memory if appropriate.\n\n"
        "Focus on:\n"
        "1. Has the user revealed things about themselves — their persona, desires, "
        "preferences, or personal details worth remembering?\n"
        "2. Has the user expressed expectations about how you should behave, their work "
        "style, or ways they want you to operate?\n\n"
        "If something stands out, save it using the memory tool. "
        "If nothing is worth saving, just say 'Nothing to save.' and stop."
    )

    _SKILL_REVIEW_PROMPT = (
        "Review the conversation above and update the skill library. Be "
        "ACTIVE — most sessions produce at least one skill update, even if "
        "small. A pass that does nothing is a missed learning opportunity, "
        "not a neutral outcome.\n\n"
        "Target shape of the library: CLASS-LEVEL skills, each with a rich "
        "SKILL.md and a `references/` directory for session-specific detail. "
        "Not a long flat list of narrow one-session-one-skill entries. This "
        "shapes HOW you update, not WHETHER you update.\n\n"
        "Signals to look for (any one of these warrants action):\n"
        "  • User corrected your style, tone, format, legibility, or "
        "verbosity. Frustration signals like 'stop doing X', 'this is too "
        "verbose', 'don't format like this', 'why are you explaining', "
        "'just give me the answer', 'you always do Y and I hate it', or an "
        "explicit 'remember this' are FIRST-CLASS skill signals, not just "
        "memory signals. Update the relevant skill(s) to embed the "
        "preference so the next session starts already knowing.\n"
        "  • User corrected your workflow, approach, or sequence of steps. "
        "Encode the correction as a pitfall or explicit step in the skill "
        "that governs that class of task.\n"
        "  • Non-trivial technique, fix, workaround, debugging path, or "
        "tool-usage pattern emerged that a future session would benefit "
        "from. Capture it.\n"
        "  • A skill that got loaded or consulted this session turned out "
        "to be wrong, missing a step, or outdated. Patch it NOW.\n\n"
        "Preference order — prefer the earliest action that fits, but do "
        "pick one when a signal above fired:\n"
        "  1. UPDATE A CURRENTLY-LOADED SKILL. Look back through the "
        "conversation for skills the user loaded via /skill-name or you "
        "read via skill_view. If any of them covers the territory of the "
        "new learning, PATCH that one first. It is the skill that was in "
        "play, so it's the right one to extend.\n"
        "  2. UPDATE AN EXISTING UMBRELLA (via skills_list + skill_view). "
        "If no loaded skill fits but an existing class-level skill does, "
        "patch it. Add a subsection, a pitfall, or broaden a trigger.\n"
        "  3. ADD A SUPPORT FILE under an existing umbrella. Skills can be "
        "packaged with three kinds of support files — use the right "
        "directory per kind:\n"
        "     • `references/<topic>.md` — session-specific detail (error "
        "transcripts, reproduction recipes, provider quirks) AND "
        "condensed knowledge banks: quoted research, API docs, external "
        "authoritative excerpts, or domain notes you found while working "
        "on the problem. Write it concise and for the value of the task, "
        "not as a full mirror of upstream docs.\n"
        "     • `templates/<name>.<ext>` — starter files meant to be "
        "copied and modified (boilerplate configs, scaffolding, a "
        "known-good example the agent can `reproduce with modifications`).\n"
        "     • `scripts/<name>.<ext>` — statically re-runnable actions "
        "the skill can invoke directly (verification scripts, fixture "
        "generators, deterministic probes, anything the agent should run "
        "rather than hand-type each time).\n"
        "     Add support files via skill_manage action=write_file with "
        "file_path starting 'references/', 'templates/', or 'scripts/'. "
        "The umbrella's SKILL.md should gain a one-line pointer to any "
        "new support file so future agents know it exists.\n"
        "  4. CREATE A NEW CLASS-LEVEL UMBRELLA SKILL when no existing "
        "skill covers the class. The name MUST be at the class level. "
        "The name MUST NOT be a specific PR number, error string, feature "
        "codename, library-alone name, or 'fix-X / debug-Y / audit-Z-today' "
        "session artifact. If the proposed name only makes sense for "
        "today's task, it's wrong — fall back to (1), (2), or (3).\n\n"
        "User-preference embedding (important): when the user expressed a "
        "style/format/workflow preference, the update belongs in the "
        "SKILL.md body, not just in memory. Memory captures 'who the user "
        "is and what the current situation and state of your operations "
        "are'; skills capture 'how to do this class of task for this "
        "user'. When they complain about how you handled a task, the "
        "skill that governs that task needs to carry the lesson.\n\n"
        "If you notice two existing skills that overlap, note it in your "
        "reply — the background curator handles consolidation at scale.\n\n"
        "Do NOT capture (these become persistent self-imposed constraints "
        "that bite you later when the environment changes):\n"
        "  • Environment-dependent failures: missing binaries, fresh-install "
        "errors, post-migration path mismatches, 'command not found', "
        "unconfigured credentials, uninstalled packages. The user can fix "
        "these — they are not durable rules.\n"
        "  • Negative claims about tools or features ('browser tools do not "
        "work', 'X tool is broken', 'cannot use Y from execute_code'). These "
        "harden into refusals the agent cites against itself for months "
        "after the actual problem was fixed.\n"
        "  • Session-specific transient errors that resolved before the "
        "conversation ended. If retrying worked, the lesson is the retry "
        "pattern, not the original failure.\n"
        "  • One-off task narratives. A user asking 'summarize today's "
        "market' or 'analyze this PR' is not a class of work that warrants "
        "a skill.\n\n"
        "If a tool failed because of setup state, capture the FIX (install "
        "command, config step, env var to set) under an existing setup or "
        "troubleshooting skill — never 'this tool does not work' as a "
        "standalone constraint.\n\n"
        "'Nothing to save.' is a real option but should NOT be the "
        "default. If the session ran smoothly with no corrections and "
        "produced no new technique, just say 'Nothing to save.' and stop. "
        "Otherwise, act."
    )

    _COMBINED_REVIEW_PROMPT = (
        "Review the conversation above and update two things:\n\n"
        "**Memory**: who the user is. Did the user reveal persona, "
        "desires, preferences, personal details, or expectations about "
        "how you should behave? Save facts about the user and durable "
        "preferences with the memory tool.\n\n"
        "**Skills**: how to do this class of task. Be ACTIVE — most "
        "sessions produce at least one skill update. A pass that does "
        "nothing is a missed learning opportunity, not a neutral outcome.\n\n"
        "Target shape of the skill library: CLASS-LEVEL skills with a rich "
        "SKILL.md and a `references/` directory for session-specific detail. "
        "Not a long flat list of narrow one-session-one-skill entries.\n\n"
        "Signals that warrant a skill update (any one is enough):\n"
        "  • User corrected your style, tone, format, legibility, "
        "verbosity, or approach. Frustration is a FIRST-CLASS skill "
        "signal, not just a memory signal. 'stop doing X', 'don't format "
        "like this', 'I hate when you Y' — embed the lesson in the skill "
        "that governs that task so the next session starts fixed.\n"
        "  • Non-trivial technique, fix, workaround, or debugging path "
        "emerged.\n"
        "  • A skill that was loaded or consulted turned out wrong, "
        "missing, or outdated — patch it now.\n\n"
        "Preference order for skills — pick the earliest that fits:\n"
        "  1. UPDATE A CURRENTLY-LOADED SKILL. Check what skills were "
        "loaded via /skill-name or skill_view in the conversation. If one "
        "of them covers the learning, PATCH it first. It was in play; "
        "it's the right place.\n"
        "  2. UPDATE AN EXISTING UMBRELLA (skills_list + skill_view to "
        "find the right one). Patch it.\n"
        "  3. ADD A SUPPORT FILE under an existing umbrella via "
        "skill_manage action=write_file. Three kinds: "
        "`references/<topic>.md` for session-specific detail OR condensed "
        "knowledge banks (quoted research, API docs excerpts, domain "
        "notes) written concise and task-focused; `templates/<name>.<ext>` "
        "for starter files meant to be copied and modified; "
        "`scripts/<name>.<ext>` for statically re-runnable actions "
        "(verification, fixture generators, probes). Add a one-line "
        "pointer in SKILL.md so future agents find them.\n"
        "  4. CREATE A NEW CLASS-LEVEL UMBRELLA when nothing exists. "
        "Name at the class level — NOT a PR number, error string, "
        "codename, library-alone name, or 'fix-X / debug-Y' session "
        "artifact. If the name only fits today's task, fall back to (1), "
        "(2), or (3).\n\n"
        "User-preference embedding: when the user complains about how "
        "you handled a task, update the skill that governs that task — "
        "memory alone isn't enough. Memory says 'who the user is and "
        "what the current situation and state of your operations are'; "
        "skills say 'how to do this class of task for this user'. Both "
        "should carry user-preference lessons when relevant.\n\n"
        "If you notice overlapping existing skills, mention it — the "
        "background curator handles consolidation.\n\n"
        "Do NOT capture as skills (these become persistent self-imposed "
        "constraints that bite you later when the environment changes):\n"
        "  • Environment-dependent failures: missing binaries, fresh-install "
        "errors, post-migration path mismatches, 'command not found', "
        "unconfigured credentials, uninstalled packages. The user can fix "
        "these — they are not durable rules.\n"
        "  • Negative claims about tools or features ('browser tools do not "
        "work', 'X tool is broken', 'cannot use Y from execute_code'). These "
        "harden into refusals the agent cites against itself for months "
        "after the actual problem was fixed.\n"
        "  • Session-specific transient errors that resolved before the "
        "conversation ended. If retrying worked, the lesson is the retry "
        "pattern, not the original failure.\n"
        "  • One-off task narratives. A user asking 'summarize today's "
        "market' or 'analyze this PR' is not a class of work that warrants "
        "a skill.\n\n"
        "If a tool failed because of setup state, capture the FIX (install "
        "command, config step, env var to set) under an existing setup or "
        "troubleshooting skill — never 'this tool does not work' as a "
        "standalone constraint.\n\n"
        "Act on whichever of the two dimensions has real signal. If "
        "genuinely nothing stands out on either, say 'Nothing to save.' "
        "and stop — but don't reach for that conclusion as a default."
    )

    def _apply_persist_user_message_override(self, messages: List[Dict]) -> None:
        """Rewrite the current-turn user message before persistence/return.

        Some call paths need an API-only user-message variant without letting
        that synthetic text leak into persisted transcripts or resumed session
        history. When an override is configured for the active turn, mutate the
        in-memory messages list in place so both persistence and returned
        history stay clean.
        """
        idx = getattr(self, "_persist_user_message_idx", None)
        override = getattr(self, "_persist_user_message_override", None)
        if override is None or idx is None:
            return
        if 0 <= idx < len(messages):
            msg = messages[idx]
            if isinstance(msg, dict) and msg.get("role") == "user":
                msg["content"] = override


    def _flush_messages_to_session_db(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """Persist any un-flushed messages to the SQLite session store.

        Uses _last_flushed_db_idx to track which messages have already been
        written, so repeated calls (from multiple exit paths) only write
        truly new messages — preventing the duplicate-write bug (#860).
        """
        if not self._session_db:
            return
        self._apply_persist_user_message_override(messages)
        try:
            # Retry row creation if the earlier attempt failed transiently.
            if not self._session_db_created:
                self._ensure_db_session()
            start_idx = len(conversation_history) if conversation_history else 0
            flush_from = max(start_idx, self._last_flushed_db_idx)
            for msg in messages[flush_from:]:
                role = msg.get("role", "unknown")
                content = msg.get("content")
                # Persist multimodal tool results as their text summary only —
                # base64 images would bloat the session DB and aren't useful
                # for cross-session replay.
                if _is_multimodal_tool_result(content):
                    content = _multimodal_text_summary(content)
                elif isinstance(content, list):
                    # List of OpenAI-style content parts: strip images, keep text.
                    _txt = []
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            _txt.append(str(p.get("text", "")))
                        elif isinstance(p, dict) and p.get("type") in ("image", "image_url", "input_image"):
                            _txt.append("[screenshot]")
                    content = "\n".join(_txt) if _txt else None
                tool_calls_data = None
                if hasattr(msg, "tool_calls") and isinstance(msg.tool_calls, list) and msg.tool_calls:
                    tool_calls_data = [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in msg.tool_calls
                    ]
                elif isinstance(msg.get("tool_calls"), list):
                    tool_calls_data = msg["tool_calls"]
                self._session_db.append_message(
                    session_id=self.session_id,
                    role=role,
                    content=content,
                    tool_name=msg.get("tool_name"),
                    tool_calls=tool_calls_data,
                    tool_call_id=msg.get("tool_call_id"),
                    finish_reason=msg.get("finish_reason"),
                    reasoning=msg.get("reasoning") if role == "assistant" else None,
                    reasoning_content=msg.get("reasoning_content") if role == "assistant" else None,
                    reasoning_details=msg.get("reasoning_details") if role == "assistant" else None,
                    codex_reasoning_items=msg.get("codex_reasoning_items") if role == "assistant" else None,
                    codex_message_items=msg.get("codex_message_items") if role == "assistant" else None,
                )
            self._last_flushed_db_idx = len(messages)
        except Exception as e:
            logger.warning("Session DB append_message failed: %s", e)


    def _save_session_log(self, messages: List[Dict[str, Any]] = None):
        """
        Save the full raw session to a JSON file.

        Stores every message exactly as the agent sees it: user messages,
        assistant messages (with reasoning, finish_reason, tool_calls),
        tool responses (with tool_call_id, tool_name), and injected system
        messages (compression summaries, todo snapshots, etc.).

        REASONING_SCRATCHPAD tags are converted to <think> blocks for consistency.
        Overwritten after each turn so it always reflects the latest state.
        """
        messages = messages or self._session_messages
        if not messages:
            return

        try:
            # Clean assistant content for session logs
            cleaned = []
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    msg = dict(msg)
                    msg["content"] = self._clean_session_content(msg["content"])
                cleaned.append(msg)

            # Guard: never overwrite a larger session log with fewer messages.
            # This protects against data loss when --resume loads a session whose
            # messages weren't fully written to SQLite — the resumed agent starts
            # with partial history and would otherwise clobber the full JSON log.
            if self.session_log_file.exists():
                try:
                    existing = json.loads(self.session_log_file.read_text(encoding="utf-8"))
                    existing_count = existing.get("message_count", len(existing.get("messages", [])))
                    if existing_count > len(cleaned):
                        logging.debug(
                            "Skipping session log overwrite: existing has %d messages, current has %d",
                            existing_count, len(cleaned),
                        )
                        return
                except Exception:
                    pass  # corrupted existing file — allow the overwrite

            entry = {
                "session_id": self.session_id,
                "model": self.model,
                "base_url": self.base_url,
                "platform": self.platform,
                "session_start": self.session_start.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "system_prompt": self._cached_system_prompt or "",
                "tools": self.tools or [],
                "message_count": len(cleaned),
                "messages": cleaned,
            }

            atomic_json_write(
                self.session_log_file,
                entry,
                indent=2,
                default=str,
            )

        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to save session log: {e}")


    def commit_memory_session(self, messages: list = None) -> None:
        """Trigger end-of-session extraction without tearing providers down.
        Called when session_id rotates (e.g. /new, context compression);
        providers keep their state and continue running under the old
        session_id — they just flush pending extraction now."""
        if self._memory_manager:
            try:
                self._memory_manager.on_session_end(messages or [])
            except Exception:
                pass
        # Notify context engine of session end too — same lifecycle moment as
        # the memory manager's on_session_end. Without this, engines that
        # accumulate per-session state (DAGs, summaries) leak that state from
        # the rotated-out session into whatever comes next under the same
        # compressor instance. Mirrors the call in shutdown_memory_provider().
        # See issue #22394.
        if hasattr(self, "context_compressor") and self.context_compressor:
            try:
                self.context_compressor.on_session_end(
                    self.session_id or "",
                    messages or [],
                )
            except Exception:
                pass


    def _cleanup_dead_connections(self) -> bool:
        """Detect and clean up dead TCP connections on the primary client.

        Inspects the httpx connection pool for sockets in unhealthy states
        (CLOSE-WAIT, errors).  If any are found, force-closes all sockets
        and rebuilds the primary client from scratch.

        Returns True if dead connections were found and cleaned up.
        """
        client = getattr(self, "client", None)
        if client is None:
            return False
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None:
                return False
            transport = getattr(http_client, "_transport", None)
            if transport is None:
                return False
            pool = getattr(transport, "_pool", None)
            if pool is None:
                return False
            connections = (
                getattr(pool, "_connections", None)
                or getattr(pool, "_pool", None)
                or []
            )
            dead_count = 0
            for conn in list(connections):
                # Check for connections that are idle but have closed sockets
                stream = (
                    getattr(conn, "_network_stream", None)
                    or getattr(conn, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None:
                    continue
                # Probe socket health with a non-blocking recv peek
                import socket as _socket
                try:
                    sock.setblocking(False)
                    data = sock.recv(1, _socket.MSG_PEEK | _socket.MSG_DONTWAIT)
                    if data == b"":
                        dead_count += 1
                except BlockingIOError:
                    pass  # No data available — socket is healthy
                except OSError:
                    dead_count += 1
                finally:
                    try:
                        sock.setblocking(True)
                    except OSError:
                        pass
            if dead_count > 0:
                logger.warning(
                    "Found %d dead connection(s) in client pool — rebuilding client",
                    dead_count,
                )
                self._replace_primary_openai_client(reason="dead_connection_cleanup")
                return True
        except Exception as exc:
            logger.debug("Dead connection check error: %s", exc)
        return False
