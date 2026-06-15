"""Session management — 層級：Core Logic Layer (Layer 2)

Extracts session-related logic from run_agent.py:
- SessionDB lifecycle (creation, recall, message flushing)
- Session log JSON snapshot writing
- Memory/context engine lifecycle (shutdown, commit)
- Session state reset

Forwarder pattern: actual logic lives here; run_agent.py delegates.
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.redact import redact_sensitive_text
from agent.trajectory import convert_scratchpad_to_think

logger = logging.getLogger(__name__)


def _launch_cwd_for_session(source: str) -> Optional[str]:
    """Working directory to stamp on a new session row, or None.

    Only local CLI sessions get a recorded cwd: the directory the process was
    launched from is meaningful for ``hermes -c`` / ``--resume`` (relaunch
    where you left off). Gateway/cron/remote-backend sessions have no stable
    host cwd to restore, so they record nothing.

    ``TERMINAL_ENV`` is set by the CLI's config bridge (``load_cli_config``);
    a non-"local" backend (docker/ssh/modal/...) means the host cwd is
    irrelevant to the agent's tools, so we skip it there too.
    """
    if source != "cli":
        return None
    backend = (os.environ.get("TERMINAL_ENV") or "local").strip().lower()
    if backend and backend != "local":
        return None
    try:
        return os.getcwd()
    except OSError:
        # cwd was unlinked out from under us — nothing meaningful to record.
        return None


def _is_multimodal_tool_result(content) -> bool:
    """Check if content is a multimodal tool result (base64 image)."""
    if not isinstance(content, dict):
        return False
    if content.get("type") != "image_url":
        return False
    return bool(content.get("image_url", {}).get("url", "").startswith("data:image"))


def _multimodal_text_summary(content: dict) -> str:
    """Extract text summary from a multimodal tool result."""
    url = content.get("image_url", {}).get("url", "")
    if url.startswith("data:image"):
        return "[screenshot]"
    return "[image]"


def _clean_session_content(content: str) -> str:
    """Convert REASONING_SCRATCHPAD to think tags and clean up whitespace."""
    if not content:
        return content
    content = convert_scratchpad_to_think(content)
    content = re.sub(r'\n+(<think>)', r'\n\1', content)
    content = re.sub(r'(</think>)\n+', r'\1\n', content)
    return content.strip()


def _redact_message_content(content):
    """Apply secret redaction to message content (str or list-of-parts).

    Handles both plain-string content and the OpenAI/Anthropic multimodal
    shape where ``content`` is a list of ``{"type": "text", "text": ...}``
    / ``{"type": "image_url", ...}`` / ``{"type": "input_text", "content": ...}``
    parts. Image / binary parts are left untouched; only text fields are
    passed through ``redact_sensitive_text``.

    Respects ``HERMES_REDACT_SECRETS`` via ``redact_sensitive_text`` —
    when disabled the helper is effectively a no-op.
    """
    if content is None:
        return content
    if isinstance(content, str):
        return redact_sensitive_text(content)
    if isinstance(content, list):
        redacted = []
        for part in content:
            if isinstance(part, dict):
                part = dict(part)
                if isinstance(part.get("text"), str):
                    part["text"] = redact_sensitive_text(part["text"])
                if isinstance(part.get("content"), str):
                    part["content"] = redact_sensitive_text(part["content"])
            redacted.append(part)
        return redacted
    return content


def _atomic_json_write(path: Path, data: dict, indent: int = 2, default=None) -> None:
    """Write JSON atomically using a temp file + rename."""
    from utils import atomic_json_write
    atomic_json_write(path, data, indent=indent)


class SessionManager:
    """Handles session lifecycle: DB, logging, memory/context engine coordination."""

    def __init__(self, agent):
        """Store weak ref to agent to avoid circular reference."""
        import weakref
        self._agent_ref = weakref.ref(agent)

    @property
    def _agent(self):
        """Dereference the weak ref."""
        return self._agent_ref()

    # -------------------------------------------------------------------------
    # SessionDB lifecycle
    # -------------------------------------------------------------------------

    def get_session_db_for_recall(self):
        """Return a SessionDB for recall, lazily creating it if an entrypoint forgot.

        Most frontends pass ``session_db`` into ``AIAgent`` explicitly, but recall
        is important enough that a missing constructor argument should degrade by
        opening the default state DB instead of making the advertised
        ``session_search`` tool unusable.
        """
        agent = self._agent
        if agent._session_db is not None:
            return agent._session_db
        try:
            from hermes_state import SessionDB
            agent._session_db = SessionDB()
            return agent._session_db
        except Exception as exc:
            logger.debug("SessionDB unavailable for recall", exc_info=True)
            return None

    def ensure_db_session(self) -> None:
        """Create session DB row on first use. Disables _session_db on failure."""
        agent = self._agent
        if agent._session_db_created or not agent._session_db:
            return
        source = agent.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli")
        try:
            agent._session_db.create_session(
                session_id=agent.session_id,
                source=source,
                model=agent.model,
                model_config=agent._session_init_model_config,
                system_prompt=agent._cached_system_prompt,
                user_id=None,
                parent_session_id=agent._parent_session_id,
                cwd=_launch_cwd_for_session(source),
            )
            agent._session_db_created = True
        except Exception as e:
            # Transient failure (e.g. SQLite lock). Keep _session_db alive —
            # _session_db_created stays False so next run_conversation() retries.
            logger.warning(
                "Session DB creation failed (will retry next turn): %s", e
            )

    def flush_messages_to_session_db(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """Persist any un-flushed messages to the SQLite session store.

        Uses _last_flushed_db_idx to track which messages have already been
        written, so repeated calls (from multiple exit paths) only write
        truly new messages — preventing the duplicate-write bug (#860).
        """
        agent = self._agent
        if not agent._session_db:
            return
        self._apply_persist_user_message_override(messages)
        try:
            # Retry row creation if the earlier attempt failed transiently.
            if not agent._session_db_created:
                self.ensure_db_session()
            start_idx = len(conversation_history) if conversation_history else 0
            flush_from = max(start_idx, agent._last_flushed_db_idx)
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
                        elif isinstance(p, dict) and p.get("type") in {"image", "image_url", "input_image"}:
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
                agent._session_db.append_message(
                    session_id=agent.session_id,
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
            agent._last_flushed_db_idx = len(messages)
        except Exception as e:
            logger.warning("Session DB append_message failed: %s", e)

    def _apply_persist_user_message_override(self, messages: List[Dict]) -> None:
        """Apply user-message content override if configured via HERMES_PERSIST_USER_MSG."""
        agent = self._agent
        idx = getattr(agent, "_persist_user_message_idx", None)
        override = getattr(agent, "_persist_user_message_override", None)
        if override is None or idx is None:
            return
        if 0 <= idx < len(messages):
            msg = messages[idx]
            if isinstance(msg, dict) and msg.get("role") == "user":
                msg["content"] = override

    # -------------------------------------------------------------------------
    # Session log JSON
    # -------------------------------------------------------------------------

    def save_session_log(self, messages: List[Dict[str, Any]] = None):
        """Optional per-session JSON snapshot writer.

        Gated by ``sessions.write_json_snapshots`` (default False).  state.db
        is the canonical message store; this writer exists only for users
        whose external tooling consumes ``~/.hermes/sessions/session_{sid}.json``
        directly.  When the flag is off this is a fast no-op.

        When enabled, rewrites the snapshot after every persistence point with
        the full message list (assistant content normalized via
        ``_clean_session_content`` to convert REASONING_SCRATCHPAD to think
        tags).  The truncation guard ("don't overwrite a larger log with
        fewer messages") is preserved so resume + branch don't clobber a
        fuller existing snapshot.
        """
        agent = self._agent
        if not getattr(agent, "_session_json_enabled", False):
            return
        messages = messages or agent._session_messages
        if not messages:
            return

        # Re-derive the target path each call so /branch and /compress
        # session-id changes land in the right file without any re-point
        # bookkeeping at the call sites.
        try:
            log_file = agent.logs_dir / f"session_{agent.session_id}.json"
        except Exception:
            return

        try:
            cleaned = []
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    msg = dict(msg)
                    msg["content"] = _clean_session_content(msg["content"])
                # Defence-in-depth: redact credentials from every message
                # content before persistence. Catches PATs / API keys / Bearer
                # tokens that may have leaked into assistant responses, tool
                # output, or user paste. Respects HERMES_REDACT_SECRETS via
                # redact_sensitive_text — no-op when disabled. (#19798, #19845)
                if "content" in msg:
                    msg = dict(msg)
                    msg["content"] = _redact_message_content(msg.get("content"))
                cleaned.append(msg)

            # Guard: never overwrite a larger session log with fewer messages.
            # Protects against data loss when a resumed agent starts with
            # partial history and would otherwise clobber the full JSON log.
            if log_file.exists():
                try:
                    existing = json.loads(log_file.read_text(encoding="utf-8"))
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
                "session_id": agent.session_id,
                "model": agent.model,
                "base_url": agent.base_url,
                "platform": agent.platform,
                "session_start": agent.session_start.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "system_prompt": redact_sensitive_text(agent._cached_system_prompt or ""),
                "tools": agent.tools or [],
                "message_count": len(cleaned),
                "messages": cleaned,
            }

            _atomic_json_write(
                log_file,
                entry,
                indent=2,
                default=str,
            )

        except Exception as e:
            if agent.verbose_logging:
                logging.warning(f"Failed to save session log: {e}")

    # -------------------------------------------------------------------------
    # Session state reset
    # -------------------------------------------------------------------------

    def reset_session_state(
        self,
        previous_messages: Optional[list] = None,
        old_session_id: Optional[str] = None,
        carry_over_context: bool = False,
    ):
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

        When ``previous_messages`` / ``old_session_id`` / ``carry_over_context``
        are provided, the active context engine is notified through the
        full transition lifecycle (``_transition_context_engine_session``)
        instead of a bare reset. Default callers pass nothing and keep the
        existing reset-only behavior.
        """
        agent = self._agent

        # Token usage counters
        agent.session_total_tokens = 0
        agent.session_input_tokens = 0
        agent.session_output_tokens = 0
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_cache_read_tokens = 0
        agent.session_cache_write_tokens = 0
        agent.session_reasoning_tokens = 0
        agent.session_api_calls = 0
        agent.session_estimated_cost_usd = 0.0
        agent.session_cost_status = "unknown"
        agent.session_cost_source = "none"

        # Turn counter (added after reset_session_state was first written — #2635)
        agent._user_turn_count = 0

        # Context engine reset/transition (works for built-in compressor and plugins)
        self._transition_context_engine_session(
            old_session_id=old_session_id,
            new_session_id=getattr(agent, "session_id", None),
            previous_messages=previous_messages,
            carry_over_context=carry_over_context,
            reset_engine=True,
        )

    def _transition_context_engine_session(
        self,
        *,
        old_session_id: Optional[str] = None,
        new_session_id: Optional[str] = None,
        previous_messages: Optional[list] = None,
        carry_over_context: bool = False,
        reset_engine: bool = True,
        **extra_context,
    ) -> None:
        """Notify the active context engine about a host session transition.

        Forwarder — actual logic lives in ``ContextEngine.transition_session``.
        """
        agent = self._agent
        engine = getattr(agent, "context_compressor", None)
        if not engine:
            return
        engine.transition_session(
            old_session_id=old_session_id,
            new_session_id=new_session_id or getattr(agent, "session_id", "") or "",
            previous_messages=previous_messages,
            carry_over_context=carry_over_context,
            reset_engine=reset_engine,
            platform=getattr(agent, "platform", None),
            model=getattr(agent, "model", ""),
            conversation_id=getattr(agent, "_gateway_session_key", None),
            **extra_context,
        )

    # -------------------------------------------------------------------------
    # Memory/context engine lifecycle
    # -------------------------------------------------------------------------

    def shutdown_memory_provider(self, messages: list = None) -> None:
        """Shut down the memory provider and context engine — call at actual session boundaries.

        This calls on_session_end() then shutdown_all() on the memory
        manager, and on_session_end() on the context engine.
        NOT called per-turn — only at CLI exit, /reset, gateway
        session expiry, etc.
        """
        agent = self._agent
        if agent._memory_manager:
            try:
                agent._memory_manager.on_session_end(messages or [])
            except Exception:
                pass
            try:
                agent._memory_manager.shutdown_all()
            except Exception:
                pass
        # Notify context engine of session end (flush DAG, close DBs, etc.)
        if hasattr(agent, "context_compressor") and agent.context_compressor:
            try:
                agent.context_compressor.on_session_end(
                    agent.session_id or "",
                    messages or [],
                )
            except Exception:
                pass

    def commit_memory_session(self, messages: list = None) -> None:
        """Trigger end-of-session extraction without tearing providers down.

        Called when session_id rotates (e.g. /new, context compression);
        providers keep their state and continue running under the old
        session_id — they just flush pending extraction now."""
        agent = self._agent
        if agent._memory_manager:
            try:
                agent._memory_manager.on_session_end(messages or [])
            except Exception:
                pass
        # Notify context engine of session end too — same lifecycle moment as
        # the memory manager's on_session_end. Without this, engines that
        # accumulate per-session state (DAGs, summaries) leak that state from
        # the rotated-out session into whatever comes next under the same
        # compressor instance. Mirrors the call in shutdown_memory_provider().
        # See issue #22394.
        if hasattr(agent, "context_compressor") and agent.context_compressor:
            try:
                agent.context_compressor.on_session_end(
                    agent.session_id or "",
                    messages or [],
                )
            except Exception:
                pass

    def clear_session_messages(self) -> None:
        """Free conversation history at hard session boundaries.

        Mirrors _release_evicted_agent_soft's soft-eviction clear — close()
        is the hard teardown for true session boundaries (/new, /reset,
        session expiry), so the message list won't be reused. Drops the
        reference proactively rather than waiting for the agent object itself
        to be collected, which matters when a caller still holds the closed
        agent (e.g. a draining background task).
        """
        agent = self._agent
        try:
            agent._session_messages = []
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Persist session (combined save to DB + JSON log)
    # -------------------------------------------------------------------------

    def persist_session(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """Save session state to both JSON log and SQLite on any exit path.

        Ensures conversations are never lost, even on errors or early returns.
        """
        agent = self._agent
        self._drop_trailing_empty_response_scaffolding(messages)
        self._apply_persist_user_message_override(messages)
        agent._session_messages = messages
        self.save_session_log(messages)
        self.flush_messages_to_session_db(messages, conversation_history)

    def _drop_trailing_empty_response_scaffolding(self, messages: List[Dict]) -> None:
        """Remove private empty-response retry/failure scaffolding from transcript tails.

        Also rewinds past any trailing tool-result / assistant(tool_calls) pair
        that the failed iteration left hanging. Without this, the tail ends at
        a raw ``tool`` message and the next user turn lands as
        ``...tool, user, user`` — a protocol-invalid sequence that most
        providers silently reject (returns empty content), causing the
        empty-retry loop to fire forever. See #<TBD>.
        """
        # Pass 1: strip the flagged scaffolding messages themselves.
        dropped_scaffolding = False
        while (
            messages
            and isinstance(messages[-1], dict)
            and (
                messages[-1].get("_empty_recovery_synthetic")
                or messages[-1].get("_empty_terminal_sentinel")
            )
        ):
            messages.pop()
            dropped_scaffolding = True

        # Pass 2: if we stripped scaffolding, rewind through any trailing
        # tool-result messages plus the assistant(tool_calls) message that
        # produced them. This preserves role alternation so the next user
        # message follows a user or assistant message, not an orphan tool
        # result. Only runs when scaffolding was actually present — normal
        # conversation tails (real tool loops mid-progress) are untouched.
        if not dropped_scaffolding:
            return

        # Drop any trailing tool-result messages
        while (
            messages
            and isinstance(messages[-1], dict)
            and messages[-1].get("role") == "tool"
        ):
            messages.pop()

        # Drop the assistant message that issued the tool calls, if the tail
        # now ends in an assistant-with-tool_calls (the pair that owned the
        # just-popped tool results). Without this, the tail is
        # ``assistant(tool_calls=...)`` with no tool answers, which some
        # providers also reject.
        if (
            messages
            and isinstance(messages[-1], dict)
            and messages[-1].get("role") == "assistant"
            and messages[-1].get("tool_calls")
        ):
            messages.pop()