"""Headless slash commands for ACP sessions (``/help``, ``/model``, ``/compress`` ...)."""

from __future__ import annotations

import contextvars
import logging
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from acp.schema import AvailableCommand, AvailableCommandsUpdate, UnstructuredCommandInput

from acp_adapter.session import SessionState, _expand_acp_enabled_toolsets

logger = logging.getLogger("acp_adapter.server")


def _estimate_tokens(history: list, agent: Any, system_prompt: str | None = None, tools: Any = None) -> int:
    """Rough request-token estimate over history + system prompt + tool schemas."""
    from agent.model_metadata import estimate_request_tokens_rough

    if system_prompt is None:
        system_prompt = getattr(agent, "_cached_system_prompt", "") or ""
    if tools is None:
        tools = getattr(agent, "tools", None) or None
    return estimate_request_tokens_rough(history, system_prompt=system_prompt, tools=tools)


def _queue_prompt(state: SessionState, text: str) -> int:
    with state.runtime_lock:
        state.queued_prompts.append(text)
        return len(state.queued_prompts)


# Commands that mutate shared turn state must not run beside a live turn or beside each
# other: slash dispatch happens on a worker thread while the turn iterates state.history and
# reads agent._session_db, so clearing, rebinding, or swapping state.agent underneath
# run_conversation tears the running turn. Gateway parity: all three are idle-only there.
# The command_op flag is held for the whole handler so a turn cannot claim the session in
# the check-then-act window (the /compress LLM call and /model agent rebuild take seconds).
_MID_TURN_BLOCKED_COMMANDS = frozenset({"reset", "compress", "model", "resume"})

# `/skills` is a picker, not a catalogue dump: a full install runs to hundreds of entries.
_SKILL_LIST_LIMIT = 60

# `/resume` prints the transcript as markdown; cap it so a long CLI session stays readable.
_RESUME_TRANSCRIPT_LIMIT = 40
_RESUME_MESSAGE_CHARS = 600

# `/sessions` search: how many sessions to index, how many message hits to read, and how
# many result rows to print.
_SESSION_INDEX_LIMIT = 3000
_SESSION_SEARCH_HITS = 200
_SESSION_LIST_LIMIT = 20


class SlashCommandsMixin:
    """Slash-command surface for ``HermesACPAgent``; relies on ``_conn``, ``_send``, ``_schedule_soon``,
    ``session_manager`` and ``_switch_model`` from the host class."""

    # `/resume` state, keyed by ACP session id: the list a bare `/resume` last showed, so a
    # number means the same session on the next call.
    _resume_choices: dict[str, list[str]] = {}

    # name -> (help text, advertised description, input hint)
    _COMMANDS: dict[str, tuple[str, str, str | None]] = {
        "help": ("Show available commands", "List available commands", None),
        "model": (
            "Show or change current model",
            "Show current model and provider, or switch models",
            "model name to switch to",
        ),
        "tools": ("List available tools", "List available tools with descriptions", None),
        "context": ("Show conversation context info", "Show conversation message counts by role", None),
        "reset": ("Clear conversation history", "Clear conversation history", None),
        "compress": ("Compress conversation context", "Compress conversation context", None),
        "steer": (
            "Inject guidance into the currently running agent turn",
            "Inject guidance into the currently running agent turn",
            "guidance for the active turn",
        ),
        "queue": (
            "Queue a prompt to run after the current turn finishes",
            "Queue a prompt to run after the current turn finishes",
            "prompt to run next",
        ),
        "btw": (
            "Ask a side question about this conversation without touching it",
            "Ask a side question about this conversation",
            "question to answer beside the thread",
        ),
        "skills": (
            "List installed skills",
            "List installed skills usable with /skill",
            None,
        ),
        "sessions": (
            "Search every session's title and transcript",
            "Search all sessions by title and conversation, newest first",
            "text to search for",
        ),
        "resume": (
            "List Hermes sessions, or load one into this thread",
            "Resume a Hermes session from any surface (CLI, TUI, gateway, ACP)",
            "session number or id prefix",
        ),
        "skill": (
            "Run an installed skill",
            "Run an installed skill, optionally with an instruction",
            "skill name and optional instruction",
        ),
        "version": ("Show Hermes version", "Show Hermes version", None),
    }


    @classmethod
    def _available_commands(cls) -> list[AvailableCommand]:
        commands = [
            AvailableCommand(name=name, description=desc, input=UnstructuredCommandInput(hint=hint) if hint else None)
            for name, (_help, desc, hint) in cls._COMMANDS.items()
        ]
        commands.extend(cls._skill_available_commands())
        return commands

    @classmethod
    def _skill_available_commands(cls) -> list[AvailableCommand]:
        """Every installed skill as its own command, so `/1password` is offered directly.

        The client only lists what is advertised here, and typing a bare `/slug` already runs
        the skill; without this the menu hid all of them behind `/skill`. Core commands win a
        name collision (a skill called ``help`` never replaces ``/help``), which is the same
        rule the CLI applies when it registers skill commands.
        """
        try:
            from agent.skill_commands import get_skill_commands

            skills = get_skill_commands()
        except Exception:
            logger.debug("Could not enumerate skill commands for ACP advertisement", exc_info=True)
            return []
        advertised = []
        for key, info in sorted(skills.items()):
            name = key.lstrip("/").strip()
            if not name or name.lower() in cls._COMMANDS:
                continue
            description = str(info.get("description") or "").strip() or f"Run the {name} skill"
            advertised.append(AvailableCommand(name=name, description=description[:300]))
        return advertised

    async def _send_available_commands_update(self, session_id: str) -> None:
        """Advertise supported slash commands to the connected ACP client."""
        if not self._conn:
            return
        update = AvailableCommandsUpdate(
            session_update="available_commands_update", available_commands=self._available_commands()
        )
        await self._send(session_id, update, fail_msg="Failed to advertise ACP slash commands for session %s")

    def _schedule_available_commands_update(self, session_id: str) -> None:
        self._schedule_soon(lambda: self._send_available_commands_update(session_id))

    def _handle_slash_command(self, text: str, state: SessionState) -> str | None:
        """Dispatch a slash command; ``None`` for unknown ones so they fall through to the LLM."""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lstrip("/").lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if cmd not in self._COMMANDS:
            return None
        from hermes_cli.observability.shared_metrics_events import record_slash_command
        record_slash_command(command=cmd, surface="acp")
        mutating = cmd in _MID_TURN_BLOCKED_COMMANDS
        if mutating:
            with state.runtime_lock:
                if state.is_running or state.command_op:
                    return (f"⏳ Session is busy; /{cmd} only works while the session is "
                            "idle. Wait for the current response or cancel first.")
                state.command_op = True
        handler = getattr(self, f"_cmd_{cmd}")

        # Handlers run outside the per-turn cwd-pinning context. ``/compress``
        # and ``/model`` REBUILD the system prompt, so unpinned they'd bake the Hermes install tree
        # into the persisted cached prompt. Pin inside a fresh context: no leak, no teardown.
        def _dispatch() -> str | None:
            try:
                from agent.runtime_cwd import set_session_cwd

                set_session_cwd(state.cwd)
            except Exception:
                logger.debug("Could not pin ACP session cwd for slash command", exc_info=True)
            return handler(args, state)

        try:
            return contextvars.copy_context().run(_dispatch)
        except Exception as e:
            logger.error("Slash command /%s error: %s", cmd, e, exc_info=True)
            return f"Error executing /{cmd}: {e}"
        finally:
            if mutating:
                with state.runtime_lock:
                    state.command_op = False

    def _cmd_help(self, args: str, state: SessionState) -> str:
        lines = ["Available commands:", ""]
        lines.extend(f"  /{cmd:10s}  {desc}" for cmd, (desc, _adv, _hint) in self._COMMANDS.items())
        lines.extend(["", "Unrecognized /commands are sent to the model as normal messages."])
        return "\n".join(lines)

    def _cmd_model(self, args: str, state: SessionState) -> str:
        if not args:
            model = state.model or getattr(state.agent, "model", "unknown")
            provider = getattr(state.agent, "provider", None) or "auto"
            return f"Current model: {model}\nProvider: {provider}"

        current_provider, target_provider, new_model = self._switch_model(state, args)
        provider_label = getattr(state.agent, "provider", None) or target_provider or current_provider or "openrouter"
        logger.info("Session %s: model switched to %s", state.session_id, new_model)
        return f"Model switched to: {new_model}\nProvider: {provider_label}"

    def _cmd_tools(self, args: str, state: SessionState) -> str:
        try:
            from model_tools import get_tool_definitions
            from types import SimpleNamespace
            from agent.memory_manager_tools import inject_memory_provider_tools

            toolsets = _expand_acp_enabled_toolsets(getattr(state.agent, "enabled_toolsets", None))
            tools = get_tool_definitions(
                enabled_toolsets=toolsets,
                disabled_toolsets=getattr(state.agent, "disabled_toolsets", None),
                quiet_mode=True,
            )
            tool_view = SimpleNamespace(
                tools=list(tools or []),
                valid_tool_names={t.get("function", {}).get("name") for t in tools or [] if isinstance(t, dict)},
                enabled_toolsets=toolsets,
                disabled_toolsets=getattr(state.agent, "disabled_toolsets", None),
                _memory_manager=getattr(state.agent, "_memory_manager", None),
            )
            inject_memory_provider_tools(tool_view)
            tools = tool_view.tools
            if not tools:
                return "No tools available."
            lines = [f"Available tools ({len(tools)}):"]
            for t in tools:
                name = (t.get("function") or {}).get("name", "?")
                desc = (t.get("function") or {}).get("description", "")
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                lines.append(f"  {name}: {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"Could not list tools: {e}"

    def _cmd_context(self, args: str, state: SessionState) -> str:
        """Show ACP session context pressure and compression guidance."""
        n_messages = len(state.history)
        roles = Counter(msg.get("role", "unknown") for msg in state.history)

        agent = state.agent
        model = state.model or getattr(agent, "model", "")
        provider = getattr(agent, "provider", None) or "auto"
        compressor = getattr(agent, "context_compressor", None)
        context_length = int(getattr(compressor, "context_length", 0) or 0)
        threshold_tokens = int(getattr(compressor, "threshold_tokens", 0) or 0)

        try:
            approx_tokens = _estimate_tokens(state.history, agent)
        except Exception:
            logger.debug("Could not estimate ACP context usage", exc_info=True)
            approx_tokens = 0

        if threshold_tokens <= 0 and context_length > 0:
            threshold_tokens = int(context_length * 0.80)

        lines = [
            f"Conversation: {n_messages} messages" if n_messages else "Conversation is empty (no messages yet).",
            f"  user: {roles.get('user', 0)}, assistant: {roles.get('assistant', 0)}, "
            f"tool: {roles.get('tool', 0)}, system: {roles.get('system', 0)}",
        ]
        if model:
            lines.append(f"Model: {model}")
        lines.append(f"Provider: {provider}")

        if approx_tokens > 0 and context_length > 0:
            usage_pct = (approx_tokens / context_length) * 100
            lines.append(f"Context usage: ~{approx_tokens:,} / {context_length:,} tokens ({usage_pct:.1f}%)")
        elif approx_tokens > 0:
            lines.append(f"Context usage: ~{approx_tokens:,} tokens")

        if threshold_tokens > 0 and approx_tokens > 0:
            threshold_pct = (threshold_tokens / context_length) * 100 if context_length > 0 else 0
            pct_note = f", {threshold_pct:.0f}%" if threshold_pct else ""
            if approx_tokens >= threshold_tokens:
                lines.append(f"Compression: due now (threshold ~{threshold_tokens:,}{pct_note}). Run /compress.")
            else:
                remaining = max(threshold_tokens - approx_tokens, 0)
                lines.append(f"Compression: ~{remaining:,} tokens until threshold (~{threshold_tokens:,}{pct_note}).")
        elif threshold_tokens > 0:
            lines.append(f"Compression threshold: ~{threshold_tokens:,} tokens")

        lines.append(
            "Auto-compaction is disabled (compression.enabled: false); /compress still compresses manually."
            if getattr(agent, "compression_enabled", True) is False
            else "Tip: run /compress to compress manually before the threshold."
        )
        return "\n".join(lines)

    def _cmd_reset(self, args: str, state: SessionState) -> str:
        state.history.clear()
        try:
            reset_session_state = getattr(state.agent, "reset_session_state", None)
            if callable(reset_session_state):
                reset_session_state()
        except Exception:
            logger.warning("ACP session state reset failed for %s", state.session_id, exc_info=True)
            return "Conversation history cleared. Agent session state reset failed; see logs."
        finally:
            self.session_manager.save_session(state.session_id)
        return "Conversation history cleared."

    def _cmd_compress(self, args: str, state: SessionState) -> str:
        """``/compress [here [N] | <focus>] [--preview] [--aggressive]`` through the shared core."""
        from agent.conversation_compression import finalize_context_engine_compression_notification
        from agent.conversation_compression_manual import (
            AGGRESSIVE_UNSUPPORTED, compress_now, parse_compress_args, render_compress_result)

        if not state.history:
            return "Nothing to compress — conversation is empty."
        agent = state.agent
        # No compression_enabled gate: it only disables *automatic* compaction (CLI/gateway parity).
        if not hasattr(agent, "_compress_context"):
            return "Context compression not available for this agent."
        request = parse_compress_args(args)
        if request.aggressive:
            return AGGRESSIVE_UNSUPPORTED
        original_session_db = getattr(agent, "_session_db", None)
        try:
            # Stable ACP session id: suppress _compress_context's SQLite session split.
            agent._session_db = None
            result = compress_now(
                agent, state.history, request, system_message=getattr(agent, "_cached_system_prompt", "") or "",
                task_id=state.session_id)
        except Exception as e:
            return f"Compression failed: {e}"
        finally:
            agent._session_db = original_session_db
        if result.status != "compressed":
            return "\n".join(render_compress_result(result))
        state.history = result.after_messages
        self.session_manager.save_session(state.session_id)
        finalize_context_engine_compression_notification(agent, committed=True)
        return (
            f"Context compressed: {len(result.before_messages)} -> {len(state.history)} messages\n"
            f"~{result.before_tokens:,} -> ~{result.after_tokens:,} tokens"
        )

    def _cmd_steer(self, args: str, state: SessionState) -> str:
        steer_text = args.strip()
        if not steer_text:
            return "Usage: /steer <guidance>"

        if state.is_running and hasattr(state.agent, "steer"):
            try:
                if state.agent.steer(steer_text):
                    preview = steer_text[:80] + ("..." if len(steer_text) > 80 else "")
                    return f"⏩ Steer queued for the active turn: {preview}"
            except Exception as exc:
                logger.warning("ACP steer failed for session %s: %s", state.session_id, exc)
                return f"⚠️ Steer failed: {exc}"

        return f"No active turn — queued for the next turn. ({_queue_prompt(state, steer_text)} queued)"

    def _cmd_queue(self, args: str, state: SessionState) -> str:
        queued_text = args.strip()
        if not queued_text:
            return "Usage: /queue <prompt>"
        return f"Queued for the next turn. ({_queue_prompt(state, queued_text)} queued)"

    def _cmd_skills(self, args: str, state: SessionState) -> str:
        from agent.skill_commands import get_skill_commands

        commands = get_skill_commands()
        if not commands:
            return "No skills installed."
        names = sorted(command.lstrip("/") for command in commands)
        shown = names[:_SKILL_LIST_LIMIT]
        # One paragraph of code spans: the client renders markdown, which folds plain newlines
        # into a single line, so a one-per-line list would arrive as an unreadable block.
        listing = ", ".join(f"`{name}`" for name in shown)
        more = f" … and {len(names) - len(shown)} more" if len(names) > len(shown) else ""
        return f"**{len(names)} skills installed.** Run one with `/skill <name> [instruction]`.\n\n{listing}{more}"

    def _cmd_btw(self, args: str, state: SessionState) -> str:
        question = args.strip()
        if not question:
            return "Usage: /btw <question>"
        from agent.side_question import answer_side_question

        # Snapshot now: the live turn may still be appending to history on another thread.
        history_snapshot = list(getattr(state, "history", None) or [])
        agent = state.agent
        try:
            return answer_side_question(question, history_snapshot, parent_agent=agent)
        except Exception as exc:
            logger.warning("ACP /btw failed for session %s: %s", state.session_id, exc, exc_info=True)
            return f"⚠️ /btw failed: {exc}"

    def _take_idle_queue_prompt(self, text: str, state: SessionState) -> str | None:
        """Text to run as this turn when `/queue` arrives on an idle session, else ``None``.

        ``/queue`` means "after the current turn". With no turn running there is nothing to
        wait for, and replying with a queue note only to drain it into the same request
        produced one message of concatenated text. Hand the prompt back instead so it runs as
        the turn, and keep queueing for a busy session, where the note is the honest answer.
        """
        parts = text.split(maxsplit=1)
        if parts[0].lstrip("/").lower() != "queue":
            return None
        queued = parts[1].strip() if len(parts) > 1 else ""
        if not queued or state.is_running or state.command_op:
            return None
        return queued

    # ---- /sessions ---------------------------------------------------------

    def _session_index(self) -> dict[str, dict[str, Any]]:
        """Every session Hermes kept, keyed by id: the join target for search hits.

        One listing feeds titles, sources and last-activity, so a search does not do a
        per-session lookup for each hit.
        """
        db = self._resume_db()
        if db is None:
            return {}
        try:
            rows = db.list_sessions_rich(
                limit=_SESSION_INDEX_LIMIT, order_by_last_active=True, min_message_count=0,
                include_hidden=True, include_archived=True, include_subagents=False,
            )
        except Exception:
            logger.debug("Could not list sessions for /sessions", exc_info=True)
            return {}
        # Subagent children are parts of a parent's turn, not conversations to resume; their
        # titles read "Subagent: ..." and searching them buries the session that spawned them.
        return {
            str(row.get("id") or ""): row
            for row in rows
            if row.get("id")
            and str(row.get("source") or "") != "subagent"
            and not str(row.get("title") or "").startswith("Subagent:")
        }

    def _session_row(self, session_id: str, fallback_ts: float = 0.0) -> dict[str, Any] | None:
        """One session's display fields, for a hit the listing did not return.

        The listing is filtered (archived, empty, subagent sessions), but a transcript match
        still deserves a row, so fall back to a direct read and the matched message's time.
        """
        db = self._resume_db()
        if db is None:
            return None
        try:
            row = db.get_session(session_id)
        except Exception:
            logger.debug("Could not read session %s for /sessions", session_id, exc_info=True)
            return None
        if not row:
            return None
        if str(row.get("source") or "") == "subagent" or str(row.get("title") or "").startswith("Subagent:"):
            return None
        return {
            "id": session_id,
            "title": row.get("title") or row.get("preview") or "",
            "source": row.get("source") or "?",
            "last_active": row.get("last_active") or fallback_ts or row.get("started_at") or 0,
            "message_count": row.get("message_count") or 0,
        }

    def _search_session_messages(self, query: str) -> tuple[dict[str, list[str]], bool]:
        """Session id -> up to two matching snippets, best first.

        The phrase is searched first (quoted), because the index otherwise matches any of the
        words: "hermes agent setup" would return every session mentioning "hermes". Only when
        the phrase matches nothing does it fall back to the loose query, and the second return
        value says which one ran so the reply can be honest about it.
        """
        db = self._resume_db()
        if db is None:
            return {}, True
        attempts = [(f'"{query}"', True), (query, False)] if len(query.split()) > 1 else [(query, False)]
        for text, phrase in attempts:
            try:
                hits = db.search_messages(text, limit=_SESSION_SEARCH_HITS)
            except Exception:
                logger.debug("Message search failed for /sessions", exc_info=True)
                return {}, phrase
            found: dict[str, list[str]] = {}
            for hit in hits:
                session_id = str(hit.get("session_id") or "")
                snippet = str(hit.get("snippet") or "").strip()
                if not session_id or not snippet:
                    continue
                # An earlier search is not a result, and an auto-title call is housekeeping
                # traffic that quotes the thread's own text. Both would otherwise surface every
                # thread that ever ran a search or had its title regenerated.
                flat = " ".join(snippet.split())
                if "User message: /sessions" in flat or "User message: /resume" in flat:
                    continue
                if "needsRefinement" in flat:
                    continue
                bucket = found.setdefault(session_id, [])
                if len(bucket) < 2:
                    bucket.append(snippet)
            if found:
                return found, phrase
        return {}, True

    @staticmethod
    def _clean_snippet(snippet: str, limit: int = 110, *, window: bool = False) -> str:
        """Table-safe snippet: markdown bold around the match, no pipes, escapes or newlines.

        ``window`` trims to the match itself, which is what a search result should show: a
        raw snippet can open with a page of file paths before the words that matched.
        """
        text = snippet
        for escaped, plain in (("\\r\\n", " "), ("\\n", " "), ("\\t", " "), ("\\\"", '"'), ("\\\\", "\\")):
            text = text.replace(escaped, plain)
        # Neutralise the source's own asterisks before marking the match: a literal ``*`` next
        # to the >>>/<<< markers pairs with them into stray bold. A numeric entity renders as
        # the character itself, which a backslash escape does not in a table cell.
        text = text.replace("*", "&#42;")
        # Double remaining backslashes: a Windows path ending in "\" directly before the match
        # markers would otherwise read as an escaped asterisk and break the bold pair.
        text = text.replace("\\", "\\\\")
        text = " ".join(text.replace(">>>", "**").replace("<<<", "**").split())
        text = text.replace("|", "/")
        if window:
            start = text.find("**")
            if start > 30:
                lead = text.rfind(" ", 0, max(0, start - 30))
                text = "…" + text[lead if lead > 0 else start - 30:].lstrip()
        if len(text) > limit:
            text = text[:limit].rstrip() + "…"
        return text

    @staticmethod
    def _unique_id_prefixes(ids: list[str]) -> dict[str, str]:
        """id -> shortest prefix that no other listed session shares, at least 8 characters.

        A fixed 8-character cut collides: several sessions started on the same day share
        ``20260811``, so the id printed in the table could not be pasted into `/resume`.
        """
        widths: dict[str, int] = {}
        for session_id in ids:
            width = 8
            while width < len(session_id) and any(
                other != session_id and other.startswith(session_id[:width]) for other in ids
            ):
                width += 1
            widths[session_id] = session_id[:width]
        return widths

    @staticmethod
    def _format_last_active(value: Any) -> str:
        try:
            stamp = float(value)
        except (TypeError, ValueError):
            return "?"
        if stamp <= 0:
            return "?"
        moment = datetime.fromtimestamp(stamp, tz=UTC).astimezone()
        age = time.time() - stamp
        if age < 3600:
            return f"{max(1, int(age // 60))}m ago"
        if age < 86400:
            return f"{int(age // 3600)}h ago"
        if age < 7 * 86400:
            return f"{int(age // 86400)}d ago"
        return moment.strftime("%Y-%m-%d")

    def _cmd_sessions(self, args: str, state: SessionState) -> str:
        """Search every session's title and transcript, newest activity first.

        ``/sessions <text>`` matches titles and message bodies; with no argument it lists the
        most recently active sessions so the command is useful on its own.
        """
        query = args.strip()
        # The composer re-inserts the command name, so a bare submit arrives as "/sessions /sessions".
        if query.startswith("/"):
            query = ""
        index = self._session_index()
        if not index:
            return "No Hermes sessions found."

        matched: dict[str, str] = {}
        phrase = False
        if query:
            lowered = query.lower()
            for session_id, row in index.items():
                title = str(row.get("title") or "")
                if lowered in title.lower():
                    matched[session_id] = "**title**: " + self._clean_snippet(title, 90)
            hits, phrase = self._search_session_messages(query)
            # The search runs from inside a session, so that session always matches its own
            # prompt text; it is the caller, not a result.
            hits.pop(state.session_id, None)
            for session_id, snippets in hits.items():
                existing = index.get(session_id) or {}
                if existing.get("source") == "subagent" or str(existing.get("title") or "").startswith("Subagent:"):
                    continue
                if session_id not in index:
                    row = self._session_row(session_id)
                    if row is None:
                        continue
                    index[session_id] = row
                cell = " · ".join(self._clean_snippet(snippet, window=True) for snippet in snippets)
                matched[session_id] = f"{matched[session_id]}<br>{cell}" if session_id in matched else cell
            if not matched:
                return f"No session matches **{query}** in its title or transcript."
            rows = [index[session_id] for session_id in matched]
        else:
            rows = list(index.values())[:_SESSION_LIST_LIMIT]

        def _activity(row: dict[str, Any]) -> float:
            try:
                return float(row.get("last_active") or 0)
            except (TypeError, ValueError):
                return 0.0

        rows.sort(key=_activity, reverse=True)
        rows = rows[:_SESSION_LIST_LIMIT]

        if not query:
            heading = "**Recent Hermes sessions** — newest activity first"
        elif phrase:
            heading = f"**{len(matched)} session(s) containing \"{query}\"** — newest activity first"
        else:
            heading = (
                f"**{len(matched)} session(s) mentioning any of \"{query}\"** — newest activity first"
            )
        prefixes = self._unique_id_prefixes([str(row.get("id") or "") for row in rows])
        lines = [
            heading,
            "",
            "| # | Session | Match | From | Last active | Id |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for position, row in enumerate(rows, start=1):
            session_id = str(row.get("id") or "")
            title = self._clean_snippet(str(row.get("title") or "(untitled)"), 60)
            match = matched.get(session_id, "—")
            source = str(row.get("source") or "?")
            lines.append(
                f"| {position} | {title} | {match} | {source} | "
                f"{self._format_last_active(row.get('last_active'))} | `{prefixes.get(session_id, session_id[:8])}` |"
            )
        self._resume_choices[state.session_id] = [str(row.get("id") or "") for row in rows]
        lines += ["", "Load one with `/resume <#>` or `/resume <id>`."]
        return "\n".join(lines)

    # ---- /resume -----------------------------------------------------------

    def _hermes_session_rows(self, limit: int = 15) -> list[dict[str, Any]]:
        """Recent Hermes sessions from every surface, newest first.

        T3 threads are their own store; these are the sessions Hermes itself kept, so a
        conversation started in the CLI or the TUI is reachable from here too.
        """
        db = self._resume_db()
        if db is None:
            return []
        try:
            rows = db.list_sessions_rich(
                limit=limit, order_by_last_active=True, min_message_count=1, include_hidden=False,
            )
        except Exception:
            logger.debug("Could not list Hermes sessions for /resume", exc_info=True)
            return []
        return [row for row in rows if str(row.get("source") or "") != "subagent"]

    def _resume_db(self) -> Any:
        try:
            from hermes_state_registry import acquire
            from hermes_constants import get_hermes_home

            return acquire(get_hermes_home() / "state.db")
        except Exception:
            logger.debug("Hermes session DB unavailable for /resume", exc_info=True)
            return None

    def _format_resume_list(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No Hermes sessions found."
        # A markdown table, not one line per row: the client renders markdown, which folds
        # single newlines into one paragraph.
        lines = [
            "**Resume a Hermes session** — `/resume <number>` or `/resume <id>`",
            "",
            "| # | Session | Msgs | From | Id |",
            "| --- | --- | --- | --- | --- |",
        ]
        prefixes = self._unique_id_prefixes([str(row.get("id") or "") for row in rows])
        for index, row in enumerate(rows, start=1):
            session_id = str(row.get("id") or "")
            title = str(row.get("title") or row.get("preview") or "(untitled)").strip()
            title = title.replace("|", "/").replace("\n", " ")[:70]
            count = int(row.get("message_count") or 0)
            source = str(row.get("source") or "?")
            short_id = prefixes.get(session_id, session_id[:8])
            lines.append(f"| {index} | {title} | {count} | {source} | `{short_id}` |")
        return "\n".join(lines)

    def _cmd_resume(self, args: str, state: SessionState) -> str | None:
        """List sessions, or load one into this ACP session.

        Loading replaces this thread's history with the chosen transcript and replays it to
        the client, so the conversation continues in T3 with its full context intact.
        """
        rows = self._hermes_session_rows()
        target = args.strip()
        # The T3 composer completes a command by re-inserting its name, so a submitted
        # "/resume" can arrive as "/resume /resume". Treat that as no argument.
        if target.startswith("/"):
            target = ""
        if not target:
            self._resume_choices[state.session_id] = [str(row.get("id") or "") for row in rows]
            return self._format_resume_list(rows)

        session_id = self._resolve_session_target(target, state, rows)
        if session_id is None:
            ambiguous = self._ambiguous_session_ids(target, state, rows)
            if ambiguous:
                listing = ", ".join(f"`{candidate}`" for candidate in ambiguous[:5])
                return f"'{target}' matches {len(ambiguous)} sessions: {listing}. Use a longer id."
            return (
                f"No session matching '{target}'. `/resume` numbers the recent ones; "
                "`/sessions <text>` finds the rest, and its ids work here."
            )
        return self._load_resume_history(session_id, state)

    def _resolve_session_target(
        self, target: str, state: SessionState, rows: list[dict[str, Any]]
    ) -> str | None:
        """Turn `/resume`'s argument into a session id.

        A number means a row of the last list this session printed — `/sessions` seeds that
        list too, so the number after a search is the row the user just read. An id or id
        prefix is matched against that list first, then against every session Hermes has,
        because a search can surface a session that is older than the recent listing.
        """
        choices = self._resume_choices.get(state.session_id) or [str(row.get("id") or "") for row in rows]
        if target.isdigit():
            index = int(target) - 1
            return choices[index] if 0 <= index < len(choices) else None

        for pool in (choices, list(self._session_index())):
            matches = [candidate for candidate in pool if candidate.startswith(target)]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                return None  # ambiguous: make the user pick, never guess a transcript
        return self._session_id_by_prefix(target)

    def _ambiguous_session_ids(self, prefix: str, state: SessionState, rows: list[dict[str, Any]]) -> list[str]:
        """Every session id starting with *prefix*, for an honest "which one?" reply."""
        choices = self._resume_choices.get(state.session_id) or [str(row.get("id") or "") for row in rows]
        pool = {*choices, *self._session_index()}
        return sorted(candidate for candidate in pool if candidate.startswith(prefix))

    def _session_id_by_prefix(self, prefix: str) -> str | None:
        """Direct lookup for a session the listings filtered out (archived, hidden)."""
        db = self._resume_db()
        if db is None:
            return None
        try:
            # Not `id_query`: that filter matches elsewhere and misses plain id prefixes, so
            # the listing is scanned here instead.
            rows = db.list_sessions_rich(
                limit=_SESSION_INDEX_LIMIT, min_message_count=0, include_hidden=True,
                include_archived=True, include_subagents=False,
            )
        except Exception:
            logger.debug("Could not look up session prefix %r", prefix, exc_info=True)
            return None
        matches = [str(row.get("id") or "") for row in rows if str(row.get("id") or "").startswith(prefix)]
        return matches[0] if len(matches) == 1 else None

    def _load_resume_history(self, session_id: str, state: SessionState) -> str:
        db = self._resume_db()
        if db is None:
            return "Hermes session DB unavailable."
        try:
            history = db.get_messages_as_conversation(session_id, repair_alternation=True)
        except Exception as exc:
            logger.warning("Could not load Hermes session %s for /resume: %s", session_id, exc)
            return f"⚠️ Could not load session `{session_id}`: {exc}"
        if not history:
            return f"Session `{session_id}` has no messages."
        # Loaded into the live session, so the next turn continues with this context.
        state.history = history
        return self._format_resume_transcript(session_id, history)

    def _format_resume_transcript(self, session_id: str, history: list[dict[str, Any]]) -> str:
        """Render the loaded session as markdown for the client.

        The transcript is what the user sees: an ACP agent cannot rewrite a T3 thread's stored
        messages, and streaming the history as `session/update` chunks lands inside the current
        assistant bubble instead of as past turns. A markdown reply is the surface that works,
        so it carries the conversation and the context is loaded for follow-up turns too.
        """
        roles = Counter(str(message.get("role") or "?") for message in history)
        lines = [
            f"**Resumed `{session_id}`** — {len(history)} messages loaded into this thread, so the "
            "next turn continues this conversation.",
            "",
            f"{roles.get('user', 0)} user · {roles.get('assistant', 0)} assistant",
            "",
        ]
        shown = history[-_RESUME_TRANSCRIPT_LIMIT:]
        if len(history) > len(shown):
            lines.append(f"_Showing the last {len(shown)} of {len(history)} messages._")
            lines.append("")
        for message in shown:
            role = str(message.get("role") or "?")
            body = message.get("content")
            if isinstance(body, list):
                body = " ".join(
                    str(block.get("text") or "") for block in body if isinstance(block, dict)
                )
            text = " ".join(str(body or "").split())
            if role == "tool":
                # Raw tool payloads are mostly JSON blobs; a transcript wants the gist.
                text = f"_{text[:160]}…_" if len(text) > 160 else f"_{text}_"
            if not text:
                tool_calls = message.get("tool_calls")
                names = [
                    str(call.get("function", {}).get("name") or "tool")
                    for call in (tool_calls if isinstance(tool_calls, list) else ())
                    if isinstance(call, dict)
                ]
                text = f"(tool calls: {', '.join(names)})" if names else "(empty)"
            if len(text) > _RESUME_MESSAGE_CHARS:
                text = text[:_RESUME_MESSAGE_CHARS].rstrip() + " …"
            label = {"user": "**User**", "assistant": "**Assistant**", "system": "**System**"}.get(role, f"**{role}**")
            lines.append(f"{label}: {text}")
            lines.append("")
        return "\n".join(lines).strip()

    def _expand_skill_command(self, text: str, state: SessionState) -> tuple[str, str] | None:
        """Expand a skill invocation into the prompt the agent should run.

        Returns ``("prompt", expanded)`` when *text* names an installed skill,
        ``("message", reply)`` when it is a skill command the user got wrong, and
        ``None`` when it is not a skill invocation at all (so normal slash handling
        continues). Skill content has to reach the model, not the client, which is
        why this cannot be an ordinary ``_cmd_*`` handler.
        """
        from agent.skill_commands import build_skill_invocation_message, resolve_skill_command_key

        parts = text.split(maxsplit=1)
        name = parts[0].lstrip("/")
        rest = parts[1].strip() if len(parts) > 1 else ""
        if name.lower() == "skill":
            bits = rest.split(maxsplit=1)
            if not bits:
                return ("message", "Usage: /skill <name> [instruction]. See /skills for the list.")
            name, rest = bits[0], (bits[1].strip() if len(bits) > 1 else "")
        elif name.lower() in self._COMMANDS:
            # Core commands keep their name; a skill of the same slug runs via /skill <name>.
            return None
        elif resolve_skill_command_key(name) is None:
            return None

        key = resolve_skill_command_key(name)
        if key is None:
            return ("message", f"No skill named '{name}'. See /skills for the installed list.")
        message = build_skill_invocation_message(key, user_instruction=rest, task_id=state.session_id)
        if message is None:
            return ("message", f"Could not load skill '{name}'.")
        return ("prompt", message)

    def _cmd_version(self, args: str, state: SessionState) -> str:
        from hermes_cli.version_info import get_version_info

        return f"Hermes Agent v{get_version_info().derived_version}"
