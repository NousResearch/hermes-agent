"""Headless slash commands for ACP sessions (``/help``, ``/model``, ``/compress`` ...)."""

from __future__ import annotations

import contextvars
import logging
from collections import Counter
from typing import Any

from acp.schema import AvailableCommand, AvailableCommandsUpdate, UnstructuredCommandInput

from acp_adapter.session import SessionState, _expand_acp_enabled_toolsets

logger = logging.getLogger("acp_adapter.server")

try:
    from hermes_cli import __version__ as HERMES_VERSION
except Exception:
    HERMES_VERSION = "0.0.0"


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


class SlashCommandsMixin:
    """Slash-command surface for ``HermesACPAgent``; relies on ``_conn``, ``_send``, ``_schedule_soon``,
    ``session_manager`` and ``_switch_model`` from the host class."""

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
        "version": ("Show Hermes version", "Show Hermes version", None),
    }


    @classmethod
    def _available_commands(cls) -> list[AvailableCommand]:
        return [
            AvailableCommand(name=name, description=desc, input=UnstructuredCommandInput(hint=hint) if hint else None)
            for name, (_help, desc, hint) in cls._COMMANDS.items()
        ]

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
        handler = getattr(self, f"_cmd_{cmd}")

        # Handlers run on the loop thread, outside the per-turn cwd-pinning context. ``/compress``
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
            from agent.memory_manager import inject_memory_provider_tools

            toolsets = _expand_acp_enabled_toolsets(getattr(state.agent, "enabled_toolsets", None) or ["hermes-acp"])
            tools = get_tool_definitions(enabled_toolsets=toolsets, quiet_mode=True)
            tool_view = SimpleNamespace(
                tools=list(tools or []),
                valid_tool_names={t.get("function", {}).get("name") for t in tools or [] if isinstance(t, dict)},
                enabled_toolsets=toolsets, _memory_manager=getattr(state.agent, "_memory_manager", None),
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
        if not state.history:
            return "Nothing to compress — conversation is empty."
        try:
            agent = state.agent
            # No compression_enabled gate: it only disables *automatic* compaction (CLI/gateway parity).
            if not hasattr(agent, "_compress_context"):
                return "Context compression not available for this agent."

            original_count = len(state.history)
            # Include system prompt + tool schemas so the figure reflects real request pressure.
            # See #6217.
            # See #6217.
            _sys_prompt = getattr(agent, "_cached_system_prompt", "") or ""
            _tools = getattr(agent, "tools", None) or None
            approx_tokens = _estimate_tokens(state.history, agent, _sys_prompt, _tools)
            original_session_db = getattr(agent, "_session_db", None)

            try:
                # Stable ACP session id: suppress _compress_context's SQLite session split.
                agent._session_db = None
                compressed, _ = agent._compress_context(
                    state.history, _sys_prompt, approx_tokens=approx_tokens, task_id=state.session_id, force=True,
                )
            finally:
                agent._session_db = original_session_db

            state.history = compressed
            self.session_manager.save_session(state.session_id)

            new_tokens = _estimate_tokens(
                state.history, agent, getattr(agent, "_cached_system_prompt", "") or _sys_prompt,
                getattr(agent, "tools", None) or _tools,
            )
            return (
                f"Context compressed: {original_count} -> {len(state.history)} messages\n"
                f"~{approx_tokens:,} -> ~{new_tokens:,} tokens"
            )
        except Exception as e:
            return f"Compression failed: {e}"

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

    def _cmd_version(self, args: str, state: SessionState) -> str:
        return f"Hermes Agent v{HERMES_VERSION}"
