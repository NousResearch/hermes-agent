"""Session lifecycle for the interactive CLI: new/resume/save, undo/retry rewinds, yolo
persistence, manual compression, and exit summary.

Mixin bound onto ``HermesCLI`` via the MRO. cli.py-internal symbols are imported LAZILY
inside each method (``from cli import ...``) — never at module load time (import cycle).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import uuid

from hermes_constants import get_hermes_home
from pathlib import Path
from rich.console import Console
from rich.markup import escape as _escape
from typing import Any, Dict, List, Optional


def _user_turn_indices(history: list) -> list[int]:
    """Indices of *real* user turns: excludes ephemeral scaffolding, display_kind timeline
    rows and compaction handoffs — the same predicate as resume turn counting."""
    from agent.context_compressor import user_originated_turn_view
    from agent.session_persistence import _is_ephemeral_scaffolding

    return [
        i for i, m in enumerate(history)
        if not _is_ephemeral_scaffolding(m) and user_originated_turn_view(m) is not None]


def _timestamp_or(value, default):
    """``datetime.fromtimestamp(value)`` or *default* when the value is missing/unparseable."""
    from cli import datetime

    if not value:
        return default
    try:
        return datetime.fromtimestamp(float(value))
    except Exception:
        return default


def _squash(text: str, limit: int = 120) -> str:
    """Collapse whitespace and cap to *limit* cells with an ellipsis."""
    text = " ".join(text.split())
    return text[: limit - 1] + "…" if len(text) > limit else text


def _dim_notice(cli, msg: str, quiet: bool) -> None:
    """Print a dim notice — raw to stderr on quiet (pre-TUI) paths. Module-level so tests can
    call the resume helpers unbound against a minimal stand-in."""
    if quiet:
        print(msg, file=sys.stderr)
    else:
        cli._console_print(f"[dim]{_escape(msg)}[/dim]")


def _reset_model_to_config_default(cli, silent: bool) -> None:
    """/new is a full boundary: re-derive model/provider from config.yaml so a
    session-only ``/model --session`` switch never leaks into the next session.
    Best-effort — an unreachable default must never block /new. Module-level helper (like
    ``_apply_new_session_title``): tests drive ``new_session`` unbound on a SimpleNamespace."""
    from cli import CLI_CONFIG, _cprint, _split_model_config_default, logger
    _model_config = CLI_CONFIG.get("model", {})
    if isinstance(_model_config, dict):
        _raw_default = _model_config.get("default") or _model_config.get("model") or ""
        _config_provider = _model_config.get("provider", "")
    else:
        _raw_default, _config_provider = (_model_config or ""), ""
    _config_model, _ = _split_model_config_default(_raw_default)
    if not _config_model or _config_model == getattr(cli, "model", None):
        return
    try:
        from hermes_cli.model_switch import switch_model as _switch_model

        r = _switch_model(
            raw_input=_config_model,
            current_provider=cli.provider or "",
            current_model=cli.model or "",
            current_base_url=cli.base_url or "",
            current_api_key=cli.api_key or "",
            is_global=False,
            explicit_provider=_config_provider or "")
        if not r.success:
            return
        if cli.agent:
            cli.agent.switch_model(
                new_model=r.new_model, new_provider=r.target_provider, api_key=r.api_key,
                base_url=r.base_url, api_mode=r.api_mode,
                capabilities=getattr(r, "runtime_capabilities", None))
        cli.model = r.new_model
        cli.provider = r.target_provider
        cli.requested_provider = r.target_provider
        cli._explicit_api_key = r.api_key
        cli._explicit_base_url = r.base_url
        if r.api_key:
            cli.api_key = r.api_key
        if r.base_url:
            cli.base_url = r.base_url
        if r.api_mode:
            cli.api_mode = r.api_mode
        if not silent:
            _cprint(f"  (model reset to config default: {r.new_model})")
    except Exception:
        logger.debug("/new model reset to config default failed", exc_info=True)


def _apply_new_session_title(cli, title: str) -> Optional[str]:
    """Sanitize + persist a /new title; returns the stored title or None (untitled)."""
    from cli import _cprint
    from hermes_state import SessionDB
    try:
        sanitized = SessionDB.sanitize_title(title)
    except ValueError as e:
        _cprint(f"  Title rejected: {e}")
        return None
    if not sanitized:
        _cprint("  Title is empty after cleanup — session started untitled.")
        return None
    try:
        cli._session_db.set_session_title(cli.session_id, sanitized)
    except ValueError as e:
        _cprint(f"  {e} — session started untitled.")
        return None
    except Exception:
        return None
    cli._pending_title = None
    cli._status_bar_title_checked_at = 0.0
    return sanitized


class CLISessionMixin:
    """Session lifecycle for the interactive CLI: new/resume/save, undo/retry rewinds, yolo
    persistence, manual compression, and exit summary."""

    def _restore_session_cwd(self, session_meta: dict, *, quiet: bool = False) -> None:
        """Relaunch a resumed session in the directory it was started from.

        Idempotent; called from every resume path. Both ``os.chdir()`` and ``TERMINAL_CWD``
        are retargeted so the process and the terminal/code-exec tools agree (the local
        terminal backend snapshots cwd on first use, after this). No-op when no cwd was
        recorded, the directory is gone (dim warning, never a crash), or we're already there.
        """
        recorded = (session_meta or {}).get("cwd")
        if not recorded:
            return
        recorded = os.path.expanduser(str(recorded))
        try:
            current = os.getcwd()
        except OSError:
            current = None
        if current and os.path.realpath(recorded) == os.path.realpath(current):
            return
        if not os.path.isdir(recorded):
            msg = f"⚠ Session's working directory is gone: {recorded} — staying in {current or '.'}"
        else:
            try:
                os.chdir(recorded)
                os.environ["TERMINAL_CWD"] = recorded
                msg = f"↻ Working directory: {recorded}"
            except OSError as e:
                msg = f"⚠ Could not enter session's working directory {recorded}: {e}"
        _dim_notice(self, msg, quiet)

    def _restore_session_yolo(self, session_meta: dict, *, quiet: bool = False) -> None:
        """Re-enable YOLO bypass on resume when the session row's ``model_config.yolo_mode``
        says so — the in-memory ``tools.approval._session_yolo`` set starts empty in a fresh
        process. No-op when already active or when the process was launched with ``--yolo``."""
        try:
            from hermes_state import SessionDB
            from tools.approval import (
                _YOLO_MODE_FROZEN, enable_session_yolo, is_session_yolo_enabled)
        except Exception:
            return
        if _YOLO_MODE_FROZEN or not SessionDB.session_yolo_enabled(session_meta):
            return
        session_key = self.session_id or "default"
        if is_session_yolo_enabled(session_key):
            return
        enable_session_yolo(session_key)
        _dim_notice(self,
            "⚡ YOLO mode restored from session — all commands auto-approved. /yolo to turn off.",
            quiet)

    def _render_resume_history_panel_lines(self, panel) -> list[str]:
        """Render the resume panel at the current terminal width for resize replay."""
        from cli import _suspend_output_history
        from io import StringIO

        buf = StringIO()
        console = Console(
            file=buf, force_terminal=True, color_system="truecolor", highlight=False,
            width=shutil.get_terminal_size((80, 24)).columns)
        with _suspend_output_history():
            console.print(panel)
        return buf.getvalue().rstrip("\n").splitlines()

    def _resolve_checkpoint_ref(self, ref: str, checkpoints: list) -> str | None:
        """Resolve a 1-indexed checkpoint number (or pass through a git hash)."""
        try:
            idx = int(ref) - 1
        except ValueError:
            return ref
        if 0 <= idx < len(checkpoints):
            return checkpoints[idx]["hash"]
        print(f"  Invalid checkpoint number. Use 1-{len(checkpoints)}.")
        return None

    def _show_status(self):
        """Show compact startup status line."""
        from cli import get_tool_definitions
        # Avoid pulling the full tool registry into the bare Termux prompt path.
        if os.environ.get("HERMES_DEFER_AGENT_STARTUP") == "1":
            tool_status = "tools deferred"
        else:
            tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets, quiet_mode=True)
            tool_status = f"{len(tools) if tools else 0} tools"

        model_short = self.model.split("/")[-1] if "/" in self.model else self.model
        if len(model_short) > 30:
            model_short = model_short[:27] + "..."
        api_indicator = "[green bold]●[/]" if self.api_key else "[red bold]●[/]"
        try:
            from hermes_cli.skin_engine import get_active_skin
            skin = get_active_skin()
            separator_color = skin.get_color("banner_dim", "#B8860B")
            accent_color = skin.get_color("ui_accent", "#FFBF00")
            label_color = skin.get_color("ui_label", "#DAA520")
        except Exception:
            separator_color, accent_color, label_color = "#B8860B", "#FFBF00", "cyan"
        sep = f" [dim {separator_color}]·[/] "
        toolsets_info = ""
        if self.enabled_toolsets and "all" not in self.enabled_toolsets:
            toolsets_info = f"{sep}[{label_color}]toolsets: {', '.join(self.enabled_toolsets)}[/]"
        provider_info = f"{sep}[dim]provider: {self.provider}[/]"
        if self._provider_source:
            provider_info += f"{sep}[dim]auth: {self._provider_source}[/]"
        self._console_print(
            f"  {api_indicator} [{accent_color}]{model_short}[/]{sep}"
            f"[bold {label_color}]{tool_status}[/]{toolsets_info}{provider_info}")

    def _show_session_status(self):
        """Show gateway-style status for the current CLI session."""
        from hermes_constants import display_hermes_home
        session_meta = {}
        if self._session_db:
            with contextlib.suppress(Exception):
                session_meta = self._session_db.get_session(self.session_id) or {}

        title = (session_meta.get("title") or "").strip()
        created_at = _timestamp_or(session_meta.get("started_at"), self.session_start)
        updated_at = created_at
        for field in ("updated_at", "last_updated_at", "last_activity_at"):
            candidate = _timestamp_or(session_meta.get(field), None)
            if candidate is not None:
                updated_at = candidate
                break

        agent = getattr(self, "agent", None)
        total_tokens = getattr(agent, "session_total_tokens", 0) or 0
        provider = getattr(self, "provider", None) or "unknown"
        model = getattr(self, "model", None) or "(unknown)"
        is_running = bool(getattr(self, "_agent_running", False))

        reasoning_label = None
        rc = getattr(agent, "reasoning_config", None) or getattr(self, "reasoning_config", None)
        if isinstance(rc, dict):
            if rc.get("enabled") is False:
                reasoning_label = "off"
            elif rc.get("effort"):
                reasoning_label = str(rc.get("effort"))
        show_r = getattr(self, "show_reasoning", None)
        if reasoning_label and show_r is not None:
            reasoning_label += f" (display: {'on' if show_r else 'off'})"

        approval_label = None
        try:
            from tools.approval import is_approval_bypass_active_for_session
            from tools.approval_context import _get_approval_mode
            approval_label = _get_approval_mode()
            if is_approval_bypass_active_for_session(getattr(self, "session_key", "") or ""):
                approval_label += " (YOLO bypass active)"
        except Exception:
            pass

        # Context window usage: reuse the status-bar snapshot (tokens / max / percent).
        ctx_label = None
        try:
            snap = self._get_status_bar_snapshot()
            ctx_max = snap.get("context_length")
            ctx_pct = snap.get("context_percent")
            if ctx_max:
                left = ""
                if isinstance(ctx_pct, (int, float)):
                    left = f"{max(0, 100 - int(ctx_pct))}% left · "
                ctx_label = f"{left}{snap.get('context_tokens') or 0:,} / {ctx_max:,} tokens used"
        except Exception:
            ctx_label = None

        lines = [
            "Hermes CLI Status", "", f"Session ID: {self.session_id}", f"Path: {display_hermes_home()}",
        ]
        if title:
            lines.append(f"Title: {title}")
        lines.append(f"Model: {model} ({provider})")
        optional = (("Reasoning", reasoning_label), ("Approvals", approval_label), ("Context", ctx_label))
        for label, value in optional:
            if value:
                lines.append(f"{label}: {value}")
        lines.extend([
            f"Created: {created_at.strftime('%Y-%m-%d %H:%M')}",
            f"Last Activity: {updated_at.strftime('%Y-%m-%d %H:%M')}",
            f"Tokens: {total_tokens:,}",
            f"Agent Running: {'Yes' if is_running else 'No'}"])
        self._console_print("\n".join(lines), highlight=False, markup=False)

    def _list_recent_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent CLI sessions for in-chat browsing/resume affordances."""
        if not self._session_db:
            return []
        try:
            from hermes_cli.session_listing import query_session_listing

            return query_session_listing(
                self._session_db, source="cli", current_session_id=self.session_id,
                include_all_sources=False, include_unnamed=True, limit=limit,
                exclude_sources=["kanban", "tool"])
        except Exception:
            return []

    def _show_recent_sessions(self, *, reason: str = "history", limit: int = 10) -> bool:
        """Render recent sessions inline from the active chat TUI.

        Returns True when something was shown, False if no session list was available.
        """
        from cli import _cli_visible_print
        sessions = self._list_recent_sessions(limit=limit)
        if not sessions:
            return False

        from hermes_cli.timefmt import relative_time as _relative_time

        _cli_visible_print()
        if reason == "history":
            _cli_visible_print("(._.) No messages in the current chat yet — here are recent sessions you can resume:")
        else:
            _cli_visible_print("  Recent sessions:")
        _cli_visible_print()
        _cli_visible_print(f"  {'#':<3} {'Title':<32} {'Preview':<40} {'Last Active':<13} {'ID'}")
        _cli_visible_print(f"  {'─' * 3} {'─' * 32} {'─' * 40} {'─' * 13} {'─' * 24}")
        for idx, session in enumerate(sessions, start=1):
            title = session.get("title") or "—"
            preview = (session.get("preview") or "")[:38]
            last_active = _relative_time(session.get("last_active"))
            _cli_visible_print(f"  {idx:<3} {title:<32} {preview:<40} {last_active:<13} {session['id']}")
        _cli_visible_print()
        _cli_visible_print("  Use /resume <number>, /resume <session id>, or /resume <session title> to continue.")
        _cli_visible_print("  Example: /resume 2")
        _cli_visible_print()
        return True

    def show_history(self):
        """Display conversation history."""
        from cli import _cli_visible_print
        if not self.conversation_history:
            if not self._show_recent_sessions(reason="history"):
                _cli_visible_print("(._.) No conversation history yet.")
            return

        preview_limit = 400
        visible_index = 0
        hidden_tool_messages = 0
        show_ts = bool(getattr(self, "show_timestamps", False))

        def _ts_suffix(message: dict) -> str:
            # Only annotate when the toggle is on AND the turn has a stored unix
            # `timestamp` (SessionDB-restored rows do; live turns may not) — never fabricate.
            ts = message.get("timestamp") if show_ts else None
            if not ts:
                return ""
            try:
                stamp = _timestamp_or(ts, None)
                return f"  [{stamp.strftime(getattr(self, 'timestamp_format', '%H:%M'))}]" if stamp else ""
            except (ValueError, OSError, TypeError):
                return ""

        def flush_tool_summary():
            nonlocal hidden_tool_messages
            if not hidden_tool_messages:
                return
            noun = "message" if hidden_tool_messages == 1 else "messages"
            _cli_visible_print("\n  [Tools]")
            _cli_visible_print(f"    ({hidden_tool_messages} tool {noun} hidden)")
            hidden_tool_messages = 0

        rule = "+" + "-" * 50 + "+"
        for line in ("", rule, "|" + " " * 12 + "(^_^) Conversation History" + " " * 11 + "|", rule):
            _cli_visible_print(line)

        for msg in self.conversation_history:
            role = msg.get("role", "unknown")
            if role == "tool":
                hidden_tool_messages += 1
                continue
            if role not in {"user", "assistant"}:
                continue
            flush_tool_summary()
            visible_index += 1

            content = msg.get("content")
            content_text = "" if content is None else str(content)
            preview = content_text[:preview_limit]
            suffix = "..." if len(content_text) > preview_limit else ""
            if role == "user":
                _cli_visible_print(f"\n  [You #{visible_index}]{_ts_suffix(msg)}")
                _cli_visible_print(f"    {preview}{suffix}")
                continue

            _cli_visible_print(f"\n  [Hermes #{visible_index}]{_ts_suffix(msg)}")
            n_calls = len(msg.get("tool_calls") or [])
            if not content_text:
                suffix = ""
                preview = "(no text response)"
                if n_calls:
                    preview = f"(requested {n_calls} tool {'call' if n_calls == 1 else 'calls'})"
            _cli_visible_print(f"    {preview}{suffix}")

        flush_tool_summary()
        _cli_visible_print()

    def _notify_session_boundary(self, event_type: str) -> None:
        """Fire a session-boundary plugin hook (on_session_finalize / on_session_reset).
        Non-blocking; errors swallowed. Safe from shutdown, /new, /reset."""
        with contextlib.suppress(Exception):
            from hermes_cli.lifecycle import finalize_session, invoke_hook

            context = {
                "session_id": self.agent.session_id if self.agent else None,
                "platform": getattr(self, "platform", None) or "cli",
                "reason": "new_session" if event_type == "on_session_reset" else "session_boundary"}
            if event_type == "on_session_finalize":
                finalize_session(**context)
            else:
                invoke_hook(event_type, **context)

    def _discard_session_if_empty(self, session_id: Optional[str]) -> bool:
        """Drop a just-ended session row that never gained content (quit-immediately, /new,
        /clear) so it doesn't clutter ``/resume``. ``SessionDB.delete_session_if_empty`` only
        removes rows with no messages, no title and no children (gemini-cli#27770 port)."""
        from cli import logger
        if not self._session_db or not session_id:
            return False
        # In-memory transcript is authoritative: a real conversation whose DB flush failed
        # or hasn't happened yet must never be pruned.
        if getattr(self, "conversation_history", None):
            return False
        try:
            from hermes_constants import get_hermes_home as _ghh
            return self._session_db.delete_session_if_empty(
                session_id, sessions_dir=_ghh() / "sessions")
        except Exception:
            logger.debug("Could not prune empty session %s", session_id, exc_info=True)
            return False

    def _launch_session_boundary_memory_flush(
        self, history_snapshot: list, *, session_id: Optional[str] = None) -> Optional[list]:
        """Stage old-session memory extraction so /new stays responsive.

        The context-engine ``on_session_end`` is delivered synchronously here: cheap (no LLM)
        and ordering-sensitive — it must land before ``reset_session_state()`` rebinds the
        engine. The memory-provider half (LLM-bound, seconds) is NOT run here: the returned
        snapshot goes to ``MemoryManager.commit_session_boundary_async`` as one end→switch
        task on the serialized worker, so a late ``on_session_end`` can never run after
        ``on_session_switch`` and misattribute the old transcript to the new session.

        Returns the snapshot to queue, or ``None`` when there is nothing to extract.
        """
        from cli import logger
        agent = getattr(self, "agent", None)
        if not agent or not history_snapshot:
            return None
        engine = getattr(agent, "context_compressor", None)
        if engine is not None and hasattr(engine, "on_session_end"):
            try:
                engine.on_session_end(session_id or "", history_snapshot)
            except Exception:
                logger.debug("Context engine on_session_end failed at /new boundary", exc_info=True)
        # No memory manager → new_session() falls back to the inline switch path.
        if getattr(agent, "_memory_manager", None) is None:
            return None
        return history_snapshot

    def new_session(self, silent=False, title=None):
        """Start a fresh session with a new session ID and cleared agent state."""
        from cli import (
            CLI_CONFIG, _parse_reasoning_config, _parse_service_tier_config,
            _sync_process_session_id, datetime)
        old_session_id = self.session_id
        _boundary_snapshot = None
        if self.agent:
            if self.conversation_history:
                # Context-engine boundary now; provider extraction is queued below (after
                # rotation) so /new never blocks on the LLM-bound call.
                _boundary_snapshot = self._launch_session_boundary_memory_flush(
                    list(self.conversation_history), session_id=old_session_id)
            self._notify_session_boundary("on_session_finalize")

        if self._session_db and old_session_id:
            # /new can arrive mid-turn before _flush_messages_to_session_db() ran — flush
            # the current turn to the OLD session before rotating or it is silently lost.
            if self.agent:
                with contextlib.suppress(Exception):
                    # Flush any un-persisted messages from the current turn to the old session *before*
                    # rotating.  /new can be called mid-turn when _flush_messages_to_session_db() has not
                    # yet run — without this, messages generated during the current turn are silently lost
                    # on session rotation (#47202).
                    # See #47202.
                    # See #47202.
                    self.agent._flush_messages_to_session_db(
                        self.conversation_history, conversation_history=self.conversation_history)
            with contextlib.suppress(Exception):
                self._session_db.end_session(old_session_id, "new_session")
            self._discard_session_if_empty(old_session_id)

        self.session_start = datetime.now()
        self.session_id = f"{self.session_start.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        # getattr: tests drive new_session unbound against a SimpleNamespace stand-in.
        getattr(self, "_write_terminal_breadcrumb", lambda: None)()
        self.conversation_history = []
        self._pending_title = None
        self._resumed = False
        # An explicit -m/--model was for the previous session only.
        self._explicit_model_override = False
        self.reasoning_config = _parse_reasoning_config(
            CLI_CONFIG["agent"].get("reasoning_effort", ""))
        # Session-scoped overrides (/model --session, /fast, one-turn restores) don't carry over.
        # Re-derive model/provider and service tier from config.yaml so a session-only switch never leaks
        # into the next session (#48055, #23131).
        self._pending_one_turn_model_restore = None
        self.service_tier = _parse_service_tier_config(CLI_CONFIG["agent"].get("service_tier", ""))
        _reset_model_to_config_default(self, silent)
        _sync_process_session_id(self.session_id)

        if self.agent:
            self.agent.session_id = self.session_id
            self.agent.session_start = self.session_start
            self.agent.reasoning_config = self.reasoning_config
            self.agent.reset_session_state()
            if hasattr(self.agent, "_last_flushed_db_idx"):
                self.agent._last_flushed_db_idx = 0
            if hasattr(self.agent, "_todo_store"):
                with contextlib.suppress(Exception):
                    from tools.todo_tool import TodoStore
                    self.agent._todo_store = TodoStore()
            if hasattr(self.agent, "_invalidate_system_prompt"):
                self.agent._invalidate_system_prompt()

            if self._session_db:
                with contextlib.suppress(Exception):
                    self.agent._session_db_created = False
                    self._session_db.create_session(
                        session_id=self.session_id,
                        source=os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                        model=self.model,
                        model_config={
                            "max_iterations": self.max_turns, "reasoning_config": self.reasoning_config,
                        })
                    self.agent._session_db_created = True
                if title:
                    title = _apply_new_session_title(self, title)
            # Tell memory providers the session_id rotated (reset=True flushes per-session
            # state) BEFORE the plugin on_session_reset hook. With old history, end-of-session
            # extraction and this switch are queued as ONE task on the serialized worker —
            # end strictly before switch, without blocking /new. No history → switch inline.
            _mm = getattr(self.agent, "_memory_manager", None)
            with contextlib.suppress(Exception):
                if _mm is not None and _boundary_snapshot:
                    _mm.commit_session_boundary_async(
                        _boundary_snapshot, new_session_id=self.session_id,
                        parent_session_id=old_session_id or "", reason="new_session")
                elif _mm is not None:
                    _mm.on_session_switch(
                        self.session_id, parent_session_id=old_session_id or "",
                        reset=True, reason="new_session")
            self._notify_session_boundary("on_session_reset")

        if not silent:
            if title:
                print(f"(^_^)v New session started: {title}")
            else:
                print("(^_^)v New session started!")

    def _consume_pending_resume_selection(self, text: str) -> bool:
        """Resolve a bare numeric reply following a bare ``/resume`` prompt.

        ``/resume`` (no args) arms ``self._pending_resume_sessions``; the next input gets one
        chance to be a bare session number. The pending state is one-shot — cleared on the
        first input regardless of outcome, so a stray later number is never hijacked.
        Returns True if the input was consumed (caller must not treat it as chat).

        See #34584.
        """
        from cli import _cprint
        pending = self._pending_resume_sessions
        if not pending:
            return False
        self._pending_resume_sessions = None
        if not isinstance(text, str):
            return False
        # Only a pure number selects; "/resume 3", titles etc. fall through.
        if not text.strip().isdigit():
            return False
        index = int(text.strip())
        if not 1 <= index <= len(pending):
            _cprint(f"  Resume index {index} is out of range.")
            _cprint("  Use /resume with no arguments to see available sessions.")
            return True
        self._handle_resume_command(f"/resume {index}")
        return True

    def save_conversation(self, cmd: str = "/save"):
        """Handle ``/save [json|md|html] [filename] [redact]``.

        A convenience export only — every message is already persisted to the session DB, so
        the live session stays resumable regardless. ``redact`` runs the force-mode secret
        redaction pass before writing.
        """
        from cli import datetime
        from hermes_cli.session_export import (
            SAVE_USAGE, normalize_save_format, render_session_for_save)

        parts = cmd.split()[1:]
        redact = bool(parts) and parts[-1].lower() in ("redact", "--redact")
        if redact:
            parts = parts[:-1]
        if not parts:
            print(SAVE_USAGE)
            return
        try:
            fmt = normalize_save_format(parts[0])
        except ValueError as e:
            print(f"(._.) {e}")
            print(SAVE_USAGE)
            return
        filename = parts[1] if len(parts) > 1 else None

        # Prefer the durable DB row (metadata + tool calls); fall back to in-memory history.
        # getattr: test doubles may not carry _session_db / session_id.
        session_data = None
        _db = getattr(self, "_session_db", None)
        _sid = getattr(self, "session_id", None)
        if _db and _sid:
            try:
                session_data = _db.export_session(_sid)
            except Exception:
                session_data = None
        if not session_data:
            if not self.conversation_history:
                print("(;_;) No conversation to save.")
                return
            session_data = {
                "id": self.session_id, "model": self.model,
                "started_at": self.session_start.timestamp(), "messages": self.conversation_history}
        if redact:
            from hermes_cli.session_export_md import redact_session_data

            session_data = redact_session_data(session_data)

        saved_dir = get_hermes_home() / "sessions" / "saved"
        try:
            saved_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"(x_x) Failed to create save directory {saved_dir}: {e}")
            return
        if filename:
            path = Path(filename).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = saved_dir / f"hermes_conversation_{timestamp}.{fmt}"

        try:
            content = render_session_for_save(session_data, fmt)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            label = {"json": "JSON", "md": "Markdown", "html": "HTML"}[fmt]
            print(f"(^_^)v Conversation saved to: {path} ({label})")
            # #76354 review F5: the worker thread also rebound the session ContextVar inside its own
            # (copied) context, which the caller never sees — and get_session_env() prefers an already-bound
            # ContextVar over os.environ. Rebind in the CALLER's context so post-compression
            # tools/subprocesses on this thread resolve HERMES_SESSION_ID to the child id after an
            # out-of-place rotation (idempotent when no rotation happened).
            if self.session_id:
                print(f"       Resume the live session with: hermes --resume {self.session_id}")
        except Exception as e:
            print(f"(x_x) Failed to save: {e}")

    def _rewind_persisted_user_turn(
        self,
        *,
        warm_history: List[Dict[str, Any]],
        user_ordinal: int,
        warm_live_view: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """Bind one warm user ordinal to a durable row and rewind it atomically."""
        if self._session_db is None or not self.session_id:
            raise RuntimeError("session database is unavailable")

        from agent.context_compressor import (
            history_before_user_originated_turn,
            split_user_originated_turn,
            user_originated_turn_view)
        from agent.memory_manager import sanitize_context
        from agent.tool_dispatch_helpers import _is_multimodal_tool_result, _multimodal_text_summary
        from agent.session_persistence import _is_ephemeral_scaffolding

        def _persistence_content(content: Any) -> Any:
            """Project warm content exactly as the session DB flush does."""
            if _is_multimodal_tool_result(content):
                return _multimodal_text_summary(content)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                    elif isinstance(part, dict) and part.get("type") in {
                        "image", "image_url", "input_image"}:
                        text_parts.append("[screenshot]")
                return "\n".join(text_parts) if text_parts else None
            return content

        def _comparison_content(message: Dict[str, Any]) -> Any:
            content = _persistence_content(message.get("content"))
            if message.get("role") in {"user", "assistant"} and isinstance(content, str):
                return sanitize_context(content).strip()
            return content

        def _user_indices(messages):
            return [i for i, m in enumerate(messages) if user_originated_turn_view(m) is not None]

        changed = RuntimeError("session history changed before the rewind could be persisted")
        expected_active_ids = self._session_db.get_active_message_ids(self.session_id)
        durable = self._session_db.get_messages_as_conversation(
            self.session_id, include_row_ids=True)
        warm_persistence_history = [m for m in warm_history if not _is_ephemeral_scaffolding(m)]
        warm_user_indices = _user_indices(warm_persistence_history)
        durable_user_indices = _user_indices(durable)
        if len(durable_user_indices) != len(warm_user_indices):
            raise changed
        if user_ordinal < 0 or user_ordinal >= len(durable_user_indices):
            raise RuntimeError("persisted rewind target is no longer available")

        warm_prefix, _ = history_before_user_originated_turn(
            warm_persistence_history, warm_user_indices[user_ordinal])
        durable_target_index = durable_user_indices[user_ordinal]
        durable_target = durable[durable_target_index]
        durable_prefix, durable_live_view = history_before_user_originated_turn(
            durable, durable_target_index)
        if _comparison_content(durable_live_view) != _comparison_content(warm_live_view):
            raise changed
        target_row_id = durable_target.get("_row_id")
        if not isinstance(target_row_id, int):
            raise RuntimeError("persisted rewind target has no row identity")
        scaffold, _ = split_user_originated_turn(durable_target)
        result = self._session_db.rewind_to_message(
            self.session_id, target_row_id,
            preserve_compaction_handoff=scaffold is not None,
            expected_active_ids=expected_active_ids,
            expected_target_content=durable_live_view.get("content"))
        if scaffold is not None:
            replacement_id = result.get("replacement_message_id")
            if not isinstance(replacement_id, int) or not durable_prefix:
                raise RuntimeError("rewind did not retain its compaction handoff")
            durable_prefix[-1]["_row_id"] = replacement_id
            durable_prefix[-1]["_db_persisted"] = True
            warm_prefix[-1] = durable_prefix[-1]
        return warm_prefix, durable_live_view, result

    def _publish_truncated_history(self, truncated: list, *, invalidate_prompt: bool) -> None:
        """Install a rewound history and mirror it onto the agent (flush index reset so the
        next turn re-flushes from the truncated head)."""
        self.conversation_history = truncated
        agent = self.agent
        if agent is None:
            return
        if invalidate_prompt and hasattr(agent, "_invalidate_system_prompt"):
            with contextlib.suppress(Exception):
                agent._invalidate_system_prompt()
        if hasattr(agent, "_last_flushed_db_idx"):
            with contextlib.suppress(Exception):
                agent._last_flushed_db_idx = len(self.conversation_history)
        if hasattr(agent, "_session_messages"):
            agent._session_messages = self.conversation_history
        if hasattr(agent, "_db_flush_scan_prefix"):
            agent._db_flush_scan_prefix = self.conversation_history[:]

    def retry_last(self):
        """Retry the last user message: drop the last exchange and return the text to re-send
        (None when there is nothing to retry)."""
        if not self.conversation_history:
            print("(._.) No messages to retry.")
            return None

        from agent.context_compressor import (
            history_before_user_originated_turn, retryable_user_text)
        from agent.memory_manager import sanitize_context

        warm_history = list(self.conversation_history)
        user_indices = _user_turn_indices(warm_history)
        if not user_indices:
            print("(._.) No user message found to retry.")
            return None

        # Resolve a lossless live payload before touching persistence or memory. A
        # force-user-leading compaction row is one physical carrier: its handoff stays in
        # the prefix while only the embedded human ask is retried. Media cannot be replayed
        # by /retry, so fail closed before archiving anything.
        try:
            truncated, live_view = history_before_user_originated_turn(
                warm_history, user_indices[-1])
            live_content = live_view.get("content")
            if isinstance(live_content, str):
                live_content = sanitize_context(live_content).strip()
            last_message = retryable_user_text(live_content)
        except ValueError as exc:
            print(f"(._.) Cannot retry that message safely: {exc}")
            return None

        # Persist the rewind before publishing the shorter in-memory view: the DB owns the
        # physical carrier split so archived original + retained scaffold commit atomically.
        if self._session_db is not None and self.session_id:
            try:
                truncated, _, _ = self._rewind_persisted_user_turn(
                    warm_history=warm_history,
                    user_ordinal=len(user_indices) - 1,
                    warm_live_view=live_view)
            except Exception as exc:
                print(f"(x_x) Retry rewind failed; history was not changed: {exc}")
                return None

        self._publish_truncated_history(truncated, invalidate_prompt=False)
        print(f"(^_^)b Retrying: \"{last_message[:60]}{'...' if len(last_message) > 60 else ''}\"")
        return last_message

    def undo_last(self, n: int = 1, prefill: bool = True):
        """Back up N user turns: truncate history, soft-delete on disk, prefill the composer.

        Discards everything from the Nth-from-last user message onward (clamped to the oldest
        turn). Rows are soft-deleted in SessionDB (``active=0``, kept for audit), memory
        providers get ``on_session_switch(rewound=True)``, and the agent is patched like
        /branch does. ``prefill=False`` is for programmatic callers (checkpoint rollback)
        that must not touch the input buffer.
        """
        from cli import logger
        if not self.conversation_history:
            print("(._.) No messages to undo.")
            return
        n = max(n, 1)

        from agent.context_compressor import history_before_user_originated_turn

        warm_history = list(self.conversation_history)
        user_indices = _user_turn_indices(warm_history)
        if not user_indices:
            print("(._.) No user message found to undo.")
            return

        turns_undone = min(n, len(user_indices))
        target_ordinal = len(user_indices) - turns_undone
        cut_idx = user_indices[target_ordinal]
        removed_count = len(warm_history) - cut_idx
        truncated, live_view = history_before_user_originated_turn(warm_history, cut_idx)
        removed_text = self._undo_content_to_text(live_view.get("content"))

        rewound_rows = 0
        if self._session_db is not None and self.session_id:
            try:
                truncated, durable_live_view, result = self._rewind_persisted_user_turn(
                    warm_history=warm_history,
                    user_ordinal=target_ordinal,
                    warm_live_view=live_view)
                # Canonical editable prefill: the raw carrier holds the reference-summary wrapper.
                durable_text = self._undo_content_to_text(durable_live_view.get("content"))
                if durable_text:
                    removed_text = durable_text
                rewound_rows = result.get("rewound_count", 0)
            except Exception as e:
                logger.debug("undo: durable rewind failed: %s", e)
                print(f"(x_x) Undo failed; history was not changed: {e}")
                return

        # Publish only after the durable rewind succeeds (or no store exists).
        self._publish_truncated_history(truncated, invalidate_prompt=True)
        # Same hook /branch fires; rewound=True invalidates per-turn document caches.
        _mm = getattr(self.agent, "_memory_manager", None)
        # See #21910, #6672.
        if _mm is not None and self.session_id:
            with contextlib.suppress(Exception):
                _mm.on_session_switch(self.session_id, parent_session_id="", reset=False, rewound=True)

        turn_word = "turn" if turns_undone == 1 else "turns"
        print(
            f"(^_^)b Undid {turns_undone} {turn_word} ({rewound_rows or removed_count} message(s)). "
            f"Backed up to: \"{removed_text[:60]}{'...' if len(removed_text) > 60 else ''}\"")
        print(f"  {len(self.conversation_history)} message(s) remaining in history.")
        # Editable, not auto-sent (Claude-Code-style).
        if prefill and removed_text:
            self._prefill_input_buffer(removed_text)

    @staticmethod
    def _undo_content_to_text(content) -> str:
        """Flatten message content (str or content-part list) to plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            ]
            return "\n".join(t for t in parts if t)
        return ""

    def _write_terminal_breadcrumb(self) -> None:
        """Record this terminal's live session for bare ``hermes -c``. Called whenever
        ``self.session_id`` is (re)assigned so a later bare ``-c`` in THIS terminal resumes
        this conversation's live tip. Best-effort; no-op without a terminal identity."""
        with contextlib.suppress(Exception):
            from hermes_cli.terminal_breadcrumbs import write_breadcrumb

            write_breadcrumb(self.session_id)

    def _transfer_session_yolo(self, old_session_id: str, new_session_id: str) -> None:
        """Move YOLO bypass state to a new session key when ``self.session_id`` is reassigned
        mid-run (/branch, auto-compression rotation) — ``_session_yolo`` is keyed by id, so
        without this the toggle silently reverts. Mirrors tui_gateway's rename path."""
        if not old_session_id or not new_session_id or old_session_id == new_session_id:
            return
        try:
            from tools.approval import (
                disable_session_yolo, enable_session_yolo, is_session_yolo_enabled)
        except Exception:
            return
        if is_session_yolo_enabled(old_session_id):
            enable_session_yolo(new_session_id)
            disable_session_yolo(old_session_id)
            # Carry the persisted flag onto the continuation row so a later --resume restores
            # it too. getattr: tests call this unbound against a minimal stand-in.
            _persist = getattr(self, "_persist_session_yolo", None)
            if _persist:
                _persist(new_session_id, True)

    def _is_session_yolo_active(self) -> bool:
        """Whether YOLO bypass is on for this session: reads ``tools.approval._session_yolo``
        (not a stale env var) and honors the frozen process-start ``--yolo`` flag."""
        try:
            from tools.approval import _YOLO_MODE_FROZEN, is_session_yolo_enabled
        except Exception:
            return False
        if _YOLO_MODE_FROZEN:
            return True
        # getattr: __new__-built test fixtures skip __init__; the status-bar builders
        # swallow exceptions but would lose every field after the failure.
        return is_session_yolo_enabled(getattr(self, "session_id", None) or "default")

    def _toggle_yolo(self):
        """Toggle per-session YOLO mode (skip dangerous-command approvals).

        Mirrors the gateway/TUI ``/yolo`` handlers. Deliberately does NOT touch
        ``HERMES_YOLO_MODE``: that env var is frozen into ``tools.approval._YOLO_MODE_FROZEN``
        at import (so prompt-injected skills can't flip the bypass), making a later set a
        silent no-op. ``run_conversation`` binds ``self.session_id`` as the active approval
        key, so the bypass applies to the very next dangerous command.
        """
        from cli import _cprint
        from hermes_cli.colors import Colors as _Colors
        from tools.approval import (
            _YOLO_MODE_FROZEN, disable_session_yolo, enable_session_yolo, is_session_yolo_enabled)

        # A frozen process-level bypass short-circuits the approval gate ahead of the session
        # check — toggling "OFF" would be a false safety claim. Say so instead.
        if _YOLO_MODE_FROZEN:
            _cprint(
                f"  ⚡ YOLO is {_Colors.BOLD}{_Colors.RED}locked ON{_Colors.RESET}"
                " for this process (started with --yolo / HERMES_YOLO_MODE)."
                " /yolo cannot disable it — restart without the flag to"
                " re-enable approvals.")
            return

        session_key = self.session_id or "default"
        # getattr: tests call this unbound against a minimal stand-in; persistence is best-effort.
        _persist = getattr(self, "_persist_session_yolo", None)
        if is_session_yolo_enabled(session_key):
            disable_session_yolo(session_key)
            if _persist:
                _persist(session_key, False)
            _cprint(
                f"  ⚠ YOLO mode {_Colors.BOLD}{_Colors.RED}OFF{_Colors.RESET}"
                " — dangerous commands will require approval.")
        else:
            enable_session_yolo(session_key)
            if _persist:
                _persist(session_key, True)
            _cprint(
                f"  ⚡ YOLO mode {_Colors.BOLD}{_Colors.GREEN}ON{_Colors.RESET}"
                " — all commands auto-approved. Use with caution.")

    def _persist_session_yolo(self, session_key: str, enabled: bool) -> None:
        """Persist the YOLO flag to the session row so --resume restores it. Best-effort; the
        in-memory toggle is authoritative. Skipped without a store or before the row exists
        (rows are created lazily on the first turn)."""
        db = getattr(self, "_session_db", None)
        if db is None or not session_key or session_key == "default":
            return
        with contextlib.suppress(Exception):
            db.set_session_yolo(session_key, enabled)

    def _manual_compress(self, cmd_original: str = ""):
        """Manually trigger context compression.

        * ``/compress [<focus>]`` — compress the whole history; an optional focus topic tells
          the summariser what to preserve while discarding the rest more aggressively.
        * ``/compress here [N]`` — boundary-aware: summarize everything except the most recent
          ``N`` exchanges (default 2), kept verbatim.
        No ``compression_enabled`` gate: that flag disables *automatic* compaction only, and
        the context-overflow error path directs users here when it is off.
        """
        if len(self.conversation_history or ()) < 4:
            print("(._.) Not enough conversation to compress (need at least 4 messages).")
            return
        if not self.agent:
            print("(._.) No active agent -- send a message first.")
            return

        from hermes_cli.partial_compress import (
            extract_compress_flags, parse_partial_compress_args, rejoin_compressed_head_and_tail,
            split_history_for_partial_compress, summarize_compress_preview)
        from agent.conversation_compression import finalize_context_engine_compression_notification
        from agent.model_metadata import estimate_request_tokens_rough

        _parts = (cmd_original or "").strip().split(None, 1)
        raw_args = _parts[1].strip() if len(_parts) > 1 else ""
        # Strip --preview/--dry-run/--aggressive before positional parsing.
        raw_args, preview, aggressive = extract_compress_flags(raw_args)
        partial, keep_last, focus_topic = parse_partial_compress_args(raw_args)
        focus_topic = focus_topic or ""

        if aggressive:
            # LLM-free hard truncation would need its own persistence path outside the
            # guarded _compress_context rotation; surface that instead of mis-parsing.
            print("(._.) --aggressive is not supported; use '/compress here [N]' "
                  "to keep only recent exchanges, or /undo to drop turns.")
            if not preview:
                return

        # Include system prompt + tool schemas in estimates — a transcript-only number
        # understates real request pressure and can even appear to grow after compression.
        _estimate_kw = {
            "system_prompt": getattr(self.agent, "_cached_system_prompt", "") or "",
            "tools": getattr(self.agent, "tools", None) or None}
        if preview:
            approx_tokens = estimate_request_tokens_rough(self.conversation_history, **_estimate_kw)
            report = summarize_compress_preview(
                self.conversation_history, partial, keep_last, focus_topic or None, approx_tokens)
            for line in report["lines"]:
                print(f"🗜️  {line}")
            return

        original_count = len(self.conversation_history)
        with self._busy_command("Compressing context...", blocks_input=False):
            try:
                from agent.manual_compression_feedback import summarize_manual_compression
                original_history = list(self.conversation_history)

                # Boundary-aware split: only the head is summarized. A degenerate split
                # (nothing to keep / no head) falls back to full compression.
                tail: list = []
                head = original_history
                if partial:
                    head, tail = split_history_for_partial_compress(original_history, keep_last)
                    if not tail:
                        partial = False
                        head = original_history

                approx_tokens = estimate_request_tokens_rough(original_history, **_estimate_kw)
                if partial:
                    print(f"🗜️  Summarizing up to here: compressing {len(head)} of "
                          f"{original_count} messages (~{approx_tokens:,} tokens), "
                          f"keeping last {keep_last} exchange(s) verbatim...")
                elif focus_topic:
                    print(f"🗜️  Compressing {original_count} messages (~{approx_tokens:,} tokens), "
                          f"focus: \"{focus_topic}\"...")
                else:
                    print(f"🗜️  Compressing {original_count} messages (~{approx_tokens:,} tokens)...")

                # system_message=None so _compress_context rebuilds the prompt from scratch;
                # passing _cached_system_prompt duplicated the identity block.
                # Passing _cached_system_prompt caused duplication because _build_system_prompt appends
                # system_message to prompt_parts which already contain the agent identity — resulting in the
                # identity block appearing twice (issue #15281).
                compressed, _ = self.agent._compress_context(
                    head, None, approx_tokens=approx_tokens, focus_topic=focus_topic or None,
                    force=True, defer_context_engine_notification=True)

                # Unchanged because a concurrent compression lock is held: say so instead of
                # the misleading "No changes" no-op text. Type-pinned check (is True / str) —
                # a bare truthiness test is fooled by MagicMock auto-attributes on test doubles.
                _lock_skip_signal = getattr(self.agent, "_compression_skipped_due_to_lock", None)
                if _lock_skip_signal is True or isinstance(_lock_skip_signal, str):
                    from agent.manual_compression_feedback import describe_compression_lock_skip
                    print(
                        "  " + describe_compression_lock_skip(self.agent._compression_skipped_due_to_lock)
                    )
                    self.agent._compression_skipped_due_to_lock = None
                    # No boundary committed → discard the deferred notification (exactly-once).
                    finalize_context_engine_compression_notification(self.agent, committed=False)
                    return

                if partial and tail:
                    compressed = rejoin_compressed_head_and_tail(compressed, tail)
                self.conversation_history = compressed
                # _compress_context ends the old session and creates a child session on the
                # agent. Sync the CLI's session_id so /status, /resume, exit summary and title
                # generation point at the live continuation, not the ended parent.
                agent_sid = getattr(self.agent, "session_id", None)
                if agent_sid and agent_sid != self.session_id:
                    self.session_id = self.agent.session_id
                    self._write_terminal_breadcrumb()
                    self._pending_title = None
                    # Persist the new handoff from offset 0 so resume can recover it after exit.
                    self.agent._flush_messages_to_session_db(self.conversation_history, None)
                finalize_context_engine_compression_notification(self.agent, committed=True)
                new_tokens = estimate_request_tokens_rough(
                    self.conversation_history, **_estimate_kw)
                summary = summarize_manual_compression(
                    original_history, self.conversation_history, approx_tokens, new_tokens,
                    compression_state=getattr(self.agent, "context_compressor", None))
                if (
                    summary.get("aborted")
                    or summary.get("fallback_used")
                    or summary.get("refused_would_grow")):
                    icon = "⚠️"
                else:
                    icon = "🗜️" if summary["noop"] else "✅"
                print(f"  {icon} {summary['headline']}")
                print(f"     {summary['token_line']}")
                if summary["note"]:
                    print(f"     {summary['note']}")
            except Exception as e:
                finalize_context_engine_compression_notification(self.agent, committed=False)
                print(f"  ❌ Compression failed: {e}")

    def _persist_prompt_summary(self, icon: str, label: str, detail: str, outcome: str) -> None:
        """Print a one-line scrollback summary of a resolved modal prompt (approval/clarify
        panels vanish on repaint); gated by ``display.persist_prompts``."""
        from cli import CLI_CONFIG, _DIM, _RST, _cprint
        if not CLI_CONFIG.get("display", {}).get("persist_prompts", True):
            return
        detail, outcome = (_squash(s) for s in (detail, outcome))
        _cprint(f"\n{_DIM}{icon} {label}: {detail} → {outcome}{_RST}")

    def _clear_terminal_on_exit(self):
        """Clear screen + scrollback (``ESC[3J ESC[2J ESC[H``) so nothing is stranded above
        the exit summary. Only safe after ``app.run()`` returned and prompt_toolkit restored
        terminal modes. Skips when stdout isn't a console; falls back to ``clear``/``cls``."""
        try:
            stream = sys.stdout
            if stream is None or not stream.isatty():
                return
        except Exception:
            return
        try:
            stream.write("\033[3J\033[2J\033[H")
            stream.flush()
        except Exception:
            try:
                os.system("cls" if os.name == "nt" else "clear")
            except Exception:
                pass

    def _persist_active_session_before_close(self):
        """Best-effort flush of the agent's live ``_session_messages`` before ``end_session()``
        — a terminal close/SIGHUP can unwind the app while the agent thread still holds the
        current turn only in memory."""
        from cli import logger
        agent = getattr(self, "agent", None)
        if not agent or not hasattr(agent, "_persist_session"):
            return

        persist_lock = getattr(agent, "_session_persist_lock", None)

        def _snapshot_and_persist() -> None:
            # Must share the staging lock with ``chat()``: otherwise close can retain a
            # history baseline just before chat appends its pending dict, and the later flush
            # stamps that dict durable without writing a row.
            messages = getattr(agent, "_session_messages", None)
            pending_cli_message = getattr(agent, "_pending_cli_user_message", None)
            if not isinstance(messages, list):
                messages = getattr(self, "conversation_history", None)
            if not isinstance(messages, list):
                return
            if isinstance(pending_cli_message, dict) and not any(
                m is pending_cli_message for m in messages):
                # The UI accepted a new input but the worker still exposes its prior snapshot.
                messages = [*messages, pending_cli_message]
            if not messages:
                return

            # Baseline: the CLI history a normal turn built its new list from, so a signal
            # between assigning ``_session_messages`` and the DB flush cannot append the
            # durable prefix twice. When both names alias the same live list, marker-only
            # persistence would mark an unflushed tail durable — pass None instead.
            conversation_history = getattr(self, "conversation_history", None)
            if (
                isinstance(conversation_history, list)
                and conversation_history
                and conversation_history[-1] is pending_cli_message):
                # Accepted but not yet durable: exclude it from the resumed-history baseline.
                conversation_history = conversation_history[:-1]
            elif not isinstance(conversation_history, list) or conversation_history is messages:
                conversation_history = None

            # A first-turn close can precede the cached prompt; build it so the durable
            # transcript never gets a NULL system_prompt cache entry.
            if getattr(agent, "_cached_system_prompt", None) is None:
                try:
                    from agent.conversation_loop import _restore_or_build_system_prompt

                    _restore_or_build_system_prompt(agent, None, conversation_history)
                except Exception:
                    logger.debug("Could not build system prompt during CLI close", exc_info=True)
                    return
            if getattr(agent, "_cached_system_prompt", None) is None:
                return

            agent._ensure_db_session()
            agent._persist_session(messages, conversation_history)
            if getattr(agent, "session_id", None):
                self.session_id = agent.session_id
                self._write_terminal_breadcrumb()

        try:
            # Create the DB session row now that _cached_system_prompt is populated, so the persisted
            # snapshot is written non-NULL on the first turn (Issue #45499). Idempotent:
            # _ensure_db_session() no-ops once the row exists. Must run BEFORE preflight compression:
            # in-place compaction inserts message rows referencing this session (archive_and_compact), and
            # rotation creates a child with parent_session_id pointing at it — with PRAGMA foreign_keys=ON,
            # a missing parent row fails both INSERTs on a fresh oversized first turn. The user-turn crash
            # persist itself runs LATER (after memory prefetch / pre_llm_call), so the row is written once
            # with its final api_content — both steps take the same per-agent persist lock as CLI close
            # persistence.
            if persist_lock is None:
                _snapshot_and_persist()
            else:
                with persist_lock:
                    _snapshot_and_persist()
        except (Exception, KeyboardInterrupt) as e:
            logger.debug("Could not persist active CLI session before close: %s", e)

    def _print_exit_summary(self, clear_screen: bool = True):
        """Print session resume info on exit. ``clear_screen`` (interactive TUI teardown)
        wipes screen + scrollback first; single-query mode passes False to keep the answer.

        Args: clear_screen: When True (default), clear the terminal screen and scrollback before printing
        the summary. See #38252, #53009.
        """
        from cli import datetime
        if clear_screen:
            # Clear the screen + scrollback before printing the summary so the live bottom chrome (status
            # bar, input box, separator rules) and the rest of the session transcript don't get stranded
            # above the exit summary (#38252). By this point app.run() has returned and prompt_toolkit has
            # restored terminal modes, so writing raw escapes to stdout is safe. ESC[3J clears scrollback,
            # ESC[2J clears the visible screen, ESC[H homes the cursor — so the summary prints at a clean
            # top-left. Falls back to the platform clear command if stdout isn't a TTY-capable stream.
            # Honors NO_COLOR/dumb terminals by skipping silently when there's no real console.
            self._clear_terminal_on_exit()
        print()
        msg_count = len(self.conversation_history)
        if not msg_count:
            try:
                from hermes_cli.skin_engine import get_active_goodbye
                goodbye = get_active_goodbye("Goodbye! ⚕")
            except Exception:
                goodbye = "Goodbye! ⚕"
            print(goodbye)
            return

        user_msgs = len([m for m in self.conversation_history if m.get("role") == "user"])
        tool_calls = len([
            m for m in self.conversation_history if m.get("role") == "tool" or m.get("tool_calls")])
        elapsed = datetime.now() - self.session_start
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{seconds}s"
        if hours > 0:
            duration_str = f"{hours}h {minutes}m {duration_str}"
        elif minutes > 0:
            duration_str = f"{minutes}m {duration_str}"

        session_title = None
        if self._session_db:
            with contextlib.suppress(Exception):
                session_title = self._session_db.get_session_title(self.session_id)

        print("Resume this session with:")
        # Session IDs are profile-constrained: non-default profiles need `-p <profile>` in
        # the hint ("default"/"custom" use the standard HERMES_HOME).
        try:
            from hermes_cli.profiles import get_active_profile_name
            _active_profile = get_active_profile_name()
        except Exception:
            _active_profile = "default"
        profile_flag = "" if _active_profile in ("default", "custom") else f" -p {_active_profile}"
        print(f"  hermes --resume {self.session_id}{profile_flag}")
        if session_title:
            print(f"  hermes -c \"{session_title}\"{profile_flag}")
        print()
        print(f"Session:        {self.session_id}")
        if session_title:
            print(f"Title:          {session_title}")
        print(f"Duration:       {duration_str}")
        print(f"Messages:       {msg_count} ({user_msgs} user, {tool_calls} tool calls)")
