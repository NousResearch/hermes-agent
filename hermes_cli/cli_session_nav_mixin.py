"""Session navigation commands for the Hermes CLI: handoff, resume, sessions, branch.

Extracted from ``hermes_cli/cli_commands_mixin.py`` as part of the god-file
decomposition campaign.

``CLICommandsMixin`` is itself a product of that campaign, lifted out of
``cli.py``, and has since grown to 3,218 lines and 54 methods, so it is now a
decomposition target of its own. This takes the one cluster in it that is
about moving between sessions rather than acting inside the current one:
handing the session to a gateway platform, resuming an earlier one, listing
what can be resumed, and forking a branch.

Mixin contract: a plain mixin, mixed into ``CLICommandsMixin`` and so reaching
``HermesCLI`` through the existing MRO. It defines no ``__init__`` and no state
of its own; the host's attributes and methods resolve through the MRO. It never
imports ``cli`` or ``hermes_cli.cli_commands_mixin`` at module level, so there
is no cycle.

The lifted methods reach ``cli`` module state the way they already did, through
lazy imports inside the method bodies (``from cli import _cprint``,
``from cli import _sync_process_session_id``). Those resolve from ``cli``'s
namespace at call time, which is what keeps ``patch("cli._cprint")`` working
from here exactly as it did before.

Behavior-neutral: every method is lifted verbatim.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

from agent.turn_context import extract_api_content_sidecar


class CLISessionNavigationMixin:
    """See module docstring - session-navigation cluster lifted verbatim."""

    def _handle_handoff_command(self, cmd_original: str) -> bool:
        """Handle ``/handoff <platform>`` — transfer this CLI session to a gateway platform.

        Flow:
          1. Validate platform name + the gateway has a home channel for it.
          2. Reject if the agent is currently running (the in-flight turn
             would race with the gateway's switch_session).
          3. Write ``handoff_state='pending'`` on this session row.
          4. Block-poll ``state.db`` for terminal state (timeout 60s).
          5. On ``completed`` → print resume hint and signal CLI exit by
             returning False (the caller honors that like ``/quit``).
          6. On ``failed`` / timeout → print error and return True so the
             user keeps their CLI session.

        Returns:
            False to signal CLI exit, True to keep going.
        """
        from cli import _cprint
        from hermes_state import format_session_db_unavailable

        parts = cmd_original.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            _cprint("  Usage: /handoff <platform>")
            _cprint("  Hands the current session off to that platform's home channel.")
            _cprint("  The CLI session ends here; resume it later with /resume.")
            return True

        platform_name = parts[1].strip().lower()

        # Validate platform name + home channel via the live gateway config.
        try:
            from gateway.config import load_gateway_config, Platform
        except Exception as exc:  # pragma: no cover — gateway pkg always shipped
            _cprint(f"  Could not load gateway config: {exc}")
            return True

        try:
            platform = Platform(platform_name)
        except (ValueError, KeyError):
            _cprint(f"  Unknown platform '{platform_name}'.")
            return True

        try:
            gw_config = load_gateway_config()
        except Exception as exc:
            _cprint(f"  Could not load gateway config: {exc}")
            return True

        pcfg = gw_config.platforms.get(platform)
        if not pcfg or not pcfg.enabled:
            # Relay aliasing: a relay-fronted gateway has no per-platform
            # config block for the logical platform ("discord" etc.) — only a
            # RELAY entry — yet /handoff discord is deliverable when the relay
            # fronts it. The fronted set is deploy config
            # (GATEWAY_RELAY_PLATFORMS), readable here without the live
            # adapter; the gateway watcher re-checks against the authenticated
            # transport (resolve_delivery_transport) before dispatch, so this
            # is a UX pre-check, not the security gate.
            relay_fronts = False
            try:
                from gateway.relay import relay_platform_identities
                relay_cfg = gw_config.platforms.get(Platform.RELAY)
                if relay_cfg and relay_cfg.enabled:
                    fronted = {p for p, _ in relay_platform_identities()}
                    relay_fronts = platform_name in fronted
            except Exception:
                relay_fronts = False
            if not relay_fronts:
                _cprint(f"  Platform '{platform_name}' is not configured/enabled in the gateway.")
                return True

        home = gw_config.get_home_channel(platform)
        if not home or not home.chat_id:
            _cprint(f"  No home channel configured for {platform_name}.")
            _cprint("  Set one with /sethome on the destination chat first.")
            return True

        # Refuse mid-turn: an in-flight agent run would race with the
        # gateway's switch_session and the synthetic turn dispatch.
        if getattr(self, "_agent_running", False):
            _cprint("  Agent is busy. Wait for the current turn to finish, then retry /handoff.")
            return True

        # Make sure we have a SessionDB handle.
        if not self._session_db:
            try:
                from hermes_state import SessionDB
                self._session_db = SessionDB()
            except Exception:
                pass
        if not self._session_db:
            _cprint(f"  {format_session_db_unavailable()}")
            return True

        # Make sure the session row exists in state.db. Most CLI sessions
        # are written via _flush_messages_to_session_db on the first turn
        # already, but if the user tries to hand off an empty session we
        # still want a row to mark.
        try:
            row = self._session_db.get_session(self.session_id)
            if not row:
                # Nothing has flushed yet. Create a stub so the gateway has
                # something to switch_session onto. Inserting via title-set
                # is the simplest path because set_session_title's INSERT OR
                # IGNORE creates the row.
                placeholder_title = f"handoff-{self.session_id[:8]}"
                self._session_db.set_session_title(self.session_id, placeholder_title)
        except Exception as exc:
            _cprint(f"  Could not ensure session row in state.db: {exc}")
            return True

        # Display title for messaging.
        session_title = ""
        try:
            row = self._session_db.get_session(self.session_id)
            if row:
                session_title = row.get("title") or ""
        except Exception:
            pass
        if not session_title:
            session_title = self.session_id[:8]

        # Mark pending — gateway watcher will pick this up.
        ok = self._session_db.request_handoff(self.session_id, platform_name)
        if not ok:
            _cprint("  Session is already in flight for handoff. Wait for it to settle, then retry.")
            return True

        _cprint(f"  Queued handoff of '{session_title}' → {platform_name} (home: {home.name}).")
        _cprint("  Waiting for the gateway to pick it up...")

        # Poll-block on terminal state. Tick every 0.5s; bail at ~60s.
        import time as _time
        deadline = _time.time() + 60.0
        last_state = "pending"
        while _time.time() < deadline:
            try:
                state_row = self._session_db.get_handoff_state(self.session_id)
            except Exception:
                state_row = None
            current = (state_row or {}).get("state") or "pending"
            if current != last_state:
                if current == "running":
                    _cprint("  Gateway picked it up; transferring...")
                last_state = current
            if current == "completed":
                _cprint("")
                _cprint(f"  ↻ Handoff complete. The session is now active on {platform_name}.")
                _cprint(f"  Resume it on this CLI later with: /resume {session_title}")
                _cprint("")
                # End the CLI cleanly — same exit semantics as /quit.
                self._should_exit = True
                return False
            if current == "failed":
                err = (state_row or {}).get("error") or "unknown error"
                _cprint(f"  Handoff failed: {err}")
                _cprint("  Your CLI session is intact. Try /handoff again, or /resume on the platform manually.")
                return True
            _time.sleep(0.5)

        # Timed out. Clear the pending flag so the user can retry.
        try:
            self._session_db.fail_handoff(self.session_id, "timed out waiting for gateway")
        except Exception:
            pass
        _cprint("  Timed out waiting for the gateway. Is `hermes gateway` running?")
        _cprint("  Your CLI session is intact.")
        return True

    def _handle_resume_command(self, cmd_original: str) -> None:
        """Handle /resume <session_id_or_title> — switch to a previous session mid-conversation."""
        from cli import _cprint, _sync_process_session_id
        parts = cmd_original.split(None, 1)
        target = parts[1].strip() if len(parts) > 1 else ""

        # Strip common outer brackets/quotes users may type literally from the
        # usage hint (e.g. ``/resume <abc123>`` or ``/resume [abc123]``).  The
        # `/resume` help text shows angle brackets as a placeholder and a few
        # users copy them through verbatim.  Stripping them keeps the lookup
        # working without changing the help string.
        if len(target) >= 2 and (
            (target[0] == "<" and target[-1] == ">")
            or (target[0] == "[" and target[-1] == "]")
            or (target[0] == '"' and target[-1] == '"')
            or (target[0] == "'" and target[-1] == "'")
        ):
            target = target[1:-1].strip()

        if not target:
            _cprint("  Usage: /resume <number|session_id_or_title>")
            if self._show_recent_sessions(reason="resume"):
                # Arm a one-shot pending-resume selection so the user can type
                # just the number (`3`) on the next line instead of having to
                # retype `/resume 3`. The list here must match the one shown by
                # _show_recent_sessions and used for index resolution below —
                # all three go through _list_recent_sessions(limit=10). See
                # #34584.
                self._pending_resume_sessions = self._list_recent_sessions(limit=10)
                return
            _cprint("  Tip:   Use /history or `hermes sessions list` to find sessions.")
            return

        # Any explicit /resume <target> supersedes a previously-armed bare
        # numbered prompt.
        self._pending_resume_sessions = None

        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            _cprint(f"  {format_session_db_unavailable()}")
            return

        # Resolve numbered selection, title, or ID
        if target.isdigit():
            sessions = self._list_recent_sessions(limit=10)
            index = int(target)
            if index < 1 or index > len(sessions):
                _cprint(f"  Resume index {index} is out of range.")
                _cprint("  Use /resume with no arguments to see available sessions.")
                return
            selected = sessions[index - 1]
            target_id = selected["id"]
        else:
            from hermes_cli.main import _resolve_session_by_name_or_id
            resolved = _resolve_session_by_name_or_id(target)
            target_id = resolved or target

        session_meta = self._session_db.get_session(target_id)
        if not session_meta:
            _cprint(f"  Session not found: {target}")
            _cprint("  Use /history or `hermes sessions list` to see available sessions.")
            return

        # If the target is the empty head of a compression chain, redirect to
        # the descendant that actually holds the transcript. See #15000.
        try:
            resolved_id = self._session_db.resolve_resume_session_id(target_id)
        except Exception:
            resolved_id = target_id
        if resolved_id and resolved_id != target_id:
            _cprint(
                f"  Session {target_id} was compressed into {resolved_id}; "
                f"resuming the descendant with your transcript."
            )
            target_id = resolved_id
            resolved_meta = self._session_db.get_session(target_id)
            if resolved_meta:
                session_meta = resolved_meta

        if target_id == self.session_id:
            _cprint("  Already on that session.")
            return

        old_session_id = self.session_id
        # Flush un-persisted messages before ending the old session (#47202).
        if self.agent:
            try:
                self.agent._flush_messages_to_session_db(
                    self.conversation_history,
                    conversation_history=self.conversation_history,
                )
            except Exception:
                pass
        # End current session
        try:
            self._session_db.end_session(self.session_id, "resumed_other")
        except Exception:
            pass

        # Switch to the target session
        self.session_id = target_id
        self._resumed = True
        self._pending_title = None
        _sync_process_session_id(target_id)

        # Load conversation history (strip transcript-only metadata entries).
        # repair_alternation: this /resume feeds LIVE REPLAY — ``restored``
        # becomes ``self.conversation_history`` for subsequent turns. Heal a
        # durable ``user;user`` violation once here instead of re-firing the
        # pre-request repair on every request for the rest of the session.
        #
        # Both projections come from one lineage SELECT: model_history is
        # alternation-repaired for live replay; display_history is the full
        # lineage verbatim, used by _display_resumed_history() so timeline
        # events and ancestor rows render correctly (matching the startup
        # --resume path in _preload_resumed_session).
        model_history, display_history = self._session_db.get_resume_conversations(
            target_id
        )
        restored = [m for m in (model_history or []) if m.get("role") != "session_meta"]
        self.conversation_history = restored
        self._resume_display_history = [
            m for m in (display_history or []) if m.get("role") != "session_meta"
        ]

        # Re-open the target session so it's not marked as ended
        try:
            self._session_db.reopen_session(target_id)
        except Exception:
            pass

        # Sync the agent if already initialised
        if self.agent:
            self.agent.session_id = target_id
            self.agent.reset_session_state()
            if hasattr(self.agent, "_last_flushed_db_idx"):
                self.agent._last_flushed_db_idx = len(self.conversation_history)
            if hasattr(self.agent, "_todo_store"):
                try:
                    from tools.todo_tool import TodoStore
                    self.agent._todo_store = TodoStore()
                except Exception:
                    pass
            if hasattr(self.agent, "_invalidate_system_prompt"):
                self.agent._invalidate_system_prompt()

            # Notify memory providers that session_id rotated to a resumed
            # session. reset=False — the provider's accumulated state is
            # still valid; it just needs to target the new session_id for
            # subsequent writes. See #6672.
            try:
                _mm = getattr(self.agent, "_memory_manager", None)
                if _mm is not None:
                    _mm.on_session_switch(
                        target_id,
                        parent_session_id=old_session_id or "",
                        reset=False,
                        reason="resume",
                    )
            except Exception:
                pass

        title_part = f" \"{session_meta['title']}\"" if session_meta.get("title") else ""
        msg_count = len([m for m in self._resume_display_history if m.get("role") == "user" and not m.get("display_kind")])
        if self.conversation_history:
            _cprint(
                f"  ↻ Resumed session {target_id}{title_part}"
                f" ({msg_count} user message{'s' if msg_count != 1 else ''},"
                f" {len(self.conversation_history)} total)"
            )
            self._display_resumed_history()
        else:
            _cprint(f"  ↻ Resumed session {target_id}{title_part} — no messages, starting fresh.")

        # Retarget the process + tool cwd to where the session was started, so a
        # mid-chat /resume (and /sessions <id>, which delegates here) lands in the
        # same directory as a startup `hermes -c`/`--resume`. The startup resume
        # paths already call this; without it, the terminal/code-exec tools and
        # relative-path resolution keep operating in the wrong repo. Idempotent
        # and a no-op when the session recorded no cwd. See #38562.
        self._restore_session_cwd(session_meta)

    def _handle_sessions_command(self, cmd_original: str) -> None:
        """Handle /sessions [list|<id_or_title>] — browse or resume previous sessions.

        Without arguments, prints the same recent-sessions table that /resume
        shows when called without a target, and tells the user how to resume.
        With an explicit subcommand or target, delegates to the resume flow so
        ``/sessions <id>`` and ``/resume <id>`` behave identically.

        The TUI ships an interactive picker overlay for this command; the
        classic CLI prints an inline list because there is no equivalent
        overlay primitive here. Without this handler the canonical name
        ``sessions`` falls through ``process_command``'s elif chain and
        prints ``Unknown command: sessions`` even though the command is
        registered in the central COMMAND_REGISTRY.
        """
        from cli import _cprint
        parts = cmd_original.split(None, 1)
        arg = parts[1].strip() if len(parts) > 1 else ""
        sub = arg.lower()

        # Bare /sessions or /sessions list — show recent sessions inline.
        if not arg or sub in {"list", "ls", "browse"}:
            if not self._session_db:
                from hermes_state import format_session_db_unavailable
                _cprint(f"  {format_session_db_unavailable()}")
                return
            if not self._show_recent_sessions(reason="sessions"):
                _cprint("  (._.) No previous sessions yet.")
            return

        # /sessions <id_or_title> behaves the same as /resume <id_or_title>.
        self._handle_resume_command(f"/resume {arg}")

    def _handle_branch_command(self, cmd_original: str) -> None:
        """Handle /branch [name] — fork the current session into a new independent copy.

        Copies the full conversation history to a new session so the user can
        explore a different approach without losing the original session state.
        Inspired by Claude Code's /branch command.
        """
        from cli import _cprint, _sync_process_session_id
        if not self.conversation_history:
            _cprint("  No conversation to branch — send a message first.")
            return

        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            _cprint(f"  {format_session_db_unavailable()}")
            return

        parts = cmd_original.split(None, 1)
        branch_name = parts[1].strip() if len(parts) > 1 else ""

        # Generate the new session ID
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        new_session_id = f"{timestamp_str}_{short_uuid}"

        # Determine branch title
        if branch_name:
            branch_title = branch_name
        else:
            # Auto-generate from the current session title
            current_title = None
            if self._session_db:
                current_title = self._session_db.get_session_title(self.session_id)
            base = current_title or "branch"
            branch_title = self._session_db.get_next_title_in_lineage(base)

        # Save the current session's state before branching
        parent_session_id = self.session_id

        # Flush un-persisted messages before ending the old session (#47202).
        if self.agent:
            try:
                self.agent._flush_messages_to_session_db(
                    self.conversation_history,
                    conversation_history=self.conversation_history,
                )
            except Exception:
                pass

        # End the old session
        try:
            self._session_db.end_session(self.session_id, "branched")
        except Exception:
            pass

        # Create the new session with parent link.
        # Persist a stable ``_branched_from`` marker in model_config so
        # list_sessions_rich() can keep the branch visible in /resume and
        # /sessions even after the parent is reopened and re-ended with a
        # different end_reason (e.g. tui_shutdown overwriting 'branched').
        try:
            self._session_db.create_session(
                session_id=new_session_id,
                source=os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                model=self.model,
                model_config={
                    "max_iterations": self.max_turns,
                    "reasoning_config": self.reasoning_config,
                    "_branched_from": parent_session_id,
                },
                parent_session_id=parent_session_id,
            )
        except Exception as e:
            _cprint(f"  Failed to create branch session: {e}")
            return

        # Copy conversation history to the new session
        for msg in self.conversation_history:
            try:
                self._session_db.append_message(
                    session_id=new_session_id,
                    role=msg.get("role", "user"),
                    content=msg.get("content"),
                    tool_name=msg.get("tool_name") or msg.get("name"),
                    tool_calls=msg.get("tool_calls"),
                    tool_call_id=msg.get("tool_call_id"),
                    reasoning=msg.get("reasoning"),
                    # Keep the api_content sidecar so the branch's first turn
                    # replays the parent's exact wire bytes (warm provider
                    # prompt cache) instead of a full cold prefill.
                    api_content=extract_api_content_sidecar(msg),
                    timestamp=msg.get("timestamp"),
                )
            except Exception:
                pass  # Best-effort copy

        # Set title on the branch
        try:
            self._session_db.set_session_title(new_session_id, branch_title)
        except Exception:
            pass

        # Switch to the new session
        self._transfer_session_yolo(self.session_id, new_session_id)
        self.session_id = new_session_id
        self.session_start = now
        self._pending_title = None
        self._resumed = True  # Prevents auto-title generation
        _sync_process_session_id(new_session_id)

        # Sync the agent
        if self.agent:
            self.agent.session_id = new_session_id
            self.agent.session_start = now
            self.agent.reset_session_state()
            if hasattr(self.agent, "_last_flushed_db_idx"):
                self.agent._last_flushed_db_idx = len(self.conversation_history)
            if hasattr(self.agent, "_todo_store"):
                try:
                    from tools.todo_tool import TodoStore
                    self.agent._todo_store = TodoStore()
                except Exception:
                    pass
            if hasattr(self.agent, "_invalidate_system_prompt"):
                self.agent._invalidate_system_prompt()

            # Notify memory providers that session_id forked to a new branch.
            # reset=False — the branched session carries the transcript
            # forward, so provider state tracks the lineage. parent_session_id
            # links the branch back to the original. See #6672.
            try:
                _mm = getattr(self.agent, "_memory_manager", None)
                if _mm is not None:
                    _mm.on_session_switch(
                        new_session_id,
                        parent_session_id=parent_session_id or "",
                        reset=False,
                        reason="branch",
                    )
            except Exception:
                pass

        msg_count = len([m for m in self.conversation_history if m.get("role") == "user"])
        _cprint(
            f"  ⑂ Branched session \"{branch_title}\""
            f" ({msg_count} user message{'s' if msg_count != 1 else ''})"
        )
        _cprint(f"  Original session: {parent_session_id}")
        _cprint(f"  Branch session:   {new_session_id}")
