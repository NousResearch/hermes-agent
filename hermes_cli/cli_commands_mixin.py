"""Slash-command handlers for the interactive CLI (god-file decomposition Phase 4).

This module hosts the ``_handle_*_command`` slash-command handlers lifted out of
``cli.py``'s ``HermesCLI`` class. ``HermesCLI`` inherits ``CLICommandsMixin`` so
every ``self.<handler>`` call resolves unchanged via the MRO — behavior-neutral.

Import discipline (mirrors gateway/slash_commands.py, PR #41886):
  * Neutral, non-cyclic deps are imported at module top-level below.
  * cli.py-internal symbols (the ``_cprint``/``_ACCENT``/``save_config_value``…
    module-level helpers and constants) are imported LAZILY inside each handler
    via ``from cli import ...`` — that resolves at call time when ``cli`` is fully
    loaded, so the mixin module never imports ``cli`` at top level (no cycle).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from urllib.parse import urlparse

from rich import box as rich_box
from rich.markup import escape as _escape
from rich.panel import Panel

from hermes_constants import display_hermes_home, is_termux as _is_termux_environment
from agent.turn_context import extract_api_content_sidecar
from hermes_cli.browser_connect import (
    DEFAULT_BROWSER_CDP_URL,
    discover_local_cdp_url,
    find_free_debug_port,
    is_browser_debug_ready,
    launch_chrome_debug,
    local_port_in_use,
    manual_chrome_debug_command,
)


class CLICommandsMixin:
    """Mixin holding the interactive-CLI slash-command handlers.

    All methods use only ``self`` state plus the imports above and per-method
    lazy ``from cli import ...`` lines, so they compose cleanly onto
    ``HermesCLI`` via the MRO.
    """

    def _handle_rollback_command(self, command: str):
        """Handle /rollback — list, diff, or restore filesystem checkpoints.

        Syntax:
            /rollback                 — list checkpoints
            /rollback <N>             — restore checkpoint N (also undoes last chat turn)
            /rollback diff <N>        — preview changes since checkpoint N
            /rollback <N> <file>      — restore a single file from checkpoint N
        """
        from tools.checkpoint_manager import format_checkpoint_list

        if not hasattr(self, 'agent') or not self.agent:
            print("  No active agent session.")
            return

        mgr = self.agent._checkpoint_mgr
        if not mgr.enabled:
            print("  Checkpoints are not enabled.")
            print("  Enable with: hermes --checkpoints")
            print("  Or in config.yaml: checkpoints: { enabled: true }")
            return

        cwd = os.getenv("TERMINAL_CWD", os.getcwd())
        parts = command.split()
        args = parts[1:] if len(parts) > 1 else []

        if not args:
            # List checkpoints
            checkpoints = mgr.list_checkpoints(cwd)
            print(format_checkpoint_list(checkpoints, cwd))
            return

        # Handle /rollback diff <N>
        if args[0].lower() == "diff":
            if len(args) < 2:
                print("  Usage: /rollback diff <N>")
                return
            checkpoints = mgr.list_checkpoints(cwd)
            if not checkpoints:
                print(f"  No checkpoints found for {cwd}")
                return
            target_hash = self._resolve_checkpoint_ref(args[1], checkpoints)
            if not target_hash:
                return
            result = mgr.diff(cwd, target_hash)
            if result["success"]:
                stat = result.get("stat", "")
                diff = result.get("diff", "")
                if not stat and not diff:
                    print("  No changes since this checkpoint.")
                else:
                    if stat:
                        print(f"\n{stat}")
                    if diff:
                        # Limit diff output to avoid terminal flood
                        diff_lines = diff.splitlines()
                        if len(diff_lines) > 80:
                            print("\n".join(diff_lines[:80]))
                            print(f"\n  ... ({len(diff_lines) - 80} more lines, showing first 80)")
                        else:
                            print(f"\n{diff}")
            else:
                print(f"  ❌ {result['error']}")
            return

        # Resolve checkpoint reference (number or hash)
        checkpoints = mgr.list_checkpoints(cwd)
        if not checkpoints:
            print(f"  No checkpoints found for {cwd}")
            return

        target_hash = self._resolve_checkpoint_ref(args[0], checkpoints)
        if not target_hash:
            return

        # Check for file-level restore: /rollback <N> <file>
        file_path = args[1] if len(args) > 1 else None

        result = mgr.restore(cwd, target_hash, file_path=file_path)
        if result["success"]:
            if file_path:
                print(f"  ✅ Restored {file_path} from checkpoint {result['restored_to']}: {result['reason']}")
            else:
                print(f"  ✅ Restored to checkpoint {result['restored_to']}: {result['reason']}")
            print("  A pre-rollback snapshot was saved automatically.")

            # Also undo the last conversation turn so the agent's context
            # matches the restored filesystem state
            if self.conversation_history:
                self.undo_last(prefill=False)
                print("  Chat turn undone to match restored file state.")
        else:
            print(f"  ❌ {result['error']}")

    def _handle_snapshot_command(self, command: str):
        """Handle /snapshot — lightweight state snapshots for Hermes config/state.

        Syntax:
            /snapshot                  — list recent snapshots
            /snapshot create [label]   — create a snapshot
            /snapshot restore <id>     — restore state from snapshot
            /snapshot prune [N]        — prune to N snapshots (default 20)
        """
        from hermes_cli.backup import (
            create_quick_snapshot, list_quick_snapshots,
            restore_quick_snapshot, prune_quick_snapshots,
        )
        from hermes_constants import display_hermes_home

        parts = command.split()
        subcmd = parts[1].lower() if len(parts) > 1 else "list"

        if subcmd in {"list", "ls"}:
            snaps = list_quick_snapshots()
            if not snaps:
                print("  No state snapshots yet.")
                print("  Create one: /snapshot create [label]")
                return
            print(f"  State snapshots ({display_hermes_home()}/state-snapshots/):\n")
            print(f"  {'#':>3}  {'ID':<35} {'Files':>5} {'Size':>10} {'Label'}")
            print(f"  {'─'*3}  {'─'*35} {'─'*5} {'─'*10} {'─'*20}")
            for i, s in enumerate(snaps, 1):
                size = s.get("total_size", 0)
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.0f} KB"
                else:
                    size_str = f"{size / 1024 / 1024:.1f} MB"
                label = s.get("label") or ""
                print(f"  {i:3}  {s['id']:<35} {s.get('file_count', 0):>5} {size_str:>10} {label}")

        elif subcmd == "create":
            label = " ".join(parts[2:]) if len(parts) > 2 else None
            snap_id = create_quick_snapshot(label=label)
            if snap_id:
                print(f"  Snapshot created: {snap_id}")
            else:
                print("  No state files found to snapshot.")

        elif subcmd in {"restore", "rewind"}:
            if len(parts) < 3:
                print("  Usage: /snapshot restore <snapshot-id>")
                # Show hint with most recent snapshot
                snaps = list_quick_snapshots(limit=1)
                if snaps:
                    print(f"  Most recent: {snaps[0]['id']}")
                return
            snap_id = parts[2]
            # Allow restore by number (1-indexed)
            try:
                idx = int(snap_id)
                snaps = list_quick_snapshots()
                if 1 <= idx <= len(snaps):
                    snap_id = snaps[idx - 1]["id"]
                else:
                    print(f"  Invalid snapshot number. Use 1-{len(snaps)}.")
                    return
            except ValueError:
                pass
            if restore_quick_snapshot(snap_id):
                print(f"  Restored state from: {snap_id}")
                print("  Restart recommended for state.db changes to take effect.")
            else:
                print(f"  Snapshot not found: {snap_id}")

        elif subcmd == "prune":
            keep = 20
            if len(parts) > 2:
                try:
                    keep = int(parts[2])
                except ValueError:
                    print("  Usage: /snapshot prune [keep-count]")
                    return
            deleted = prune_quick_snapshots(keep=keep)
            print(f"  Pruned {deleted} old snapshot(s) (keeping {keep}).")

        else:
            print(f"  Unknown subcommand: {subcmd}")
            print("  Usage: /snapshot [list|create [label]|restore <id>|prune [N]]")

    def _handle_stop_command(self):
        """Handle /stop — kill all running background processes and
        background (async) delegations.

        Inspired by OpenAI Codex's separation of interrupt (stop current turn)
        from /stop (clean up background processes). See openai/codex#14602.
        """
        from tools.process_registry import process_registry

        processes = process_registry.list_sessions()
        running = [p for p in processes if p.get("status") == "running"]

        # Background subagents dispatched via delegate_task(background=true)
        # live in their own registry, not the process registry.
        try:
            from tools.async_delegation import active_count, interrupt_all
            n_async = active_count()
        except Exception:
            n_async = 0
            interrupt_all = None

        if not running and not n_async:
            print("  No running background processes.")
            return

        if running:
            print(f"  Stopping {len(running)} background process(es)...")
            killed = process_registry.kill_all()
            print(f"  ✅ Stopped {killed} process(es).")
        if n_async and interrupt_all is not None:
            stopped = interrupt_all(reason="/stop")
            print(f"  ✅ Interrupted {stopped} background delegation(s).")

    def _handle_agents_command(self):
        """Handle /agents — show background processes and agent status."""
        from cli import _cprint
        from tools.process_registry import format_uptime_short, process_registry

        processes = process_registry.list_sessions()
        running = [p for p in processes if p.get("status") == "running"]
        finished = [p for p in processes if p.get("status") != "running"]

        _cprint(f"  Running processes: {len(running)}")
        for p in running:
            cmd = p.get("command", "")[:80]
            up = format_uptime_short(p.get("uptime_seconds", 0))
            _cprint(f"    {p.get('session_id', '?')} · {up} · {cmd}")

        if finished:
            _cprint(f"  Recently finished: {len(finished)}")

        # Background (async) delegations — delegate_task(background=true)
        try:
            from tools.async_delegation import list_async_delegations
            delegations = list_async_delegations()
        except Exception:
            delegations = []
        running_d = [d for d in delegations if d.get("status") == "running"]
        if delegations:
            _cprint(f"  Background delegations: {len(running_d)} running")
            for d in delegations:
                goal = (d.get("goal") or "")[:60]
                _cprint(
                    f"    {d.get('delegation_id', '?')} · "
                    f"{d.get('status', '?')} · {goal}"
                )

        agent_running = getattr(self, "_agent_running", False)
        _cprint(f"  Agent: {'running' if agent_running else 'idle'}")

    def _handle_journey_command(self, cmd_original: str) -> None:
        """Handle /journey — the learning timeline (see `hermes journey`).

        The read-only views (default + ``list``) render Rich color, which
        patch_stdout would swallow as raw escapes; capture with forced ANSI and
        re-emit through ``_cprint``. ``delete``/``edit`` are interactive
        (confirm prompt / ``$EDITOR``) so they keep the real stdio.
        """
        import argparse
        import io
        import shlex
        from contextlib import redirect_stdout

        from cli import _cprint
        from hermes_cli.journey import register_cli

        parser = argparse.ArgumentParser(prog="/journey", add_help=False)
        register_cli(parser)
        rest = cmd_original.split(None, 1)
        try:
            args = parser.parse_args(shlex.split(rest[1]) if len(rest) > 1 else [])
        except SystemExit:
            return

        interactive = getattr(args, "journey_action", None) in ("delete", "edit")
        try:
            if interactive:
                args.func(args)
                return
            args.force_color = True
            buf = io.StringIO()
            with redirect_stdout(buf):
                args.func(args)
            _cprint(buf.getvalue().rstrip("\n"))
        except Exception as exc:
            _cprint(f"  /journey failed: {exc}")

    def _handle_paste_command(self):
        """Handle /paste — explicitly check clipboard for an image.

        This is the reliable fallback for terminals where BracketedPaste
        doesn't fire for image-only clipboard content (e.g., VSCode terminal,
        Windows Terminal with WSL2).
        """
        from cli import _DIM, _RST, _cprint, _termux_example_image_path
        if _is_termux_environment():
            _cprint(
                f"  {_DIM}Clipboard image paste is not available on Termux — "
                f"use /image <path> or paste a local image path like "
                f"{_termux_example_image_path()}{_RST}"
            )
            return

        from hermes_cli.clipboard import has_clipboard_image
        if has_clipboard_image():
            if self._try_attach_clipboard_image():
                n = len(self._attached_images)
                _cprint(f"  📎 Image #{n} attached from clipboard")
            else:
                _cprint(f"  {_DIM}(>_<) Clipboard has an image but extraction failed{_RST}")
        else:
            _cprint(f"  {_DIM}(._.) No image found in clipboard{_RST}")

    def _handle_copy_command(self, cmd_original: str) -> None:
        """Handle /copy [number] — copy assistant output to clipboard."""
        from cli import _assistant_copy_text, _cprint
        parts = cmd_original.split(maxsplit=1)
        arg = parts[1].strip() if len(parts) > 1 else ""

        assistant = [m for m in self.conversation_history if m.get("role") == "assistant"]
        if not assistant:
            _cprint("  Nothing to copy yet.")
            return

        if arg:
            try:
                idx = int(arg) - 1
            except ValueError:
                _cprint("  Usage: /copy [number]")
                return
            if idx < 0 or idx >= len(assistant):
                _cprint(f"  Invalid response number. Use 1-{len(assistant)}.")
                return
        else:
            idx = len(assistant) - 1
            while idx >= 0 and not _assistant_copy_text(assistant[idx].get("content")):
                idx -= 1
            if idx < 0:
                _cprint("  Nothing to copy in assistant responses yet.")
                return

        text = _assistant_copy_text(assistant[idx].get("content"))
        if not text:
            _cprint("  Nothing to copy in that assistant response.")
            return

        try:
            self._write_osc52_clipboard(text)
            _cprint(f"  Copied assistant response #{idx + 1} to clipboard")
        except Exception as e:
            _cprint(f"  Clipboard copy failed: {e}")

    def _handle_image_command(self, cmd_original: str):
        """Handle /image <path> — attach a local image file for the next prompt."""
        from cli import _DIM, _IMAGE_EXTENSIONS, _RST, _cprint, _resolve_attachment_path, _split_path_input, _termux_example_image_path
        raw_args = (cmd_original.split(None, 1)[1].strip() if " " in cmd_original else "")
        if not raw_args:
            hint = _termux_example_image_path() if _is_termux_environment() else "/path/to/image.png"
            _cprint(f"  {_DIM}Usage: /image <path>  e.g. /image {hint}{_RST}")
            return

        path_token, _remainder = _split_path_input(raw_args)
        image_path = _resolve_attachment_path(path_token)
        if image_path is None:
            _cprint(f"  {_DIM}(>_<) File not found: {path_token}{_RST}")
            return
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            _cprint(f"  {_DIM}(._.) Not a supported image file: {image_path.name}{_RST}")
            return

        self._attached_images.append(image_path)
        _cprint(f"  📎 Attached image: {image_path.name}")
        if _remainder:
            _cprint(f"  {_DIM}Now type your prompt (or use --image in single-query mode): {_remainder}{_RST}")
        elif _is_termux_environment():
            _cprint(f"  {_DIM}Tip: type your next message, or run hermes chat -q --image {_termux_example_image_path(image_path.name)} \"What do you see?\"{_RST}")

    def _handle_tools_command(self, cmd: str):
        """Handle /tools [list|disable|enable] slash commands.

        /tools (no args) shows the tool list.
        /tools list shows enabled/disabled status per toolset.
        /tools disable/enable saves the change to config and resets
        the session so the new tool set takes effect cleanly (no
        prompt-cache breakage mid-conversation).
        """
        from cli import _ACCENT, _DIM, _RST, _cprint
        import shlex
        from argparse import Namespace
        from contextlib import redirect_stdout
        from io import StringIO
        from hermes_cli.tools_config import tools_disable_enable_command

        def _run_capture(ns: Namespace) -> None:
            """Run tools_disable_enable_command, routing its ANSI-colored
            print() output through _cprint when inside the interactive TUI
            so escapes aren't mangled by patch_stdout's StdoutProxy into
            garbled '?[32m...?[0m' text.

            Outside the TUI (standalone mode, tests), call straight through
            so real stdout / pytest capture works as expected.
            """
            # Standalone/tests, run as usual
            if getattr(self, "_app", None) is None:
                tools_disable_enable_command(ns)
                return

            # Buffer reports isatty()=True so color() in hermes_cli/colors.py
            # still emits ANSI escapes. StringIO.isatty() is False, which
            # would otherwise strip all colors before we re-render them.
            class _TTYBuf(StringIO):
                def isatty(self) -> bool:
                    return True

            buf = _TTYBuf()
            with redirect_stdout(buf):
                tools_disable_enable_command(ns)
            for line in buf.getvalue().splitlines():
                _cprint(line)

        try:
            parts = shlex.split(cmd)
        except ValueError:
            parts = cmd.split()

        subcommand = parts[1] if len(parts) > 1 else ""
        if subcommand not in {"list", "disable", "enable"}:
            self.show_tools()
            return

        if subcommand == "list":
            _run_capture(Namespace(tools_action="list", platform="cli"))
            return

        names = parts[2:]
        if not names:
            print(f"(._.) Usage: /tools {subcommand} <name> [name ...]")
            print(f"  Built-in toolset:  /tools {subcommand} web")
            print(f"  MCP tool:          /tools {subcommand} github:create_issue")
            return

        # Apply the change directly — the user typing the command is implicit
        # consent.  Do NOT use input() here; it hangs inside prompt_toolkit's
        # TUI event loop (known pitfall).
        verb = "Disabling" if subcommand == "disable" else "Enabling"
        label = ", ".join(names)
        _cprint(f"{_ACCENT}{verb} {label}...{_RST}")

        _run_capture(Namespace(tools_action=subcommand, names=names, platform="cli"))

        # Reset session so the new tool config is picked up from a clean state
        from hermes_cli.tools_config import _get_platform_tools
        from hermes_cli.config import load_config
        self.enabled_toolsets = _get_platform_tools(load_config(), "cli")
        self.new_session()
        _cprint(f"{_DIM}Session reset. New tool configuration is active.{_RST}")

    def _handle_profile_command(self):
        """Display active profile name and home directory."""
        from hermes_constants import display_hermes_home
        from hermes_cli.profiles import get_active_profile_name

        display = display_hermes_home()
        profile_name = get_active_profile_name()

        print()
        print(f"  Profile: {profile_name}")
        print(f"  Home:    {display}")
        print()

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
                    self.conversation_history
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
        restored = self._session_db.get_messages_as_conversation(
            target_id, repair_alternation=True
        )
        restored = [m for m in (restored or []) if m.get("role") != "session_meta"]
        self.conversation_history = restored

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
        msg_count = len([m for m in self.conversation_history if m.get("role") == "user"])
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
                    self.conversation_history
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

    def _handle_personality_command(self, cmd: str):
        """Handle the /personality command to set predefined personalities."""
        from cli import save_config_value
        parts = cmd.split(maxsplit=1)
        
        if len(parts) > 1:
            # Set personality
            personality_name = parts[1].strip().lower()
            
            if personality_name in {"none", "default", "neutral"}:
                self.system_prompt = ""
                self.agent = None  # Force re-init
                if save_config_value("agent.system_prompt", ""):
                    print("(^_^)b Personality cleared (saved to config)")
                else:
                    print("(^_^) Personality cleared (session only)")
                print("  No personality overlay — using base agent behavior.")
            elif personality_name in self.personalities:
                self.system_prompt = self._resolve_personality_prompt(self.personalities[personality_name])
                self.agent = None  # Force re-init
                if save_config_value("agent.system_prompt", self.system_prompt):
                    print(f"(^_^)b Personality set to '{personality_name}' (saved to config)")
                else:
                    print(f"(^_^) Personality set to '{personality_name}' (session only)")
                print(f"  \"{self.system_prompt[:60]}{'...' if len(self.system_prompt) > 60 else ''}\"")
            else:
                print(f"(._.) Unknown personality: {personality_name}")
                print(f"  Available: none, {', '.join(self.personalities.keys())}")
        else:
            # Show available personalities
            print()
            print("+" + "-" * 50 + "+")
            print("|" + " " * 12 + "(^o^)/ Personalities" + " " * 15 + "|")
            print("+" + "-" * 50 + "+")
            print()
            print(f"  {'none':<12} - (no personality overlay)")
            for name, prompt in self.personalities.items():
                if isinstance(prompt, dict):
                    preview = prompt.get("description") or prompt.get("system_prompt", "")[:50]
                else:
                    preview = str(prompt)[:50]
                print(f"  {name:<12} - {preview}")
            print()
            print("  Usage: /personality <name>")
            print()

    def _handle_pet_command(self, cmd: str):
        """Toggle, browse, or adopt a petdex mascot.

        ``/pet`` / ``/pet toggle`` → flip ``display.pet.enabled`` on/off
        ``/pet list``            → browse the petdex gallery
        ``/pet scale <n>``       → resize the pet everywhere (e.g. 0.5)
        ``/pet <slug>``          → adopt (install if needed) + make active
        ``/pet off``             → disable (alias for toggle-off)

        Writes ``display.pet.*`` to config; the CLI/TUI/desktop pet surfaces
        pick the change up on their next poll, so the pet appears shortly.
        """
        from agent.pet import store
        from agent.pet.manifest import ManifestError
        from hermes_cli.pets import _set_active, _set_enabled, print_pet_gallery, set_pet_scale, toggle_pet_display

        parts = cmd.split(maxsplit=1)
        arg = parts[1].strip() if len(parts) > 1 else ""
        low = arg.lower()

        if not arg or low == "toggle":
            enabled, name, err = toggle_pet_display()
            if err:
                print(f"(x_x) {err}")
                return
            if enabled:
                print(f"(^_^)b {name} is out — it'll pop in shortly.")
            else:
                print(f"(-_-)zzZ {name} put away." if name else "(-_-)zzZ Pet put away.")
            return

        if low in ("list", "gallery", "browse", "all"):
            print_pet_gallery()
            return

        if low == "scale" or low.startswith("scale "):
            value = arg[len("scale"):].strip()
            if not value:
                print("(o_o) Usage: /pet scale <factor>  (e.g. /pet scale 0.5)")
                return
            scale, err = set_pet_scale(value)
            print(f"(x_x) {err}" if err else f"(^_^) Pet scale → {scale:g}.")
            return

        if low == "off":
            _set_enabled(False)
            print("(-_-)zzZ Pet put away.")
            return

        print(f"(o_o) Fetching '{arg}' from petdex…")
        try:
            pet = store.install_pet(arg)
        except (store.PetStoreError, ManifestError) as exc:
            print(f"(x_x) Couldn't adopt '{arg}': {exc}")
            return
        _set_active(arg)
        print(f"(^_^)b {pet.display_name} is out — it'll pop in shortly.")

    def _handle_hatch_command(self, cmd: str):
        """Generate ("hatch") a brand-new petdex pet from a description.

        ``/hatch <description>`` runs the full pet pipeline in-process: a base
        look, then one grounded animation row per state, sliced + normalized into
        a spritesheet, then adopted as the active mascot. Progress streams inline
        (it's ~a minute of image-model calls). In the desktop app this command
        opens the richer generate overlay instead; here we run it directly.
        """
        from agent.pet import store
        from agent.pet.generate import orchestrate
        from agent.pet.generate.imagegen import GenerationError
        from hermes_cli.pets import _set_active

        parts = cmd.split(maxsplit=1)
        concept = parts[1].strip() if len(parts) > 1 else ""

        if not concept:
            try:
                concept = input("(o_o) Describe your pet: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

        if not concept:
            print("(o_o) Usage: /hatch <description>  (e.g. /hatch a tiny cyber fox)")
            return

        # A short, friendly display name from the first few words of the concept.
        display_name = " ".join(w.capitalize() for w in concept.split()[:3])[:28].strip() or "Pet"
        slug = store.slugify(display_name) or store.slugify(concept) or "pet"

        print(f"(o_o) Designing '{concept}'… (a minute of image-model calls)")
        try:
            drafts = orchestrate.generate_base_drafts(concept, n=1)
        except GenerationError as exc:
            print(f"(x_x) Couldn't generate a base look: {exc}")
            return

        if not drafts:
            print("(x_x) No base draft came back — try again.")
            return

        def _progress(event: str, detail: str) -> None:
            if event == "row":
                # detail is "<state>:<done>:<total>"; show the state name.
                state = detail.split(":", 1)[0]
                print(f"  ┊ drawing {state}…")
            elif event == "compose":
                print("  ┊ composing spritesheet…")
            elif event == "save":
                print("  ┊ saving…")

        try:
            result = orchestrate.hatch_pet(
                base_image=drafts[0],
                slug=slug,
                display_name=display_name,
                concept=concept,
                on_progress=_progress,
            )
        except GenerationError as exc:
            print(f"(x_x) Hatch failed: {exc}")
            return

        _set_active(result.slug)
        print(f"(^_^)b {result.display_name} hatched and adopted — it'll pop in shortly!")

    def _handle_cron_command(self, cmd: str):
        """Handle the /cron command to manage scheduled tasks."""
        from cli import get_job
        import shlex
        from tools.cronjob_tools import cronjob as cronjob_tool

        def _cron_api(**kwargs):
            return json.loads(cronjob_tool(**kwargs))

        def _normalize_skills(values):
            normalized = []
            for value in values:
                text = str(value or "").strip()
                if text and text not in normalized:
                    normalized.append(text)
            return normalized

        def _parse_flags(tokens):
            opts = {
                "name": None,
                "deliver": None,
                "repeat": None,
                "skills": [],
                "add_skills": [],
                "remove_skills": [],
                "clear_skills": False,
                "all": False,
                "prompt": None,
                "schedule": None,
                "positionals": [],
            }
            i = 0
            while i < len(tokens):
                token = tokens[i]
                if token == "--name" and i + 1 < len(tokens):
                    opts["name"] = tokens[i + 1]
                    i += 2
                elif token == "--deliver" and i + 1 < len(tokens):
                    opts["deliver"] = tokens[i + 1]
                    i += 2
                elif token == "--repeat" and i + 1 < len(tokens):
                    try:
                        opts["repeat"] = int(tokens[i + 1])
                    except ValueError:
                        print("(._.) --repeat must be an integer")
                        return None
                    i += 2
                elif token == "--skill" and i + 1 < len(tokens):
                    opts["skills"].append(tokens[i + 1])
                    i += 2
                elif token == "--add-skill" and i + 1 < len(tokens):
                    opts["add_skills"].append(tokens[i + 1])
                    i += 2
                elif token == "--remove-skill" and i + 1 < len(tokens):
                    opts["remove_skills"].append(tokens[i + 1])
                    i += 2
                elif token == "--clear-skills":
                    opts["clear_skills"] = True
                    i += 1
                elif token == "--all":
                    opts["all"] = True
                    i += 1
                elif token == "--prompt" and i + 1 < len(tokens):
                    opts["prompt"] = tokens[i + 1]
                    i += 2
                elif token == "--schedule" and i + 1 < len(tokens):
                    opts["schedule"] = tokens[i + 1]
                    i += 2
                else:
                    opts["positionals"].append(token)
                    i += 1
            return opts

        tokens = shlex.split(cmd)

        if len(tokens) == 1:
            print()
            print("+" + "-" * 68 + "+")
            print("|" + " " * 22 + "(^_^) Scheduled Tasks" + " " * 23 + "|")
            print("+" + "-" * 68 + "+")
            print()
            print("  Commands:")
            print("    /cron list")
            print('    /cron add "every 2h" "Check server status" [--skill blogwatcher]')
            print('    /cron edit <job_id> --schedule "every 4h" --prompt "New task"')
            print("    /cron edit <job_id> --skill blogwatcher --skill maps")
            print("    /cron edit <job_id> --remove-skill blogwatcher")
            print("    /cron edit <job_id> --clear-skills")
            print("    /cron pause <job_id>")
            print("    /cron resume <job_id>")
            print("    /cron run <job_id>")
            print("    /cron remove <job_id>")
            print()
            result = _cron_api(action="list")
            jobs = result.get("jobs", []) if result.get("success") else []
            if jobs:
                print("  Current Jobs:")
                print("  " + "-" * 63)
                for job in jobs:
                    repeat_str = job.get("repeat", "?")
                    print(f"    {job['job_id'][:12]:<12} | {job['schedule']:<15} | {repeat_str:<8}")
                    if job.get("skills"):
                        print(f"      Skills: {', '.join(job['skills'])}")
                    print(f"      {job.get('prompt_preview', '')}")
                    if job.get("next_run_at"):
                        print(f"      Next: {job['next_run_at']}")
                    print()
            else:
                print("  No scheduled jobs. Use '/cron add' to create one.")
            print()
            return

        subcommand = tokens[1].lower()
        opts = _parse_flags(tokens[2:])
        if opts is None:
            return

        if subcommand == "list":
            result = _cron_api(action="list", include_disabled=opts["all"])
            jobs = result.get("jobs", []) if result.get("success") else []
            if not jobs:
                print("(._.) No scheduled jobs.")
                return

            print()
            print("Scheduled Jobs:")
            print("-" * 80)
            for job in jobs:
                print(f"  ID: {job['job_id']}")
                print(f"  Name: {job['name']}")
                print(f"  State: {job.get('state', '?')}")
                print(f"  Schedule: {job['schedule']} ({job.get('repeat', '?')})")
                print(f"  Next run: {job.get('next_run_at', 'N/A')}")
                if job.get("skills"):
                    print(f"  Skills: {', '.join(job['skills'])}")
                print(f"  Prompt: {job.get('prompt_preview', '')}")
                if job.get("last_run_at"):
                    print(f"  Last run: {job['last_run_at']} ({job.get('last_status', '?')})")
                print()
            return

        if subcommand in {"add", "create"}:
            positionals = opts["positionals"]
            if not positionals:
                print("(._.) Usage: /cron add <schedule> <prompt>")
                return
            schedule = opts["schedule"] or positionals[0]
            prompt = opts["prompt"] or " ".join(positionals[1:])
            skills = _normalize_skills(opts["skills"])
            if not prompt and not skills:
                print("(._.) Please provide a prompt or at least one skill")
                return
            result = _cron_api(
                action="create",
                schedule=schedule,
                prompt=prompt or None,
                name=opts["name"],
                deliver=opts["deliver"],
                repeat=opts["repeat"],
                skills=skills or None,
            )
            if result.get("success"):
                print(f"(^_^)b Created job: {result['job_id']}")
                print(f"  Schedule: {result['schedule']}")
                if result.get("skills"):
                    print(f"  Skills: {', '.join(result['skills'])}")
                print(f"  Next run: {result['next_run_at']}")
            else:
                print(f"(x_x) Failed to create job: {result.get('error')}")
            return

        if subcommand == "edit":
            positionals = opts["positionals"]
            if not positionals:
                print("(._.) Usage: /cron edit <job_id> [--schedule ...] [--prompt ...] [--skill ...]")
                return
            job_id = positionals[0]
            existing = get_job(job_id)
            if not existing:
                print(f"(._.) Job not found: {job_id}")
                return

            final_skills = None
            replacement_skills = _normalize_skills(opts["skills"])
            add_skills = _normalize_skills(opts["add_skills"])
            remove_skills = set(_normalize_skills(opts["remove_skills"]))
            existing_skills = list(existing.get("skills") or ([] if not existing.get("skill") else [existing.get("skill")]))
            if opts["clear_skills"]:
                final_skills = []
            elif replacement_skills:
                final_skills = replacement_skills
            elif add_skills or remove_skills:
                final_skills = [skill for skill in existing_skills if skill not in remove_skills]
                for skill in add_skills:
                    if skill not in final_skills:
                        final_skills.append(skill)

            result = _cron_api(
                action="update",
                job_id=job_id,
                schedule=opts["schedule"],
                prompt=opts["prompt"],
                name=opts["name"],
                deliver=opts["deliver"],
                repeat=opts["repeat"],
                skills=final_skills,
            )
            if result.get("success"):
                job = result["job"]
                print(f"(^_^)b Updated job: {job['job_id']}")
                print(f"  Schedule: {job['schedule']}")
                if job.get("skills"):
                    print(f"  Skills: {', '.join(job['skills'])}")
                else:
                    print("  Skills: none")
            else:
                print(f"(x_x) Failed to update job: {result.get('error')}")
            return

        if subcommand in {"pause", "resume", "run", "remove", "rm", "delete"}:
            positionals = opts["positionals"]
            if not positionals:
                print(f"(._.) Usage: /cron {subcommand} <job_id>")
                return
            job_id = positionals[0]
            action = "remove" if subcommand in {"remove", "rm", "delete"} else subcommand
            result = _cron_api(action=action, job_id=job_id, reason="paused from /cron" if action == "pause" else None)
            if not result.get("success"):
                print(f"(x_x) Failed to {action} job: {result.get('error')}")
                return
            if action == "pause":
                print(f"(^_^)b Paused job: {result['job']['name']} ({job_id})")
            elif action == "resume":
                print(f"(^_^)b Resumed job: {result['job']['name']} ({job_id})")
                print(f"  Next run: {result['job'].get('next_run_at')}")
            elif action == "run":
                print(f"(^_^)b Triggered job: {result['job']['name']} ({job_id})")
                print("  It will run on the next scheduler tick.")
            else:
                removed = result.get("removed_job", {})
                print(f"(^_^)b Removed job: {removed.get('name', job_id)} ({job_id})")
            return

        print(f"(._.) Unknown cron command: {subcommand}")
        print("  Available: list, add, edit, pause, resume, run, remove")

    def _handle_suggestions_command(self, cmd: str):
        """Handle /suggestions — review/accept/dismiss suggested automations.

        Delegates to the shared handler so CLI and gateway never drift. CLI
        origin is the local platform so an accepted job's "origin" delivery
        resolves to a configured home channel.
        """
        import shlex

        try:
            tokens = shlex.split(cmd)[1:] if cmd else []
        except ValueError:
            tokens = (cmd or "").split()[1:]
        args = " ".join(tokens)
        try:
            from hermes_cli.suggestions_cmd import handle_suggestions_command
            output = handle_suggestions_command(args)
        except Exception as e:
            output = f"Suggestions command failed: {e}"
        self._console_print(output)

    def _handle_blueprint_command(self, cmd: str):
        """Handle /blueprint — set up an automation from a blueprint template.

        Delegates to the shared handler. A bare ``/blueprint`` lists the
        catalog; ``/blueprint <name>`` name-matches a blueprint and seeds the
        agent to ask the user for each value conversationally (the result's
        ``agent_seed``); ``/blueprint <name> slot=val …`` creates the job
        directly. When a seed is returned it is stashed as a one-shot pending
        message the interactive loop runs as the next agent turn.
        """
        import shlex

        try:
            tokens = shlex.split(cmd)[1:] if cmd else []
        except ValueError:
            tokens = (cmd or "").split()[1:]
        args = " ".join(shlex.quote(t) for t in tokens)
        try:
            from hermes_cli.blueprint_cmd import handle_blueprint_command
            result = handle_blueprint_command(args)
        except Exception as e:
            self._console_print(f"Cron blueprint command failed: {e}")
            return
        self._console_print(result.text)
        seed = getattr(result, "agent_seed", None)
        if seed:
            # One-shot: the interactive loop picks this up right after the
            # slash command returns and runs it as a normal agent turn.
            self._pending_agent_seed = seed

    def _handle_curator_command(self, cmd: str):
        """Handle /curator slash command.

        Delegates to hermes_cli.curator so the CLI and the `hermes curator`
        subcommand share the same handler set.
        """
        import shlex

        tokens = shlex.split(cmd)[1:] if cmd else []
        if not tokens:
            tokens = ["status"]

        try:
            from hermes_cli.curator import cli_main
            cli_main(tokens)
        except SystemExit:
            # argparse calls sys.exit() on --help or errors; swallow so we
            # don't kill the interactive session.
            pass
        except Exception as exc:
            print(f"(._.) curator: {exc}")

    def _handle_kanban_command(self, cmd: str):
        """Handle the /kanban command — delegate to the shared kanban CLI.

        The string form passed here is the user's full ``/kanban ...``
        including the leading slash; we strip it and hand the remainder
        to ``kanban.run_slash`` which returns a single formatted string.
        """
        from hermes_cli.kanban import run_slash

        rest = cmd.strip()
        if rest.startswith("/"):
            rest = rest.lstrip("/")
        if rest.startswith("kanban"):
            rest = rest[len("kanban"):].lstrip()
        try:
            output = run_slash(rest)
        except Exception as exc:  # pragma: no cover - defensive
            output = f"(._.) kanban error: {exc}"
        if output:
            print(output)

    def _handle_skills_command(self, cmd: str):
        """Handle /skills slash command — delegates to hermes_cli.skills_hub."""
        from cli import ChatConsole
        # Intercept write-approval review subcommands first (pending/approve/
        # reject/diff/mode); everything else goes to the skills hub.
        parts = cmd.strip().split()
        args = parts[1:] if len(parts) > 1 else []
        if args and args[0].lower() in {"pending", "approve", "apply", "reject",
                                        "deny", "drop", "diff", "approval", "mode"}:
            from hermes_cli.write_approval_commands import handle_pending_subcommand
            from tools import write_approval as wa
            out = handle_pending_subcommand(
                wa.SKILLS, args,
                set_mode_fn=lambda enabled: self._save_write_approval("skills", enabled),
            )
            if out is not None:
                print(out)
                return
        from hermes_cli.skills_hub import handle_skills_slash
        handle_skills_slash(cmd, ChatConsole())

    def _handle_learn_command(self, cmd: str):
        """Handle /learn — distill a reusable skill from anything the user describes.

        Open-ended: the argument is free text describing the source(s) — a
        directory, a URL, "what we just did", pasted notes. We build a
        standards-guided prompt and inject it onto the agent's input queue; the
        live agent gathers the material with the tools it already has and
        authors the skill via ``skill_manage``. No engine, no model-tool
        footprint, works on any terminal backend.
        """
        from agent.learn_prompt import build_learn_prompt

        # Everything after the command word is the open-ended request.
        parts = cmd.strip().split(None, 1)
        user_request = parts[1].strip() if len(parts) > 1 else ""

        msg = build_learn_prompt(user_request)
        if user_request:
            print("\n⚡ Learning a skill from what you described...")
        else:
            print("\n⚡ Learning a skill from this conversation...")
        if hasattr(self, "_pending_input"):
            self._pending_input.put(msg)
        else:  # pragma: no cover - defensive (no live input loop)
            print("  /learn needs an active chat session to run.")

    def _handle_memory_command(self, cmd: str):
        """Handle /memory slash command — pending review + approval-gate toggle."""
        from hermes_cli.write_approval_commands import handle_pending_subcommand
        from tools import write_approval as wa
        parts = cmd.strip().split()
        args = parts[1:] if len(parts) > 1 else []
        store = getattr(self.agent, "_memory_store", None) if getattr(self, "agent", None) else None
        if store is None:
            # No live agent store (e.g. /memory approve invoked from the Desktop
            # GUI, or any context without an active agent). Apply against a freshly
            # loaded on-disk store, mirroring the gateway path
            # (gateway/slash_commands.py): it persists to the same MEMORY/USER.md
            # and creates MEMORY.md on the first approved write. Without this the
            # shared handler returns "memory store unavailable". See #46783.
            # load_on_disk_store() honors the user's configured char limits, so
            # an approval here enforces the same caps as the live agent would.
            from tools.memory_tool import load_on_disk_store
            store = load_on_disk_store()
        out = handle_pending_subcommand(
            wa.MEMORY, args,
            memory_store=store,
            set_mode_fn=lambda enabled: self._save_write_approval("memory", enabled),
        )
        if out is None:
            out = ("Unknown /memory subcommand. "
                   "Use: pending, approve <id>, reject <id>, approval <on|off>.")
        print(out)

    def _save_write_approval(self, subsystem: str, enabled: bool):
        """Persist <subsystem>.write_approval to config (for /memory|/skills approval)."""
        from cli import save_config_value
        save_config_value(f"{subsystem}.write_approval", bool(enabled))

    def _handle_background_command(self, cmd: str):
        """Handle /background <prompt> — run a prompt in a separate background session.

        Spawns a new AIAgent in a background thread with its own session.
        When it completes, prints the result to the CLI without modifying
        the active session's conversation history.
        """
        from cli import AIAgent, ChatConsole, _accent_hex, _cprint, _maybe_remap_for_light_mode, _render_final_assistant_content, set_approval_callback, set_secret_capture_callback, set_sudo_password_callback
        parts = cmd.strip().split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            _cprint("  Usage: /background <prompt>")
            _cprint("  Example: /background Summarize the top HN stories today")
            _cprint("  The task runs in a separate session and results display here when done.")
            return

        prompt = parts[1].strip()
        self._background_task_counter += 1
        task_num = self._background_task_counter
        task_id = f"bg_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Make sure we have valid credentials
        if not self._ensure_runtime_credentials():
            _cprint("  (>_<) Cannot start background task: no valid credentials.")
            return

        _cprint(f"  🔄 Background task #{task_num} started: \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")
        _cprint(f"  Task ID: {task_id}")
        _cprint("  You can continue chatting — results will appear when done.\n")

        turn_route = self._resolve_turn_agent_config(prompt)

        def run_background():
            set_sudo_password_callback(self._sudo_password_callback)
            set_approval_callback(self._approval_callback)
            try:
                set_secret_capture_callback(self._secret_capture_callback)
            except Exception:
                pass
            try:
                bg_agent = AIAgent(
                    model=turn_route["model"],
                    api_key=turn_route["runtime"].get("api_key"),
                    base_url=turn_route["runtime"].get("base_url"),
                    provider=turn_route["runtime"].get("provider"),
                    api_mode=turn_route["runtime"].get("api_mode"),
                    acp_command=turn_route["runtime"].get("command"),
                    acp_args=turn_route["runtime"].get("args"),
                    max_tokens=turn_route["runtime"].get("max_tokens"),
                    max_iterations=self.max_turns,
                    enabled_toolsets=self.enabled_toolsets,
                    quiet_mode=True,
                    verbose_logging=False,
                    session_id=task_id,
                    platform="cli",
                    session_db=self._session_db,
                    reasoning_config=self.reasoning_config,
                    service_tier=self.service_tier,
                    request_overrides=turn_route.get("request_overrides"),
                    providers_allowed=self._providers_only,
                    providers_ignored=self._providers_ignore,
                    providers_order=self._providers_order,
                    provider_sort=self._provider_sort,
                    provider_require_parameters=self._provider_require_params,
                    provider_data_collection=self._provider_data_collection,
                    openrouter_min_coding_score=self._openrouter_min_coding_score,
                    fallback_model=self._fallback_model,
                )
                # Silence raw spinner; route thinking through TUI widget when no foreground agent is active.
                bg_agent._print_fn = lambda *_a, **_kw: None

                def _bg_thinking(text: str) -> None:
                    # Concurrent bg tasks may race on _spinner_text; acceptable for best-effort UI.
                    if not self._agent_running:
                        self._spinner_text = text
                        if self._app:
                            self._app.invalidate()

                bg_agent.thinking_callback = _bg_thinking

                result = bg_agent.run_conversation(
                    user_message=prompt,
                    task_id=task_id,
                )

                response = result.get("final_response", "") if result else ""
                if not response and result and result.get("error"):
                    response = f"Error: {result['error']}"

                # Display result in the CLI (thread-safe via patch_stdout).
                # Force a TUI refresh first so spinner/status bar don't overlap
                # with the output (fixes #2718).
                if self._app:
                    self._app.invalidate()
                    time.sleep(0.05)  # brief pause for refresh
                print()
                ChatConsole().print(f"[{_accent_hex()}]{'─' * 40}[/]")
                _cprint(f"  ✅ Background task #{task_num} complete")
                _cprint(f"  Prompt: \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")
                ChatConsole().print(f"[{_accent_hex()}]{'─' * 40}[/]")
                if response:
                    try:
                        from hermes_cli.skin_engine import get_active_skin
                        _skin = get_active_skin()
                        label = _skin.get_branding("response_label", "⚕ Hermes")
                        _resp_color = _maybe_remap_for_light_mode(_skin.get_color("response_border", "#CD7F32"))
                        _resp_text = _maybe_remap_for_light_mode(_skin.get_color("banner_text", "#FFF8DC"))
                    except Exception:
                        label = "⚕ Hermes"
                        _resp_color = "#CD7F32"
                        _resp_text = "#FFF8DC"

                    _chat_console = ChatConsole()
                    _chat_console.print(Panel(
                        _render_final_assistant_content(response, mode=self.final_response_markdown),
                        title=f"[{_resp_color} bold]{label} (background #{task_num})[/]",
                        title_align="left",
                        border_style=_resp_color,
                        style=_resp_text,
                        box=rich_box.HORIZONTALS,
                        padding=(1, 4),
                        width=self._scrollback_box_width(),
                    ))
                else:
                    _cprint("  (No response generated)")

                # Play bell if enabled
                if self.bell_on_complete:
                    sys.stdout.write("\a")
                    sys.stdout.flush()

            except Exception as e:
                # Same TUI refresh pattern as success path (#2718)
                if self._app:
                    self._app.invalidate()
                    time.sleep(0.05)
                print()
                _cprint(f"  ❌ Background task #{task_num} failed: {e}")
            finally:
                try:
                    set_sudo_password_callback(None)
                    set_approval_callback(None)
                    set_secret_capture_callback(None)
                except Exception:
                    pass
                self._background_tasks.pop(task_id, None)
                # Clear spinner only if no foreground agent owns it
                if not self._agent_running:
                    self._spinner_text = ""
                if self._app:
                    self._invalidate(min_interval=0)

        thread = threading.Thread(target=run_background, daemon=True, name=f"bg-task-{task_id}")
        self._background_tasks[task_id] = thread
        thread.start()

    def _handle_bundles_command(self, cmd: str) -> None:
        """In-session ``/bundles`` — show installed skill bundles.

        Mirrors ``hermes bundles list`` but renders inside the running
        CLI so users can discover what's available without dropping out
        of their session. Bundles are loaded via ``/<bundle-name>``.
        """
        from cli import ChatConsole, _BOLD, _DIM, _RST, _accent_hex, _cprint
        try:
            from agent.skill_bundles import list_bundles, _bundles_dir
        except Exception as exc:
            _cprint(f"\033[1;31mBundle subsystem unavailable: {exc}{_RST}")
            return

        bundles = list_bundles()
        if not bundles:
            _cprint("  No skill bundles installed.")
            _cprint(
                f"  {_DIM}Create one with: hermes bundles create "
                f"<name> --skill <s1> --skill <s2>{_RST}"
            )
            _cprint(f"  {_DIM}Directory: {_bundles_dir()}{_RST}")
            return

        _cprint(f"\n  ▣ {_BOLD}Skill Bundles{_RST} ({len(bundles)} installed):")
        for info in bundles:
            skill_count = len(info.get("skills", []))
            desc = info.get("description") or f"Load {skill_count} skills"
            ChatConsole().print(
                f"    [bold {_accent_hex()}]/{info['slug']:<20}[/] "
                f"[dim]-[/] {_escape(desc)} [dim]({skill_count} skills)[/]"
            )
            for s in info.get("skills", []):
                ChatConsole().print(f"        [dim]· {_escape(s)}[/]")
        _cprint(
            f"\n  {_DIM}Invoke a bundle with /<slug>. "
            f"Manage with `hermes bundles`.{_RST}"
        )

    def _handle_browser_command(self, cmd: str):
        """Handle /browser connect|disconnect|status — manage live Chromium-family CDP connection."""
        import platform as _plat

        parts = cmd.strip().split(None, 1)
        sub = parts[1].lower().strip() if len(parts) > 1 else "status"

        _DEFAULT_CDP = DEFAULT_BROWSER_CDP_URL
        current = os.environ.get("BROWSER_CDP_URL", "").strip()

        if sub.startswith("connect"):
            # Optionally accept a custom CDP URL: /browser connect ws://host:port
            connect_parts = cmd.strip().split(None, 2)  # ["/browser", "connect", "ws://..."]
            cdp_url = connect_parts[2].strip() if len(connect_parts) > 2 else _DEFAULT_CDP
            parsed_cdp = urlparse(cdp_url if "://" in cdp_url else f"http://{cdp_url}")
            if parsed_cdp.scheme not in {"http", "https", "ws", "wss"}:
                print()
                print(
                    f"   ⚠ Unsupported browser url scheme: {parsed_cdp.scheme or '(missing)'} "
                    "(expected one of: http, https, ws, wss)"
                )
                print()
                return
            try:
                _port = parsed_cdp.port or (443 if parsed_cdp.scheme in {"https", "wss"} else 80)
            except ValueError:
                print()
                print(f"   ⚠ Invalid port in browser url: {cdp_url}")
                print()
                return
            if not parsed_cdp.hostname:
                print()
                print(f"   ⚠ Missing host in browser url: {cdp_url}")
                print()
                return
            _host = parsed_cdp.hostname
            if parsed_cdp.path.startswith("/devtools/browser/"):
                cdp_url = parsed_cdp.geturl()
            else:
                cdp_url = parsed_cdp._replace(
                    path="",
                    params="",
                    query="",
                    fragment="",
                ).geturl()

            # Clear any existing browser sessions so the next tool call uses the new backend
            try:
                from tools.browser_tool import cleanup_all_browsers
                cleanup_all_browsers()
            except Exception:
                pass

            print()

            # Check if a Chromium-family browser is already serving CDP on the debug port.
            # For the default-local URL, probe both loopbacks (IPv4 + IPv6): a
            # squatter on 127.0.0.1:<port> (e.g. an IDE's JS debugger) can push
            # the debug browser to bind [::1] only.
            _is_default = cdp_url == _DEFAULT_CDP
            if _is_default:
                _found = discover_local_cdp_url(_port, timeout=1.0)
                _already_open = _found is not None
                if _found:
                    cdp_url = _found
            else:
                _already_open = is_browser_debug_ready(cdp_url, timeout=1.0)

            if _already_open:
                print(f"   ✓ Chromium-family browser is already listening at {cdp_url}")
            elif _is_default:
                _launch_port = _port
                if local_port_in_use(_port):
                    _launch_port = find_free_debug_port(_port)
                    print(
                        f"   ⚠ Port {_port} is occupied by another application that isn't a CDP browser"
                    )
                    print(
                        f"     (an IDE debugger or dev server may be using it) — launching on port {_launch_port} instead..."
                    )
                else:
                    # Try to auto-launch a Chromium-family browser with remote debugging
                    print("   Chromium-family browser isn't running with remote debugging — attempting to launch...")
                _launch = launch_chrome_debug(_launch_port, _plat.system())
                if _launch.launched:
                    # Wait for the DevTools discovery endpoint to come up
                    for _wait in range(10):
                        _found = discover_local_cdp_url(_launch_port, timeout=1.0)
                        if _found:
                            cdp_url = _found
                            _already_open = True
                            break
                        time.sleep(0.5)
                    if _already_open:
                        print(f"   ✓ Chromium-family browser launched and listening on port {_launch_port}")
                    else:
                        print(f"   ⚠ Browser launched but port {_launch_port} isn't responding yet")
                        print("     Try again in a few seconds — the debug instance may still be starting")
                else:
                    print("   ⚠ Could not auto-launch a Chromium-family browser")
                    _hint = _launch.hint
                    if _hint:
                        print(f"     {_hint}")
                    sys_name = _plat.system()
                    chrome_cmd = manual_chrome_debug_command(_launch_port, sys_name)
                    if chrome_cmd:
                        print("     Launch a Chromium-family browser manually:")
                        print(f"     {chrome_cmd}")
                    else:
                        print("     No supported Chromium-family browser executable found in this environment")
            else:
                print(f"   ⚠ Port {_port} is not reachable at {cdp_url}")

            if not _already_open:
                print()
                print("Browser not connected — start a Chromium-family browser with remote debugging and retry /browser connect")
                print()
                return

            os.environ["BROWSER_CDP_URL"] = cdp_url
            # Eagerly start the CDP supervisor so pending_dialogs + frame_tree
            # show up in the next browser_snapshot.  No-op if already started.
            try:
                from tools.browser_tool import _ensure_cdp_supervisor  # type: ignore[import-not-found]
                _ensure_cdp_supervisor("default")
            except Exception:
                pass
            print()
            print("🌐 Browser connected to live Chromium-family browser via CDP")
            print(f"   Endpoint: {cdp_url}")
            print()

            # Inject context message so the model knows this slash command
            # intentionally makes the dev/debug CDP browser available for use.
            if hasattr(self, '_pending_input'):
                self._pending_input.put(
                    "[System note: The user invoked /browser connect and connected your browser tools to "
                    "a Chromium-family dev/debug browser via Chrome DevTools Protocol. "
                    "Your browser_navigate, browser_snapshot, browser_click, and other browser tools now "
                    "control that CDP browser. The command itself is a signal that using browser tools for "
                    "their current browser-related request is expected; do not wait for separate permission "
                    "just because CDP is connected. This is typically a Hermes-managed isolated debug "
                    "profile, not the user's main everyday browser. It is still user-visible and may contain "
                    "pages, logged-in sessions, or cookies in that debug profile, so avoid destructive actions, "
                    "closing tabs, or navigating away unless the user's task calls for it.]"
                )

        elif sub == "disconnect":
            if current:
                os.environ.pop("BROWSER_CDP_URL", None)
                try:
                    from tools.browser_tool import cleanup_all_browsers, _stop_cdp_supervisor
                    _stop_cdp_supervisor("default")
                    cleanup_all_browsers()
                except Exception:
                    pass
                print()
                print("🌐 Browser disconnected from live Chromium-family browser")
                print("   Browser tools reverted to default mode (local headless or cloud provider)")
                print()

                if hasattr(self, '_pending_input'):
                    self._pending_input.put(
                        "[System note: The user has disconnected the browser tools from their live Chromium-family browser. "
                        "Browser tools are back to default mode (headless local browser or cloud provider).]"
                    )
            else:
                print()
                print("Browser is not connected to a live Chromium-family browser (already using default mode)")
                print()

        elif sub == "status":
            print()
            if current:
                print("🌐 Browser: connected to live Chromium-family browser via CDP")
                print(f"   Endpoint: {current}")

                _port = 9222
                try:
                    _port = int(current.rsplit(":", 1)[-1].split("/")[0])
                except (ValueError, IndexError):
                    pass
                try:
                    import socket
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(1)
                    s.connect(("127.0.0.1", _port))
                    s.close()
                    print("   Status: ✓ reachable")
                except (OSError, Exception):
                    print("   Status: ⚠ not reachable (browser may not be running)")
            else:
                try:
                    from tools.browser_tool import _get_cloud_provider
                    provider = _get_cloud_provider()
                except Exception:
                    provider = None

                if provider is not None:
                    print(f"🌐 Browser: {provider.provider_name()} (cloud)")
                else:
                    # Show engine info for local mode
                    try:
                        from tools.browser_tool import _get_browser_engine
                        engine = _get_browser_engine()
                    except Exception:
                        engine = "auto"
                    if engine == "lightpanda":
                        print("🌐 Browser: local Lightpanda (agent-browser --engine lightpanda)")
                        print("   ⚡ Lightpanda: faster navigation, no screenshot support")
                        print("   Automatic Chromium fallback for screenshots and failed commands")
                    elif engine == "chrome":
                        print("🌐 Browser: local headless Chromium (agent-browser --engine chrome)")
                    else:
                        print("🌐 Browser: local headless Chromium (agent-browser)")
            print()
            print("   /browser connect      — connect to your live Chromium-family browser")
            print("   /browser disconnect   — revert to default")
            print()

        else:
            print()
            print("Usage: /browser connect|disconnect|status")
            print()
            print("   connect      Connect browser tools to your live Chromium-family browser session")
            print("   disconnect   Revert to default browser backend")
            print("   status       Show current browser mode")
            print()

    def _handle_goal_command(self, cmd: str) -> None:
        """Dispatch /goal subcommands: set / draft / show / status / pause / resume / clear."""
        from cli import _DIM, _RST, _cprint
        parts = (cmd or "").strip().split(None, 1)
        arg = parts[1].strip() if len(parts) > 1 else ""

        mgr = self._get_goal_manager()
        if mgr is None:
            _cprint(f"  {_DIM}Goals unavailable (no active session).{_RST}")
            return

        lower = arg.lower()

        # Bare /goal or /goal status → show current state
        if not arg or lower == "status":
            _cprint(f"  {mgr.status_line()}")
            return

        # /goal show → print the active goal's completion contract
        if lower == "show":
            _cprint(f"  {mgr.status_line()}")
            _cprint(f"  {mgr.render_contract()}")
            return

        # /goal draft <objective> → expand plain text into a structured
        # completion contract (outcome / verification / constraints /
        # boundaries / stop_when) and set it as the active goal. Adapted
        # from Codex's "let the agent draft the goal" guidance: the contract
        # makes "done" evidence-based instead of a loose vibe check.
        if lower.startswith("draft"):
            objective = arg[len("draft"):].strip()
            if not objective:
                _cprint("  Usage: /goal draft <objective in plain language>")
                return
            self._handle_goal_draft(objective)
            return

        if lower == "pause":
            state = mgr.pause(reason="user-paused")
            if state is None:
                _cprint(f"  {_DIM}No goal set.{_RST}")
            else:
                _cprint(f"  ⏸ Goal paused: {state.goal}")
            return

        if lower == "resume":
            state = mgr.resume()
            if state is None:
                _cprint(f"  {_DIM}No goal to resume.{_RST}")
            else:
                _cprint(f"  ▶ Goal resumed: {state.goal}")
                _cprint(
                    f"  {_DIM}Send any message (or press Enter on an empty prompt "
                    f"is a no-op; type 'continue' to kick it off).{_RST}"
                )
            return

        if lower in {"clear", "stop", "done"}:
            had = mgr.has_goal()
            mgr.clear()
            if had:
                _cprint("  ✓ Goal cleared.")
            else:
                _cprint(f"  {_DIM}No active goal.{_RST}")
            return

        # /goal wait <pid> [reason] — park the loop on a background process so
        # it stops re-poking the agent every turn while it waits on CI / a
        # build / a long job. The barrier auto-clears when the PID exits.
        if lower == "wait" or lower.startswith("wait "):
            wait_arg = arg[len("wait"):].strip()
            if not wait_arg:
                _cprint("  Usage: /goal wait <pid> [reason]")
                return
            wtokens = wait_arg.split(None, 1)
            try:
                pid = int(wtokens[0])
            except ValueError:
                _cprint("  /goal wait: <pid> must be an integer process id.")
                return
            reason = wtokens[1].strip() if len(wtokens) > 1 else ""
            try:
                mgr.wait_on(pid, reason=reason)
            except (RuntimeError, ValueError) as exc:
                _cprint(f"  /goal wait: {exc}")
                return
            rtxt = f" ({reason})" if reason else ""
            _cprint(f"  ⏳ Goal parked on pid {pid}{rtxt}. Loop pauses until it exits.")
            return

        # /goal unwait — drop the wait barrier and resume normal looping.
        if lower == "unwait":
            if mgr.stop_waiting():
                _cprint("  ▶ Wait barrier cleared — goal loop resumes.")
            else:
                _cprint(f"  {_DIM}No wait barrier set.{_RST}")
            return

        # Otherwise treat the arg as the goal text. Inline `field: value`
        # lines (verify:, constraints:, boundaries:, stop when:) are parsed
        # into a completion contract; the remaining prose is the headline.
        # A plain free-form goal with no such lines behaves exactly as before.
        from hermes_cli.goals import parse_contract

        headline, contract = parse_contract(arg)
        goal_text = headline or arg
        try:
            state = mgr.set(goal_text, contract=contract if not contract.is_empty() else None)
        except ValueError as exc:
            _cprint(f"  Invalid goal: {exc}")
            return

        _cprint(f"  ⊙ Goal set ({state.max_turns}-turn budget): {state.goal}")
        if state.has_contract():
            _cprint(f"  {_DIM}Completion contract:{_RST}")
            for line in state.contract.render_block().splitlines():
                _cprint(f"    {line}")
        _cprint(
            f"  {_DIM}After each turn, a judge model checks if the goal is done"
            f"{' against the contract above' if state.has_contract() else ''}. "
            f"Hermes keeps working until it is, you pause/clear it, or the budget is "
            f"exhausted. Use /goal status, /goal show, /goal pause, /goal resume, /goal clear.{_RST}"
        )
        # Kick the loop off immediately so the user doesn't have to send a
        # separate message after setting the goal.
        try:
            self._pending_input.put(state.goal)
        except Exception:
            pass

    def _handle_goal_draft(self, objective: str) -> None:
        """Draft a structured completion contract from a plain objective and
        set it as the active goal. Falls back to a bare goal if the aux model
        can't produce a contract."""
        from cli import _DIM, _RST, _cprint
        from hermes_cli.goals import draft_contract

        mgr = self._get_goal_manager()
        if mgr is None:
            _cprint(f"  {_DIM}Goals unavailable (no active session).{_RST}")
            return

        _cprint(f"  {_DIM}Drafting completion contract…{_RST}")
        try:
            contract = draft_contract(objective)
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).debug("goal draft failed: %s", exc)
            contract = None

        try:
            state = mgr.set(objective, contract=contract)
        except ValueError as exc:
            _cprint(f"  Invalid goal: {exc}")
            return

        _cprint(f"  ⊙ Goal set ({state.max_turns}-turn budget): {state.goal}")
        if state.has_contract():
            _cprint(f"  {_DIM}Drafted completion contract:{_RST}")
            for line in state.contract.render_block().splitlines():
                _cprint(f"    {line}")
            _cprint(
                f"  {_DIM}Tighten any field by re-setting the goal with inline "
                f"lines (e.g. verify: <command>), then /goal resume. "
                f"Use /goal show to review.{_RST}"
            )
        else:
            _cprint(
                f"  {_DIM}Couldn't draft a contract (aux model unavailable) — "
                f"running as a free-form goal. The per-turn judge still applies.{_RST}"
            )
        try:
            self._pending_input.put(state.goal)
        except Exception:
            pass

    def _handle_subgoal_command(self, cmd: str) -> None:
        """Dispatch /subgoal subcommands.

        Forms:
          /subgoal                              show current subgoals
          /subgoal <text>                       append a criterion
          /subgoal remove <n>                   drop subgoal n (1-based)
          /subgoal clear                        wipe all subgoals

        Subgoals are extra criteria the user adds mid-loop. They get
        appended to both the judge prompt (verdict must consider them)
        and the continuation prompt (agent sees them) on the next turn
        boundary. No special kick — the running turn finishes, the next
        judge call includes them.
        """
        from cli import _DIM, _RST, _cprint
        parts = (cmd or "").strip().split(None, 2)
        arg = " ".join(parts[1:]).strip() if len(parts) > 1 else ""

        mgr = self._get_goal_manager()
        if mgr is None:
            _cprint(f"  {_DIM}Goals unavailable (no active session).{_RST}")
            return

        if not mgr.has_goal():
            _cprint(f"  {_DIM}No active goal. Set one with /goal <text>.{_RST}")
            return

        # No args → list current subgoals.
        if not arg:
            _cprint(f"  {mgr.status_line()}")
            _cprint(f"  {mgr.render_subgoals()}")
            return

        tokens = arg.split(None, 1)
        verb = tokens[0].lower()
        rest = tokens[1].strip() if len(tokens) > 1 else ""

        if verb == "remove":
            if not rest:
                _cprint("  Usage: /subgoal remove <n>")
                return
            try:
                idx = int(rest.split()[0])
            except ValueError:
                _cprint("  /subgoal remove: <n> must be an integer (1-based index).")
                return
            try:
                removed = mgr.remove_subgoal(idx)
            except (IndexError, RuntimeError) as exc:
                _cprint(f"  /subgoal remove: {exc}")
                return
            _cprint(f"  ✓ Removed subgoal {idx}: {removed}")
            return

        if verb == "clear":
            try:
                prev = mgr.clear_subgoals()
            except RuntimeError as exc:
                _cprint(f"  /subgoal clear: {exc}")
                return
            if prev:
                _cprint(f"  ✓ Cleared {prev} subgoal{'s' if prev != 1 else ''}.")
            else:
                _cprint(f"  {_DIM}No subgoals to clear.{_RST}")
            return

        # Otherwise — append the whole arg as a new subgoal.
        try:
            text = mgr.add_subgoal(arg)
        except (ValueError, RuntimeError) as exc:
            _cprint(f"  /subgoal: {exc}")
            return
        idx = len(mgr.state.subgoals) if mgr.state else 0
        _cprint(f"  ✓ Added subgoal {idx}: {text}")

    def _handle_skin_command(self, cmd: str):
        """Handle /skin [name] — show or change the display skin."""
        from cli import _ACCENT, save_config_value
        try:
            from hermes_cli.skin_engine import list_skins, set_active_skin, get_active_skin_name
        except ImportError:
            print("Skin engine not available.")
            return

        parts = cmd.strip().split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            # Show current skin and list available
            current = get_active_skin_name()
            skins = list_skins()
            print(f"\n  Current skin: {current}")
            print("  Available skins:")
            for s in skins:
                marker = " ●" if s["name"] == current else "  "
                source = f" ({s['source']})" if s["source"] == "user" else ""
                print(f"   {marker} {s['name']}{source} — {s['description']}")
            print("\n  Usage: /skin <name>")
            print(f"  Custom skins: drop a YAML file in {display_hermes_home()}/skins/\n")
            return

        new_skin = parts[1].strip().lower()
        available = {s["name"] for s in list_skins()}
        if new_skin not in available:
            print(f"  Unknown skin: {new_skin}")
            print(f"  Available: {', '.join(sorted(available))}")
            return

        set_active_skin(new_skin)
        _ACCENT.reset()  # Re-resolve ANSI color for the new skin
        # _DIM is now a fixed dim+italic ANSI escape (terminal-default fg)
        # so it doesn't need re-resolving on skin switch.
        if save_config_value("display.skin", new_skin):
            print(f"  Skin set to: {new_skin} (saved)")
        else:
            print(f"  Skin set to: {new_skin}")
        print("  Note: banner colors will update on next session start.")
        if self._apply_tui_skin_style():
            print("  Prompt + TUI colors updated.")

    def _compose_in_editor(self, initial_text: str = "") -> str:
        """Open ``$VISUAL``/``$EDITOR`` on a temp markdown file and return the
        saved buffer (comment lines starting with ``#!`` stripped).

        Returns the composed prompt text, or an empty string if the editor
        could not be launched or the buffer was left empty. Factored out so
        the read-back/strip logic is unit-testable without spawning an editor.
        """
        import os
        import shlex
        import subprocess
        import tempfile

        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if not editor:
            editor = "notepad" if os.name == "nt" else "nano"

        header = (
            "#! Compose your prompt below. Lines starting with '#!' are ignored.\n"
            "#! Save and quit to send; leave empty to cancel.\n\n"
        )
        fd, path = tempfile.mkstemp(suffix=".md", prefix="hermes_prompt_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(header)
                if initial_text:
                    fh.write(initial_text)
            try:
                subprocess.call([*shlex.split(editor), path])
            except Exception:
                # Fall back to a bare invocation (editor value may not be a
                # simple argv-splittable string on some platforms).
                subprocess.call(f"{editor} {shlex.quote(path)}", shell=True)
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        lines = [ln for ln in raw.splitlines() if not ln.startswith("#!")]
        return "\n".join(lines).strip()

    def _handle_prompt_compose_command(self, cmd_original: str) -> None:
        """Handle /prompt — compose the next prompt in $EDITOR and send it.

        Opens the user's editor on a temporary markdown file (optionally
        seeded with text passed after the command), then queues the saved
        buffer as the next agent turn via the one-shot ``_pending_agent_seed``
        the interactive loop already consumes (same path as /blueprint).
        """
        from cli import _DIM, _RST, _cprint

        initial = ""
        parts = (cmd_original or "").strip().split(None, 1)
        if len(parts) > 1:
            initial = parts[1]

        try:
            composed = self._compose_in_editor(initial)
        except Exception as exc:
            _cprint(f"  {_DIM}(>_<) Could not open editor: {exc}{_RST}")
            return

        if not composed:
            _cprint(f"  {_DIM}(._.) Empty prompt — nothing sent.{_RST}")
            return

        # One-shot seed: the interactive loop runs this as the next agent turn
        # right after process_command() returns (see cli.py main loop).
        self._pending_agent_seed = composed

    def _handle_footer_command(self, cmd_original: str) -> None:
        """Toggle or inspect ``display.runtime_footer.enabled`` from the CLI.

        Usage:
            /footer           → toggle
            /footer on|off    → explicit
            /footer status    → show current state
        """
        from cli import _cprint, save_config_value
        from hermes_cli.config import load_config
        from hermes_cli.colors import Colors as _Colors

        # Parse arg
        arg = ""
        try:
            parts = (cmd_original or "").strip().split(None, 1)
            if len(parts) > 1:
                arg = parts[1].strip().lower()
        except Exception:
            arg = ""

        cfg = load_config() or {}
        footer_cfg = ((cfg.get("display") or {}).get("runtime_footer") or {})
        current = bool(footer_cfg.get("enabled", False))
        fields = footer_cfg.get("fields") or ["model", "context_pct", "cwd"]

        if arg in {"status", "?"}:
            state = "ON" if current else "OFF"
            _cprint(
                f"  {_Colors.BOLD}Runtime footer:{_Colors.RESET} {state}\n"
                f"  Fields: {', '.join(fields)}"
            )
            return

        if arg in {"on", "enable", "true", "1"}:
            new_state = True
        elif arg in {"off", "disable", "false", "0"}:
            new_state = False
        elif arg == "":
            new_state = not current
        else:
            _cprint("  Usage: /footer [on|off|status]")
            return

        if save_config_value("display.runtime_footer.enabled", new_state):
            state = (
                f"{_Colors.GREEN}ON{_Colors.RESET}" if new_state
                else f"{_Colors.DIM}OFF{_Colors.RESET}"
            )
            _cprint(f"  Runtime footer: {state}")
        else:
            _cprint("  Failed to save runtime_footer setting to config.yaml")

    def _handle_timestamps_command(self, cmd_original: str) -> None:
        """Toggle or inspect ``display.timestamps`` from the CLI.

        When on, submitted and streamed message labels carry an ``[HH:MM]``
        suffix and ``/history`` prefixes each turn with its time (for turns
        that carry a stored timestamp).

        Usage:
            /timestamps           → toggle
            /timestamps on|off    → explicit
            /timestamps status    → show current state
        """
        from cli import _cprint, save_config_value
        from hermes_cli.colors import Colors as _Colors

        arg = ""
        try:
            parts = (cmd_original or "").strip().split(None, 1)
            if len(parts) > 1:
                arg = parts[1].strip().lower()
        except Exception:
            arg = ""

        current = bool(getattr(self, "show_timestamps", False))

        if arg in {"status", "?"}:
            state = "ON" if current else "OFF"
            _cprint(f"  {_Colors.BOLD}Message timestamps:{_Colors.RESET} {state}")
            return

        if arg in {"on", "enable", "true", "1"}:
            new_state = True
        elif arg in {"off", "disable", "false", "0"}:
            new_state = False
        elif arg == "":
            new_state = not current
        else:
            _cprint("  Usage: /timestamps [on|off|status]")
            return

        self.show_timestamps = new_state
        if save_config_value("display.timestamps", new_state):
            state = (
                f"{_Colors.GREEN}ON{_Colors.RESET}" if new_state
                else f"{_Colors.DIM}OFF{_Colors.RESET}"
            )
            _cprint(f"  Message timestamps: {state}")
        else:
            _cprint("  Failed to save timestamps setting to config.yaml")

    def _handle_reasoning_command(self, cmd: str):
        """Handle /reasoning — manage effort level and display toggle.

        Usage:
            /reasoning              Show current effort level and display state
            /reasoning <level>      Set effort for this session only (none, minimal, low, medium, high, xhigh, max, ultra)
            /reasoning <level> --global  Persist reasoning effort to config.yaml
            /reasoning show|on      Show model thinking/reasoning in output
            /reasoning hide|off     Hide model thinking/reasoning from output
            /reasoning full         Show complete thinking (no 10-line clamp)
            /reasoning clamp        Collapse long thinking to the first 10 lines
        """
        from cli import CLI_CONFIG, _ACCENT, _DIM, _RST, _cprint, _parse_reasoning_config, save_config_value
        parts = cmd.strip().split(maxsplit=1)

        if len(parts) < 2:
            # Show current state
            rc = self.reasoning_config
            if rc is None:
                level = "medium (default)"
            elif rc.get("enabled") is False:
                level = "none (disabled)"
            else:
                level = rc.get("effort", "medium")
            display_state = "on ✓" if self.show_reasoning else "off"
            full_state = "full" if getattr(self, "reasoning_full", False) else "clamped to 10 lines"
            _cprint(f"  {_ACCENT}Reasoning effort:  {level}{_RST}")
            _cprint(f"  {_ACCENT}Reasoning display: {display_state} ({full_state}){_RST}")
            _cprint(f"  {_DIM}Usage: /reasoning <none|minimal|low|medium|high|xhigh|max|ultra|show|hide|full|clamp> [--global]{_RST}")
            return

        arg = parts[1].strip().lower()
        arg_tokens = arg.split()
        # Session scope is the default; --global opts into persisting to
        # config.yaml. --session is accepted as an explicit no-op for parity
        # with /model and the gateway /reasoning handler.
        explicit_global = "--global" in arg_tokens
        if explicit_global or "--session" in arg_tokens:
            arg = " ".join(
                token for token in arg_tokens
                if token not in ("--global", "--session")
            )

        # Display toggle
        if arg in {"show", "on"}:
            self.show_reasoning = True
            if self.agent:
                self.agent.reasoning_callback = self._current_reasoning_callback()
            save_config_value("display.show_reasoning", True)
            _cprint(f"  {_ACCENT}✓ Reasoning display: ON (saved){_RST}")
            _cprint(f"  {_DIM}  Model thinking will be shown during and after each response.{_RST}")
            return
        if arg in {"hide", "off"}:
            self.show_reasoning = False
            if self.agent:
                self.agent.reasoning_callback = self._current_reasoning_callback()
            save_config_value("display.show_reasoning", False)
            _cprint(f"  {_ACCENT}✓ Reasoning display: OFF (saved){_RST}")
            return

        # Full / clamped recap toggle
        if arg in {"full", "all"}:
            self.reasoning_full = True
            save_config_value("display.reasoning_full", True)
            _cprint(f"  {_ACCENT}✓ Reasoning display: FULL (saved){_RST}")
            _cprint(f"  {_DIM}  The post-response recap box will print complete thinking.{_RST}")
            if not self.show_reasoning:
                _cprint(f"  {_DIM}  Note: reasoning display is OFF — run /reasoning show to see it.{_RST}")
            return
        if arg in {"clamp", "collapse", "short"}:
            self.reasoning_full = False
            save_config_value("display.reasoning_full", False)
            _cprint(f"  {_ACCENT}✓ Reasoning display: CLAMPED to 10 lines (saved){_RST}")
            return

        # Effort level change
        parsed = _parse_reasoning_config(arg)
        if parsed is None:
            _cprint(f"  {_DIM}(._.) Unknown argument: {arg}{_RST}")
            _cprint(f"  {_DIM}Valid levels: none, minimal, low, medium, high, xhigh, max, ultra{_RST}")
            _cprint(f"  {_DIM}Display:      show, hide{_RST}")
            _cprint(f"  {_DIM}Scope:        session-scoped by default, --global to persist{_RST}")
            return

        self.reasoning_config = parsed
        self.agent = None  # Force agent re-init with new reasoning config

        if explicit_global and save_config_value("agent.reasoning_effort", arg):
            agent_cfg = CLI_CONFIG.get("agent")
            if not isinstance(agent_cfg, dict):
                agent_cfg = {}
                CLI_CONFIG["agent"] = agent_cfg
            agent_cfg["reasoning_effort"] = arg
            _cprint(f"  {_ACCENT}✓ Reasoning effort set to '{arg}' (saved to config){_RST}")
        elif explicit_global:
            _cprint(f"  {_ACCENT}✓ Reasoning effort set to '{arg}' (session only; config save failed){_RST}")
        else:
            _cprint(f"  {_ACCENT}✓ Reasoning effort set to '{arg}' (this session — use --global to persist){_RST}")

    def _handle_busy_command(self, cmd: str):
        """Handle /busy — control what Enter does while Hermes is working.

        Usage:
            /busy               Show current busy input mode
            /busy status        Show current busy input mode
            /busy queue         Queue input for the next turn instead of interrupting
            /busy steer         Inject Enter mid-run via /steer (after next tool call)
            /busy interrupt     Interrupt the current run on Enter (default)
        """
        from cli import _ACCENT, _DIM, _RST, _cprint, save_config_value
        parts = cmd.strip().split(maxsplit=1)
        if len(parts) < 2 or parts[1].strip().lower() == "status":
            _cprint(f"  {_ACCENT}Busy input mode: {self.busy_input_mode}{_RST}")
            if self.busy_input_mode == "queue":
                _behavior = "queues for next turn"
            elif self.busy_input_mode == "steer":
                _behavior = "steers into current run (after next tool call)"
            else:
                _behavior = "interrupts current run"
            _cprint(f"  {_DIM}Enter while busy: {_behavior}{_RST}")
            _cprint(f"  {_DIM}Usage: /busy [queue|steer|interrupt|status]{_RST}")
            return

        arg = parts[1].strip().lower()
        if arg not in {"queue", "interrupt", "steer"}:
            _cprint(f"  {_DIM}(._.) Unknown argument: {arg}{_RST}")
            _cprint(f"  {_DIM}Usage: /busy [queue|steer|interrupt|status]{_RST}")
            return

        self.busy_input_mode = arg
        if save_config_value("display.busy_input_mode", arg):
            if arg == "queue":
                behavior = "Enter will queue follow-up input while Hermes is busy."
            elif arg == "steer":
                behavior = "Enter will steer your message into the current run (after the next tool call)."
            else:
                behavior = "Enter will interrupt the current run while Hermes is busy."
            _cprint(f"  {_ACCENT}✓ Busy input mode set to '{arg}' (saved to config){_RST}")
            _cprint(f"  {_DIM}{behavior}{_RST}")
        else:
            _cprint(f"  {_ACCENT}✓ Busy input mode set to '{arg}' (session only){_RST}")

    def _handle_fast_command(self, cmd: str):
        """Handle /fast — toggle fast mode (OpenAI Priority Processing / Anthropic Fast Mode).

        Session-scoped by default; ``--global`` persists agent.service_tier
        to config.yaml (parity with /model and /reasoning).
        """
        from cli import _ACCENT, _DIM, _RST, _cprint, save_config_value
        if not self._fast_command_available():
            _cprint("  (._.) /fast is only available for models that support fast mode (OpenAI Priority Processing or Anthropic Fast Mode).")
            return

        # Determine the branding for the current model
        try:
            from hermes_cli.models import _is_anthropic_fast_model
            agent = getattr(self, "agent", None)
            model = getattr(agent, "model", None) or getattr(self, "model", None)
            feature_name = "Anthropic Fast Mode" if _is_anthropic_fast_model(model) else "Priority Processing"
        except Exception:
            feature_name = "Fast mode"

        parts = cmd.strip().split(maxsplit=1)
        if len(parts) < 2 or parts[1].strip().lower() == "status":
            status = "fast" if self.service_tier == "priority" else "normal"
            _cprint(f"  {_ACCENT}{feature_name}: {status}{_RST}")
            _cprint(f"  {_DIM}Usage: /fast [normal|fast|status] [--global]{_RST}")
            return

        arg_tokens = parts[1].strip().lower().split()
        explicit_global = "--global" in arg_tokens
        arg = " ".join(
            token for token in arg_tokens
            if token not in ("--global", "--session")
        )

        if arg in {"fast", "on"}:
            self.service_tier = "priority"
            saved_value = "fast"
            label = "FAST"
        elif arg in {"normal", "off"}:
            self.service_tier = None
            saved_value = "normal"
            label = "NORMAL"
        else:
            _cprint(f"  {_DIM}(._.) Unknown argument: {arg}{_RST}")
            _cprint(f"  {_DIM}Usage: /fast [normal|fast|status] [--global]{_RST}")
            return

        self.agent = None  # Force agent re-init with new service-tier config
        if explicit_global and save_config_value("agent.service_tier", saved_value):
            _cprint(f"  {_ACCENT}✓ {feature_name} set to {label} (saved to config){_RST}")
        elif explicit_global:
            _cprint(f"  {_ACCENT}✓ {feature_name} set to {label} (session only; config save failed){_RST}")
        else:
            _cprint(f"  {_ACCENT}✓ {feature_name} set to {label} (this session — use --global to persist){_RST}")

    def _handle_debug_command(self, cmd_original: str = ""):
        """Handle /debug — upload debug report + logs and print share URLs.

        Accepts optional destination words after the command:

        - ``/debug``        → upload to the public paste service (default)
        - ``/debug nous``   → upload to Nous-internal storage (private, staff-only)
        - ``/debug local``  → render the report to stdout, no upload

        ``nous`` and ``local`` are mutually exclusive; if both are given,
        ``local`` wins (it never touches the network).
        """
        from hermes_cli.debug import run_debug_share
        from types import SimpleNamespace

        words = {w.lower() for w in cmd_original.split()[1:]}
        local = "local" in words
        nous = "nous" in words and not local
        # Typing the /debug slash command is itself the explicit consent to
        # upload, so we pass yes=True to skip run_debug_share's [y/N] prompt.
        # input() would hang inside prompt_toolkit's event loop anyway.
        args = SimpleNamespace(
            lines=200, expire=7, local=local, nous=nous, yes=True
        )
        run_debug_share(args)

    def _handle_update_command(self) -> bool:
        """Handle /update — update Hermes Agent to the latest version.

        In the classic CLI this exits the session and relaunches as
        ``hermes update`` so the user sees update output directly and gets
        the new version on next launch.

        Returns ``True`` when the update was confirmed (caller should trigger
        app exit so the relaunch is deferred to the main thread after
        prompt_toolkit cleans up terminal modes).  Returns ``False`` / falsy
        when cancelled.
        """
        from hermes_cli.config import is_managed, format_managed_message

        if is_managed():
            print(f"  ✗ {format_managed_message('update Hermes Agent')}")
            return False

        # Use the prompt_toolkit-native modal so the confirmation panel
        # renders properly above the composer and avoids raw input() races
        # with the prompt_toolkit event loop (same pattern as
        # _confirm_destructive_slash).
        choices = [
            ("once", "Update Now", "exit the current session and update Hermes Agent"),
            ("cancel", "Cancel", "keep the current session"),
        ]
        raw = self._prompt_text_input_modal(
            title="⚕  Update Hermes Agent",
            detail="This will exit the current session and run `hermes update`.",
            choices=choices,
        )
        if raw is None:
            print("  🟡 /update cancelled.")
            return False
        choice = self._normalize_slash_confirm_choice(raw, choices)
        if choice != "once":
            print("  🟡 /update cancelled.")
            return False

        print()
        print("  ⚕ Launching update...")
        print()

        # Store the relaunch args so run() can exec them from the main thread
        # after prompt_toolkit exits and restores terminal modes.  Calling
        # relaunch() directly here (from the process_loop daemon thread) would
        # skip terminal cleanup on POSIX (execvp replaces the process mid-TUI)
        # and only exit the worker thread on Windows (subprocess.run +
        # sys.exit inside a non-main thread does not exit the process).
        self._pending_relaunch = ["update"]
        return True

    def _handle_voice_command(self, command: str):
        """Handle /voice [on|off|tts|status] command."""
        from cli import _cprint
        parts = command.strip().split(maxsplit=1)
        subcommand = parts[1].lower().strip() if len(parts) > 1 else ""

        if subcommand == "on":
            self._enable_voice_mode()
        elif subcommand == "off":
            self._disable_voice_mode()
        elif subcommand == "tts":
            self._toggle_voice_tts()
        elif subcommand == "status":
            self._show_voice_status()
        elif subcommand == "":
            # Toggle
            if self._voice_mode:
                self._disable_voice_mode()
            else:
                self._enable_voice_mode()
        else:
            _cprint(f"Unknown voice subcommand: {subcommand}")
            _cprint("Usage: /voice [on|off|tts|status]")
