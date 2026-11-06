"""``/tree`` — the session-family navigator's classic-CLI command.

A topical sibling of ``cli_commands_mixin.py``: that module is over the file-line ratchet and may
only shrink, so new command surface goes in a sibling (root AGENTS.md, "Facade + siblings").
"""

from __future__ import annotations

from hermes_cli.cli_commands_session_tools import _command_arg, _cp, _t


class CLICommandsTreeMixin:
    """The classic CLI's ``/tree``: print or pick from the session family around this session."""

    def _handle_tree_command(self, cmd_original: str) -> None:
        """Handle /tree [--all] — show this session's family and switch to any session in it.

        The family is the whole tree around the current session: its root, every branch and
        descendant, and (behind ``--all``) the compression continuations folded into their parent
        and the subagent runs hidden by default. Enter switches through the ordinary /resume path,
        so a switch mid-conversation behaves exactly like typing ``/resume <id>``.
        """
        import sys
        from hermes_cli.cli_commands_mixin import _db_unavailable_line
        from hermes_cli.session_tree import empty_line, load_family, parse_show_all, pick_session, tree_lines, usage_line

        show_all = parse_show_all(_command_arg(cmd_original))
        if show_all is None:
            return _cp(f"  {usage_line()}")
        if not self._session_db:
            return _cp(_db_unavailable_line())
        # The picker owns the screen: a streaming turn would repaint over it, so refuse mid-turn
        # rather than leave the user with a half-drawn menu (same guard /branch and /rewind take).
        if getattr(self, "_agent_running", False):
            return _cp(f"  {_t('shared.agent_busy', command='/tree')}")
        family = load_family(self._session_db, self.session_id, show_all=show_all)
        if not family.nodes:
            return _cp(f"  {empty_line()}")
        # A terminal gets the picker and nothing printed first (painting rows and then taking the
        # screen for curses races the prompt's queued repaints); every other reader gets text.
        if not sys.stdin.isatty() or len(family.nodes) < 2:
            for line in tree_lines(family):
                _cp(f"  {line}")
            return None
        picked = pick_session(family)
        if picked is None or picked == self.session_id:
            return None
        self._handle_resume_command(f"/resume {picked}")
