"""``/cron`` on the shared gateway: the classic CLI's handler (hermes_cli/cli_cron_lines.py), run by the owner.

The profile's cron store belongs to the gateway whose ticker fires its jobs, so no client writes it. Bare
``/cron`` and ``list`` read, for every local session (main's TUI ran the same text in its slash worker). Every
other form writes the store: it runs here when the classic CLI sends it on its own command socket, and a
viewer's connection (Ink, Desktop, ACP) keeps main's refusal (gateway/session_commands.py).
"""
from __future__ import annotations


def cron_read(tokens):
    """``'overview'`` / ``'list'`` / ``'list-all'`` for the read forms of the shlex-split ``/cron`` argument;
    None for a write."""
    if not tokens:
        return 'overview'
    if tokens[0].lower() == 'list' and set(tokens[1:]) <= {'--all', '-a'}:
        return 'list-all' if tokens[1:] else 'list'
    return None


async def run_cron(authority, arg) -> dict:
    """``/cron <arg>`` off the owner loop in the owning profile: the store is written here, never by a client."""
    from gateway.session_info_reads import owner_lines
    from hermes_cli.cli_cron_lines import cron_command_lines
    # A /cron run waits for its job, and the job's admission and execution take threads from the loop's
    # default pool (tracked_write, session_cron.execute). The wait therefore holds a thread of the runner's
    # unbounded turn pool, never one of the pool the job needs, however many runs (or Ctrl+C'd waits) pile up.
    return await owner_lines(authority, lambda: cron_command_lines(arg),
                             getattr(authority.runner, '_run_in_executor_with_context', None))
