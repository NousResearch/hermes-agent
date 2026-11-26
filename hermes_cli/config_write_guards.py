"""Write-side guards for the ``hermes config`` write commands.

Split out of :mod:`hermes_cli.config` (code-health FILE_LINES ratchet); the CLI
command handlers late-import these. Companion to :mod:`hermes_cli.config_section_guard`.
"""

from __future__ import annotations

import os

# The terminal tool bridges the gateway session's HERMES_SESSION_* context into the child
# environment (tools/environments/local.py), so their presence marks a `hermes config`
# invocation as agent-owned. A human shell carries none of them; in-process callers
# (approval mode, console engine) never go through these CLI command handlers at all.
_AGENT_SESSION_ENV_MARKERS = ("HERMES_SESSION_KEY", "HERMES_SESSION_ID", "HERMES_SESSION_PLATFORM")


def exit_if_agent_session(action: str) -> None:
    """Refuse `hermes config set/unset` launched from an agent-owned session (fail closed).

    The file-tool guard already blocks agent edits of config.yaml and names this CLI as the
    human alternative — without this check that advice is an open relay: an agent pins
    ``delegation.model`` on a paid route itself and every spawned child inherits it (#128524)."""
    # Membership, not truthiness: the session bridge writes "" for an engaged-but-None
    # value, and that must still refuse.
    present = [name for name in _AGENT_SESSION_ENV_MARKERS if name in os.environ]
    if not present:
        return
    from hermes_cli.config import _exit_invalid
    _exit_invalid(
        f"✗ Refused: 'hermes config {action}' is running inside an agent session "
        f"({', '.join(present)} is set).\n"
        "  Agents cannot modify security-sensitive configuration — the same rule that blocks "
        "file-tool writes to config.yaml (#128524).\n"
        "  Ask the user to run this command in their own shell instead.")


def run_write_command(action: str, fn, *args) -> None:
    """Run a config writer behind the agent-session guard, surfacing the fail-closed write
    guard's RuntimeError as a clean CLI error instead of a traceback."""
    exit_if_agent_session(action)
    try:
        fn(*args)
    except RuntimeError as exc:
        from hermes_cli.config import _exit_invalid
        _exit_invalid(f"✗ {exc}")
