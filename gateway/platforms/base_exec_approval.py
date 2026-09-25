"""Shared wording for the exec-approval prompt every messaging surface renders.

The button card (``BasePlatformAdapter._format_exec_approval``) and the plain-text
``/approve`` fallback (``gateway.run._format_exec_approval_fallback``) must tell the user
the same three things: what Hermes wants to run, why it was flagged, and that silence means
the command does NOT run once ``approvals.timeout`` elapses. Keeping the text here means one
edit changes every platform; adapters only wrap these strings in their own markup.

No imports from ``gateway.platforms.base`` or ``gateway.run`` — both import this module.
"""

from __future__ import annotations

# Bare strings; adapters add their own bold/HTML around them.
EA_HEADER_TEXT = "Hermes wants to run a command that needs your OK"
EA_REASON_LABEL_TEXT = "Why it was flagged"

# The plain-text fallback shows this many characters of the command before cutting. It matches
# the button card's ``BasePlatformAdapter._EA_CMD_BUDGET``: a human approving a command has to be
# able to read it, and 200 characters (the old cap) hid the end of ordinary multi-line commands.
EA_FALLBACK_CMD_BUDGET = 3000
# On an adapter whose message cap cannot hold the full budget the preview shrinks to fit, but
# never below the old fixed cut.
EA_FALLBACK_CMD_FLOOR = 200


def fit_command_preview(command: str, budget: int = EA_FALLBACK_CMD_BUDGET) -> str:
    """``command`` whole when it fits ``budget``; otherwise the first ``budget`` characters and a
    marker that says how much was cut, so the reader knows the prompt is not the whole command."""
    if len(command) <= budget:
        return command
    return f"{command[:budget]}\n... [{len(command) - budget} more characters not shown]"

# Timeout notice posted when nobody answered the prompt (``{window}`` = "5 minutes").
APPROVAL_TIMED_OUT_NOTICE = (
    "⌛ Approval timed out after {window} — the command was NOT run. "
    "Ask me to try again if you still want it, or raise approvals.timeout in config.yaml.")


def approval_timeout_seconds() -> int:
    """The configured ``approvals.timeout`` (default 300s); module attribute so tests can pin it."""
    from tools.approval_context import _get_approval_timeout
    return _get_approval_timeout()


def format_approval_window(seconds: int) -> str:
    """Human wording for a timeout (300 → "5 minutes"); one formatter shared with the CLI notice and
    the tool result's ``user_summary`` — see ``tools.approval_context.format_approval_window``."""
    from tools.approval_context import format_approval_window as _shared
    return _shared(seconds)


def format_approval_deadline_line(timeout_s: int) -> str:
    """The last line of every approval prompt: doing nothing is a safe no."""
    return f"If you don't answer within {format_approval_window(timeout_s)} it will NOT run."


def format_approval_timed_out_notice(timeout_s: int) -> str:
    return APPROVAL_TIMED_OUT_NOTICE.format(window=format_approval_window(timeout_s))
