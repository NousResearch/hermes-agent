"""Shared persistent approval-mode command logic.

Approval mode is profile-scoped configuration, not conversation state. Changing
it affects subsequent terminal guard checks immediately because approval.py
loads config on each check; it must not rebuild a live agent or mutate its
system prompt/tool schema, preserving the prompt-cache prefix.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Optional

VALID_APPROVAL_MODES = ("manual", "smart", "off")


@dataclass(frozen=True)
class ApprovalModeResult:
    ok: bool
    mode: str
    changed: bool
    message: str


def _effective_mode() -> str:
    """Return the exact mode enforced by the terminal approval guard."""
    from tools.approval import _get_approval_mode

    return _get_approval_mode()


def run_approval_mode_command(requested_mode: Optional[str]) -> ApprovalModeResult:
    """Inspect or persist ``approvals.mode`` through canonical config APIs."""
    current = _effective_mode()
    requested = (requested_mode or "").strip().lower()

    if not requested:
        return ApprovalModeResult(
            True,
            current,
            False,
            f"Approval mode: {current} (persistent profile setting).",
        )
    if requested not in VALID_APPROVAL_MODES:
        return ApprovalModeResult(
            False,
            current,
            False,
            "Usage: /approvals [manual|smart|off]",
        )

    # set_config_value is the canonical managed-scope/write-safety chokepoint.
    # It reports managed policy through stderr + SystemExit, so capture that for
    # slash-command output instead of terminating the interactive worker.
    from hermes_cli.config import set_config_value

    output = StringIO()
    try:
        with redirect_stdout(output), redirect_stderr(output):
            set_config_value("approvals.mode", requested)
    except SystemExit:
        detail = output.getvalue().strip() or "Approval mode is managed and cannot be changed."
        return ApprovalModeResult(False, current, False, detail)
    except Exception as exc:
        return ApprovalModeResult(
            False,
            current,
            False,
            f"Failed to save approval mode: {exc}",
        )

    effective = _effective_mode()
    if effective != requested:
        return ApprovalModeResult(
            False,
            effective,
            False,
            f"Approval mode remains {effective}; the requested value did not become effective.",
        )
    return ApprovalModeResult(
        True,
        effective,
        effective != current,
        f"Approval mode: {effective} (persistent profile setting).",
    )
