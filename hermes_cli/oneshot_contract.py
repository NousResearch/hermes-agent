"""Per-run scripting flags of a finite run (``hermes -z`` / ``hermes chat -q``).

``--on-approval`` overrides ``approvals.single_query_mode`` for this run only and travels with the
submission (the turn runs in the gateway, so a process environment variable would not reach it);
``--timeout`` bounds the client's wait. The exit codes these produce are in
``hermes_cli.turn_exit.EXIT_CODES``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

OnApproval = Literal["deny", "approve", "wait", "fail"]
ON_APPROVAL_CHOICES = ("deny", "approve", "wait", "fail")

# What the submission commits (``gateway.session_finite.FINITE_APPROVAL_MODES``): ``wait`` and
# ``fail`` both ask the gateway to send the approval to this viewer; the client then holds for an
# answer from another surface or stops the run.
_WIRE_APPROVAL = {"deny": "deny", "approve": "approve", "wait": "ask", "fail": "ask"}


@dataclass(frozen=True)
class RunPolicy:
    on_approval: OnApproval | None = None
    timeout: float | None = None

    @classmethod
    def from_args(cls, args) -> RunPolicy:
        return cls(on_approval=getattr(args, "on_approval", None), timeout=getattr(args, "run_timeout", None))

    def submit_fields(self, *, oneshot_z: bool) -> dict:
        """The ``prompt.submit`` approval fields. ``hermes -z`` keeps its legacy auto-approval
        (``unattended``) unless ``--on-approval`` says otherwise; ``{}`` lets
        ``approvals.single_query_mode`` decide."""
        if oneshot_z and self.on_approval in (None, "approve"):
            return {"unattended": True}
        return {"on_approval": _WIRE_APPROVAL[self.on_approval]} if self.on_approval else {}


def positive_seconds(raw: str) -> float:
    """argparse ``type`` for ``--timeout``: a positive number of seconds."""
    import argparse
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid seconds: {raw!r}") from None
    if not value > 0:
        raise argparse.ArgumentTypeError("--timeout must be greater than 0")
    return value


def scripted_flags_requested(args) -> bool:
    """True when ``--on-approval`` or ``--timeout`` was passed to ``hermes chat``. They only mean
    something for a run that answers one query and exits, so without a query (or under ``--tui``)
    this is a usage error: exit 2."""
    flags = [flag for flag, dest in (("--on-approval", "on_approval"), ("--timeout", "run_timeout"))
             if getattr(args, dest, None) is not None]
    if not flags:
        return False
    if not (getattr(args, "query", None) or getattr(args, "query_file", None)):
        print(f"Error: {' / '.join(flags)} requires -q/--query (or hermes -z).", file=sys.stderr)
        raise SystemExit(2)
    if getattr(args, "tui", False):
        print(f"Error: {' / '.join(flags)} cannot be used with --tui.", file=sys.stderr)
        raise SystemExit(2)
    return True


def transport_chat_requested(args) -> bool:
    """``hermes chat`` flags only the gateway transport client serves (``--format stream-json|json``,
    ``--on-approval``, ``--timeout``); each exits 2 when combined with no query or with ``--tui``."""
    from hermes_cli.stream_json import stream_json_requested
    return stream_json_requested(args) | scripted_flags_requested(args)
