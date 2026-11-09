"""Implementation of the gateway run CLI command."""

from __future__ import annotations

import os


def record_exit_event(exit_diag, tag: str, **details) -> None:
    if exit_diag is not None:
        exit_diag(tag, **details)


def run_command(args) -> None:
    from hermes_cli.gateway import (
        EXTERNAL_GATEWAY_SUPERVISOR_ENV, _maybe_redirect_run_to_s6_supervision, run_gateway)

    control_only = getattr(args, "control_only", False)
    if control_only and (getattr(args, "replace", False) or getattr(args, "force", False)):
        raise SystemExit("--control-only cannot be combined with --replace or --force")
    if not control_only and _maybe_redirect_run_to_s6_supervision(args):
        return  # unreachable; execvp doesn't return
    if getattr(args, "external_supervisor", False):
        os.environ[EXTERNAL_GATEWAY_SUPERVISOR_ENV] = "1"
    run_gateway(
        getattr(args, "verbose", 0), quiet=getattr(args, "quiet", False),
        replace=getattr(args, "replace", False), force=getattr(args, "force", False),
        idle_exit=getattr(args, "idle_exit", False), control_only=control_only)
