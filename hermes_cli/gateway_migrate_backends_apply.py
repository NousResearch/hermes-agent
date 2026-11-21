"""``hermes gateway migrate --backends --apply --dry-run``: the plan for moving this home from the old
``hermes dashboard`` / ``hermes serve`` backend to the gateway's Webapp on the same address, and every
reason it cannot happen. Changes nothing.

Exit codes: 0 the move could run, or nothing to do; 2 invalid invocation; 3 blocked by preflight; 6 this
kind of backend cannot be migrated automatically (the manual steps are printed).
"""

from __future__ import annotations

import json
import sys
from typing import Optional

from hermes_cli.gateway_migrate_backends_plan import (
    EXIT_BLOCKED, EXIT_OK, EXIT_UNSUPPORTED, EXIT_USAGE, BackendPlan, build_plan, format_plan, parse_address,
)


def _usage_error(args) -> Optional[str]:
    if getattr(args, "multiplex", False):
        return "--backends cannot be combined with --multiplex."
    apply_only = ("address", "keep_webapp_address", "force_interrupt")
    if not getattr(args, "apply", False) and any(getattr(args, f, None) for f in apply_only):
        return "--address, --keep-webapp-address and --force-interrupt go with --apply."
    if not getattr(args, "dry_run", False):
        return "--apply needs --dry-run in this version: it prints the migration plan and changes nothing."
    if getattr(args, "address", None) and parse_address(args.address) is None:
        return f"--address must be HOST:PORT, got {args.address!r}."
    return None


def _print_plan(plan: BackendPlan, *, as_json: bool) -> None:
    if as_json:
        view = plan.to_dict()
        print(json.dumps({"applied": False, "dry_run": True, "steps": view["steps"], "plan": view}, indent=2))
    else:
        print("\n".join(format_plan(plan, dry_run=True)))


def cmd_apply(host, args) -> int:
    """``--apply --dry-run``: print the plan and its blockers, change nothing; the exit code."""
    address = parse_address(args.address) if getattr(args, "address", None) else None
    plan = build_plan(host, address=address, keep_webapp=bool(getattr(args, "keep_webapp_address", False)),
                      force_interrupt=bool(getattr(args, "force_interrupt", False)))
    _print_plan(plan, as_json=bool(getattr(args, "json", False)))
    return plan.exit_code()


def cmd_backends(args) -> None:
    """Dispatch for every ``--backends`` mode beyond the read-only report."""
    from hermes_cli.gateway_migrate_backends_host import SystemHost
    from hermes_constants import get_hermes_home
    if (err := _usage_error(args)) is not None:
        print(f"✗ {err}", file=sys.stderr)
        sys.exit(EXIT_USAGE)
    sys.exit(cmd_apply(SystemHost(get_hermes_home()), args))


__all__ = ["EXIT_BLOCKED", "EXIT_OK", "EXIT_UNSUPPORTED", "cmd_backends"]
