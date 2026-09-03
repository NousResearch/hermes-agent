"""``hermes grant`` — manage explicitly approved one-shot destructive grants.

Issuance is NOT available from this CLI.  A destructive grant is minted
exclusively inside the long-lived Hermes process that will consume it: the
model requests the governed operation, the process captures the live guest
incarnation, asks the human for an explicit ``approve once`` decision
through the approval surface, and issues the grant with its own
process-local authority.  A short-lived CLI process cannot sign a grant the
running Hermes process would accept (different authority generation), so
``hermes grant issue`` refuses explicitly and points to the governed tool.

``hermes grant list`` / ``revoke`` / ``audit`` remain available for
operational visibility.

Example::

    hermes grant list
    hermes grant revoke <grant_id>
    hermes grant audit [--limit 200]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict

from tools.destructive_grants import GrantError


def build_grant_parser(subparsers, *, cmd_grant=None) -> None:
    parser = subparsers.add_parser(
        "grant",
        help="Manage explicitly approved one-shot destructive grants.",
    )
    grant_sub = parser.add_subparsers(dest="grant_command")

    issue = grant_sub.add_parser(
        "issue",
        help="REFUSED: destructive grants are minted by the running Hermes process, not the CLI.",
    )
    issue.add_argument(
        "--operation",
        help="Operation, e.g. CREATE_FILESYSTEM.",
    )
    issue.add_argument("--vm", dest="vm_id", help="Target VM id.")
    issue.add_argument("--hostname", help="Expected guest hostname.")
    issue.add_argument(
        "--device",
        help="Exact validated block-device path, e.g. /dev/sdb1.",
    )
    issue.add_argument(
        "--fs",
        dest="fs_type",
        help="Filesystem type (ext4|xfs).",
    )
    issue.add_argument("--label", help="Filesystem label (1-16 chars).")
    issue.add_argument(
        "--subject",
        help="Human-readable audit subject (not an authority credential).",
    )
    issue.add_argument(
        "--session",
        dest="session_id",
        help="Hermes session id bound to the approval and grant.",
    )
    issue.add_argument(
        "--ttl",
        type=int,
        default=600,
        help="Requested lifetime in seconds (default 600, max 3600).",
    )
    issue.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    grant_sub.add_parser("list", help="List live grants.")
    grant_sub.add_parser("audit", help="Show the grant audit trail (no secrets).")

    revoke = grant_sub.add_parser("revoke", help="Revoke a live grant.")
    revoke.add_argument("grant_id", help="Opaque grant id.")

    if cmd_grant is not None:
        for sub in (issue, revoke):
            sub.set_defaults(func=cmd_grant)
        for name in ("list", "audit"):
            grant_sub.choices[name].set_defaults(func=cmd_grant)


def run_grant_command(args) -> int:
    from tools.destructive_grants import (
        list_grants,
        read_audit_trail,
        revoke_grant,
    )

    cmd = getattr(args, "grant_command", None)

    if cmd == "issue":
        # Track A (review #100694 process-boundary blocker): a short-lived
        # CLI process must NOT mint a grant the running Hermes process would
        # consume — the authority generations differ, so the grant would be
        # rejected at claim time anyway.  Issuance happens exclusively inside
        # the long-lived Hermes process via the governed tool, which asks
        # the human through the approval surface.
        print(
            "REFUSED: destructive grants are minted by the running Hermes "
            "process, not by this CLI. Request the governed operation "
            "(e.g. governed_mkfs) in a live Hermes session; the process will "
            "ask the human for an explicit one-shot approval of the exact "
            "target and issue the grant in-process.",
            file=sys.stderr,
        )
        return 2

    if cmd == "list":
        grants = list_grants()
        if not grants:
            print("No live grants.")
            return 0
        for grant in grants:
            print(
                f"{grant.grant_id}  {grant.operation}  vm={grant.vm_id}  "
                f"{grant.device}  {grant.fs_type}  label={grant.label}  "
                f"subject={grant.authorization_subject}  "
                f"expires_at={grant.expires_at}"
            )
        return 0

    if cmd == "audit":
        entries = read_audit_trail(limit=getattr(args, "limit", 200))
        if not entries:
            print("No audit entries.")
            return 0
        for entry in entries:
            print(json.dumps(entry, sort_keys=True, ensure_ascii=False))
        return 0

    if cmd == "revoke":
        try:
            ok = revoke_grant(args.grant_id)
        except (OSError, GrantError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if ok:
            print(f"revoked={args.grant_id}")
            return 0
        print(f"not_found={args.grant_id}", file=sys.stderr)
        return 1

    print("usage: hermes grant {issue,list,revoke,audit}", file=sys.stderr)
    return 2
