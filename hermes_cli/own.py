"""`hermes own` — manage automation ownership and scaffold automation bundles."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from agent.automation_ownership import (
    Identity, artifact_key, add_collaborator, claim, get_record,
    list_for_user, register_creator, remove_collaborator, transfer,
)
from hermes_constants import get_hermes_home

_BUNDLE_MANIFEST = """\
# Automation bundle manifest
name: {name}
owner: {owner}
description: ""
collaborators: []
# Every toolset the automation's cron job needs to do its work (e.g.
# jira_write, github_rw, slack_post). A job whose workdir is this bundle is
# refused at create/update — and its runs are blocked with an alert to the
# owner — while any of these would be unavailable to it.
requires_toolsets: []
links:
  skills: []
  cron_jobs: []
  scripts: []
"""

_BUNDLE_WORKFLOW = """\
# {name}

What this automation does, how the pieces fit together, and how to run it.
"""


def _ident(user: str, name: str | None, platform: str = "slack") -> Identity:
    return Identity(platform=platform, user_id=user, display_name=name or user)


def run_own(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="hermes own")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("--user", required=True)

    p_claim = sub.add_parser("claim")
    p_claim.add_argument("key")
    p_claim.add_argument("--user", default=None)
    p_claim.add_argument("--name", default=None)

    p_tr = sub.add_parser("transfer")
    p_tr.add_argument("key")
    p_tr.add_argument("--to", required=True)
    p_tr.add_argument("--to-name", default=None)
    p_tr.add_argument("--by", default="")
    p_tr.add_argument("--by-name", default=None)
    p_tr.add_argument("--admin", action="store_true")

    p_co = sub.add_parser("collab")
    p_co.add_argument("op", choices=["add", "remove"])
    p_co.add_argument("key")
    p_co.add_argument("--user", required=True)
    p_co.add_argument("--name", default=None)

    p_init = sub.add_parser("init")
    p_init.add_argument("name")
    p_init.add_argument("--user", default=None)
    p_init.add_argument("--name", dest="display", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "list":
        out = list_for_user(args.user)
        for k in out["owned"]:
            print(f"owner        {k}")
        for k in out["collaborator"]:
            print(f"collaborator {k}")
        return 0

    if args.cmd == "claim":
        if args.user:
            ident = _ident(args.user, args.name)
        else:
            import agent.automation_ownership as _ao_mod
            ident = _ao_mod.current_identity()
            if ident is None:
                print("could not determine your identity; pass --user <id>")
                return 1
        try:
            rec = claim(args.key, args.key.split(":", 1)[0], ident)
        except PermissionError as e:
            print(f"Error: {e}")
            return 1
        print(f"Claimed {args.key} for {rec['owner']['display_name']}")
        return 0

    if args.cmd == "transfer":
        try:
            rec = transfer(
                args.key, _ident(args.to, args.to_name),
                by=_ident(args.by or "cli", args.by_name), by_is_admin=args.admin,
            )
        except (KeyError, PermissionError) as e:
            print(f"Error: {e}")
            return 1
        print(f"Transferred {args.key} to {rec['owner']['display_name']}")
        return 0

    if args.cmd == "collab":
        if get_record(args.key) is None:
            print(f"Error: no ownership record for {args.key}")
            return 1
        # A shell caller is an admin by definition (reaching this CLI from a
        # messaging platform requires the ``terminal`` toolset), so the CLI
        # keeps its unrestricted behavior while the ownership tool's callers
        # go through the owner-or-admin check.
        if args.op == "add":
            add_collaborator(args.key, _ident(args.user, args.name), by_is_admin=True)
            print(f"Added collaborator {args.user} to {args.key}")
        else:
            remove_collaborator(args.key, args.user, by_is_admin=True)
            print(f"Removed collaborator {args.user} from {args.key}")
        return 0

    if args.cmd == "init":
        base = get_hermes_home() / "automations" / args.name
        (base / "scripts").mkdir(parents=True, exist_ok=True)
        (base / "assets").mkdir(parents=True, exist_ok=True)
        if args.user:
            owner = _ident(args.user, args.display)
        else:
            import agent.automation_ownership as _ao_mod
            owner = _ao_mod.current_identity()
            if owner is None:
                print("could not determine your identity; pass --user <id>")
                return 1
        (base / "automation.yaml").write_text(
            _BUNDLE_MANIFEST.format(name=args.name, owner=owner.display_name), encoding="utf-8")
        (base / "workflow.md").write_text(
            _BUNDLE_WORKFLOW.format(name=args.name), encoding="utf-8")
        register_creator(artifact_key("automation", args.name), "automation", owner)
        print(f"Initialized automation bundle at {base}")
        return 0

    return 2
