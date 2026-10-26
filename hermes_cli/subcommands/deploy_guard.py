"""``hermes deploy-guard`` — CLI surface for the deploy-branch guard.

Split out of ``hermes_cli.main`` so the facade does not grow: the CLI
wiring for the deploy guard lives here, and the guard's own inspection
logic stays in ``hermes_cli.deploy_guard``.

Subcommands:
  - ``check``  — print the current state, exit 0 on match, exit 2 on drift.
  - ``init``   — write ``HERMES_DEPLOY_BRANCH`` to ``~/.hermes/.env`` if not
                 already present, so operators can pick up the canonical-branch
                 invariant with one shell command after pulling a fix.

The dispatcher calls ``check_deploy_branch`` on every tick already; this
subcommand is for ad-hoc verification and for shell scripts that want a
reliable exit code (the cron pump can't easily read the dispatcher's stderr).
See t_7161d9a9, t_4722df6f.
"""

from __future__ import annotations

import os
import sys

# The canonical deploy branch this guard compares against. The deploy guard
# reads HERMES_DEPLOY_BRANCH from the environment first; this is the
# fallback and the value ``init`` writes.
DEFAULT_DEPLOY_BRANCH = "canonical-deploy-t_7161d9a9"


def cmd_deploy_guard(args):
    """``hermes deploy-guard <subcommand>`` — gate the running tree against the canonical deploy branch."""
    sub = getattr(args, "deploy_guard_command", None) or "check"
    if sub == "check":
        from hermes_cli.deploy_guard import check_deploy_state
        result = check_deploy_state()
        print(f"agent_home       = {result.agent_home}")
        print(f"expected_branch  = {result.expected_branch}")
        print(f"observed_branch  = {result.observed_branch or '(none)'}")
        print(f"observed_sha     = {result.observed_sha or '(none)'}")
        print(f"reason           = {result.reason}")
        print(f"ok               = {result.ok}")
        sys.exit(0 if result.ok else 2)
    if sub == "init":
        env_path = os.path.expanduser("~/.hermes/.env")  # health: allow HX001 -- the deploy branch invariant lives in the SHARED hermes home, not a per-profile home
        line = f"HERMES_DEPLOY_BRANCH={DEFAULT_DEPLOY_BRANCH}\n"
        if os.path.exists(env_path):
            existing = open(env_path, encoding="utf-8-sig").read()
            if "HERMES_DEPLOY_BRANCH=" in existing:
                print(f"HERMES_DEPLOY_BRANCH already set in {env_path}; leaving as-is.")
                sys.exit(0)
            with open(env_path, "a", encoding="utf-8") as fh:
                fh.write(line)
        else:
            with open(env_path, "w", encoding="utf-8") as fh:
                fh.write(line)
            os.chmod(env_path, 0o600)
        print(f"wrote {line.strip()} to {env_path}")
        sys.exit(0)
    print(f"unknown subcommand: {sub}", file=sys.stderr)
    sys.exit(2)


def build_deploy_guard_parser(subparsers):
    """Register ``hermes deploy-guard <subcommand>``. See t_7161d9a9."""
    p = subparsers.add_parser(
        "deploy-guard",
        help="Gate the running tree against the canonical deploy branch (t_7161d9a9).",
        description=(
            "Inspects the live working tree's HEAD branch against the canonical "
            "deploy branch (env HERMES_DEPLOY_BRANCH, default canonical-deploy-t_7161d9a9). "
            "Returns 0 on match, 2 on drift. The dispatcher already runs this check "
            "on every tick; this command is for ad-hoc verification and shell scripts."
        ),
    )
    subs = p.add_subparsers(dest="deploy_guard_command")
    p_check = subs.add_parser("check", help="Print state + exit 0/2.")
    p_check.set_defaults(func=cmd_deploy_guard)
    p_init = subs.add_parser(
        "init",
        help=f"Append HERMES_DEPLOY_BRANCH={DEFAULT_DEPLOY_BRANCH} to ~/.hermes/.env if missing.",
    )
    p_init.set_defaults(func=cmd_deploy_guard)
    # When the user runs ``hermes deploy-guard`` with no subcommand, fall back
    # to ``check`` — argparse subparsers dest= with no value lands as None.
    p.set_defaults(func=cmd_deploy_guard)
    return p
