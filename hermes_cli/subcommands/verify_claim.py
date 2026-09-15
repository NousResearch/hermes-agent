"""``hermes verify-claim`` subcommand parser.

Follows the pattern of ``hermes_cli/subcommands/verify.py``: the parser is built
here and the handler is injected, so nothing in this module imports ``main``.
"""

from __future__ import annotations

from typing import Callable

from hermes_cli.subcommands._shared import add_json_flag

# Keep in sync with hermes_cli/verify_claim.py; not imported here so `hermes`
# startup does not pay for the evidence-collection module.
DEFAULT_REPO = "NousResearch/hermes-agent"
DEFAULT_BASE_REF = "origin/main"


def build_verify_claim_parser(subparsers, *, cmd_verify_claim: Callable) -> None:
    """Attach the ``verify-claim`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "verify-claim",
        help="Build a landing-evidence bundle for an issue/PR claim (is-ancestor proof)",
        description="Given an issue or PR number, collect a machine-checkable evidence "
            "bundle: what is claimed (fixed / duplicate / fix PR), the canonical's "
            "landing merge commit plus a local `git merge-base --is-ancestor <sha> "
            "<base>` proof, and the repro record (test names, red-on-base -> green-after) "
            "parsed out of the fix PR body. `state_reason: completed` is recorded as a "
            "label only and is never treated as landing evidence. Exit codes: 0 = evidence "
            "complete, 1 = missing/unverifiable evidence, 2 = usage error, 3 = refuted "
            "(a landing commit exists but is not reachable from the base ref).")
    parser.add_argument("number", type=int, help="Issue or pull-request number")
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, metavar="OWNER/REPO",
        help=f"Repository to ask (default: {DEFAULT_REPO})")
    parser.add_argument(
        "-C", "--repo-dir", default=".", metavar="PATH",
        help="Local clone used for the is-ancestor proof (default: current directory)")
    parser.add_argument(
        "--base", default=DEFAULT_BASE_REF, metavar="REF",
        help=f"Base ref the landing commit must be reachable from (default: {DEFAULT_BASE_REF})")
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="Never `git fetch`; report unverified when the landing commit is absent locally")
    parser.add_argument(
        "--repro-optional", action="store_true",
        help="Report a PR's repro record without letting it gate the exit code "
            "(proportionality valve for docs/typo PRs)")
    add_json_flag(parser, "Emit the bundle as JSON (the machine-checkable artifact)")
    parser.add_argument(
        "--compact", action="store_true", help="With --json: single-line output")
    parser.set_defaults(func=cmd_verify_claim)
