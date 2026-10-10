"""``hermes sandbox`` subcommand — run code you do not trust in a throwaway container.

``hermes sandbox run`` copies a PR, ref or directory without executing anything it carries and
runs a command on it in a locked container (no network, no credentials, read-only root). The
parser lives here; the logic lives in ``hermes_cli.sandbox_cmd``.
"""

from __future__ import annotations

import argparse


def cmd_sandbox_run(args) -> int:  # noqa: ANN001
    from hermes_cli.sandbox_cmd import run_sandbox

    return run_sandbox(args)


def cmd_sandbox(args) -> int:  # noqa: ANN001
    sub = getattr(args, "sandbox_command", None)
    if sub in ("run",):
        return cmd_sandbox_run(args)
    print("usage: hermes sandbox run (--pr N | --ref REF | --path DIR) [options] -- COMMAND")
    print("  see: hermes sandbox run --help")
    return 0


def build_sandbox_parser(subparsers) -> None:
    """Attach the ``sandbox`` subcommand to ``subparsers``."""
    sandbox_parser = subparsers.add_parser(
        "sandbox",
        help="Run code from a repo or PR you do not trust in a throwaway container",
        description=(
            "Run another author's code (a PR, a ref, a downloaded tree) in a throwaway "
            "Docker/Podman container: no network, no credentials or host environment, read-only "
            "root, unprivileged user, no host mount but a scratch copy that is deleted afterwards. "
            "Exits 69 when no container runtime is available; never run the code on the host instead. "
            "A tool to use, not a security boundary. See: "
            "https://hermes-agent.nousresearch.com/docs/user-guide/security#untrusted-code"
        ),
    )
    sandbox_subparsers = sandbox_parser.add_subparsers(dest="sandbox_command")
    run = sandbox_subparsers.add_parser(
        "run",
        help="Copy the code without running any of it, then run COMMAND in a locked container",
        description=(
            "Copy the code into a scratch directory without executing anything it carries "
            "(no checkout, no hooks, symlinks kept as links), optionally run --setup, then run COMMAND "
            "with no network. Exit status is COMMAND's (124 = timed out, 69 = no container runtime)."
        ),
        epilog=(
            "examples:\n"
            "  hermes sandbox run --pr 123 -- python -m pytest -q tests/test_x.py\n"
            "  hermes sandbox run --ref origin/feature --setup 'pip install --user -e .' --setup-network open "
            "-- python -m pytest -q\n"
            "  hermes sandbox run --path ./downloaded-repo --setup 'npm ci' --setup-network open -- npm test"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--pr", metavar="N",
                        help="Pull request number; pull/N/head is fetched into a private ref, never checked out")
    source.add_argument("--ref", metavar="REF", help="Commit, branch or tag of --repo to export with git archive")
    source.add_argument("--path", metavar="DIR",
                        help="Directory to copy (.git and special files skipped, symlinks kept as links)")
    run.add_argument("--repo", default=".", metavar="DIR",
                     help="Git repository for --pr/--ref (default: current directory)")
    run.add_argument("--remote", default="origin", help="Remote to fetch --pr from (default: origin)")
    run.add_argument("--setup", metavar="CMD",
                     help="Shell command run first in its own container (no credentials), e.g. dependency installs")
    run.add_argument("--setup-network", choices=("none", "open"), default="none",
                     help="Network for --setup: none (default) or open (the runtime's default network: install "
                          "scripts can then reach services on this host, the local network and the internet). "
                          "The run step never has a network")
    run.add_argument("--image", metavar="IMAGE",
                     help="Container image (default: terminal.docker_image)")
    run.add_argument("--timeout", type=float, default=900, metavar="SECONDS",
                     help="Per-step timeout in seconds (default: 900)")
    run.add_argument("run_command", nargs=argparse.REMAINDER, metavar="COMMAND",
                     help="Command to run, after --")
    sandbox_parser.set_defaults(func=cmd_sandbox)
