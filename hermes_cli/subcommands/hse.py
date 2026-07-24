"""``hermes hse`` / ``hermes evolve`` parser.

Thin operating-copy bridge into the standalone ``hermes-agent-self-evolution``
repository.  The default status command is read-only; module execution is
restricted to explicit ``evolution.*`` modules.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable


def _add_common_repo_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hse-repo",
        help="Path to hermes-agent-self-evolution repo (default: ~/.hermes/evolution/repos/hermes-agent-self-evolution)",
    )
    parser.add_argument(
        "--active-hermes-repo",
        help="Path to target Hermes checkout (default: this Hermes source checkout or HERMES_AGENT_REPO)",
    )


def _build_single_parser(subparsers, name: str, *, cmd_hse: Callable, help_text: str) -> None:
    parser = subparsers.add_parser(name, help=help_text, description=help_text)
    parser.set_defaults(func=cmd_hse, hse_command="status")
    _add_common_repo_flags(parser)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON status")

    nested = parser.add_subparsers(dest="hse_command")

    status = nested.add_parser("status", help="Show read-only HSE bridge status")
    _add_common_repo_flags(status)
    status.add_argument("--json", action="store_true", help="Emit machine-readable JSON status")
    status.set_defaults(func=cmd_hse)

    module = nested.add_parser(
        "module",
        help="Run an explicit evolution.* module from the HSE repo",
        description="Run an explicit evolution.* module from the HSE repo. Use -- before module arguments.",
    )
    _add_common_repo_flags(module)
    module.add_argument("module", help="Python module name; must start with evolution.")
    module.add_argument("module_args", nargs=argparse.REMAINDER, help="Arguments passed through to the evolution module")
    module.set_defaults(func=cmd_hse)


def build_hse_parser(subparsers, *, cmd_hse: Callable) -> None:
    """Attach ``hse`` and ``evolve`` top-level commands."""

    _build_single_parser(
        subparsers,
        "hse",
        cmd_hse=cmd_hse,
        help_text="Hermes Self-Evolution operating-copy bridge",
    )
    _build_single_parser(
        subparsers,
        "evolve",
        cmd_hse=cmd_hse,
        help_text="Alias for `hermes hse`",
    )
