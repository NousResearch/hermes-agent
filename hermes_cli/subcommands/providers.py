"""Parser for the tier-0 provider compatibility smoke command."""

from __future__ import annotations

from typing import Callable


def build_providers_parser(subparsers, *, cmd_providers: Callable) -> None:
    """Attach ``providers`` and its verbs to the shared CLI parser."""

    parser = subparsers.add_parser(
        "providers",
        help="Run the tier-0 provider compatibility smoke",
        description=(
            "Run the explicitly labeled tier-0 provider compatibility smoke. "
            "It is not Hermes qualification, replacement evidence, a benchmark, "
            "or a routing decision."
        ),
    )
    verbs = parser.add_subparsers(dest="providers_command")

    validate = verbs.add_parser(
        "validate",
        help="Run the six-case tier-0 compatibility smoke",
        description=(
            "Run the explicitly labeled tier-0 compatibility smoke through "
            "real `hermes chat -Q` turns and persisted SessionDB receipts. "
            "This is not qualification or replacement evidence."
        ),
    )
    validate.add_argument("--provider", help="Inference provider")
    validate.add_argument("--model", help="Model identifier")
    validate.add_argument(
        "--toolsets",
        default="file",
        choices=["file"],
        help="Tier-0 toolset (only file is supported)",
    )
    validate.add_argument(
        "--suite",
        default="agent-readiness",
        choices=["agent-readiness"],
        help="Tier-0 suite (default: agent-readiness)",
    )
    validate.add_argument("--out", help="Directory for smoke receipts")
    validate.add_argument(
        "--timeout", type=float, default=120.0, help="Per-case timeout in seconds"
    )
    validate.add_argument(
        "--hermes-executable",
        help="Hermes executable override for local/fake integration tests",
    )
    validate.set_defaults(func=cmd_providers)
    parser.set_defaults(func=cmd_providers)
