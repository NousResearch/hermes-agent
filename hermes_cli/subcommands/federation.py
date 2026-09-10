"""``hermes federation`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_federation_parser(subparsers, *, cmd_federation: Callable) -> None:
    parser = subparsers.add_parser(
        "federation",
        help="Audit and seed the governed specialist federation",
        description=(
            "Audit federation role coverage or plan explicit creation of missing "
            "specialist profiles. Seed is a dry-run unless --apply is supplied."
        ),
    )
    actions = parser.add_subparsers(dest="federation_action")

    audit = actions.add_parser(
        "audit", help="Report installed, covered, and missing roles"
    )
    audit.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    seed = actions.add_parser("seed", help="Plan or create missing federation profiles")
    seed.add_argument("--department", help="Limit seeding to one department id")
    seed.add_argument("--role", action="append", help="Limit seeding to a role id; repeatable")
    seed.add_argument("--apply", action="store_true", help="Create missing profiles")
    seed.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Adopt exact existing role profiles, preserving their current souls",
    )
    seed.add_argument(
        "--create-alias", action="store_true", help="Create wrapper aliases for seeded profiles"
    )
    seed.add_argument(
        "--groups", action="store_true", help="Seed durable Bot Mode groups and memberships"
    )
    seed.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    parser.set_defaults(func=cmd_federation)
