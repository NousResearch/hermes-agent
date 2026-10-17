"""``hermes provider`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_provider_parser(subparsers, *, cmd_provider: Callable) -> None:
    """Attach the ``provider`` subcommand to ``subparsers``."""
    provider_parser = subparsers.add_parser(
        "provider", help="Inspect and remove configured inference providers",
        description="Inspect and remove configured inference providers")
    provider_sub = provider_parser.add_subparsers(dest="provider_action")

    remove_parser = provider_sub.add_parser(
        "remove", help="Remove a provider from config.yaml and its credential pool",
        description=(
            "Remove a provider definition from its owning config schema (legacy "
            "custom_providers or v12+ providers), rewrite or clear the references that "
            "point at it, and prune its auth.json credential pool."))
    remove_parser.add_argument(
        "name", help="Provider name or providers.<key> slug to remove")
    remove_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show every store and reference that would change, and write nothing")
    remove_parser.add_argument(
        "--reassign", action="append", default=[], metavar="PATH=PROVIDER",
        help=("Reference to repoint, e.g. --reassign model.provider=openrouter "
              "(repeatable)"))
    remove_parser.add_argument(
        "--clear", action="append", default=[], metavar="PATH",
        help="Reference to drop outright, e.g. --clear auxiliary.vision.provider (repeatable)")
    remove_parser.add_argument(
        "--force", action="store_true",
        help="Clear every remaining reference instead of failing on unresolved ones")
    remove_parser.add_argument(
        "--keep-credentials", action="store_true",
        help="Leave the auth.json credential pool in place")
    remove_parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON")
    remove_parser.set_defaults(func=cmd_provider)
