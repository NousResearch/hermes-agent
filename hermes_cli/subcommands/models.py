"""``hermes models`` machine-readable model discovery parser."""

from __future__ import annotations

from typing import Callable


def build_models_parser(subparsers, *, cmd_models: Callable) -> None:
    parser = subparsers.add_parser(
        "models",
        help="Discover provider model catalogs as JSON",
        description="Noninteractively discover registered provider model metadata",
    )
    parser.add_argument("--json", action="store_true", required=True, help="Emit one JSON document")
    parser.add_argument("--provider", metavar="ID", help="Limit discovery to one registered provider")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh the selected provider's live catalog (requires --provider)",
    )
    parser.add_argument("--offline", action="store_true", help="Prohibit network access and allow stale cache data")
    parser.set_defaults(func=cmd_models)
