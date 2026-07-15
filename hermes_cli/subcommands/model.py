"""``hermes model`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

import argparse
from typing import Callable

MODEL_COMMAND_HELP = "Select and configure the default AI model"


def _positive_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("timeout must be greater than zero")
    return parsed


def build_model_parser(subparsers, *, cmd_model: Callable) -> None:
    """Attach the ``model`` subcommand to ``subparsers``."""
    # =========================================================================
    # model command
    # =========================================================================
    model_parser = subparsers.add_parser(
        "model",
        help="Select default model and provider",
        description="Interactively select your inference provider and default model",
    )
    model_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Wipe the model picker disk cache and re-fetch every provider's live /v1/models list.",
    )
    model_parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Read-only check that profile config, gateway process, source tree, "
            "and provider auth structure refer to the same runtime"
        ),
    )
    model_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit --preflight or transaction result as stable JSON",
    )
    model_parser.add_argument(
        "--provider",
        dest="transaction_provider",
        help="Target provider for a non-interactive transaction preview/apply",
    )
    model_parser.add_argument(
        "--model",
        dest="transaction_model",
        help="Target model for a non-interactive transaction preview/apply",
    )
    model_parser.add_argument(
        "--confirm-profile",
        help="Required profile name binding for transaction preview/apply",
    )
    model_parser.add_argument(
        "--apply-transaction",
        action="store_true",
        help=(
            "Apply the previewed provider/model, restart the active gateway, "
            "verify a real API event, and roll back automatically on failure"
        ),
    )
    model_parser.add_argument(
        "--config-lock-timeout",
        type=_positive_seconds,
        default=10.0,
        help="Cross-process config lock timeout for --apply-transaction",
    )
    model_parser.add_argument(
        "--restart-timeout",
        type=_positive_seconds,
        default=90.0,
        help="Gateway restart/readiness timeout for --apply-transaction",
    )
    model_parser.add_argument(
        "--smoke-timeout",
        type=_positive_seconds,
        default=180.0,
        help="Active-interpreter API smoke timeout for --apply-transaction",
    )
    model_parser.add_argument(
        "--portal-url",
        help="Portal base URL for Nous login (default: production portal)",
    )
    model_parser.add_argument(
        "--inference-url",
        help="Inference API base URL for Nous login (default: production inference API)",
    )
    model_parser.add_argument(
        "--client-id",
        default=None,
        help="OAuth client id to use for Nous login (default: hermes-cli)",
    )
    model_parser.add_argument(
        "--scope", default=None, help="OAuth scope to request for Nous login"
    )
    model_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not attempt to open the browser automatically during Nous login",
    )
    model_parser.add_argument(
        "--timeout",
        type=_positive_seconds,
        default=15.0,
        help="HTTP request timeout in seconds for Nous login (default: 15)",
    )
    model_parser.add_argument(
        "--ca-bundle", help="Path to CA bundle PEM file for Nous TLS verification"
    )
    model_parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification for Nous login (testing only)",
    )
    model_parser.set_defaults(func=cmd_model)
