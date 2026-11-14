"""``hermes model`` subcommand parser."""

from __future__ import annotations

import sys
from typing import Callable


def build_model_parser(subparsers, *, cmd_model: Callable) -> None:
    """Attach the ``model`` subcommand to ``subparsers``."""
    model_parser = subparsers.add_parser(
        "model", help="Select default model and provider",
        description="Interactively select your inference provider and default model, or list a "
                    "provider's models with --list")
    model_parser.add_argument(
        "--refresh", action="store_true",
        help="Wipe the model picker disk cache and re-fetch every provider's live /v1/models list.")
    model_parser.add_argument(
        "--list", action="store_true",
        help="Print the provider's model ids and exit (no picker; works without a TTY). "
             "The configured default is marked with *.")
    model_parser.add_argument(
        "--provider", default=None,
        help="Provider to list with --list (default: the configured provider)")
    model_parser.add_argument(
        "--json", action="store_true", help="With --list, print one JSON object instead of text")
    model_parser.add_argument(
        "--portal-url", help="Portal base URL for Nous login (default: production portal)")
    model_parser.add_argument(
        "--inference-url",
        help="Inference API base URL for Nous login (default: production inference API)")
    model_parser.add_argument(
        "--client-id", default=None,
        help="OAuth client id to use for Nous login (default: hermes-cli)")
    model_parser.add_argument("--scope", default=None, help="OAuth scope to request for Nous login")
    model_parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not attempt to open the browser automatically during Nous login")
    model_parser.add_argument(
        "--timeout", type=float, default=15.0,
        help="HTTP request timeout in seconds for Nous login (default: 15)")
    model_parser.add_argument(
        "--ca-bundle", help="Path to CA bundle PEM file for Nous TLS verification")
    model_parser.add_argument(
        "--insecure", action="store_true",
        help="Disable TLS verification for Nous login (testing only)")
    def _run(args):
        if args.list:  # the picker needs a TTY; --list must work from scripts and CI
            from hermes_cli.model_list_cmd import run_model_list
            sys.exit(run_model_list(args))
        return cmd_model(args)

    model_parser.set_defaults(func=_run)
