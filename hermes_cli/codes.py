"""``hermes codes put`` — park an SMS/email verification code server-side (#119683).

Prints only an opaque ``otp_…`` handle. The raw code never appears on stdout,
in logs, or in the model's context; it is registered with the vault redaction
boundary at mint time.
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, Namespace
from typing import Optional


def register_cli(subparser: ArgumentParser) -> None:
    """Build the ``hermes codes`` argparse tree (called from the subcommand module)."""
    subs = subparser.add_subparsers(dest="codes_action")

    p_put = subs.add_parser(
        "put",
        help="Park a verification code server-side; print only an opaque otp_… handle",
        description=(
            "Store a short-lived one-time code so browser_vault_enter_code can inject it "
            "without the code ever entering the conversation. Reads the code from the "
            "argument, or from stdin when the argument is '-'."
        ),
    )
    p_put.add_argument(
        "code",
        nargs="?",
        default=None,
        help="The raw code (prefer '-' or a pipe so it never appears in shell history / argv logs)",
    )
    p_put.add_argument(
        "--origin",
        default="",
        help="Optional site origin to bind the handle to (e.g. https://example.com)",
    )
    p_put.add_argument(
        "--source",
        default="",
        help="Label for where the code came from (sms, email, ...)",
    )
    p_put.add_argument(
        "--extract",
        action="store_true",
        help="Pull the OTP out of a full SMS/email body on stdin (body never printed; only the handle)",
    )
    p_put.add_argument(
        "--ttl",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Lifetime in seconds (default 300; clamped to 30..900)",
    )
    p_put.set_defaults(_codes_handler=_cmd_put)

    # Bare `hermes codes` → put with stdin (same UX as a secret read).
    subparser.set_defaults(_codes_handler=_cmd_put)


def _read_code(args: Namespace) -> str:
    code = args.code
    if code is None or code == "-":
        if sys.stdin.isatty() and code is None:
            raise SystemExit("error: pass the code as an argument, or pipe it on stdin with '-'")
        code = sys.stdin.read()
    raw = (code or "").strip().replace("\n", "").replace("\r", "")
    if getattr(args, "extract", False):
        from agent.code_registry import extract_verification_code

        found = extract_verification_code(raw)
        if not found:
            raise SystemExit("error: no verification code found in input")
        return found
    return raw


def _cmd_put(args: Namespace) -> None:
    from agent.code_registry import mint

    raw = _read_code(args)
    if not raw:
        raise SystemExit("error: code is required")
    ttl = args.ttl if getattr(args, "ttl", None) is not None else None
    kwargs = {}
    if getattr(args, "origin", ""):
        kwargs["origin"] = args.origin
    if getattr(args, "source", ""):
        kwargs["source"] = args.source
    if ttl is not None:
        kwargs["ttl_s"] = ttl
    try:
        out = mint(raw, **kwargs)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    except Exception as exc:  # VaultError on bad --origin, etc.
        raise SystemExit(f"error: {exc}") from exc
    # Only the handle leaves the process — never the raw code.
    json.dump(out, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


def codes_command(args: Namespace) -> None:
    handler = getattr(args, "_codes_handler", None)
    if handler is None:
        _cmd_put(args)
        return
    handler(args)
