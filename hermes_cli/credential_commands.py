"""Masked credential-entry CLI for opaque agent credential refs."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from typing import Any

from agent.credential_store import (
    CredentialStoreError,
    delete_credential,
    list_credentials,
    request_credential,
    revoke_credential,
    set_credential_value,
    update_credential_value,
)


def _print_public(record: dict[str, Any]) -> None:
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))


def _prompt_secret(confirm: bool = True) -> str:
    first = getpass.getpass("Credential value: ")
    if not first:
        raise CredentialStoreError("credential value cannot be empty")
    if confirm:
        second = getpass.getpass("Confirm credential value: ")
        if first != second:
            raise CredentialStoreError("credential values did not match")
    return first


def cmd_credentials(args: argparse.Namespace) -> int:
    try:
        command = getattr(args, "credentials_command", None)
        if command == "request":
            _print_public(request_credential(args.name, args.type))
            return 0
        if command == "set":
            value = _prompt_secret(confirm=not args.no_confirm)
            _print_public(set_credential_value(args.name, args.type, value))
            return 0
        if command == "update":
            value = _prompt_secret(confirm=not args.no_confirm)
            _print_public(update_credential_value(args.credential_ref, value))
            return 0
        if command in {"list", "ls"}:
            _print_public({"credentials": list_credentials()})
            return 0
        if command == "revoke":
            _print_public(revoke_credential(args.credential_ref))
            return 0
        if command in {"delete", "rm"}:
            _print_public(delete_credential(args.credential_ref))
            return 0
    except CredentialStoreError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1
    print("Run `hermes credentials --help`.", file=sys.stderr)
    return 1


def build_credentials_parser(subparsers) -> argparse.ArgumentParser:  # noqa: ANN001
    parser = subparsers.add_parser(
        "credentials",
        aliases=["credential"],
        help="Enter and manage opaque agent credentials via masked terminal UI",
        description=(
            "Secure credential entry for agent workflows. The model receives "
            "only opaque credential refs; values are entered with getpass in a "
            "separate masked terminal prompt and stored encrypted per profile."
        ),
    )
    subs = parser.add_subparsers(dest="credentials_command")

    request = subs.add_parser("request", help="Create/return a pending opaque credential ref")
    request.add_argument("name", help="Credential name, e.g. github-pat")
    request.add_argument("--type", default="secret", help="Credential type, e.g. api_key/token/password")
    request.set_defaults(func=cmd_credentials)

    set_cmd = subs.add_parser("set", help="Set a credential value using masked input")
    set_cmd.add_argument("name", help="Credential name")
    set_cmd.add_argument("--type", default="secret", help="Credential type")
    set_cmd.add_argument("--no-confirm", action="store_true", help="Skip second masked confirmation prompt")
    set_cmd.set_defaults(func=cmd_credentials)

    update = subs.add_parser("update", help="Update an existing credential by opaque ref")
    update.add_argument("credential_ref", help="Opaque credential ref")
    update.add_argument("--no-confirm", action="store_true", help="Skip second masked confirmation prompt")
    update.set_defaults(func=cmd_credentials)

    list_cmd = subs.add_parser("list", aliases=["ls"], help="List refs and metadata only")
    list_cmd.set_defaults(func=cmd_credentials)

    revoke = subs.add_parser("revoke", help="Revoke a credential; future resolution fails")
    revoke.add_argument("credential_ref", help="Opaque credential ref")
    revoke.set_defaults(func=cmd_credentials)

    delete = subs.add_parser("delete", aliases=["rm"], help="Delete a credential record")
    delete.add_argument("credential_ref", help="Opaque credential ref")
    delete.set_defaults(func=cmd_credentials)

    parser.set_defaults(func=cmd_credentials)
    return parser
