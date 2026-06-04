"""Hermes-native credential helpers.

This module intentionally treats macOS Keychain as the local secret store and
keeps secret values out of return payloads, logs, and command output. Tool code
should retrieve secrets at execution time and let Hermes approvals decide
whether the action itself is safe.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable

DEFAULT_KEYCHAIN_SERVICE = "hermes-agent"
SERVICE_ENV_VAR = "HERMES_KEYCHAIN_SERVICE"
CHIEF_OF_STAFF_KEYCHAIN_SERVICES = (
    "chief-of-staff",
    "assistant-to-the-it-manager",
)


@dataclass(frozen=True)
class CredentialSpec:
    """A named Hermes credential and its macOS Keychain account mapping."""

    name: str
    account: str
    description: str
    source_accounts: tuple[str, ...] = ()

    def candidate_source_accounts(self) -> tuple[str, ...]:
        return self.source_accounts or (self.account,)


CREDENTIAL_SPECS: tuple[CredentialSpec, ...] = (
    CredentialSpec(
        name="jira.api_token",
        account="JIRA_API_TOKEN",
        description="Jira Cloud API token",
    ),
    CredentialSpec(
        name="atlassian.api_token",
        account="ATLASSIAN_API_TOKEN",
        description="Shared Atlassian API token usable by Confluence/Jira",
    ),
    CredentialSpec(
        name="confluence.api_token",
        account="CONFLUENCE_API_TOKEN",
        description="Confluence Cloud API token",
        source_accounts=("CONFLUENCE_API_TOKEN", "ATLASSIAN_API_TOKEN", "JIRA_API_TOKEN"),
    ),
    CredentialSpec(
        name="zendesk.api_token",
        account="ZENDESK_API_TOKEN",
        description="Zendesk API token",
    ),
    CredentialSpec(
        name="zendesk.oauth_token",
        account="ZENDESK_OAUTH_TOKEN",
        description="Zendesk OAuth bearer token",
    ),
)
SPECS_BY_NAME = {spec.name: spec for spec in CREDENTIAL_SPECS}
SPECS_BY_ACCOUNT = {spec.account: spec for spec in CREDENTIAL_SPECS}


def keychain_service_name(service: str | None = None) -> str:
    """Return the Hermes Keychain service name."""

    value = service or os.environ.get(SERVICE_ENV_VAR, "") or DEFAULT_KEYCHAIN_SERVICE
    return value.strip() or DEFAULT_KEYCHAIN_SERVICE


def resolve_credential_spec(name_or_account: str) -> CredentialSpec:
    """Resolve a known credential name or Keychain account name."""

    key = (name_or_account or "").strip()
    if key in SPECS_BY_NAME:
        return SPECS_BY_NAME[key]
    if key in SPECS_BY_ACCOUNT:
        return SPECS_BY_ACCOUNT[key]
    available = ", ".join(spec.name for spec in CREDENTIAL_SPECS)
    raise KeyError(f"Unknown Hermes credential: {name_or_account}. Known credentials: {available}")


def _run_security(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=False, capture_output=True, text=True)


def get_keychain_password(account: str, service: str | None = None) -> str:
    """Read a generic password from macOS Keychain, returning empty if missing.

    The caller must not print the returned value.
    """

    if sys.platform != "darwin":
        return ""
    resolved_service = keychain_service_name(service)
    result = _run_security(
        [
            "security",
            "find-generic-password",
            "-s",
            resolved_service,
            "-a",
            account,
            "-w",
        ]
    )
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def set_keychain_password(account: str, password: str, service: str | None = None) -> None:
    """Write a generic password to macOS Keychain."""

    if sys.platform != "darwin":
        raise RuntimeError("macOS Keychain credential storage is only available on macOS")
    resolved_service = keychain_service_name(service)
    result = _run_security(
        [
            "security",
            "add-generic-password",
            "-U",
            "-s",
            resolved_service,
            "-a",
            account,
            "-w",
            password,
        ]
    )
    if result.returncode != 0:
        message = (result.stderr or "security add-generic-password failed").strip()
        raise RuntimeError(message)


def delete_keychain_password(account: str, service: str | None = None) -> bool:
    """Delete a generic password. Returns True when Security reported success."""

    if sys.platform != "darwin":
        return False
    resolved_service = keychain_service_name(service)
    result = _run_security(
        [
            "security",
            "delete-generic-password",
            "-s",
            resolved_service,
            "-a",
            account,
        ]
    )
    return result.returncode == 0


def has_keychain_secret(account: str, service: str | None = None) -> bool:
    return bool(get_keychain_password(account, service=service))


def copy_keychain_secret(
    *,
    source_service: str,
    source_account: str,
    destination_service: str | None = None,
    destination_account: str | None = None,
) -> dict[str, object]:
    """Copy one Keychain secret without returning or printing the secret value."""

    destination_service = keychain_service_name(destination_service)
    destination_account = destination_account or source_account
    value = get_keychain_password(source_account, service=source_service)
    if not value:
        return {
            "copied": False,
            "source_service": source_service,
            "source_account": source_account,
            "destination_service": destination_service,
            "destination_account": destination_account,
            "reason": "source_missing",
        }
    set_keychain_password(destination_account, value, service=destination_service)
    return {
        "copied": True,
        "source_service": source_service,
        "source_account": source_account,
        "destination_service": destination_service,
        "destination_account": destination_account,
    }


def _first_source_hit(
    *,
    source_services: Iterable[str],
    source_accounts: Iterable[str],
) -> tuple[str, str, str] | None:
    for service in source_services:
        for account in source_accounts:
            value = get_keychain_password(account, service=service)
            if value:
                return service, account, value
    return None


def import_chief_of_staff_credentials(
    *,
    destination_service: str | None = None,
    source_services: Iterable[str] = CHIEF_OF_STAFF_KEYCHAIN_SERVICES,
    specs: Iterable[CredentialSpec] = CREDENTIAL_SPECS,
) -> dict[str, object]:
    """Copy known chief-of-staff credentials into Hermes' native Keychain service.

    The result only includes names/accounts and copied/missing status. Secret
    values never leave this function.
    """

    resolved_destination = keychain_service_name(destination_service)
    copied: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for spec in specs:
        hit = _first_source_hit(
            source_services=source_services,
            source_accounts=spec.candidate_source_accounts(),
        )
        if not hit:
            missing.append({"name": spec.name, "account": spec.account})
            continue
        source_service, source_account, value = hit
        set_keychain_password(spec.account, value, service=resolved_destination)
        copied.append(
            {
                "name": spec.name,
                "account": spec.account,
                "source_service": source_service,
                "source_account": source_account,
            }
        )
    return {
        "destination_service": resolved_destination,
        "copied": copied,
        "missing": missing,
    }


def credential_status(service: str | None = None) -> dict[str, object]:
    resolved_service = keychain_service_name(service)
    return {
        "service": resolved_service,
        "credentials": [
            {
                "name": spec.name,
                "account": spec.account,
                "description": spec.description,
                "present": has_keychain_secret(spec.account, service=resolved_service),
            }
            for spec in CREDENTIAL_SPECS
        ],
    }


def _print_json(payload: object) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    return _print_json(credential_status(service=args.service))


def cmd_set(args: argparse.Namespace) -> int:
    spec = resolve_credential_spec(args.name)
    value = args.value
    if not value:
        value = getpass.getpass(f"Value for {spec.name} ({spec.account}): ")
    if not value:
        raise SystemExit("No credential value provided.")
    set_keychain_password(spec.account, value, service=args.service)
    return _print_json({"stored": True, "service": keychain_service_name(args.service), "name": spec.name, "account": spec.account})


def cmd_delete(args: argparse.Namespace) -> int:
    spec = resolve_credential_spec(args.name)
    deleted = delete_keychain_password(spec.account, service=args.service)
    return _print_json({"deleted": deleted, "service": keychain_service_name(args.service), "name": spec.name, "account": spec.account})


def cmd_copy(args: argparse.Namespace) -> int:
    spec = resolve_credential_spec(args.name)
    source_account = args.from_account or spec.account
    return _print_json(
        copy_keychain_secret(
            source_service=args.from_service,
            source_account=source_account,
            destination_service=args.service,
            destination_account=spec.account,
        )
    )


def cmd_import_chief_of_staff(args: argparse.Namespace) -> int:
    return _print_json(import_chief_of_staff_credentials(destination_service=args.service))


def register_credentials_subparser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "credentials",
        help="Manage Hermes-native credentials in macOS Keychain",
        description=(
            "Store integration credentials in the Hermes macOS Keychain service. "
            "Commands never print secret values."
        ),
    )
    parser.set_defaults(func=lambda args: (parser.print_help(), 0)[1])
    credential_subparsers = parser.add_subparsers(dest="credentials_action")

    status = credential_subparsers.add_parser("status", aliases=["list"], help="Show known credential presence")
    status.add_argument("--service", help=f"Keychain service name (default: {DEFAULT_KEYCHAIN_SERVICE})")
    status.set_defaults(func=cmd_status)

    set_cmd = credential_subparsers.add_parser("set", help="Store one known credential")
    set_cmd.add_argument("name", help="Credential name or account, e.g. jira.api_token")
    set_cmd.add_argument("--service", help=f"Keychain service name (default: {DEFAULT_KEYCHAIN_SERVICE})")
    set_cmd.add_argument("--value", help="Secret value. If omitted, Hermes prompts securely.")
    set_cmd.set_defaults(func=cmd_set)

    delete_cmd = credential_subparsers.add_parser("delete", help="Delete one known credential")
    delete_cmd.add_argument("name", help="Credential name or account, e.g. jira.api_token")
    delete_cmd.add_argument("--service", help=f"Keychain service name (default: {DEFAULT_KEYCHAIN_SERVICE})")
    delete_cmd.set_defaults(func=cmd_delete)

    copy_cmd = credential_subparsers.add_parser("copy", help="Copy one credential from another Keychain service")
    copy_cmd.add_argument("name", help="Credential name or account, e.g. jira.api_token")
    copy_cmd.add_argument("--from-service", required=True, help="Source Keychain service")
    copy_cmd.add_argument("--from-account", help="Source account override")
    copy_cmd.add_argument("--service", help=f"Destination service (default: {DEFAULT_KEYCHAIN_SERVICE})")
    copy_cmd.set_defaults(func=cmd_copy)

    import_cos = credential_subparsers.add_parser(
        "import-chief-of-staff",
        help="Copy known chief-of-staff business-system credentials into Hermes",
    )
    import_cos.add_argument("--service", help=f"Destination service (default: {DEFAULT_KEYCHAIN_SERVICE})")
    import_cos.set_defaults(func=cmd_import_chief_of_staff)

    return parser


__all__ = [
    "CHIEF_OF_STAFF_KEYCHAIN_SERVICES",
    "CREDENTIAL_SPECS",
    "DEFAULT_KEYCHAIN_SERVICE",
    "CredentialSpec",
    "copy_keychain_secret",
    "credential_status",
    "delete_keychain_password",
    "get_keychain_password",
    "has_keychain_secret",
    "import_chief_of_staff_credentials",
    "keychain_service_name",
    "register_credentials_subparser",
    "resolve_credential_spec",
    "set_keychain_password",
]
