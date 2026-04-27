"""Resolve credential strings with optional schema prefixes.

Background: Hermes reads API keys from several config fields as plain strings
(`model.api_key`, `custom_providers[*].api_key`). For users who store secrets
in OS keychains (macOS Keychain, GNOME Keyring) or environment variables, this
module lets the same fields hold a *reference* to the secret instead of the
secret itself.

Recognised schemas (everything else is treated as a literal value):

    env:VAR                          — read from os.environ[VAR]
    keychain:service[@account]       — macOS Keychain via `security`
    secret-tool:service[@account]    — Linux GNOME Keyring via `secret-tool`

The optional ``@account`` suffix selects a specific keychain account.
When omitted, the current user (``$USER`` / ``getpass.getuser()``) is used —
which matches common conventions like ``security find-generic-password
-a "$USER" -s "<service>"``. This means service names can contain slashes
(e.g. ``keychain:ai-system/litellm-master-key``) without ambiguity.

Resolution failures (missing env var, missing keychain item, missing CLI tool)
raise CredentialResolveError with a clear message — Hermes should fail loudly
rather than silently fall back to an empty key, because a user who wrote
``keychain:...`` explicitly opted in to a strict secret source.

Backwards compatibility: any string without a recognised prefix is returned
unchanged, so existing plaintext configs continue to work.
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess


SCHEMA_PREFIXES = ("env:", "keychain:", "secret-tool:")


class CredentialResolveError(RuntimeError):
    """Raised when a credential reference cannot be resolved."""


def resolve_credential_string(value: str) -> str:
    """Resolve a config string to its actual secret value.

    Returns the value unchanged when no recognised schema prefix is present.
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return stripped
    if not any(stripped.startswith(p) for p in SCHEMA_PREFIXES):
        return stripped

    schema, _, rest = stripped.partition(":")
    rest = rest.strip()
    if not rest:
        raise CredentialResolveError(
            f"Empty reference after schema '{schema}:' — expected '{schema}:<target>'"
        )

    if schema == "env":
        try:
            resolved = os.environ[rest]
        except KeyError:
            raise CredentialResolveError(
                f"Environment variable '{rest}' is not set (referenced as 'env:{rest}')"
            ) from None
        if not resolved:
            raise CredentialResolveError(
                f"Environment variable '{rest}' is empty (referenced as 'env:{rest}')"
            )
        return resolved

    if schema == "keychain":
        return _lookup_macos_keychain(rest)

    if schema == "secret-tool":
        return _lookup_secret_tool(rest)

    return stripped


def _split_service_account(target: str, schema: str) -> tuple[str, str]:
    """Parse 'service[@account]'. Account defaults to the current user.

    Slashes in the service portion are allowed and preserved — the only
    delimiter is the first ``@``, which separates an explicit account.
    """
    service, sep, account = target.partition("@")
    service = service.strip()
    if sep:
        account = account.strip()
        if not account:
            raise CredentialResolveError(
                f"Reference '{schema}:{target}' has empty account after '@'"
            )
    else:
        try:
            account = getpass.getuser()
        except Exception:
            account = os.environ.get("USER", "")
    if not service:
        raise CredentialResolveError(
            f"Reference '{schema}:{target}' has empty service"
        )
    if not account:
        raise CredentialResolveError(
            f"Reference '{schema}:{target}' has no account and current user is unknown"
        )
    return service, account


def _lookup_macos_keychain(target: str) -> str:
    service, account = _split_service_account(target, "keychain")
    if shutil.which("security") is None:
        raise CredentialResolveError(
            "macOS 'security' CLI not found — 'keychain:' references work only on macOS"
        )
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-w", "-a", account, "-s", service],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        raise CredentialResolveError(
            f"Keychain lookup timed out for service='{service}' account='{account}'"
        ) from None
    if result.returncode != 0:
        raise CredentialResolveError(
            f"Keychain item not found: service='{service}' account='{account}' "
            f"(security exit {result.returncode}: {result.stderr.strip() or 'no output'})"
        )
    secret = result.stdout.rstrip("\n")
    if not secret:
        raise CredentialResolveError(
            f"Keychain item is empty: service='{service}' account='{account}'"
        )
    return secret


def _lookup_secret_tool(target: str) -> str:
    service, account = _split_service_account(target, "secret-tool")
    if shutil.which("secret-tool") is None:
        raise CredentialResolveError(
            "'secret-tool' CLI not found — install libsecret-tools (Linux/GNOME Keyring)"
        )
    try:
        result = subprocess.run(
            ["secret-tool", "lookup", "service", service, "account", account],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        raise CredentialResolveError(
            f"secret-tool lookup timed out for service='{service}' account='{account}'"
        ) from None
    if result.returncode != 0 or not result.stdout:
        raise CredentialResolveError(
            f"secret-tool item not found: service='{service}' account='{account}' "
            f"(exit {result.returncode}: {result.stderr.strip() or 'no output'})"
        )
    return result.stdout.rstrip("\n")
