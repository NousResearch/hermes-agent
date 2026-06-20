"""Account alias path helpers for Google Workspace skill scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

# Ensure sibling modules (_hermes_home) are importable when run standalone.
import sys

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _hermes_home import get_hermes_home


_ACCOUNT_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class GoogleAccountPaths:
    """Credential paths for either the legacy default account or a named alias."""

    account: str | None
    root: Path
    token: Path
    client_secret: Path
    pending_auth: Path


def normalize_account_alias(account: str | None) -> str | None:
    """Validate and normalize an optional account alias.

    ``None`` means the legacy single-account files directly under HERMES_HOME.
    Named accounts are intentionally limited to simple path-safe aliases so an
    alias can never escape ``google/accounts/<alias>/``.
    """

    if account is None:
        return None
    alias = account.strip()
    if not alias:
        raise ValueError("Google account alias cannot be empty")
    if not _ACCOUNT_RE.fullmatch(alias):
        raise ValueError(
            "Google account alias must contain only letters, numbers, underscore, or hyphen"
        )
    if alias in {".", ".."}:
        raise ValueError("Google account alias cannot be '.' or '..'")
    return alias


def resolve_account_paths(account: str | None = None) -> GoogleAccountPaths:
    """Return token/client-secret/pending-auth paths for an account alias."""

    hermes_home = get_hermes_home()
    alias = normalize_account_alias(account)
    if alias is None:
        root = hermes_home
    else:
        root = hermes_home / "google" / "accounts" / alias
    return GoogleAccountPaths(
        account=alias,
        root=root,
        token=root / "google_token.json",
        client_secret=root / "google_client_secret.json",
        pending_auth=root / "google_oauth_pending.json",
    )


def list_account_aliases() -> list[str]:
    """List configured named Google account aliases."""

    accounts_dir = get_hermes_home() / "google" / "accounts"
    if not accounts_dir.exists():
        return []
    return sorted(
        child.name
        for child in accounts_dir.iterdir()
        if child.is_dir() and _ACCOUNT_RE.fullmatch(child.name)
    )
