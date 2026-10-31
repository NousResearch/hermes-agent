"""Token locations for named Google accounts.

Without ``--account`` every script keeps using the single profile token at
``<HERMES_HOME>/google_token.json``, exactly as before. ``--account EMAIL``
selects a token of its own under ``<HERMES_HOME>/google_tokens/``, so one
Hermes profile can hold a personal and a work Google account side by side.

A named account never falls back to the profile token: a command that names
an account the user has not authorized fails instead of reading someone
else's mail.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from _hermes_home import get_hermes_home

ACCOUNTS_DIR_NAME = "google_tokens"
_PENDING_SUFFIX = ".pending.json"
# Lowercase address with no path separators, so it is always a plain file name.
_ACCOUNT_RE = re.compile(r"[a-z0-9._%+-]+@[a-z0-9-]+(\.[a-z0-9-]+)+")


def normalize_account(value: str) -> str:
    account = value.strip().lower()
    if not _ACCOUNT_RE.fullmatch(account):
        raise argparse.ArgumentTypeError(
            f"expected a Google account email address, got {value!r}"
        )
    return account


def add_account_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--account",
        metavar="EMAIL",
        type=normalize_account,
        default=None,
        help="Use the token for this Google account instead of the profile's default token",
    )


def accounts_dir() -> Path:
    return get_hermes_home() / ACCOUNTS_DIR_NAME


def ensure_accounts_dir() -> Path:
    directory = accounts_dir()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


def token_path(account: str) -> Path:
    return accounts_dir() / f"{account}.json"


def pending_auth_path(account: str) -> Path:
    return accounts_dir() / f"{account}{_PENDING_SUFFIX}"


def gws_config_dir(account: str) -> Path:
    """Private gws config dir per account.

    gws caches access tokens per config dir without keying on the credentials
    file (googleworkspace/cli#572), so two accounts sharing one dir would run
    as whichever account filled the cache first.
    """
    return accounts_dir() / "gws" / account


def list_accounts() -> list[str]:
    directory = accounts_dir()
    if not directory.is_dir():
        return []
    return sorted(
        path.name.removesuffix(".json")
        for path in directory.glob("*.json")
        if not path.name.endswith(_PENDING_SUFFIX)
    )
