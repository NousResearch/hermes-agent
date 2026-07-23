#!/usr/bin/env python3
"""Mint short-lived GitHub App installation tokens for git/gh workflows.

Part of the github-auth skill. Reads GitHub App credentials from the Hermes
.env file — never from process env, because tool subprocesses may run with
GITHUB_APP_* stripped from their environment; the file read is the path that
works everywhere, and it matches how gh-env.sh already reads .env. Builds an
RS256 app JWT, exchanges it for an installation access token (~1 hour
lifetime, auto-rotating, revocable by uninstalling the App), and caches the
result so repeated git operations don't re-mint.

Subcommands:
    mint        Print a valid installation token (bare) to stdout.
    credential  Git credential-helper protocol. Git appends an operation
                argument (get/store/erase); output is emitted only for
                "get" — store/erase are silent no-ops.

Configuration keys (flat key=value lines in ${HERMES_HOME:-~/.hermes}/.env):
    GITHUB_APP_ID                App ID (required)
    GITHUB_APP_PRIVATE_KEY_PATH  Path to the App's PEM private key (required)
    GITHUB_APP_INSTALLATION_ID   Installation ID (optional — auto-discovered
                                 when the App has exactly one installation)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import jwt  # PyJWT

API_BASE = "https://api.github.com"
HTTP_TIMEOUT = 10
JWT_LOOKBACK_SECONDS = 60  # tolerate clock skew (matches tools/skills_hub.py)
JWT_LIFETIME_SECONDS = 600  # GitHub caps app JWTs at 10 minutes
REMINT_MARGIN_SECONDS = 600  # re-mint when < 10 minutes of validity remain


class TokenError(Exception):
    """A credential/config/API problem. Message is safe to print (no secrets)."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def default_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def default_env_file() -> Path:
    return default_home() / ".env"


def default_cache_file() -> Path:
    return default_home() / "cache" / "github-app-token.json"


def parse_env_file(path: Path) -> dict:
    """Flat key=value parse. No shell expansion, no `source` semantics."""
    values: dict = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


@dataclass
class AppCredentials:
    app_id: str
    private_key_path: Path
    installation_id: Optional[str] = None


def read_app_credentials(env_file: Path) -> Optional[AppCredentials]:
    """Read GITHUB_APP_* from the .env file. None unless both required keys set."""
    values = parse_env_file(env_file)
    app_id = values.get("GITHUB_APP_ID", "")
    key_path = values.get("GITHUB_APP_PRIVATE_KEY_PATH", "")
    if not app_id or not key_path:
        return None
    return AppCredentials(
        app_id=app_id,
        private_key_path=Path(key_path).expanduser(),
        installation_id=values.get("GITHUB_APP_INSTALLATION_ID") or None,
    )


# ---------------------------------------------------------------------------
# JWT + GitHub API
# ---------------------------------------------------------------------------


def build_app_jwt(app_id: str, private_key_pem: str, now: Optional[int] = None) -> str:
    if now is None:
        now = int(time.time())
    payload = {
        "iat": now - JWT_LOOKBACK_SECONDS,
        "exp": now + JWT_LIFETIME_SECONDS,
        "iss": app_id,
    }
    return jwt.encode(payload, private_key_pem, algorithm="RS256")


def _api_headers(app_jwt: str) -> dict:
    return {
        "Authorization": f"Bearer {app_jwt}",
        "Accept": "application/vnd.github.v3+json",
    }


def discover_installation_id(app_jwt: str) -> str:
    """Resolve the installation id when GITHUB_APP_INSTALLATION_ID is unset."""
    resp = httpx.get(
        f"{API_BASE}/app/installations",
        headers=_api_headers(app_jwt),
        timeout=HTTP_TIMEOUT,
    )
    if resp.status_code != 200:
        raise TokenError(
            f"installation discovery failed (HTTP {resp.status_code} from GitHub)"
        )
    installations = resp.json()
    if not installations:
        raise TokenError(
            "GitHub App has no installations — install it on your account or org first"
        )
    if len(installations) > 1:
        accounts = ", ".join(
            sorted(
                str((inst.get("account") or {}).get("login") or inst.get("id"))
                for inst in installations
            )
        )
        raise TokenError(
            f"GitHub App has multiple installations ({accounts}) — "
            "set GITHUB_APP_INSTALLATION_ID to pick one"
        )
    return str(installations[0]["id"])


def mint_installation_token(app_jwt: str, installation_id: str) -> tuple:
    """POST /app/installations/{id}/access_tokens → (token, expires_at)."""
    resp = httpx.post(
        f"{API_BASE}/app/installations/{installation_id}/access_tokens",
        headers=_api_headers(app_jwt),
        timeout=HTTP_TIMEOUT,
    )
    if resp.status_code != 201:
        raise TokenError(f"token mint failed (HTTP {resp.status_code} from GitHub)")
    data = resp.json()
    token = data.get("token")
    if not token:
        raise TokenError("token mint response had no token field")
    return token, data.get("expires_at", "")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _parse_expires_at(iso: str) -> float:
    return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def load_cached_token(cache_file: Path) -> Optional[str]:
    """Return the cached token if it still has > REMINT_MARGIN_SECONDS left."""
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        token = data["token"]
        expires_epoch = _parse_expires_at(data["expires_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return None  # missing or corrupt cache — treat as a miss
    if not token or expires_epoch - time.time() < REMINT_MARGIN_SECONDS:
        return None
    return token


def store_cached_token(cache_file: Path, token: str, expires_at: str) -> None:
    """Write the cache atomically (tmp + rename) with owner-only permissions."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_name(f"{cache_file.name}.{os.getpid()}.tmp")
    try:
        tmp.unlink()  # clear any stale tmp (crashed prior run / planted link)
    except OSError:
        pass
    # O_EXCL: never follow or reuse a pre-existing path — refuse instead.
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"token": token, "expires_at": expires_at}, fh)
        os.chmod(tmp, 0o600)
        tmp.replace(cache_file)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def get_installation_token(
    env_file: Path, cache_file: Path, force: bool = False
) -> str:
    """Cache-first token fetch; mints and caches a fresh one when needed."""
    if not force:
        cached = load_cached_token(cache_file)
        if cached:
            return cached

    creds = read_app_credentials(env_file)
    if creds is None:
        raise TokenError(
            "GitHub App credentials not configured — set GITHUB_APP_ID and "
            f"GITHUB_APP_PRIVATE_KEY_PATH in {env_file}"
        )
    try:
        private_key_pem = creds.private_key_path.read_text(encoding="utf-8")
    except OSError:
        raise TokenError(
            f"GitHub App private key not readable at {creds.private_key_path}"
        ) from None

    app_jwt = build_app_jwt(creds.app_id, private_key_pem)
    installation_id = creds.installation_id or discover_installation_id(app_jwt)
    token, expires_at = mint_installation_token(app_jwt, installation_id)
    store_cached_token(cache_file, token, expires_at)
    return token


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# Hosts this helper will answer for. Git invokes credential helpers for EVERY
# https remote, so an unscoped helper that ignored the request's host would
# hand the GitHub token to any URL a repo (or a malicious submodule/redirect)
# points at. Both layers guard against that: SKILL.md wires the helper scoped
# to https://github.com, and this allowlist enforces it even when a user wires
# the helper globally.
ALLOWED_HOSTS = frozenset({"github.com", "gist.github.com"})


def _read_credential_request() -> dict:
    """Parse git's credential description block (key=value lines) from stdin."""
    request: dict = {}
    try:
        if sys.stdin is None or sys.stdin.isatty():
            return request
        for line in sys.stdin.read().splitlines():
            key, sep, value = line.partition("=")
            if sep:
                request[key.strip()] = value.strip()
    except OSError:
        pass
    return request


def _cmd_mint(args: argparse.Namespace) -> int:
    try:
        token = get_installation_token(args.env_file, args.cache_file)
    except TokenError as exc:
        print(f"gh-app-token: {exc}", file=sys.stderr)
        return 1
    print(token)
    return 0


def _cmd_credential(args: argparse.Namespace) -> int:
    request = _read_credential_request()
    if args.operation != "get":
        return 0  # store/erase: nothing to persist or forget — silent no-op
    # Only answer for GitHub over https. For anything else, emit nothing and
    # exit 0 so git falls through to other helpers / prompts — never send the
    # token toward a host we don't recognize.
    host = request.get("host", "")
    protocol = request.get("protocol", "")
    if host and host not in ALLOWED_HOSTS:
        return 0
    if protocol and protocol != "https":
        return 0
    try:
        token = get_installation_token(args.env_file, args.cache_file)
    except TokenError as exc:
        # No stdout on failure so git falls through to the next helper.
        print(f"gh-app-token: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(f"username=x-access-token\npassword={token}\n\n")
    return 0


def main(argv: Optional[list] = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--env-file", type=Path, default=default_env_file(),
        help="path to the .env file holding GITHUB_APP_* keys",
    )
    common.add_argument(
        "--cache-file", type=Path, default=default_cache_file(),
        help="path to the token cache file",
    )

    parser = argparse.ArgumentParser(prog="gh-app-token", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    mint = sub.add_parser("mint", parents=[common], help="print an installation token")
    mint.set_defaults(func=_cmd_mint)

    cred = sub.add_parser(
        "credential", parents=[common], help="git credential-helper protocol"
    )
    cred.add_argument(
        "operation", nargs="?", default="get", choices=["get", "store", "erase"],
        help="git credential operation (git always appends one)",
    )
    cred.set_defaults(func=_cmd_credential)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
