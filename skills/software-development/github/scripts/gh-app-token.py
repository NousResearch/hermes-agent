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
                ``--repo owner/name`` narrows the token to one repository.
    credential  Git credential-helper protocol. Git appends an operation
                argument (get/store/erase); output is emitted only for
                "get" — store/erase are silent no-ops. With
                ``credential.useHttpPath`` set, git sends the repo path and
                the token is narrowed to that repository.

Both accept ``--min-ttl SECONDS``: a cached token is reused only when it will
stay valid that long (never less than a 5-minute margin), so a caller about
to run a long command gets a token that cannot expire mid-command.

Tokens live only in the owner-only cache under the Hermes home
(``<HERMES_HOME>/cache/github-app-token.json``, one entry per scope); they
are never written to config.yaml or .env.

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
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import jwt  # PyJWT

try:
    from hermes_constants import get_hermes_home
except ImportError:  # run by git / system python outside the Hermes tree

    def get_hermes_home() -> Path:
        """Stdlib mirror of ``hermes_constants.get_hermes_home`` (profile-aware
        because profile subprocesses export HERMES_HOME)."""
        val = os.environ.get("HERMES_HOME", "").strip()
        return Path(val) if val else Path.home() / ".hermes"


API_BASE = "https://api.github.com"
HTTP_TIMEOUT = 10
JWT_LOOKBACK_SECONDS = 60  # tolerate clock skew (matches tools/skills_hub.py)
JWT_LIFETIME_SECONDS = 600  # GitHub caps app JWTs at 10 minutes
REMINT_MARGIN_SECONDS = 300  # re-mint when < 5 minutes of validity remain
INSTALLATION_SCOPE = "*"  # cache key for an installation-wide token
_REPO_SEGMENT = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


class TokenError(Exception):
    """A credential/config/API problem. Message is safe to print (no secrets)."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def default_env_file() -> Path:
    return get_hermes_home() / ".env"


def default_cache_file() -> Path:
    return get_hermes_home() / "cache" / "github-app-token.json"


def repo_from_path(path: str) -> Optional[str]:
    """``owner/name`` from a git credential ``path`` (``owner/name.git``,
    ``owner/name.git/info/lfs``), or None when it does not name a repo."""
    parts = [p for p in path.strip("/").split("/") if p]
    if len(parts) < 2:
        return None
    owner, name = parts[0], parts[1]
    name = name.removesuffix(".git")
    if not (_REPO_SEGMENT.match(owner) and _REPO_SEGMENT.match(name)):
        return None
    if name in (".", ".."):
        return None
    return f"{owner}/{name}"


def parse_env_file(path: Path) -> dict:
    """Flat key=value parse. No shell expansion, no `source` semantics."""
    values: dict = {}
    try:
        text = path.read_text(encoding="utf-8-sig")
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


def mint_installation_token(
    app_jwt: str, installation_id: str, repo: Optional[str] = None
) -> tuple:
    """POST /app/installations/{id}/access_tokens → (token, expires_at).

    With *repo* (``owner/name``) the token is restricted to that repository,
    so a leaked token can touch nothing else the installation can reach."""
    body = {"repositories": [repo.split("/", 1)[1]]} if repo else None
    resp = httpx.post(
        f"{API_BASE}/app/installations/{installation_id}/access_tokens",
        headers=_api_headers(app_jwt),
        json=body,
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
    return dt.datetime.fromisoformat(iso).timestamp()


def _read_cache_entries(cache_file: Path) -> dict:
    try:
        entries = json.loads(cache_file.read_text(encoding="utf-8-sig"))["entries"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}  # missing or corrupt cache — treat as empty
    return entries if isinstance(entries, dict) else {}


def _remaining_seconds(entry: dict) -> float:
    try:
        return _parse_expires_at(entry["expires_at"]) - time.time()
    except (KeyError, TypeError, ValueError, AttributeError):
        return 0.0


def load_cached_token(
    cache_file: Path, scope: str = INSTALLATION_SCOPE, min_ttl: int = 0
) -> Optional[str]:
    """The cached token for *scope* if it outlives max(margin, *min_ttl*)."""
    entry = _read_cache_entries(cache_file).get(scope)
    if not isinstance(entry, dict) or not entry.get("token"):
        return None
    if _remaining_seconds(entry) < max(REMINT_MARGIN_SECONDS, min_ttl):
        return None
    return entry["token"]


def store_cached_token(
    cache_file: Path, token: str, expires_at: str, scope: str = INSTALLATION_SCOPE
) -> None:
    """Write the cache atomically (tmp + rename) with owner-only permissions.

    Other scopes' live entries are kept; expired ones are dropped."""
    entries = {
        key: entry
        for key, entry in _read_cache_entries(cache_file).items()
        if isinstance(entry, dict) and _remaining_seconds(entry) > 0
    }
    entries[scope] = {"token": token, "expires_at": expires_at}
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
            json.dump({"entries": entries}, fh)
        os.chmod(tmp, 0o600)
        tmp.replace(cache_file)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def get_installation_token(
    env_file: Path,
    cache_file: Path,
    force: bool = False,
    repo: Optional[str] = None,
    min_ttl: int = 0,
) -> str:
    """Cache-first token fetch; mints and caches a fresh one when needed.

    *repo* narrows the token to one repository (cached per repo); *min_ttl*
    is how long the returned token must stay valid."""
    scope = repo or INSTALLATION_SCOPE
    if not force:
        cached = load_cached_token(cache_file, scope, min_ttl)
        if cached:
            return cached

    creds = read_app_credentials(env_file)
    if creds is None:
        raise TokenError(
            "GitHub App credentials not configured — set GITHUB_APP_ID and "
            f"GITHUB_APP_PRIVATE_KEY_PATH in {env_file}"
        )
    try:
        private_key_pem = creds.private_key_path.read_text(encoding="utf-8-sig")
    except OSError:
        raise TokenError(
            f"GitHub App private key not readable at {creds.private_key_path}"
        ) from None

    app_jwt = build_app_jwt(creds.app_id, private_key_pem)
    installation_id = creds.installation_id or discover_installation_id(app_jwt)
    token, expires_at = mint_installation_token(app_jwt, installation_id, repo)
    store_cached_token(cache_file, token, expires_at, scope)
    if min_ttl and _remaining_seconds({"expires_at": expires_at}) < min_ttl:
        raise TokenError(
            f"a fresh token is valid for less than --min-ttl {min_ttl}s "
            "(GitHub installation tokens last about one hour)"
        )
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
        token = get_installation_token(
            args.env_file, args.cache_file, repo=args.repo, min_ttl=args.min_ttl
        )
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
    # git sends path= only with credential.useHttpPath; narrow to that repo.
    repo = repo_from_path(request.get("path", ""))
    try:
        token = get_installation_token(
            args.env_file, args.cache_file, repo=repo, min_ttl=args.min_ttl
        )
    except TokenError as exc:
        # No stdout on failure so git falls through to the next helper.
        print(f"gh-app-token: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(f"username=x-access-token\npassword={token}\n\n")
    return 0


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return number


def _repo_arg(value: str) -> str:
    repo = repo_from_path(value)
    if repo is None or value.strip("/").count("/") != 1:
        raise argparse.ArgumentTypeError("expected OWNER/NAME")
    return repo


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
    common.add_argument(
        "--min-ttl", type=_non_negative_int, default=0, metavar="SECONDS",
        help="only reuse a cached token that stays valid this long",
    )

    parser = argparse.ArgumentParser(prog="gh-app-token", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    mint = sub.add_parser("mint", parents=[common], help="print an installation token")
    mint.add_argument(
        "--repo", type=_repo_arg, default=None, metavar="OWNER/NAME",
        help="restrict the token to this repository",
    )
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
