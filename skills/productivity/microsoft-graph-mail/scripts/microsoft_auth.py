#!/usr/bin/env python3
"""Microsoft Graph delegated OAuth setup for Hermes Agent.

The flow is intentionally non-interactive: the agent prints an authorization
URL, the user authorizes in a browser, then the agent exchanges the pasted
redirect URL. Tokens are scoped to the current Hermes profile and this module
only requests delegated read permissions.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import os
import secrets
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _hermes_home import display_hermes_home, get_hermes_home
from _log_sanitizer import _sanitize_for_log

HERMES_HOME = get_hermes_home()
TOKEN_PATH = HERMES_HOME / "microsoft_graph_token.json"
CLIENT_CONFIG_PATH = HERMES_HOME / "microsoft_graph_client.json"
PENDING_AUTH_PATH = HERMES_HOME / "microsoft_graph_oauth_pending.json"

DEFAULT_TENANT = "common"
DEFAULT_REDIRECT_URI = "http://localhost:1"
AUTHORITY_ROOT = "https://login.microsoftonline.com"
SCOPES = ["offline_access", "User.Read", "Mail.Read"]
REQUIRED_DELEGATED_SCOPES = ["User.Read", "Mail.Read"]

try:
    httpx = importlib.import_module("httpx")
except ModuleNotFoundError:
    httpx = None


def _token_endpoint(tenant: str) -> str:
    return f"{AUTHORITY_ROOT}/{tenant}/oauth2/v2.0/token"


def _authorize_endpoint(tenant: str) -> str:
    return f"{AUTHORITY_ROOT}/{tenant}/oauth2/v2.0/authorize"


def _new_code_verifier() -> str:
    return base64.urlsafe_b64encode(os.urandom(64)).decode("ascii").rstrip("=")


def _code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _new_state() -> str:
    return secrets.token_urlsafe(32)


def _scope_param(scopes: list[str] | None = None) -> str:
    return " ".join(scopes or SCOPES)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_json(path: Path, payload: dict, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(tmp_path, flags, mode)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
        os.chmod(tmp_path, mode)
        tmp_path.replace(path)
        os.chmod(path, mode)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _load_client_config() -> dict:
    config = _load_json(CLIENT_CONFIG_PATH)
    if not config.get("client_id"):
        print("ERROR: No Microsoft Graph client metadata stored. Run --configure first.")
        sys.exit(1)
    return {
        "client_id": config["client_id"],
        "tenant": config.get("tenant") or DEFAULT_TENANT,
        "redirect_uri": config.get("redirect_uri") or DEFAULT_REDIRECT_URI,
    }


def _token_scopes(payload: dict) -> set[str]:
    raw = payload.get("scope") or payload.get("scopes") or ""
    if isinstance(raw, str):
        values = raw.split()
    elif isinstance(raw, list):
        values = raw
    else:
        values = []
    return {str(scope).lower() for scope in values if str(scope).strip()}


def _missing_required_scopes(payload: dict) -> list[str]:
    granted = _token_scopes(payload)
    return [scope for scope in REQUIRED_DELEGATED_SCOPES if scope.lower() not in granted]


def configure_client(client_id: str, tenant: str | None = None, redirect_uri: str | None = None) -> None:
    client_id = client_id.strip()
    tenant = (tenant or DEFAULT_TENANT).strip() or DEFAULT_TENANT
    redirect_uri = (redirect_uri or DEFAULT_REDIRECT_URI).strip() or DEFAULT_REDIRECT_URI
    if not client_id:
        print("ERROR: --client-id is required.")
        sys.exit(1)
    if "client_secret" in client_id.lower():
        print("ERROR: Pass the public application client ID, not a client secret.")
        sys.exit(1)

    _write_json(
        CLIENT_CONFIG_PATH,
        {
            "client_id": client_id,
            "tenant": tenant,
            "redirect_uri": redirect_uri,
            "auth_flow": "authorization_code_pkce",
        },
    )
    print(f"OK: Microsoft Graph client metadata saved to {CLIENT_CONFIG_PATH}")


def _save_pending_auth(*, state: str, code_verifier: str, config: dict) -> None:
    _write_json(
        PENDING_AUTH_PATH,
        {
            "state": state,
            "code_verifier": code_verifier,
            "client_id": config["client_id"],
            "tenant": config["tenant"],
            "redirect_uri": config["redirect_uri"],
            "created_at": int(time.time()),
        },
    )


def _load_pending_auth() -> dict:
    if not PENDING_AUTH_PATH.exists():
        print("ERROR: No pending OAuth session found. Run --auth-url first.")
        sys.exit(1)
    try:
        data = json.loads(PENDING_AUTH_PATH.read_text())
    except Exception as exc:
        print(f"ERROR: Could not read pending OAuth session: {_sanitize_for_log(exc)}")
        print("Run --auth-url again to start a fresh OAuth session.")
        sys.exit(1)
    if not data.get("state") or not data.get("code_verifier"):
        print("ERROR: Pending OAuth session is missing PKCE data.")
        print("Run --auth-url again to start a fresh OAuth session.")
        sys.exit(1)
    return data


def _extract_code_and_state(code_or_url: str) -> tuple[str, str | None]:
    if not code_or_url.startswith("http"):
        return code_or_url, None

    params = parse_qs(urlparse(code_or_url).query)
    if "error" in params:
        error = params.get("error", ["authorization_error"])[0]
        description = params.get("error_description", [""])[0]
        print(f"ERROR: Microsoft authorization failed: {_sanitize_for_log(error)}")
        if description:
            print(_sanitize_for_log(description))
        sys.exit(1)
    if "code" not in params:
        print("ERROR: No 'code' parameter found in URL.")
        sys.exit(1)
    return params["code"][0], params.get("state", [None])[0]


def get_auth_url() -> None:
    config = _load_client_config()
    state = _new_state()
    verifier = _new_code_verifier()
    params = {
        "client_id": config["client_id"],
        "response_type": "code",
        "redirect_uri": config["redirect_uri"],
        "response_mode": "query",
        "scope": _scope_param(),
        "state": state,
        "code_challenge": _code_challenge(verifier),
        "code_challenge_method": "S256",
        "prompt": "select_account",
    }
    _save_pending_auth(state=state, code_verifier=verifier, config=config)
    print(f"{_authorize_endpoint(config['tenant'])}?{urlencode(params)}")


def _request_token(tenant: str, data: dict) -> dict:
    if httpx is None:
        print("ERROR: Missing dependency 'httpx'. Install Hermes with its core dependencies.")
        sys.exit(1)
    try:
        response = httpx.post(
            _token_endpoint(tenant),
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
    except Exception as exc:
        print(f"ERROR: Microsoft token request failed: {_sanitize_for_log(exc)}")
        sys.exit(1)

    if response.status_code >= 400:
        message = response.text
        try:
            error = response.json()
            message = error.get("error_description") or error.get("error") or message
        except Exception:
            pass
        print(f"ERROR: Microsoft token request failed ({response.status_code}): {_sanitize_for_log(message)}")
        sys.exit(1)

    try:
        return response.json()
    except Exception as exc:
        print(f"ERROR: Microsoft token response was not JSON: {_sanitize_for_log(exc)}")
        sys.exit(1)


def _persist_token_response(response: dict, config: dict, previous: dict | None = None) -> dict:
    payload = dict(response)
    if previous and not payload.get("refresh_token") and previous.get("refresh_token"):
        payload["refresh_token"] = previous["refresh_token"]
    if "expires_in" in payload:
        try:
            payload["expires_at"] = int(time.time()) + int(payload["expires_in"])
        except Exception:
            pass
    payload["client_id"] = config["client_id"]
    payload["tenant"] = config["tenant"]
    payload["redirect_uri"] = config["redirect_uri"]
    _write_json(TOKEN_PATH, payload)
    return payload


def exchange_auth_code(code: str) -> None:
    config = _load_client_config()
    pending = _load_pending_auth()
    effective_config = {
        "client_id": pending.get("client_id") or config["client_id"],
        "tenant": pending.get("tenant") or config["tenant"],
        "redirect_uri": pending.get("redirect_uri") or config["redirect_uri"],
    }
    code, returned_state = _extract_code_and_state(code)
    if returned_state is None:
        print("ERROR: Paste the full Microsoft redirect URL so the OAuth state can be validated.")
        sys.exit(1)
    if returned_state != pending["state"]:
        print("ERROR: OAuth state mismatch. Run --auth-url again to start a fresh session.")
        sys.exit(1)

    token_response = _request_token(
        effective_config["tenant"],
        {
            "client_id": effective_config["client_id"],
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": effective_config["redirect_uri"],
            "code_verifier": pending["code_verifier"],
            "scope": _scope_param(),
        },
    )
    missing = _missing_required_scopes(token_response)
    if missing:
        print("ERROR: Token missing required delegated Microsoft Graph scopes:")
        for scope in missing:
            print(f"  - {scope}")
        print("Update the Entra app permissions or consent, then run --auth-url again.")
        sys.exit(1)

    payload = _persist_token_response(token_response, effective_config)
    PENDING_AUTH_PATH.unlink(missing_ok=True)
    print(f"OK: Authenticated. Token saved to {TOKEN_PATH}")
    print(f"Profile-scoped token location: {display_hermes_home()}/microsoft_graph_token.json")
    if not payload.get("refresh_token"):
        print("WARNING: Microsoft did not return a refresh token; re-authentication may be required.")


def refresh_token(*, emit_status: bool = True) -> bool:
    config = _load_client_config()
    current = _load_json(TOKEN_PATH)
    refresh = current.get("refresh_token")
    if not refresh:
        print("TOKEN_INVALID: No refresh token. Run setup again.", file=sys.stderr)
        return False
    token_response = _request_token(
        current.get("tenant") or config["tenant"],
        {
            "client_id": current.get("client_id") or config["client_id"],
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "scope": _scope_param(),
        },
    )
    missing = _missing_required_scopes(token_response)
    if missing:
        print("REFRESH_FAILED: refreshed token missing Mail.Read/User.Read. Run setup again.", file=sys.stderr)
        return False
    persisted_config = {
        "client_id": current.get("client_id") or config["client_id"],
        "tenant": current.get("tenant") or config["tenant"],
        "redirect_uri": current.get("redirect_uri") or config["redirect_uri"],
    }
    _persist_token_response(token_response, persisted_config, previous=current)
    if emit_status:
        print(f"AUTHENTICATED: Token refreshed at {TOKEN_PATH}")
    return True


def check_auth() -> bool:
    if not CLIENT_CONFIG_PATH.exists():
        print(f"NOT_CONFIGURED: No client metadata at {CLIENT_CONFIG_PATH}")
        return False
    if not TOKEN_PATH.exists():
        print(f"NOT_AUTHENTICATED: No token at {TOKEN_PATH}")
        return False

    payload = _load_json(TOKEN_PATH)
    if _missing_required_scopes(payload):
        print("TOKEN_INVALID: Missing delegated Mail.Read/User.Read scopes. Run setup again.")
        return False

    expires_at = int(payload.get("expires_at") or 0)
    if expires_at and expires_at > int(time.time()) + 300 and payload.get("access_token"):
        print(f"AUTHENTICATED: Token valid at {TOKEN_PATH}")
        return True
    return refresh_token()


def get_valid_access_token() -> str:
    payload = _load_json(TOKEN_PATH)
    expires_at = int(payload.get("expires_at") or 0)
    if not payload.get("access_token") or expires_at <= int(time.time()) + 300:
        if not refresh_token(emit_status=False):
            sys.exit(1)
        payload = _load_json(TOKEN_PATH)
    missing = _missing_required_scopes(payload)
    if missing:
        print("ERROR: Token missing Mail.Read/User.Read. Run setup again.", file=sys.stderr)
        sys.exit(1)
    token = payload.get("access_token")
    if not token:
        print("ERROR: Token has no access token. Run setup again.", file=sys.stderr)
        sys.exit(1)
    return token


def revoke() -> None:
    TOKEN_PATH.unlink(missing_ok=True)
    PENDING_AUTH_PATH.unlink(missing_ok=True)
    print(f"Deleted local Microsoft Graph token state under {display_hermes_home()}.")
    print("To fully revoke consent, remove the app from the Microsoft account or Entra user consent records.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Microsoft Graph OAuth setup for Hermes")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--configure", action="store_true", help="Store Entra public-client app metadata")
    group.add_argument("--check", action="store_true", help="Check if auth is valid (exit 0=yes, 1=no)")
    group.add_argument("--auth-url", action="store_true", help="Print OAuth URL for user to visit")
    group.add_argument("--auth-code", metavar="REDIRECT_URL", help="Exchange the full redirect URL for a token")
    group.add_argument("--revoke", action="store_true", help="Delete local token state")
    parser.add_argument("--client-id", help="Microsoft Entra application/client ID")
    parser.add_argument("--tenant", help="Tenant ID/domain, common, organizations, or consumers")
    parser.add_argument("--redirect-uri", help=f"OAuth redirect URI (default: {DEFAULT_REDIRECT_URI})")
    args = parser.parse_args()

    if args.configure:
        if not args.client_id:
            print("ERROR: --client-id is required with --configure.")
            sys.exit(1)
        configure_client(args.client_id, tenant=args.tenant, redirect_uri=args.redirect_uri)
    elif args.check:
        sys.exit(0 if check_auth() else 1)
    elif args.auth_url:
        get_auth_url()
    elif args.auth_code:
        exchange_auth_code(args.auth_code)
    elif args.revoke:
        revoke()


if __name__ == "__main__":
    main()