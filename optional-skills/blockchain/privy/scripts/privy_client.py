#!/usr/bin/env python3
"""Read-only Privy helper for Hermes optional skill usage.

The helper intentionally supports only user and wallet inspection. It does not
prepare, sign, or broadcast transactions.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.parse
import urllib.request
from typing import Any, cast

DEFAULT_API_BASE = "https://api.privy.io"


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse every HTTP redirect before urllib can rebuild the request."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def build_basic_auth_header(app_id: str, app_secret: str) -> str:
    token = f"{app_id}:{app_secret}".encode("utf-8")
    return "Basic " + base64.b64encode(token).decode("ascii")


def load_credentials() -> tuple[str, str]:
    app_id = os.getenv("PRIVY_APP_ID")
    app_secret = os.getenv("PRIVY_APP_SECRET")
    missing = [
        name
        for name, value in (("PRIVY_APP_ID", app_id), ("PRIVY_APP_SECRET", app_secret))
        if not value
    ]
    if missing:
        raise SystemExit("Missing required Privy credentials: " + ", ".join(missing))
    return app_id or "", app_secret or ""


def build_users_query(*, limit: int = 20, cursor: str | None = None) -> str:
    if not 1 <= limit <= 100:
        raise ValueError("Privy user list limit must be between 1 and 100")
    params: dict[str, str | int] = {"limit": limit}
    if cursor:
        params["cursor"] = cursor
    return "/v1/users?" + urllib.parse.urlencode(params)


def bounded_limit(value: str) -> int:
    limit = int(value)
    if not 1 <= limit <= 100:
        raise argparse.ArgumentTypeError("limit must be between 1 and 100")
    return limit


def request_json(
    path: str,
    *,
    safe_route: str,
    method: str = "GET",
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    app_id, app_secret = load_credentials()
    url = DEFAULT_API_BASE + path
    headers = {
        "Authorization": build_basic_auth_header(app_id, app_secret),
        "privy-app-id": app_id,
        "Accept": "application/json",
    }
    request_body = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        request_body = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=request_body,
        headers=headers,
        method=method,
    )
    try:
        opener = urllib.request.build_opener(NoRedirectHandler())
        with opener.open(request, timeout=30) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)
    except Exception:  # pragma: no cover - exact failures vary by network/runtime
        raise SystemExit(f"Privy API {method.upper()} {safe_route} failed") from None


def extract_wallets(user: dict[str, Any]) -> list[dict[str, Any]]:
    wallets: list[dict[str, Any]] = []
    for account in user.get("linked_accounts") or []:
        if account.get("type") != "wallet":
            continue
        wallets.append({
            "address": account.get("address"),
            "chain_type": account.get("chain_type"),
            "wallet_client_type": account.get("wallet_client_type"),
            "connector_type": account.get("connector_type"),
            "verified_at": account.get("verified_at"),
        })
    return wallets


def summarize_user(user: dict[str, Any]) -> dict[str, Any]:
    """Return a PII-minimized user summary suitable for an agent transcript."""
    linked_accounts = user.get("linked_accounts") or []
    return {
        "id": user.get("id") or user.get("user_id") or user.get("did"),
        "created_at": user.get("created_at"),
        "linked_account_types": [
            account.get("type") for account in linked_accounts if account.get("type")
        ],
        "wallets": extract_wallets(user),
    }


def extract_user_payload(data: dict[str, Any]) -> dict[str, Any]:
    user = data.get("user")
    if isinstance(user, dict):
        return cast(dict[str, Any], user)
    return data


def cmd_user(args: argparse.Namespace) -> dict[str, Any]:
    data = request_json(
        f"/v1/users/{urllib.parse.quote(args.user_id, safe='')}",
        safe_route="/v1/users/{user_id}",
    )
    user = extract_user_payload(data)
    return summarize_user(user)


def cmd_wallets(args: argparse.Namespace) -> dict[str, Any]:
    data = request_json(
        f"/v1/users/{urllib.parse.quote(args.user_id, safe='')}",
        safe_route="/v1/users/{user_id}",
    )
    user = extract_user_payload(data)
    return {"user_id": user.get("id") or args.user_id, "wallets": extract_wallets(user)}


def cmd_users(args: argparse.Namespace) -> dict[str, Any]:
    if args.email:
        data = request_json(
            "/v1/users/email/address",
            safe_route="/v1/users/email/address",
            method="POST",
            body={"address": args.email},
        )
        raw_users = [extract_user_payload(data)]
        next_cursor = None
    else:
        data = request_json(
            build_users_query(limit=args.limit, cursor=args.cursor),
            safe_route="/v1/users",
        )
        raw_users = data.get("users") or data.get("data") or []
        next_cursor = data.get("next_cursor") or data.get("cursor")
    return {
        "users": [summarize_user(user) for user in raw_users],
        "next_cursor": next_cursor,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only Privy user and wallet inspection helper. No signing or transactions."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_user = sub.add_parser("user", help="Fetch a PII-minimized Privy user summary")
    p_user.add_argument("user_id", help="Privy DID/user id, for example did:privy:...")
    p_user.set_defaults(func=cmd_user)

    p_wallets = sub.add_parser(
        "wallets", help="List wallet linked accounts for a Privy user"
    )
    p_wallets.add_argument(
        "user_id", help="Privy DID/user id, for example did:privy:..."
    )
    p_wallets.set_defaults(func=cmd_wallets)

    p_users = sub.add_parser("users", help="List Privy users with PII-minimized output")
    p_users.add_argument("--limit", type=bounded_limit, default=20)
    p_users.add_argument("--cursor")
    p_users.add_argument(
        "--email", help="Optional server-side email filter; output remains redacted"
    )
    p_users.set_defaults(func=cmd_users)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    print(json.dumps(args.func(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
