#!/usr/bin/env python3
"""Whoop token refresh script for cron.

Touches the API to trigger auto-refresh if the access token is about to expire.
FULLY SILENT on success (no stdout, no Telegram — nothing is delivered).
On failure writes to stderr and exits non-zero; the cron job delivers the error.

Exit codes:
  0 — success (token still valid or refreshed)
  1 — failure (refresh failed or API unreachable)
"""

import sys
import time

from whoop_storage import load_tokens, save_tokens, load_client_credentials

API_BASE = "https://api.prod.whoop.com"


def main() -> int:
    tokens = load_tokens()
    if tokens is None:
        print(
            "Whoop token refresh failed: no tokens found. Run `whoop_sync.py setup` to re-authenticate.",
            file=sys.stderr,
        )
        return 1

    # Check if token expires within 10 minutes
    expires_at = tokens.get("expires_at", 0)
    remaining = expires_at - time.time()
    if remaining > 600:
        # Token still valid — silent success (no stdout, no delivery)
        return 0

    # Token expired or about to expire — force refresh
    import requests
    creds = load_client_credentials()
    if not creds:
        print(
            "Whoop token refresh failed: no client credentials found.",
            file=sys.stderr,
        )
        return 1

    try:
        response = requests.post(
            f"{API_BASE}/oauth/oauth2/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
                "client_id": creds["client_id"],
                "client_secret": creds["client_secret"],
            },
            timeout=30,
        )
    except requests.RequestException as e:
        print(f"Whoop token refresh failed: network error ({e})", file=sys.stderr)
        return 1

    if response.status_code == 401:
        print(
            "Whoop token refresh failed: refresh token expired. Re-auth needed. Run `whoop_sync.py setup`.",
            file=sys.stderr,
        )
        return 1

    if response.status_code != 200:
        print(f"Whoop token refresh failed: HTTP {response.status_code}", file=sys.stderr)
        return 1

    token_data = response.json()
    new_expires_at = time.time() + token_data.get("expires_in", 3600)
    save_tokens(
        access_token=token_data["access_token"],
        refresh_token=token_data["refresh_token"],
        expires_at=new_expires_at,
    )
    # Silent on success — no stdout, no delivery
    return 0


if __name__ == "__main__":
    sys.exit(main())
