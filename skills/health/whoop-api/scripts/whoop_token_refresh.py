#!/usr/bin/env python3
"""Whoop token refresh script for cron.

Touches the API to trigger auto-refresh if the access token is about to expire.
Silent on success (no stdout = no Telegram delivery).
Sends Telegram notification on failure via `hermes send`.

Exit codes:
  0 — success (token still valid or refreshed)
  1 — failure (refresh failed or API unreachable)
"""

import subprocess
import sys
import time

from whoop_storage import load_tokens, save_tokens, load_client_credentials

API_BASE = "https://api.prod.whoop.com"


def notify_telegram(message: str) -> None:
    """Send a Telegram notification via hermes send (best-effort)."""
    try:
        subprocess.run(
            ["hermes", "send", "--to", "telegram", "--quiet", "--message", message],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        pass  # Don't let notification failure mask the original error


def main() -> int:
    tokens = load_tokens()
    if tokens is None:
        msg = "Whoop token refresh failed: no tokens found. Run `whoop_sync.py setup` to re-authenticate."
        notify_telegram(msg)
        print(msg, file=sys.stderr)
        return 1

    # Check if token expires within 10 minutes
    expires_at = tokens.get("expires_at", 0)
    remaining = expires_at - time.time()
    if remaining > 600:
        # Token still valid — silent exit (no stdout = no delivery)
        return 0

    # Token expired or about to expire — force refresh
    import requests
    creds = load_client_credentials()
    if not creds:
        msg = "Whoop token refresh failed: no client credentials found."
        notify_telegram(msg)
        print(msg, file=sys.stderr)
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
        msg = f"Whoop token refresh failed: network error ({e})"
        notify_telegram(msg)
        print(msg, file=sys.stderr)
        return 1

    if response.status_code == 401:
        msg = "Whoop token refresh failed: refresh token expired. Re-auth needed. Run `whoop_sync.py setup`."
        notify_telegram(msg)
        print(msg, file=sys.stderr)
        return 1

    if response.status_code != 200:
        msg = f"Whoop token refresh failed: HTTP {response.status_code}"
        notify_telegram(msg)
        print(msg, file=sys.stderr)
        return 1

    token_data = response.json()
    new_expires_at = time.time() + token_data.get("expires_in", 3600)
    save_tokens(
        access_token=token_data["access_token"],
        refresh_token=token_data["refresh_token"],
        expires_at=new_expires_at,
    )
    print(f"Whoop token refreshed. Next expiry in {token_data.get('expires_in', 3600)/60:.0f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())