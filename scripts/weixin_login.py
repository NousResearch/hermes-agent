#!/usr/bin/env python3
"""
Standalone WeChat QR login script for Hermes Agent.

Run this on the Pi to authenticate the bot with WeChat:
    python3 scripts/weixin_login.py

After scanning the QR code with WeChat, credentials are saved to
~/.hermes/weixin/accounts/<account_id>.json and can be loaded by
the Hermes gateway's WeChat adapter.

The script also prints the env vars to add to ~/.hermes/.env.
"""

import asyncio
import base64
import json
import os
import struct
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Error: httpx not installed. Run: pip install httpx")
    sys.exit(1)

DEFAULT_BASE_URL = "https://ilinkai.weixin.qq.com"
DEFAULT_BOT_TYPE = "3"


def _random_uin_header() -> str:
    rand_uint32 = struct.unpack(">I", os.urandom(4))[0]
    return base64.b64encode(str(rand_uint32).encode()).decode()


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "X-WECHAT-UIN": _random_uin_header(),
    }


async def fetch_qr_code(client: httpx.AsyncClient, base_url: str) -> dict:
    """Request a QR code from the WeChat iLink API."""
    url = f"{base_url}/ilink/bot/get_bot_qrcode?bot_type={DEFAULT_BOT_TYPE}"
    resp = await client.get(url, headers=_headers(), timeout=15)
    resp.raise_for_status()
    return resp.json()


async def poll_qr_status(client: httpx.AsyncClient, base_url: str, qrcode: str) -> dict:
    """Long-poll for QR code scan status."""
    url = f"{base_url}/ilink/bot/get_qrcode_status?qrcode={qrcode}"
    headers = {**_headers(), "iLink-App-ClientVersion": "1"}
    try:
        resp = await client.get(url, headers=headers, timeout=35)
        resp.raise_for_status()
        return resp.json()
    except httpx.TimeoutException:
        return {"status": "wait"}


def save_credentials(account_id: str, token: str, base_url: str, user_id: str = "") -> Path:
    """Save credentials to ~/.hermes/weixin/accounts/<id>.json"""
    hermes_home = Path.home() / ".hermes"
    accounts_dir = hermes_home / "weixin" / "accounts"
    accounts_dir.mkdir(parents=True, exist_ok=True)

    # Normalize account ID
    normalized = account_id.strip().lower().replace("@", "-").replace(".", "-")

    data = {
        "token": token,
        "baseUrl": base_url,
        "savedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if user_id:
        data["userId"] = user_id

    filepath = accounts_dir / f"{normalized}.json"
    filepath.write_text(json.dumps(data, indent=2))
    filepath.chmod(0o600)

    # Also write account index
    index_path = hermes_home / "weixin" / "accounts.json"
    index_path.write_text(json.dumps([normalized], indent=2))

    return filepath


async def main():
    base_url = os.getenv("WEIXIN_BASE_URL", DEFAULT_BASE_URL)
    print(f"WeChat Login for Hermes Agent")
    print(f"API: {base_url}\n")

    async with httpx.AsyncClient(follow_redirects=True) as client:
        # Step 1: Get QR code
        print("Fetching QR code...")
        qr_data = await fetch_qr_code(client, base_url)
        qrcode = qr_data.get("qrcode", "")
        qrcode_url = qr_data.get("qrcode_img_content", "")

        if not qrcode:
            print("Error: Failed to get QR code from server")
            sys.exit(1)

        # Display QR code
        try:
            import qrcode as qr_lib
            qr = qr_lib.QRCode(box_size=1, border=1)
            qr.add_data(qrcode_url)
            qr.make()
            qr.print_ascii(invert=True)
        except ImportError:
            # Fallback: just print the URL
            pass

        print(f"\nQR Code URL: {qrcode_url}")
        print("\nScan this QR code with WeChat to connect.\n")

        # Step 2: Poll for scan
        scanned_printed = False
        max_attempts = 60  # ~5 minutes with long-poll

        for attempt in range(max_attempts):
            status = await poll_qr_status(client, base_url, qrcode)
            state = status.get("status", "wait")

            if state == "wait":
                if not scanned_printed:
                    print(".", end="", flush=True)
                continue

            elif state == "scaned":
                if not scanned_printed:
                    print("\n\nQR code scanned! Confirm on your phone...")
                    scanned_printed = True

            elif state == "expired":
                print("\n\nQR code expired. Please run the script again.")
                sys.exit(1)

            elif state == "confirmed":
                bot_token = status.get("bot_token", "")
                account_id = status.get("ilink_bot_id", "")
                response_base_url = status.get("baseurl", base_url)
                user_id = status.get("ilink_user_id", "")

                if not bot_token or not account_id:
                    print("\n\nLogin confirmed but missing credentials. Response:")
                    print(json.dumps(status, indent=2))
                    sys.exit(1)

                filepath = save_credentials(account_id, bot_token, response_base_url, user_id)

                print(f"\n\nConnected successfully!")
                print(f"\nCredentials saved to: {filepath}")
                print(f"\nAccount ID: {account_id}")
                if user_id:
                    print(f"User ID:    {user_id}")

                print(f"\n--- Add these to ~/.hermes/.env ---")
                print(f"WEIXIN_TOKEN={bot_token}")
                print(f"WEIXIN_ACCOUNT_ID={account_id}")
                if response_base_url != DEFAULT_BASE_URL:
                    print(f"WEIXIN_BASE_URL={response_base_url}")
                if user_id:
                    print(f"WEIXIN_ALLOWED_USERS={user_id}")
                print(f"-----------------------------------")

                # Quick connection test
                print(f"\nTesting connection...")
                try:
                    test_headers = {
                        **_headers(),
                        "AuthorizationType": "ilink_bot_token",
                        "Authorization": f"Bearer {bot_token}",
                    }
                    test_body = {
                        "get_updates_buf": "",
                        "base_info": {"channel_version": "hermes-login-0.1"},
                    }
                    test_resp = await client.post(
                        f"{response_base_url}/ilink/bot/getupdates",
                        json=test_body,
                        headers=test_headers,
                        timeout=10,
                    )
                    if test_resp.status_code == 200:
                        print("Connection test passed!")
                    else:
                        print(f"Connection test returned HTTP {test_resp.status_code}")
                except Exception as e:
                    print(f"Connection test failed: {e}")
                    print("(This may be normal if the server holds the long-poll)")

                return

            await asyncio.sleep(1)

        print("\n\nLogin timed out. Please try again.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
