#!/usr/bin/env python3
"""OneDrive OAuth setup — device code flow for headless environments."""

import argparse
import json
import os
import sys
import time
from urllib.request import urlopen, Request
from urllib.parse import urlencode

DEFAULT_TOKEN_PATH = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "onedrive_token.json")

AUTH_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/devicecode"
TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
SCOPES = "https://graph.microsoft.com/Files.ReadWrite offline_access User.Read"

def json_request(url, data=None, headers=None, method=None):
    req = Request(url, data=data, headers=headers or {}, method=method)
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())

def do_device_flow(client_id):
    device_resp = json_request(
        AUTH_URL,
        data=urlencode({"client_id": client_id, "scope": SCOPES}).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    user_code = device_resp["user_code"]
    verification_uri = device_resp["verification_uri"]
    device_code = device_resp["device_code"]
    expires_in = device_resp["expires_in"]
    interval = device_resp.get("interval", 5)

    print(json.dumps({
        "user_code": user_code,
        "verification_uri": verification_uri,
        "message": f"Open {verification_uri} in a browser and enter code: {user_code}"
    }))

    start = time.time()
    while time.time() - start < expires_in:
        time.sleep(interval)
        try:
            token_resp = json_request(
                TOKEN_URL,
                data=urlencode({
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "client_id": client_id,
                    "device_code": device_code
                }).encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
        except Exception as e:
            body = e.read().decode() if hasattr(e, 'read') else str(e)
            if "authorization_pending" in body or "authorization_pending" in str(body):
                continue
            if "expired_token" in body:
                print(json.dumps({"error": "Device code expired. Please restart --auth."}))
                sys.exit(1)
            if "invalid_client" in body:
                print(json.dumps({"error": "Invalid client_id. Check your Azure app registration — device code requires the app to be a Public client."}))
                sys.exit(1)
            print(json.dumps({"error": str(body)}))
            sys.exit(1)

        if "error" in token_resp:
            if token_resp["error"] == "authorization_pending":
                continue
            if token_resp["error"] == "expired_token":
                print(json.dumps({"error": "Device code expired. Please restart --auth."}))
                sys.exit(1)
            if token_resp["error"] == "invalid_client":
                print(json.dumps({"error": "Invalid client_id. Check your Azure app registration — device code requires the app to be a Public client."}))
                sys.exit(1)
            continue

        return token_resp

    print(json.dumps({"error": "Timed out waiting for user approval."}))
    sys.exit(1)

def refresh_access_token(token_data, client_id):
    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        return None
    try:
        return json_request(
            TOKEN_URL,
            data=urlencode({
                "client_id": client_id,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
                "scope": SCOPES
            }).encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
    except Exception:
        return None

def load_token(path=DEFAULT_TOKEN_PATH):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

def save_token(token_data, path=DEFAULT_TOKEN_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(token_data, f, indent=2)

def check_auth(token_path=DEFAULT_TOKEN_PATH):
    token = load_token(token_path)
    if not token:
        print("NOT_AUTHENTICATED")
        return 1
    try:
        me = json_request(
            "https://graph.microsoft.com/v1.0/me",
            headers={"Authorization": f"Bearer {token['access_token']}"}
        )
        print(f"AUTHENTICATED as {me.get('userPrincipalName')}")
        return 0
    except Exception:
        print("NOT_AUTHENTICATED")
        return 1

def main():
    parser = argparse.ArgumentParser(description="OneDrive OAuth setup")
    parser.add_argument("--client-id", help="Azure AD application client ID")
    parser.add_argument("--auth", action="store_true", help="Start device flow authentication")
    parser.add_argument("--check", action="store_true", help="Check if authenticated")
    parser.add_argument("--token-path", default=DEFAULT_TOKEN_PATH, help="Path to save/load token")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    args = parser.parse_args()

    if args.check:
        sys.exit(check_auth(args.token_path))

    if args.auth:
        if not args.client_id:
            print(json.dumps({"error": "--auth requires --client-id"}))
            sys.exit(1)
        token = do_device_flow(args.client_id)
        token["_client_id"] = args.client_id
        save_token(token, args.token_path)
        print(json.dumps({"status": "success", "message": "Authenticated. Token saved."}))
        return

    parser.print_help()

if __name__ == "__main__":
    main()
