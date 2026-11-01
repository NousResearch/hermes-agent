#!/usr/bin/env python3
"""CLI auth helper for 中国农历黄历吉凶 · Zhongguo Nongli Huangli Jixiong · China Lunar Almanac API token acquisition."""

import json
import os
import sys
import time
import urllib.error
import urllib.request
import webbrowser

BASE = os.environ.get("HUANGLI_BASE", "https://api.nongli.skill.4glz.com").rstrip("/")
TOKEN_FILE = os.path.expanduser(os.environ.get("HUANGLI_TOKEN_FILE", "~/.huangli_token.json"))
ENV_FILE = os.path.expanduser(os.environ.get("HUANGLI_ENV_FILE", "~/.huangli.env"))
ZSHRC_FILE = os.path.expanduser("~/.zshrc")


def post_json(url: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, headers: dict):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def shell_exports(token: str, base: str) -> str:
    return f"export HUANGLI_TOKEN='{token}'\nexport HUANGLI_BASE='{base}'\n"


def write_env(token: str, base: str):
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write(shell_exports(token, base))


def append_zshrc():
    line = f"[ -f '{ENV_FILE}' ] && source '{ENV_FILE}'"
    exists = os.path.exists(ZSHRC_FILE)
    cur = ""
    if exists:
        with open(ZSHRC_FILE, "r", encoding="utf-8") as f:
            cur = f.read()
    if line not in cur:
        with open(ZSHRC_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n# Huangli CLI\n{line}\n")
        return True
    return False


def status() -> int:
    print(f"BASE: {BASE}")
    print(f"token file: {TOKEN_FILE} ({'exists' if os.path.exists(TOKEN_FILE) else 'missing'})")
    print(f"env file: {ENV_FILE} ({'exists' if os.path.exists(ENV_FILE) else 'missing'})")

    token = os.environ.get("HUANGLI_TOKEN", "").strip()
    if not token and os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                token = json.load(f).get("access_token", "")
        except Exception:
            token = ""

    if not token:
        print("No token found.")
        return 1

    try:
        data = get_json(f"{BASE}/api/auth/verify", {"Authorization": f"Bearer {token}"})
        print(f"Token valid for: {data.get('username')}")
        return 0
    except Exception as e:
        print(f"Token check failed: {e}")
        return 1


def main():
    args = sys.argv[1:]
    action = args[0] if args and not args[0].startswith("--") else "login"
    print_shell = "--print-shell" in args
    append = "--append-zshrc" in args

    if action not in {"login", "register", "status"}:
        print("Usage: python3 huangli_auth.py [login|register|status] [--print-shell] [--append-zshrc]", file=sys.stderr)
        sys.exit(1)

    if action == "status":
        sys.exit(status())

    try:
        start = post_json(f"{BASE}/api/auth/cli/device/start", {"action": action})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Network error: {e.reason}", file=sys.stderr)
        sys.exit(1)

    verify_url = start["verification_uri_complete"]
    device_code = start["device_code"]
    interval = int(start.get("interval", 5))

    print("Open browser URL:")
    print(verify_url)
    try:
        webbrowser.open(verify_url)
    except Exception:
        pass

    while True:
        time.sleep(interval)
        try:
            data = post_json(f"{BASE}/api/auth/cli/device/poll", {"device_code": device_code})
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                msg = json.loads(body)
            except Exception:
                print(f"Auth failed: {body}", file=sys.stderr)
                sys.exit(1)

            if e.code == 428 and msg.get("error") == "authorization_pending":
                print("Waiting for authorization...")
                interval = int(msg.get("interval", interval))
                continue
            if e.code == 429 and msg.get("error") == "slow_down":
                interval = int(msg.get("interval", interval))
                continue
            print(f"Auth failed: {msg}", file=sys.stderr)
            sys.exit(1)

        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        write_env(data["access_token"], BASE)

        if append:
            changed = append_zshrc()
            print("Updated ~/.zshrc" if changed else "~/.zshrc already contains source line")

        print(f"Saved token: {TOKEN_FILE}")
        print(f"Saved env: {ENV_FILE}")
        print(f"source '{ENV_FILE}'")
        if print_shell:
            print("\n# exports")
            print(shell_exports(data["access_token"], BASE).strip())
        break


if __name__ == "__main__":
    main()
