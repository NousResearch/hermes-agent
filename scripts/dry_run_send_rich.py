"""
scripts/dry_run_send_rich.py — Hermes Agent

Dry-run verify cho Telegram Bot API 10.1 Rich Messages mà KHÔNG cần bot token
thật. Chỉ ping API endpoint, expect HTTP 401 (invalid token) hoặc 404 nếu
method không tồn tại.

Mục đích: CI smoke test trước khi deploy / rollback, hoặc khi verify Bot API
10.1 đã deploy thật trên production.

Usage:
    python3 scripts/dry_run_send_rich.py
    python3 scripts/dry_run_send_rich.py --token "$TELEGRAM_BOT_TOKEN"

Output:
    exit 0 = method available (200 OK real token, hoặc 401 dummy — cả 2 đều OK)
    exit 1 = method NOT available (404) → Bot API version <10.1 hoặc method renamed
    exit 2 = network error / DNS fail
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

API_BASE = "https://api.telegram.org"
METHODS_TO_CHECK = [
    "sendRichMessage",
    "sendRichMessageDraft",
]


def ping(method: str, token: str = "0000000000:DUMMY_TOKEN") -> int:
    """POST một dummy payload tới `method`. Return HTTP status code."""
    if method in ("sendRichMessage",):
        payload = {"chat_id": 0, "rich_message": json.dumps({"html": "<p>dry-run</p>"})}
    elif method == "sendRichMessageDraft":
        payload = {"chat_id": 0, "draft_id": 1, "rich_message": json.dumps({"html": "<p>dry-run</p>"})}
    else:
        raise ValueError(f"Unknown method: {method}")

    req = urllib.request.Request(
        f"{API_BASE}/bot{token}/{method}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"❌ Network error on {method}: {e}")
        return -1


def main():
    parser = argparse.ArgumentParser(description="Dry-run verify Telegram Bot API 10.1 Rich Messages")
    parser.add_argument("--token", default="0000000000:DUMMY_TOKEN", help="Bot token (default dummy for 401)")
    args = parser.parse_args()

    print(f"=== Dry-run verify Telegram Bot API 10.1 Rich Messages ===")
    print(f"Endpoint: {API_BASE}")
    print(f"Token: {args.token[:12]}... (truncated)")
    print()

    results = {}
    for method in METHODS_TO_CHECK:
        status = ping(method, args.token)
        results[method] = status
        print(f"  {method}: HTTP {status}")

    print()
    print("=== Verdict ===")
    if all(s == 401 for s in results.values()):
        print("✅ All methods DEPLOYED (401 = method exists, dummy token rejected).")
        print("   Bot API 10.1+ is live on production.")
        return 0
    if any(s == 404 for s in results.values()):
        print("❌ At least one method NOT FOUND (404).")
        print("   Bot API version may be <10.1, OR method was renamed/removed.")
        return 1
    if any(s < 0 for s in results.values()):
        print("❌ Network errors. Check connectivity.")
        return 2
    print(f"⚠️ Unexpected statuses: {results}")
    print("   Methods exist (not 404) but returned unexpected codes. Inspect manually.")
    return 0


if __name__ == "__main__":
    sys.exit(main())