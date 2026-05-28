#!/usr/bin/env python3
"""Daily Hermes/OpenRouter usage report → Telegram + append to log file.

Usage:
    python3 scripts/daily_usage_report.py          # send to Telegram + log
    python3 scripts/daily_usage_report.py --test   # print preview, skip Telegram

Cron (8 AM daily):
    0 8 * * * cd /path/to/hermes-agent && python3 scripts/daily_usage_report.py >> ~/.hermes/logs/cron.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
# hermes-agent lives at <monorepo>/.hermes/hermes-agent
MONOREPO_ROOT = REPO_ROOT.parent.parent
DOTENV_FILE = MONOREPO_ROOT / ".env"
LOG_DIR = Path.home() / ".hermes" / "logs"
LOG_FILE = LOG_DIR / "usage.log"

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env parser — no external dependency required."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        result[key] = val
    return result


def _env(key: str, dotenv: dict[str, str]) -> str:
    """OS env takes precedence, falls back to .env file."""
    return os.environ.get(key) or dotenv.get(key, "")


# ── OpenRouter API ────────────────────────────────────────────────────────────

def fetch_openrouter_usage(api_key: str) -> dict:
    """Fetch credits balance + per-key usage from OpenRouter."""
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    with httpx.Client(timeout=12.0) as client:
        credits_resp = client.get(f"{OPENROUTER_BASE}/credits", headers=headers)
        credits_resp.raise_for_status()
        credits = (credits_resp.json() or {}).get("data") or {}
        try:
            key_resp = client.get(f"{OPENROUTER_BASE}/auth/key", headers=headers)
            key_resp.raise_for_status()
            key_data = (key_resp.json() or {}).get("data") or {}
        except Exception:
            key_data = {}
    return {"credits": credits, "key": key_data}


# ── Formatting ────────────────────────────────────────────────────────────────

def format_telegram_message(data: dict, today: str) -> str:
    credits = data.get("credits", {})
    key = data.get("key", {})

    total_credits = float(credits.get("total_credits") or 0.0)
    total_usage = float(credits.get("total_usage") or 0.0)
    balance = max(0.0, total_credits - total_usage)

    limit = key.get("limit")
    limit_remaining = key.get("limit_remaining")
    usage_daily = key.get("usage_daily")
    usage_weekly = key.get("usage_weekly")
    usage_monthly = key.get("usage_monthly")
    usage_total = key.get("usage")
    limit_reset = str(key.get("limit_reset") or "").strip()

    lines = [
        f"🤖 *Hermes Daily Usage — {today}*",
        "───────────────────────────",
        f"💰 Balance: *${balance:.2f}* remaining",
    ]

    if isinstance(usage_daily, (int, float)):
        lines.append(f"📅 Today:      *${float(usage_daily):.4f}*")
    if isinstance(usage_weekly, (int, float)):
        lines.append(f"📆 This week:  *${float(usage_weekly):.4f}*")
    if isinstance(usage_monthly, (int, float)):
        lines.append(f"🗓 This month: *${float(usage_monthly):.4f}*")
    if isinstance(usage_total, (int, float)):
        lines.append(f"📊 Key total:  *${float(usage_total):.4f}*")

    if (
        isinstance(limit, (int, float))
        and float(limit) > 0
        and isinstance(limit_remaining, (int, float))
    ):
        limit_f = float(limit)
        remaining_f = float(limit_remaining)
        used_pct = ((limit_f - remaining_f) / limit_f) * 100
        status = "🔴 HIGH" if used_pct > 85 else ("🟡 WATCH" if used_pct > 60 else "🟢 OK")
        lines.append("")
        lines.append(
            f"{status} — *${remaining_f:.2f}* of *${limit_f:.2f}* left "
            f"({used_pct:.1f}% used)"
        )
        if limit_reset:
            lines.append(f"🔄 Resets: {limit_reset}")

    return "\n".join(lines)


# ── Telegram ─────────────────────────────────────────────────────────────────

def send_telegram(bot_token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    with httpx.Client(timeout=12.0) as client:
        resp = client.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
        )
    return resp.status_code == 200


# ── Logging ───────────────────────────────────────────────────────────────────

def append_log(data: dict, today: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    credits = data.get("credits", {})
    key = data.get("key", {})
    entry = {
        "date": today,
        "ts": datetime.now(timezone.utc).isoformat(),
        "balance": max(
            0.0,
            float(credits.get("total_credits") or 0) - float(credits.get("total_usage") or 0),
        ),
        "usage_daily": key.get("usage_daily"),
        "usage_weekly": key.get("usage_weekly"),
        "usage_monthly": key.get("usage_monthly"),
        "usage_total": key.get("usage"),
        "limit": key.get("limit"),
        "limit_remaining": key.get("limit_remaining"),
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"📝 Logged to {LOG_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Daily OpenRouter usage report")
    parser.add_argument("--test", action="store_true", help="Preview only — do not send to Telegram")
    args = parser.parse_args()

    dotenv = _load_dotenv(DOTENV_FILE)
    api_key = _env("OPENROUTER_API_KEY", dotenv)
    bot_token = _env("TELEGRAM_BOT_TOKEN", dotenv)
    chat_id = _env("TELEGRAM_CHAT_ID", dotenv)

    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in env or .env file", file=sys.stderr)
        sys.exit(1)

    today = datetime.now().strftime("%Y-%m-%d")

    print(f"Fetching OpenRouter usage for {today}…")
    try:
        data = fetch_openrouter_usage(api_key)
    except Exception as exc:
        print(f"ERROR: OpenRouter API call failed: {exc}", file=sys.stderr)
        sys.exit(1)

    append_log(data, today)
    message = format_telegram_message(data, today)

    if args.test:
        print("\n=== Telegram Message Preview ===")
        print(message)
        print("================================")
        print("(--test mode: Telegram not sent)")
        return

    if not bot_token or not chat_id:
        print("ERROR: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set", file=sys.stderr)
        sys.exit(1)

    ok = send_telegram(bot_token, chat_id, message)
    if ok:
        print(f"✅  Usage report sent to Telegram")
    else:
        print("❌  Failed to send Telegram message", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
