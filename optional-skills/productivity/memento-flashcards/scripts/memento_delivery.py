#!/usr/bin/env python3
"""Memento idle-window delivery — Phase 2.

Delivers ONE due flashcard question to Telegram when Hafs has gone idle.
Enforces quiet hours, daily cap, per-card cooldown, and a global kill switch.

Feature flag: MEMENTO_DELIVERY_ENABLED=1 required to send. Default is dry-run.

All output is JSON to stdout. Exit 0 on success (including skipped/dry-run).
Exit 1 only on hard errors (bad config that prevents any decision).

Environment variables (all optional — defaults shown):
  MEMENTO_DELIVERY_ENABLED   — "1" to enable live sends (default: disabled)
  MEMENTO_DRY_RUN            — "0" to disable dry-run when enabled (default: "1")
  TELEGRAM_BOT_TOKEN         — bot token for live sends
  TELEGRAM_CHAT_ID           — target chat ID for live sends
  MEMENTO_IDLE_MINUTES       — quiet minutes before delivery (default: 30)
  MEMENTO_QUIET_HOURS_START  — local hour to start quiet window (default: 22)
  MEMENTO_QUIET_HOURS_END    — local hour to end quiet window (default: 8)
  MEMENTO_DAILY_CAP          — max sends per calendar day (default: 5)
  MEMENTO_COOLDOWN_MINUTES   — min minutes between sends (default: 60)
  MEMENTO_SESSIONS_FILE      — override path to gateway sessions.json
  HERMES_HOME                — Hermes home directory
"""

import argparse
import json
import os
import sys
import tempfile
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

_HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))

# Card store (Phase 1)
_CARDS_FILE = _HERMES_HOME / "skills" / "productivity" / "memento-flashcards" / "data" / "cards.json"

# Delivery state
_DELIVERY_STATE_FILE = _HERMES_HOME / "skills" / "productivity" / "memento-flashcards" / "delivery_state.json"

# Gateway sessions file for idle detection
_SESSIONS_FILE = _HERMES_HOME / "sessions" / "sessions.json"

RETIRED_SENTINEL = "9999-12-31T23:59:59+00:00"


# ── Config helpers ────────────────────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name, "").strip()
    if val.lstrip("-").isdigit():
        return int(val)
    return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


def _load_config() -> dict:
    return {
        "enabled": _env_bool("MEMENTO_DELIVERY_ENABLED", False),
        "dry_run": _env_bool("MEMENTO_DRY_RUN", True),
        "bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", "").strip(),
        "chat_id": os.environ.get("TELEGRAM_CHAT_ID", "").strip(),
        "idle_minutes": _env_int("MEMENTO_IDLE_MINUTES", 30),
        "quiet_start": _env_int("MEMENTO_QUIET_HOURS_START", 22),
        "quiet_end": _env_int("MEMENTO_QUIET_HOURS_END", 8),
        "daily_cap": _env_int("MEMENTO_DAILY_CAP", 5),
        "cooldown_minutes": _env_int("MEMENTO_COOLDOWN_MINUTES", 60),
        "sessions_file": Path(
            os.environ.get("MEMENTO_SESSIONS_FILE", "").strip() or str(_SESSIONS_FILE)
        ),
        "cards_file": Path(
            os.environ.get("MEMENTO_CARDS_FILE", "").strip() or str(_CARDS_FILE)
        ),
        "state_file": Path(
            os.environ.get("MEMENTO_STATE_FILE", "").strip() or str(_DELIVERY_STATE_FILE)
        ),
    }


# ── Idle detection ────────────────────────────────────────────────────────────

def _most_recent_session_activity(sessions_file: Path) -> Optional[datetime]:
    """Return the most recent updated_at across all sessions, or None if unavailable."""
    if not sessions_file.exists():
        return None
    try:
        with open(sessions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    entries = data if isinstance(data, list) else data.get("entries", [])
    if not entries and isinstance(data, dict):
        entries = list(data.values())

    latest: Optional[datetime] = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ts_str = entry.get("updated_at")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if latest is None or ts > latest:
                latest = ts
        except ValueError:
            continue
    return latest


def _is_idle(sessions_file: Path, idle_minutes: int, now: datetime) -> tuple[bool, str]:
    """Return (is_idle, reason).

    Returns (False, reason) if idle detection is unavailable — fail-open means
    we do NOT deliver when we can't confirm idleness.
    """
    last_activity = _most_recent_session_activity(sessions_file)
    if last_activity is None:
        return False, "sessions_unavailable"
    idle_threshold = timedelta(minutes=idle_minutes)
    elapsed = now - last_activity
    if elapsed >= idle_threshold:
        return True, f"idle_for_{int(elapsed.total_seconds() // 60)}m"
    remaining = int((idle_threshold - elapsed).total_seconds() // 60)
    return False, f"active_{int(elapsed.total_seconds() // 60)}m_ago_need_{idle_minutes}m"


# ── Quiet hours ───────────────────────────────────────────────────────────────

def _quiet_hour_check(quiet_start: int, quiet_end: int, local_hour: int) -> bool:
    """Return True when local_hour falls within the quiet window.

    Handles both overnight (22-8) and same-day (e.g. 1-3) windows.
    """
    if quiet_start < quiet_end:
        return quiet_start <= local_hour < quiet_end
    elif quiet_start > quiet_end:
        return local_hour >= quiet_start or local_hour < quiet_end
    return False  # quiet_start == quiet_end means no quiet window


def _is_quiet_hour(quiet_start: int, quiet_end: int, now: datetime) -> bool:
    return _quiet_hour_check(quiet_start, quiet_end, now.astimezone().hour)


# ── Due cards ────────────────────────────────────────────────────────────────

def _load_due_cards(cards_file: Path, now: datetime) -> list[dict]:
    """Return all learning cards with next_review_at <= now, sorted oldest-first."""
    if not cards_file.exists():
        return []
    try:
        with open(cards_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    cards = data.get("cards", []) if isinstance(data, dict) else []
    due = []
    for card in cards:
        if card.get("status") == "retired":
            continue
        try:
            review_at = datetime.fromisoformat(card["next_review_at"])
            if review_at.tzinfo is None:
                review_at = review_at.replace(tzinfo=timezone.utc)
            if review_at <= now:
                due.append(card)
        except (KeyError, ValueError):
            continue
    due.sort(key=lambda c: c.get("next_review_at", ""))
    return due


# ── Delivery state ────────────────────────────────────────────────────────────

def _load_state(state_file: Path, today_str: str) -> dict:
    default = {
        "last_sent_at": None,
        "last_sent_card_id": None,
        "today_date": today_str,
        "sent_today": 0,
    }
    if not state_file.exists():
        return default
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return default
    if data.get("today_date") != today_str:
        data["today_date"] = today_str
        data["sent_today"] = 0
    return {**default, **data}


def _save_state(state_file: Path, state: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=state_file.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, state_file)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _pick_card(due_cards: list[dict], last_sent_card_id: Optional[str],
               last_sent_at: Optional[str], cooldown_minutes: int,
               now: datetime) -> tuple[Optional[dict], str]:
    """Pick one card to deliver, skipping the last-sent card if within cooldown.

    Returns (card, reason_if_skipped).
    """
    if not due_cards:
        return None, "no_due_cards"

    # Determine if the last-sent card is still in cooldown
    cooldown_card_id: Optional[str] = None
    if last_sent_card_id and last_sent_at:
        try:
            sent_at = datetime.fromisoformat(last_sent_at)
            if sent_at.tzinfo is None:
                sent_at = sent_at.replace(tzinfo=timezone.utc)
            if now - sent_at < timedelta(minutes=cooldown_minutes):
                cooldown_card_id = last_sent_card_id
        except ValueError:
            pass

    for card in due_cards:
        if card.get("id") != cooldown_card_id:
            return card, ""

    # All due cards are in cooldown (single-card deck edge case)
    return due_cards[0], ""


# ── Telegram send ─────────────────────────────────────────────────────────────

def _send_telegram(bot_token: str, chat_id: str, card: dict) -> dict:
    """Send one card question to Telegram. Returns API response dict."""
    text = f"<b>Memento flashcard:</b>\n\n{card['question']}"
    collection = card.get("collection", "")
    if collection:
        text += f"\n\n<i>Collection: {collection}</i>"

    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }).encode("utf-8")

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "error": f"HTTP {exc.code}", "body": body}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ── Output helpers ────────────────────────────────────────────────────────────

def _out(obj: dict) -> None:
    json.dump(obj, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


# ── Main logic ────────────────────────────────────────────────────────────────

def decide(config: dict, now: Optional[datetime] = None) -> dict:
    """Run the full delivery decision. Returns a result dict.

    Designed to be testable without side effects when dry_run=True.
    Does NOT write state or send Telegram when dry_run=True.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    today_str = now.astimezone().strftime("%Y-%m-%d")

    # ── Kill switch ───────────────────────────────────────────────────────────
    if not config["enabled"]:
        return {"action": "skip", "reason": "feature_disabled"}

    # ── Quiet hours ───────────────────────────────────────────────────────────
    if _is_quiet_hour(config["quiet_start"], config["quiet_end"], now):
        return {"action": "skip", "reason": "quiet_hours"}

    # ── Idle check ────────────────────────────────────────────────────────────
    is_idle, idle_reason = _is_idle(config["sessions_file"], config["idle_minutes"], now)
    if not is_idle:
        return {"action": "skip", "reason": "not_idle", "detail": idle_reason}

    # ── Load state ────────────────────────────────────────────────────────────
    state = _load_state(config["state_file"], today_str)

    # ── Daily cap ─────────────────────────────────────────────────────────────
    if state["sent_today"] >= config["daily_cap"]:
        return {"action": "skip", "reason": "daily_cap_reached",
                "sent_today": state["sent_today"], "cap": config["daily_cap"]}

    # ── Cooldown (time between sends) ─────────────────────────────────────────
    if state["last_sent_at"]:
        try:
            last_sent_ts = datetime.fromisoformat(state["last_sent_at"])
            if last_sent_ts.tzinfo is None:
                last_sent_ts = last_sent_ts.replace(tzinfo=timezone.utc)
            elapsed = now - last_sent_ts
            if elapsed < timedelta(minutes=config["cooldown_minutes"]):
                remaining_m = int((timedelta(minutes=config["cooldown_minutes"]) - elapsed).total_seconds() // 60)
                return {"action": "skip", "reason": "cooldown",
                        "cooldown_remaining_minutes": remaining_m}
        except ValueError:
            pass

    # ── Due cards ─────────────────────────────────────────────────────────────
    due_cards = _load_due_cards(config["cards_file"], now)
    card, pick_reason = _pick_card(
        due_cards,
        state.get("last_sent_card_id"),
        state.get("last_sent_at"),
        config["cooldown_minutes"],
        now,
    )
    if card is None:
        return {"action": "skip", "reason": pick_reason}

    # ── Dry-run or live send ──────────────────────────────────────────────────
    result: dict = {
        "action": "would_send" if config["dry_run"] else "sent",
        "card": {
            "id": card["id"],
            "question": card["question"],
            "collection": card.get("collection", ""),
        },
        "idle_reason": idle_reason,
    }

    if not config["dry_run"]:
        if not config["bot_token"] or not config["chat_id"]:
            result["action"] = "error"
            result["error"] = "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set"
            return result

        api_result = _send_telegram(config["bot_token"], config["chat_id"], card)
        result["telegram_ok"] = api_result.get("ok", False)
        if not result["telegram_ok"]:
            result["action"] = "error"
            result["error"] = api_result.get("error", "unknown")
            return result

        # Update delivery state
        state["last_sent_at"] = now.isoformat()
        state["last_sent_card_id"] = card["id"]
        state["sent_today"] = state.get("sent_today", 0) + 1
        _save_state(config["state_file"], state)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Memento idle-window delivery")
    parser.add_argument("--dry-run", action="store_true", default=None,
                        help="Force dry-run regardless of MEMENTO_DRY_RUN env")
    parser.add_argument("--enabled", action="store_true", default=None,
                        help="Force enabled regardless of MEMENTO_DELIVERY_ENABLED env")
    args = parser.parse_args()

    config = _load_config()
    if args.dry_run:
        config["dry_run"] = True
    if args.enabled:
        config["enabled"] = True

    result = decide(config)
    _out(result)


if __name__ == "__main__":
    main()
