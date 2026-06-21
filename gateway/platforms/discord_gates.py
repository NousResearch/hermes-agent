"""
Inbound-message guardrails for the Discord gateway.

Imported by gateway/platforms/discord.py with a single early-return call at
the top of on_message. Enforces:

  1. Guild allowlist: messages from servers not in DISCORD_ALLOWED_GUILD_IDS
     are silently dropped (and the bot leaves the server). DMs always pass.
  2. Per-user message rate limit: max DISCORD_USER_MSG_DAILY messages per
     user per UTC day. When exceeded, the bot replies once with a polite
     "you've hit your daily limit" then ignores the user until UTC midnight.

State is persisted to /home/hermes/discord-limits/state.json so counters
survive restarts.

Env vars (read fresh each call so config changes take effect without restart):
  DISCORD_ALLOWED_GUILD_IDS   comma-separated guild IDs; empty = allow all
  DISCORD_USER_MSG_DAILY      integer cap, default 50
  DISCORD_LIMITS_STATE_FILE   override state path (default /home/hermes/discord-limits/state.json)
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger("discord.gates")

_STATE_LOCK = threading.Lock()


def _state_path() -> Path:
    return Path(os.environ.get("DISCORD_LIMITS_STATE_FILE", "/home/hermes/discord-limits/state.json"))


def _allowed_guilds() -> set[str]:
    raw = os.environ.get("DISCORD_ALLOWED_GUILD_IDS", "").strip()
    if not raw:
        return set()  # empty = allow all (no allowlist active)
    return {g.strip() for g in raw.split(",") if g.strip()}


def _msg_cap() -> int:
    try:
        return int(os.environ.get("DISCORD_USER_MSG_DAILY", "50"))
    except ValueError:
        return 50


def _img_cap() -> int:
    try:
        return int(os.environ.get("DISCORD_USER_IMG_DAILY", "5"))
    except ValueError:
        return 5


def _today_utc() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")


def _load_state() -> dict:
    p = _state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(p)


def _bucket(state: dict, today: str) -> dict:
    """Return today's bucket; prune older days while we're here."""
    if today not in state:
        # Drop stale day buckets to keep file small.
        for k in list(state.keys()):
            if k != today and len(k) == 10 and k[4] == "-":
                state.pop(k, None)
        state[today] = {}
    return state[today]


# ----- Public API used by Hermes' Discord platform -----------------------


def check_inbound(guild_id: str | None, user_id: str, is_bot_self: bool) -> tuple[bool, str | None]:
    """Decide whether to process an incoming Discord message.

    Returns (allow, reject_reason). If allow is False and reject_reason is
    a string, the platform should reply to the user with that string ONCE
    and then drop. If reject_reason is None, drop silently.
    """
    if is_bot_self:
        return False, None  # never react to our own messages

    # Guild allowlist (DMs have guild_id None and always pass).
    allowed = _allowed_guilds()
    if guild_id is not None and allowed and guild_id not in allowed:
        logger.warning("rejected message from unauthorized guild_id=%s", guild_id)
        return False, None  # silent drop; bot will also leave (see leave_disallowed_guild)

    # quarantined-install v2: per-user spend cap moved to openrouter-router
    # proxy. Keep the guild-allowlist check above; skip the message-count gate.
    return True, None


def check_image_quota(user_id: str) -> tuple[bool, str | None]:
    """Per-user image-generation cap. Called by the seedream MCP wrapper."""
    cap = _img_cap()
    today = _today_utc()
    with _STATE_LOCK:
        state = _load_state()
        bucket = _bucket(state, today)
        users = bucket.setdefault("user_imgs", {})
        count = users.get(user_id, 0)
        if count >= cap:
            return False, f"image limit reached for today ({cap}/user). resets at UTC midnight."
        users[user_id] = count + 1
        _save_state(state)
    return True, None


def is_guild_allowed(guild_id: str) -> bool:
    """Used by leave-on-join handler to decide whether to leave a server."""
    allowed = _allowed_guilds()
    return not allowed or guild_id in allowed


# ----- Spend pre-check (skip the agent entirely when over cap) -----------

def _spend_state_path():
    import os
    from pathlib import Path
    return Path(os.environ.get(
        "DISCORD_LIMITS_STATE_FILE",
        "/home/hermes/discord-limits/state.json",
    ))


def _override_path():
    import os
    from pathlib import Path
    return Path(os.environ.get(
        "BUDGET_OVERRIDE_FILE",
        "/home/hermes/budget-override.json",
    ))


def _today_multiplier() -> float:
    import json
    from datetime import datetime, timezone
    p = _override_path()
    if not p.exists():
        return 1.0
    try:
        d = json.loads(p.read_text())
        if d.get("date") == datetime.now(timezone.utc).strftime("%Y-%m-%d"):
            return float(d.get("multiplier", 1.0))
    except Exception:
        pass
    return 1.0


def _user_spend_today(user_id: str) -> float:
    import json
    from datetime import datetime, timezone
    p = _spend_state_path()
    if not p.exists():
        return 0.0
    try:
        s = json.loads(p.read_text())
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return float(((s.get(today) or {}).get("user_spend") or {}).get(user_id, 0.0))
    except Exception:
        return 0.0


def _format_budget_message(user_id: str, spent: float, cap: float) -> str:
    """Same format the router proxy uses, kept in sync."""
    from datetime import datetime, timedelta, timezone
    pct = round((spent / cap) * 100, 1) if cap > 0 else 0
    now = datetime.now(timezone.utc)
    next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    delta = next_midnight - now
    hours, minutes = int(delta.total_seconds() // 3600), int((delta.total_seconds() % 3600) // 60)
    until = f"{hours}h {minutes}m" if hours else f"{minutes}m"
    try:
        from zoneinfo import ZoneInfo
        syd = next_midnight.astimezone(ZoneInfo("Australia/Sydney")).strftime("%-I:%M %p %Z (%a)")
    except Exception:
        syd = ""
    utc = next_midnight.strftime("%H:%M UTC (%a)")
    suffix = f". Sydney {syd}, {utc}" if syd else f". {utc}"
    return (
        f"<@{user_id}> you have used **${spent:.3f} / ${cap:.2f}** ({pct}%) "
        f"of today's chat budget. Resets in {until}{suffix}."
    )


def check_spend_pre_invoke(user_id: str) -> "tuple[bool, str | None]":
    """Called at on_message time, BEFORE invoking the agent.

    If user is over their per-user chat cap, returns (False, friendly_msg)
    so the platform posts that message directly and skips the whole agent
    loop. Saves the confusion of routing a synthetic budget response
    through the agent's tool-call state machine.

    Bot/system invocations (no user_id) and cron-prefixed users are
    ignored here -- their caps are enforced at the router level.
    """
    import os
    if not user_id or user_id.startswith("cron:"):
        return True, None
    try:
        base_cap = float(os.environ.get("CHAT_USER_CAP_USD", "0.10"))
    except ValueError:
        base_cap = 0.10
    cap = base_cap * _today_multiplier()
    spent = _user_spend_today(user_id)
    if spent >= cap:
        # Over cap. ALWAYS allow through — the router handles free-swap
        # downstream and posts a synthetic budget reply if no free option
        # exists. We used to call free_picker.get_free_model() here to
        # short-circuit the block, but that hits OpenRouter on every
        # message and HUNG the bot when their frontend slowed (~13:18
        # incident). Trust the router; let the request flow.
        return True, None
    return True, None
