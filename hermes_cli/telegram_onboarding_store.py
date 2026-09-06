"""Short-lived, profile-local Telegram handoffs and configuration rollback.

The poll bearer never goes to the browser. Persist it with owner-only permissions
so a dashboard restart does not orphan a bot that Telegram has already created.
"""
from dataclasses import asdict, dataclass, field
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from hermes_constants import get_hermes_home
from utils import atomic_json_write

CONFIRMATION_SECONDS = 30 * 60
lock = threading.RLock()


@dataclass
class TelegramOnboardingPairing:
    poll_token: str = field(repr=False)
    expires_at: str
    expires_at_ts: float
    bot_token: str | None = field(default=None, repr=False)
    bot_username: str | None = None
    owner_user_id: str | None = None
    requires_ack: bool = False
    setup: dict[str, Any] | None = None
    saved: bool = False
    result: dict[str, Any] | None = None


def record_path(pairing_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", pairing_id):
        raise HTTPException(404, "Telegram setup session was not found. Start a new setup.")
    return get_hermes_home() / "telegram-onboarding" / f"{pairing_id}.json"


def save(pairing_id: str, record: TelegramOnboardingPairing) -> None:
    path = record_path(pairing_id)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_json_write(path, asdict(record), mode=0o600)


def discard(pairing_id: str) -> None:
    record_path(pairing_id).unlink(missing_ok=True)


def load(pairing_id: str) -> TelegramOnboardingPairing:
    try:
        record = TelegramOnboardingPairing(**json.loads(record_path(pairing_id).read_text()))
    except FileNotFoundError:
        raise HTTPException(404, "Telegram setup session was not found. Start a new setup.") from None
    # A waiting client may first poll *after* creation expired while the Worker
    # already advanced to ready. Allow both Telegram and dashboard confirmation to
    # retrieve that result before deleting the local bearer.
    retain_until = record.expires_at_ts + (0 if record.bot_token or record.saved else 2 * CONFIRMATION_SECONDS)
    if retain_until <= time.time():
        discard(pairing_id)
        raise HTTPException(410, "Telegram setup expired. Start a new setup.")
    return record


def prune() -> None:
    directory = get_hermes_home() / "telegram-onboarding"
    for path in directory.glob("*.json"):
        try:
            load(path.stem)
        except HTTPException:
            continue


def save_configuration(bot_token: str, allowed_ids: list[str], commit) -> None:
    """Restore only the fields this attempt owns if any step fails.

    Read/validate the existing config before touching credentials. Rollback reads
    the latest files, preserving unrelated settings rather than restoring an
    entire stale .env/config snapshot over another writer's changes.
    """
    from copy import deepcopy
    from hermes_cli import config as cfg
    from utils import atomic_yaml_write

    with cfg._CONFIG_LOCK:
        env_before = cfg.load_env()
        raw_before = cfg.require_readable_config_before_write()
        if cfg.is_managed() or cfg.managed_scope.is_key_managed("platforms.telegram.enabled"):
            raise ValueError("Telegram configuration is managed by your administrator.")
        platform_before = deepcopy(raw_before.get("platforms", {}).get("telegram", {}))
        values = {"TELEGRAM_BOT_TOKEN": bot_token, "TELEGRAM_ALLOWED_USERS": ",".join(allowed_ids)}
        changed: list[str] = []
        enabled_attempted = False
        try:
            for key, value in values.items():
                changed.append(key)
                cfg.save_env_value(key, value)
                if cfg.load_env().get(key) != value:
                    raise ValueError(f"{key} is managed and cannot be changed here.")
            enabled_attempted = True
            raw = cfg.require_readable_config_before_write()
            raw.setdefault("platforms", {}).setdefault("telegram", {})["enabled"] = True
            cfg.atomic_config_write(cfg.get_config_path(), raw)
            commit()
        except Exception:
            for key in reversed(changed):
                # A concurrent editor's replacement takes precedence over rollback.
                if cfg.load_env().get(key) == values[key]:
                    if key in env_before:
                        cfg.save_env_value(key, env_before[key])
                    else:
                        cfg.remove_env_value(key)
            # A writer can fail after replacing the file. Restore enabled even if it
            # raised before returning, but never replace unrelated platform settings.
            raw = cfg.require_readable_config_before_write()
            telegram = raw.get("platforms", {}).get("telegram", {})
            if enabled_attempted and telegram.get("enabled") is True and platform_before.get("enabled") is not True:
                if "enabled" in platform_before:
                    telegram["enabled"] = platform_before["enabled"]
                else:
                    telegram.pop("enabled", None)
                if not telegram:
                    raw.get("platforms", {}).pop("telegram", None)
                if raw.get("platforms") == {} and "platforms" not in raw_before:
                    raw.pop("platforms", None)
                atomic_yaml_write(cfg.get_config_path(), raw)
            raise
