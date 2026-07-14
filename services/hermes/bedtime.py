"""Bedtime overnight-sprint prompt service (issue #8).

Provides:
- fetch_backlog_count() — GET /api/home, sum per-project backlog counts
- check_running_sprint() — GET /api/sprints/running-all
- start_sprint() — POST /api/sprints/run with bearer token
- log_bedtime_action() — append one JSONL audit entry
- BedtimeView — pure-Python discord.ui.View subclass for Start/Skip

The BedtimeView is defined here so it can be tested without a live
discord.py install. The adapter module imports and re-exports it inside
_define_discord_view_classes() so the Discord-specific button decorators
are applied only when discord.py is available.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from services.hermes.config import get_commander_api_url

_log = logging.getLogger(__name__)
_audit_write_lock = threading.Lock()

_HOME_PATH = "/api/home"
_RUNNING_ALL_PATH = "/api/sprints/running-all"
_SPRINT_RUN_PATH = "/api/sprints/run"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _resolve_bedtime_log_path() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "logs" / "bedtime-audit.log"


def _get_commander_token() -> Optional[str]:
    return os.environ.get("COMMANDER_API_TOKEN") or None


# ---------------------------------------------------------------------------
# Commander API calls
# ---------------------------------------------------------------------------


def fetch_backlog_count() -> int:
    """Return the sum of per-project backlog counts from GET /api/home.

    Raises ValueError if COMMANDER_API_URL is not configured.
    Raises urllib.error.HTTPError / urllib.error.URLError on network/HTTP failure.
    """
    base_url = get_commander_api_url()
    if not base_url:
        raise ValueError(
            "COMMANDER_API_URL is not configured — cannot fetch backlog count."
        )
    url = base_url.rstrip("/") + _HOME_PATH
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return 0
    if isinstance(data, list):
        return sum(int(item.get("backlog_count") or 0) for item in data)
    return 0


def check_running_sprint() -> dict:
    """Check GET /api/sprints/running-all.

    Returns {"running": bool, "sprint_id": str|None}.
    Raises urllib.error.HTTPError / urllib.error.URLError on failure.
    """
    base_url = get_commander_api_url()
    if not base_url:
        raise ValueError("COMMANDER_API_URL is not configured.")
    url = base_url.rstrip("/") + _RUNNING_ALL_PATH
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        data = []
    if isinstance(data, list) and data:
        sprint_id = data[0].get("id") if isinstance(data[0], dict) else None
        return {"running": True, "sprint_id": sprint_id}
    return {"running": False, "sprint_id": None}


def start_sprint() -> dict:
    """POST /api/sprints/run with the bearer token from COMMANDER_API_TOKEN.

    Returns {"success": bool, "sprint_id": str|None, "error": str|None}.
    Never raises — errors are returned as success=False entries.
    """
    base_url = get_commander_api_url()
    if not base_url:
        return {
            "success": False,
            "sprint_id": None,
            "error": "COMMANDER_API_URL is not configured.",
        }
    token = _get_commander_token()
    url = base_url.rstrip("/") + _SPRINT_RUN_PATH
    req = urllib.request.Request(url, data=b"", method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            payload = {}
        sprint_id = payload.get("id") or None
        return {"success": True, "sprint_id": sprint_id, "error": None}
    except urllib.error.HTTPError as exc:
        _log.warning("POST /api/sprints/run failed: HTTP %d %s", exc.code, exc.reason)
        return {
            "success": False,
            "sprint_id": None,
            "error": f"HTTP {exc.code} {exc.reason}",
        }
    except urllib.error.URLError as exc:
        _log.warning("POST /api/sprints/run unreachable: %s", exc.reason)
        return {
            "success": False,
            "sprint_id": None,
            "error": f"Connection error: {exc.reason}",
        }


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


def log_bedtime_action(
    *,
    user_id: str,
    username: str,
    action: str,
    sprint_id: Optional[str],
) -> None:
    """Append one bedtime-action audit entry (JSONL).

    Fields: ts (UTC ISO-8601), user_id, username, action, sprint_id.
    Write failures are logged at WARNING and never raised.
    """
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "user_id": user_id,
        "username": username,
        "action": action,
        "sprint_id": sprint_id,
    }
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    path = _resolve_bedtime_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _audit_write_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as exc:
        _log.warning("Bedtime audit log write failed: %s", exc)


# ---------------------------------------------------------------------------
# BedtimeView — pure-Python base that works without discord.py
# ---------------------------------------------------------------------------
# The real discord.ui.View subclass is assembled in
# plugins/platforms/discord/adapter._define_discord_view_classes() where
# the @discord.ui.button decorators are applied.  This class provides the
# pure async business logic so it can be unit-tested without a live bot.


class BedtimeView:
    """Business-logic layer for the bedtime overnight-sprint prompt.

    In production, this class is subclassed inside _define_discord_view_classes()
    to attach the real discord.ui.View mechanics.  For unit tests, the plain
    class is used directly.
    """

    def __init__(
        self,
        *,
        backlog_count: int,
        allowed_user_ids: set,
        allowed_role_ids: Optional[set] = None,
        timeout: int = 300,
    ) -> None:
        self.backlog_count = backlog_count
        self.allowed_user_ids = allowed_user_ids
        self.allowed_role_ids = allowed_role_ids or set()
        self.timeout = timeout
        self.resolved = False
        self.children: list = []

    def _check_auth(self, interaction) -> bool:
        # In tests the environment variable DISCORD_ALLOW_ALL_USERS=true bypasses checks.
        if os.getenv("DISCORD_ALLOW_ALL_USERS", "").strip().lower() in {"true", "1", "yes"}:
            return True
        user = getattr(interaction, "user", None)
        if user is None:
            return False
        uid = str(getattr(user, "id", ""))
        user_set = {str(u).strip() for u in self.allowed_user_ids}
        if "*" in user_set or uid in user_set:
            return True
        return False

    async def start(self, interaction, button) -> None:
        """Handle the [Start] button press."""
        if self.resolved:
            await interaction.response.send_message(
                "Already handled.", ephemeral=True
            )
            return

        if not self._check_auth(interaction):
            await interaction.response.send_message(
                "You're not authorised to use this button.", ephemeral=True
            )
            return

        self.resolved = True
        user = getattr(interaction, "user", None)
        user_id = str(getattr(user, "id", "unknown"))
        username = getattr(user, "name", "unknown")

        # Disable buttons to prevent further interaction
        for child in self.children:
            child.disabled = True

        # Check for already-running sprint
        try:
            sprint_status = await asyncio.to_thread(check_running_sprint)
        except Exception as exc:
            _log.error("check_running_sprint failed: %s", exc)
            await interaction.response.send_message(
                f"❌ Error checking sprint status: {exc}", ephemeral=True
            )
            self.resolved = False  # allow retry
            for child in self.children:
                child.disabled = False
            return

        if sprint_status["running"]:
            log_bedtime_action(
                user_id=user_id,
                username=username,
                action="skip",
                sprint_id=sprint_status.get("sprint_id"),
            )
            await interaction.response.send_message(
                "A sprint is already running — skipped.", ephemeral=True
            )
            return

        # Start the sprint
        result = await asyncio.to_thread(start_sprint)
        if not result["success"]:
            log_bedtime_action(
                user_id=user_id,
                username=username,
                action="start",
                sprint_id=None,
            )
            await interaction.response.send_message(
                f"❌ Could not start sprint: {result.get('error', 'unknown error')}",
                ephemeral=True,
            )
            return

        sprint_id = result["sprint_id"]
        log_bedtime_action(
            user_id=user_id,
            username=username,
            action="start",
            sprint_id=sprint_id,
        )
        try:
            await interaction.response.edit_message(
                content=f"✅ Overnight sprint started! (ID: {sprint_id})",
                view=None,
            )
        except Exception:
            try:
                await interaction.followup.send(
                    content=f"✅ Overnight sprint started! (ID: {sprint_id})"
                )
            except Exception:
                pass

    async def skip(self, interaction, button) -> None:
        """Handle the [Skip] button press."""
        if self.resolved:
            await interaction.response.send_message(
                "Already handled.", ephemeral=True
            )
            return

        if not self._check_auth(interaction):
            await interaction.response.send_message(
                "You're not authorised to use this button.", ephemeral=True
            )
            return

        self.resolved = True
        user = getattr(interaction, "user", None)
        user_id = str(getattr(user, "id", "unknown"))
        username = getattr(user, "name", "unknown")

        for child in self.children:
            child.disabled = True

        log_bedtime_action(
            user_id=user_id,
            username=username,
            action="skip",
            sprint_id=None,
        )
        try:
            await interaction.response.edit_message(
                content="⏭ Overnight sprint skipped.",
                view=self,
            )
        except Exception:
            try:
                await interaction.followup.send(
                    content="⏭ Overnight sprint skipped.", ephemeral=False
                )
            except Exception:
                pass

    async def on_timeout(self) -> None:
        """Called by discord.py when the view timeout expires."""
        self.resolved = True
        for child in self.children:
            child.disabled = True
        log_bedtime_action(
            user_id="system",
            username="system",
            action="timeout",
            sprint_id=None,
        )
        msg = getattr(self, "_message", None)
        if msg:
            try:
                await msg.edit(
                    content="⏰ Overnight sprint prompt timed out — no action taken.",
                    view=self,
                )
            except Exception:
                pass
