"""Discord slash command handler for /rpe (Rate of Perceived Exertion).

Registers the /rpe command on a Discord app_commands tree and provides
``handle_rpe()`` — the pure-logic function that validates, POSTs to the
perf-coach feel-entry endpoint, and writes to the audit log.

Design principles:
- Hermes ingests and forwards only; it does NOT interpret, parse, or act
  on the feedback content (AC6).
- Bearer token is always read from config; never hardcoded (AC5).
- Every invocation (success and failure) is written to the audit log;
  the bearer token never appears in the log (AC9).
- Non-2xx / unreachable endpoint → clear human-readable error; no silent
  failure and no raw exception traceback exposed to the user (AC7).
- 409 Conflict from the endpoint is treated as an idempotent no-op (AC8).
"""
from __future__ import annotations

import datetime
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Optional, TypedDict

from services.hermes import audit as _audit
from services.hermes.config import get_perf_coach_token, get_perf_coach_url

_log = logging.getLogger(__name__)

_FEEL_ENTRY_PATH = "/feel-entry"


class RpeResult(TypedDict):
    success: bool
    error: Optional[str]
    duplicate: bool


def handle_rpe(
    *,
    user_id: str,
    rpe: int,
    notes: Optional[str],
    date_str: Optional[str],
    today: Optional[datetime.date] = None,
) -> RpeResult:
    """Validate, POST to perf-coach, and write audit log.

    Returns a dict with ``success`` (bool), ``error`` (str or None), and
    ``duplicate`` (bool, True when the endpoint reported a 409 conflict).
    """
    # ── Validate rpe range (1–10) ──────────────────────────────────────────
    if not (1 <= rpe <= 10):
        return {
            "success": False,
            "error": f"rpe must be between 1 and 10 (got {rpe}).",
            "duplicate": False,
        }

    # ── Resolve config ─────────────────────────────────────────────────────
    base_url = get_perf_coach_url()
    token = get_perf_coach_token()

    if not base_url:
        return {
            "success": False,
            "error": (
                "Performance coach service is not configured. "
                "Set PERF_COACH_URL to enable /rpe."
            ),
            "duplicate": False,
        }
    if not token:
        return {
            "success": False,
            "error": (
                "Performance coach bearer token is not configured. "
                "Set PERF_COACH_BEARER_TOKEN to enable /rpe."
            ),
            "duplicate": False,
        }

    # ── Build payload ──────────────────────────────────────────────────────
    if today is None:
        today = datetime.date.today()
    target_date = date_str if date_str else today.isoformat()
    has_notes = bool(notes)

    payload: dict[str, Any] = {
        "user_id": user_id,
        "rpe": rpe,
        "date": target_date,
        "notes": notes,
    }

    # ── POST to feel-entry ─────────────────────────────────────────────────
    url = base_url.rstrip("/") + _FEEL_ENTRY_PATH
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")

    http_outcome: str
    try:
        with urllib.request.urlopen(req) as resp:
            http_outcome = f"{resp.status} OK"
        _audit.log_rpe_invocation(
            user_id=user_id,
            rpe=rpe,
            has_notes=has_notes,
            target_date=target_date,
            http_outcome=http_outcome,
        )
        return {"success": True, "error": None, "duplicate": False}

    except urllib.error.HTTPError as exc:
        if exc.code == 409:
            # Idempotent — endpoint already has a record for this user+date.
            http_outcome = f"409 Conflict (duplicate)"
            _audit.log_rpe_invocation(
                user_id=user_id,
                rpe=rpe,
                has_notes=has_notes,
                target_date=target_date,
                http_outcome=http_outcome,
            )
            return {"success": True, "error": None, "duplicate": True}

        http_outcome = f"{exc.code} {exc.reason}"
        _audit.log_rpe_invocation(
            user_id=user_id,
            rpe=rpe,
            has_notes=has_notes,
            target_date=target_date,
            http_outcome=http_outcome,
        )
        _log.warning("feel-entry returned %d for user=%s: %s", exc.code, user_id, exc.reason)
        return {
            "success": False,
            "error": (
                f"Could not reach the performance coach service "
                f"(HTTP {exc.code} {exc.reason}) — please try again later."
            ),
            "duplicate": False,
        }

    except urllib.error.URLError as exc:
        http_outcome = f"Connection Error: {exc.reason}"
        _audit.log_rpe_invocation(
            user_id=user_id,
            rpe=rpe,
            has_notes=has_notes,
            target_date=target_date,
            http_outcome=http_outcome,
        )
        _log.warning("feel-entry unreachable for user=%s: %s", user_id, exc.reason)
        return {
            "success": False,
            "error": (
                "Could not reach the performance coach service — "
                "please try again later."
            ),
            "duplicate": False,
        }


def register_rpe_command(tree: Any) -> None:
    """Register the /rpe slash command on a Discord app_commands tree.

    Call this inside ``_register_slash_commands`` after the tree is attached
    to the Discord client.

    Parameters match the Discord interaction; actual HTTP work is delegated
    to ``handle_rpe()`` so the logic stays testable without a live bot.
    """
    try:
        import discord
    except ImportError:
        _log.warning("discord package not available; /rpe command not registered.")
        return

    @tree.command(
        name="rpe",
        description="Log your Rate of Perceived Exertion after a training session",
    )
    @discord.app_commands.describe(
        rpe="Training intensity (1 = very easy, 10 = maximal effort)",
        notes="Optional free-text notes about the session",
        date="Optional date (YYYY-MM-DD); defaults to today",
    )
    async def slash_rpe(
        interaction: discord.Interaction,
        rpe: int,
        notes: str = "",
        date: str = "",
    ) -> None:
        user_id = str(interaction.user.id)

        # Defer so we have time to call the network
        try:
            await interaction.response.defer(ephemeral=True)
        except Exception:
            pass

        result = handle_rpe(
            user_id=user_id,
            rpe=rpe,
            notes=notes or None,
            date_str=date or None,
        )

        if result["success"]:
            if result.get("duplicate"):
                msg = "Already logged for this date — no duplicate created."
            else:
                msg = f"RPE {rpe} logged for {date or 'today'}. Keep it up!"
        else:
            msg = result["error"] or "An unexpected error occurred."

        try:
            await interaction.edit_original_response(content=msg)
        except Exception:
            try:
                await interaction.followup.send(content=msg, ephemeral=True)
            except Exception:
                pass
