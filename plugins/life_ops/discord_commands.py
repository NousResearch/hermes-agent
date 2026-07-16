"""Discord slash command handlers: /rpe, /done, /dismiss, /snooze, /away-on,
/away-off.

/rpe registers on a Discord app_commands tree and provides ``handle_rpe()``
— the pure-logic function that validates, POSTs to the perf-coach
feel-entry endpoint, and writes to the audit log.

/done, /dismiss, /snooze are the Discord-side counterpart to
``plugins.life_ops.todo_store``'s persistent, stable-key todo store —
each is a thin ``handle_x()`` + ``register_x_command(tree)`` pair, same
shape as /rpe, that calls ``todo_store.close_todo()``.

/away-on, /away-off wrap ``plugins.life_ops.away_mode`` (set_away/clear_away).
discord.py slash commands don't nest into ``on|off`` subcommands without a
``discord.app_commands.Group`` — this codebase doesn't use Group anywhere
else, so these stay two flat commands rather than introducing that pattern
for a single use site.

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
import urllib.parse
import urllib.request
from typing import Any, Optional, TypedDict

from plugins.life_ops import audit as _audit
from plugins.life_ops.config import (
    get_perf_coach_token,
    get_perf_coach_url,
    get_perf_coach_user,
)

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
        "feel_date": target_date,
        "rpe_1_to_10": rpe,
        "notes": notes,
    }

    # ── POST to feel-entry ─────────────────────────────────────────────────
    # user_id is Discord's identifier and is only used for the audit log —
    # the worker resolves the perf-coach user via the ``user`` query param
    # (or falls back to single-active-user resolution when unset).
    url = base_url.rstrip("/") + _FEEL_ENTRY_PATH
    perf_coach_user = get_perf_coach_user()
    if perf_coach_user:
        url = f"{url}?user={urllib.parse.quote(perf_coach_user)}"
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
            status = resp.status
            http_outcome = f"{status} OK"

        if not (200 <= status < 300):
            # Defensive: urlopen normally raises HTTPError for non-2xx, but
            # only 2xx (e.g. 201 Created) is treated as success here.
            _audit.log_rpe_invocation(
                user_id=user_id,
                rpe=rpe,
                has_notes=has_notes,
                target_date=target_date,
                http_outcome=http_outcome,
            )
            _log.warning("feel-entry returned unexpected status %d for user=%s", status, user_id)
            return {
                "success": False,
                "error": (
                    f"Could not reach the performance coach service "
                    f"(HTTP {status}) — please try again later."
                ),
                "duplicate": False,
            }

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


# ---------------------------------------------------------------------------
# /done, /dismiss, /snooze — persistent todo store closure commands
# ---------------------------------------------------------------------------


class TodoCloseResult(TypedDict):
    ok: bool
    error: Optional[str]
    status: Optional[str]


async def _autocomplete_todo_key(interaction: Any, current: str) -> list:
    """Shared autocomplete for /done, /dismiss, /snooze's ``key`` param.

    Suggests currently-open ``todo_store`` keys (label = key + truncated
    text, value = key), filtered case-insensitively against ``current``.
    Defensive: any failure (store unavailable, discord.py missing) degrades
    to an empty suggestion list rather than raising — autocomplete must
    never crash the interaction.
    """
    try:
        import discord
        from plugins.life_ops import todo_store
    except Exception:
        return []
    try:
        open_todos = todo_store.get_open_keys()
    except Exception:
        return []

    query = (current or "").strip().lower()
    choices = []
    for item in open_todos:
        key = str(item.get("key") or "")
        text = str(item.get("text") or "")
        if query and query not in key.lower() and query not in text.lower():
            continue
        label = f"{key} — {text}" if text else key
        # Discord's Choice.name is capped at 100 chars.
        if len(label) > 100:
            label = label[:97] + "..."
        choices.append(discord.app_commands.Choice(name=label, value=key))
        if len(choices) >= 25:
            break
    return choices


def handle_done(key: str) -> TodoCloseResult:
    """Mark todo ``key`` done via the persistent todo store. Never raises."""
    from plugins.life_ops import todo_store

    key = str(key or "").strip()
    if not key:
        return {"ok": False, "error": "A todo key is required.", "status": None}
    return todo_store.close_todo(key, "done", source="discord:/done")


def handle_dismiss(key: str) -> TodoCloseResult:
    """Dismiss todo ``key`` via the persistent todo store. Never raises."""
    from plugins.life_ops import todo_store

    key = str(key or "").strip()
    if not key:
        return {"ok": False, "error": "A todo key is required.", "status": None}
    return todo_store.close_todo(key, "dismiss", source="discord:/dismiss")


def handle_snooze(key: str, until: str) -> TodoCloseResult:
    """Snooze todo ``key`` until ISO date ``until`` (YYYY-MM-DD). Never raises.

    An unparseable date returns a friendly ``{"ok": False, "error": ...}``
    rather than raising — the caller (the slash command) surfaces this as
    an ephemeral reply, not a Discord-level interaction failure.
    """
    from plugins.life_ops import todo_store

    key = str(key or "").strip()
    if not key:
        return {"ok": False, "error": "A todo key is required.", "status": None}
    until = str(until or "").strip()
    try:
        parsed = datetime.date.fromisoformat(until)
    except ValueError:
        return {
            "ok": False,
            "error": f"Could not parse date {until!r} — use YYYY-MM-DD (e.g. 2026-07-22).",
            "status": None,
        }
    return todo_store.close_todo(
        key, "snooze", source="discord:/snooze", snooze_until=parsed.isoformat()
    )


def _friendly_close_message(action_label: str, key: str, result: dict) -> str:
    if result.get("ok"):
        return f"{action_label} `{key}`."
    return result.get("error") or "An unexpected error occurred."


def register_done_command(tree: Any) -> None:
    """Register the /done slash command on a Discord app_commands tree.

    Same shape as :func:`register_rpe_command`: a pure ``handle_done()`` for
    testability, wired to a thin Discord-facing wrapper here.
    """
    try:
        import discord
    except ImportError:
        _log.warning("discord package not available; /done command not registered.")
        return

    @tree.command(name="done", description="Mark an open todo as done")
    @discord.app_commands.describe(key="The todo's stable key (see the morning brief's /done <key> hint)")
    @discord.app_commands.autocomplete(key=_autocomplete_todo_key)
    async def slash_done(interaction: discord.Interaction, key: str) -> None:
        try:
            await interaction.response.defer(ephemeral=True)
        except Exception:
            pass

        result = handle_done(key)
        msg = _friendly_close_message("✅ Marked done", key, result)

        try:
            await interaction.edit_original_response(content=msg)
        except Exception:
            try:
                await interaction.followup.send(content=msg, ephemeral=True)
            except Exception:
                pass


def register_dismiss_command(tree: Any) -> None:
    """Register the /dismiss slash command on a Discord app_commands tree."""
    try:
        import discord
    except ImportError:
        _log.warning("discord package not available; /dismiss command not registered.")
        return

    @tree.command(name="dismiss", description="Dismiss an open todo")
    @discord.app_commands.describe(key="The todo's stable key (see the morning brief's /done <key> hint)")
    @discord.app_commands.autocomplete(key=_autocomplete_todo_key)
    async def slash_dismiss(interaction: discord.Interaction, key: str) -> None:
        try:
            await interaction.response.defer(ephemeral=True)
        except Exception:
            pass

        result = handle_dismiss(key)
        msg = _friendly_close_message("🗑️ Dismissed", key, result)

        try:
            await interaction.edit_original_response(content=msg)
        except Exception:
            try:
                await interaction.followup.send(content=msg, ephemeral=True)
            except Exception:
                pass


def register_snooze_command(tree: Any) -> None:
    """Register the /snooze slash command on a Discord app_commands tree."""
    try:
        import discord
    except ImportError:
        _log.warning("discord package not available; /snooze command not registered.")
        return

    @tree.command(name="snooze", description="Snooze an open todo until a date")
    @discord.app_commands.describe(
        key="The todo's stable key (see the morning brief's /done <key> hint)",
        date="Date to snooze until (YYYY-MM-DD)",
    )
    @discord.app_commands.autocomplete(key=_autocomplete_todo_key)
    async def slash_snooze(interaction: discord.Interaction, key: str, date: str) -> None:
        try:
            await interaction.response.defer(ephemeral=True)
        except Exception:
            pass

        result = handle_snooze(key, date)
        if result.get("ok"):
            msg = f"💤 Snoozed `{key}` until {date}."
        else:
            msg = result.get("error") or "An unexpected error occurred."

        try:
            await interaction.edit_original_response(content=msg)
        except Exception:
            try:
                await interaction.followup.send(content=msg, ephemeral=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# /away-on, /away-off — plugins.life_ops.away_mode wiring
# ---------------------------------------------------------------------------


def handle_away_on(until: Optional[str]) -> dict:
    """Turn away mode on, optionally until an ISO date. Never raises."""
    from plugins.life_ops import away_mode

    until = (until or "").strip() or None
    if until is not None:
        try:
            datetime.date.fromisoformat(until)
        except ValueError:
            return {
                "ok": False,
                "error": (
                    f"Could not parse date {until!r} — use YYYY-MM-DD "
                    "(e.g. 2026-07-25), or leave empty for indefinite."
                ),
            }
    away_mode.set_away(until)
    return {"ok": True, "until": until}


def handle_away_off() -> dict:
    """Turn away mode off. Never raises."""
    from plugins.life_ops import away_mode

    away_mode.clear_away()
    return {"ok": True}


def register_away_on_command(tree: Any) -> None:
    """Register /away-on. Flat command (not a subcommand) — this codebase
    doesn't use ``discord.app_commands.Group`` anywhere, so /away-on and
    /away-off stay two top-level commands rather than introducing that
    pattern for a single use site.
    """
    try:
        import discord
    except ImportError:
        _log.warning("discord package not available; /away-on command not registered.")
        return

    @tree.command(name="away-on", description="Pause overnight runs and bedtime prompts")
    @discord.app_commands.describe(
        until="Optional date to resume on (YYYY-MM-DD); leave empty for indefinite"
    )
    async def slash_away_on(interaction: discord.Interaction, until: str = "") -> None:
        try:
            await interaction.response.defer(ephemeral=True)
        except Exception:
            pass

        result = handle_away_on(until or None)
        if result.get("ok"):
            resolved_until = result.get("until")
            msg = (
                f"🌙 Away mode on until {resolved_until}."
                if resolved_until
                else "🌙 Away mode on (indefinite — /away-off to resume)."
            )
        else:
            msg = result.get("error") or "An unexpected error occurred."

        try:
            await interaction.edit_original_response(content=msg)
        except Exception:
            try:
                await interaction.followup.send(content=msg, ephemeral=True)
            except Exception:
                pass


def register_away_off_command(tree: Any) -> None:
    """Register /away-off — see :func:`register_away_on_command` for the
    flat-command-vs-Group naming note.
    """
    try:
        import discord
    except ImportError:
        _log.warning("discord package not available; /away-off command not registered.")
        return

    @tree.command(name="away-off", description="Resume overnight runs and bedtime prompts")
    async def slash_away_off(interaction: discord.Interaction) -> None:
        try:
            await interaction.response.defer(ephemeral=True)
        except Exception:
            pass

        handle_away_off()
        msg = "☀️ Away mode off — overnight runs and bedtime prompts resumed."

        try:
            await interaction.edit_original_response(content=msg)
        except Exception:
            try:
                await interaction.followup.send(content=msg, ephemeral=True)
            except Exception:
                pass
