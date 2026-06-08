"""Shared ``/cron-recipe`` command logic for CLI, TUI, and gateway.

The conversational counterpart to the dashboard's Cron Recipes form. Where a
surface has a screen, the user fills a form (dashboard / GUI app) and the API
calls ``fill_recipe`` -> ``create_job`` directly. Where a surface is just a
chat line, the user pastes a pre-filled slash command and this handler
parses it; any missing or invalid slot is reported so the agent can ask.

Subcommand shapes:
  /cron-recipe                      list the catalog (numbered + copy commands)
  /cron-recipe <key>                show that recipe's slots + a ready command
  /cron-recipe <key> slot=val …     fill + create the cron job

Parsing is shlex-based so quoted free-text values (``criteria="from my boss"``)
survive. On a fill error the message names the slot, which is exactly what the
agent needs to ask a targeted follow-up rather than re-prompting everything.
"""

from __future__ import annotations

import logging
import shlex
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _resolve_origin(explicit: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if explicit is not None:
        return explicit
    try:
        from gateway.session_context import get_session_env

        platform = get_session_env("HERMES_SESSION_PLATFORM")
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID")
        if platform and chat_id:
            return {
                "platform": platform,
                "chat_id": chat_id,
                "chat_name": get_session_env("HERMES_SESSION_CHAT_NAME") or None,
                "thread_id": get_session_env("HERMES_SESSION_THREAD_ID") or None,
            }
    except Exception:
        pass
    return None


def _parse_kv(tokens) -> Tuple[Dict[str, str], list]:
    """Split ``slot=value`` tokens from bare tokens. Returns (values, leftovers)."""
    values: Dict[str, str] = {}
    leftovers = []
    for tok in tokens:
        if "=" in tok:
            k, _, v = tok.partition("=")
            k = k.strip()
            if k:
                values[k] = v.strip()
                continue
        leftovers.append(tok)
    return values, leftovers


def _fmt_catalog() -> str:
    from cron.recipe_catalog import CATALOG, recipe_slash_command

    lines = ["Cron Recipes — `/cron-recipe <name>` to set one up:\n"]
    for r in CATALOG:
        lines.append(f"  • {r.key} — {r.title}")
        lines.append(f"    {r.description}")
        lines.append(f"    ↳ {recipe_slash_command(r)}")
    lines.append("\nEdit the values then send, or just send to use the defaults.")
    return "\n".join(lines)


def _fmt_recipe(recipe) -> str:
    from cron.recipe_catalog import recipe_slash_command

    lines = [f"{recipe.title} — {recipe.description}\n", "Fields:"]
    for s in recipe.slots:
        opts = f"  (one of: {', '.join(map(str, s.options))})" if s.options else ""
        dflt = f"  [default: {s.default}]" if s.default not in (None, "") else ""
        opt = "  (optional)" if s.optional else ""
        lines.append(f"  • {s.name}: {s.label}{opts}{dflt}{opt}")
    lines.append("\nReady-to-edit command:")
    lines.append(f"  {recipe_slash_command(recipe)}")
    return "\n".join(lines)


def handle_cron_recipe_command(
    args: str,
    *,
    origin: Optional[Dict[str, Any]] = None,
) -> str:
    """Dispatch a ``/cron-recipe`` invocation. Returns text to show the user.

    ``args`` is everything after ``/cron-recipe``. ``origin`` lets an accepted
    recipe's job deliver back to the chat it was created from; resolved from
    session env when omitted.
    """
    try:
        from cron.recipe_catalog import fill_recipe, get_recipe, RecipeFillError
    except Exception as e:  # pragma: no cover - import guard
        logger.debug("recipe catalog import failed: %s", e)
        return "Cron Recipes are unavailable in this build."

    try:
        tokens = shlex.split(args or "")
    except ValueError:
        tokens = (args or "").split()

    # Bare -> list catalog.
    if not tokens:
        return _fmt_catalog()

    key = tokens[0]
    recipe = get_recipe(key)
    if recipe is None:
        return (
            f"No cron recipe named '{key}'. Run /cron-recipe to see the catalog."
        )

    values, _leftover = _parse_kv(tokens[1:])

    # `<key>` with no slot args -> show the recipe's fields + a ready command.
    if not values:
        return _fmt_recipe(recipe)

    # `<key> slot=val …` -> fill + create.
    try:
        spec = fill_recipe(recipe, values, origin=_resolve_origin(origin))
    except RecipeFillError as e:
        return f"Can't set up '{recipe.title}': {e}\nRun /cron-recipe {key} to see its fields."

    try:
        from cron.jobs import create_job

        job = create_job(**spec)
    except Exception as e:
        logger.debug("cron-recipe create_job failed: %s", e)
        return f"Failed to create the job: {e}"

    sched = job.get("schedule_display") or spec.get("schedule", "")
    return (
        f"Scheduled '{recipe.title}'"
        + (f" ({sched})" if sched else "")
        + f", delivering to {spec.get('deliver', 'origin')}. Manage it with /cron."
    )
