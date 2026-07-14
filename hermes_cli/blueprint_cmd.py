"""Shared ``/blueprint`` command logic for CLI, TUI, and gateway.

The conversational counterpart to the dashboard's Automation Blueprints form. Where a
surface has a screen, the user fills a form (dashboard / GUI app) and the API
calls ``fill_blueprint`` -> ``create_job`` directly. Where a surface is just a
chat line, the user picks a blueprint by name and the agent asks for what it
needs — pick a blueprint by name and the agent asks you for what it needs, one
question at a time (the messaging-assistant model: pick a blueprint → it asks you
a couple things → done).

Subcommand shapes:
  /blueprint                      list the catalog
  /blueprint <name>               name-match a blueprint, then SEED THE AGENT to
                                    ask the user for each value conversationally
  /blueprint <name> slot=val …    fill + create the cron job directly
                                    (the deterministic dashboard / docs / power-
                                    user shortcut — no agent turn)

The ``<name>`` form is forgiving: exact key, unique prefix, or fuzzy match all
resolve; an ambiguous query lists the candidates; an unknown one suggests the
closest. When it resolves, the handler returns an ``agent_seed`` — a natural-
language instruction built from the blueprint's typed slots + schedule/prompt
templates — that the calling surface feeds to the agent as a normal user turn
(gateway: rewrite ``event.text`` and fall through, the ``/steer`` pattern; CLI:
a one-shot pending seed the main loop runs). The agent then asks for each slot
and calls the existing ``cronjob`` tool. No new tool, no second job engine.

Parsing is shlex-based so quoted free-text values (``criteria="from my boss"``)
survive.
"""

from __future__ import annotations

import difflib
import logging
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BlueprintCommandResult:
    """Outcome of a ``/blueprint`` invocation.

    ``text`` is always shown to the user. When ``agent_seed`` is set, the
    calling surface should ALSO hand that seed to the agent as the user's next
    turn (the blueprint was matched and now the agent gathers the slot values
    conversationally). When ``agent_seed`` is None the command is fully handled
    (catalog listing, direct create, or an error) and nothing is sent to the
    agent.
    """

    text: str
    agent_seed: Optional[str] = None


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
        logger.debug("Suppressed exception", exc_info=True)
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


def match_blueprint(query: str) -> Tuple[Optional[Any], List[Any]]:
    """Resolve a free-typed blueprint name to a blueprint.

    Returns ``(blueprint, candidates)``:
      * exact key or unique prefix / fuzzy match -> ``(blueprint, [])``
      * ambiguous (2+ plausible) -> ``(None, [candidates…])``
      * no plausible match -> ``(None, [])``

    Matching is forgiving because chat-line users type the name (unlike the
    dashboard/Discord where it's picked): exact key first, then case-insensitive
    prefix on key or title, then a difflib fuzzy pass.
    """
    from cron.blueprint_catalog import CATALOG, get_blueprint

    q = (query or "").strip().lower()
    if not q:
        return None, []

    exact = get_blueprint(q)
    if exact is not None:
        return exact, []

    # Prefix match on key or title word-start.
    prefix = [
        r for r in CATALOG
        if r.key.lower().startswith(q)
        or any(w.lower().startswith(q) for w in r.title.split())
    ]
    if len(prefix) == 1:
        return prefix[0], []
    if len(prefix) > 1:
        return None, prefix

    # Substring match anywhere in key/title/description.
    substr = [
        r for r in CATALOG
        if q in r.key.lower() or q in r.title.lower() or q in r.description.lower()
    ]
    if len(substr) == 1:
        return substr[0], []
    if len(substr) > 1:
        return None, substr

    # Fuzzy on keys (typo tolerance).
    keys = [r.key for r in CATALOG]
    close = difflib.get_close_matches(q, keys, n=3, cutoff=0.6)
    if len(close) == 1:
        return get_blueprint(close[0]), []
    if len(close) > 1:
        return None, [get_blueprint(k) for k in close]

    return None, []


def _humanize_schedule(blueprint) -> str:
    from cron.blueprint_catalog import _humanize_schedule as _h

    try:
        return _h(blueprint)
    except Exception:
        return "on a schedule"


def build_blueprint_seed(blueprint) -> str:
    """Build the natural-language fill-request the agent will act on.

    The agent reads this as a normal user turn, asks the user for each unfilled
    slot one at a time, then calls the ``cronjob`` tool with the
    cron expression it builds from the blueprint's ``schedule_template`` and the
    rendered prompt. Defaults are stated so the agent can offer them.
    """
    from cron.blueprint_catalog import WEEKDAY_PRESETS

    lines: List[str] = []
    lines.append(
        f"Set up the '{blueprint.title}' automation for me (automation blueprint "
        f"'{blueprint.key}'). {blueprint.description}"
    )
    lines.append("")
    lines.append(
        "Ask me for each of these, one at a time, offering the default in "
        "brackets if I don't have a preference:"
    )
    for s in blueprint.slots:
        bits = [f"- {s.label} ({s.name})"]
        if s.options:
            bits.append(f" — one of: {', '.join(map(str, s.options))}")
        if s.default not in (None, ""):
            bits.append(f" [default: {s.default}]")
        if s.optional:
            bits.append(" (optional)")
        if s.help:
            bits.append(f" — {s.help}")
        lines.append("".join(bits))

    lines.append("")
    lines.append(
        "Once you have my answers, create the job by calling the cronjob tool "
        "with action='create'. Build the schedule as a cron expression from "
        f"this template: `{blueprint.schedule_template}` "
        "(fill {minute}/{hour} from the chosen time, {dow} from the weekday "
        f"choice using {dict(WEEKDAY_PRESETS)}, {{interval_min}} from any "
        "interval). Use this exact prompt for the job (substituting my "
        f"answers into any {{slot}} placeholders): \"{blueprint.prompt_template}\". "
        "Confirm the schedule and what it will do before you create it."
    )
    return "\n".join(lines)


def _fmt_catalog() -> str:
    from cron.blueprint_catalog import CATALOG

    lines = ["Automation Blueprints — `/blueprint <name>` and I'll ask you what I need:\n"]
    for r in CATALOG:
        lines.append(f"  • {r.key} — {r.title}")
        lines.append(f"    {r.description}")
    lines.append(
        "\nTip: `/blueprint <name>` walks you through it. Power users can "
        "pass values inline, e.g. `/blueprint morning-brief time=08:00`."
    )
    return "\n".join(lines)


def _fmt_candidates(query: str, candidates: List[Any]) -> str:
    lines = [f"'{query}' matches several blueprints — which one?\n"]
    for r in candidates:
        lines.append(f"  • {r.key} — {r.title}")
    lines.append("\nRun `/blueprint <name>` with one of the names above.")
    return "\n".join(lines)


def _fmt_no_match(query: str) -> str:
    from cron.blueprint_catalog import CATALOG

    keys = [r.key for r in CATALOG]
    close = difflib.get_close_matches((query or "").lower(), keys, n=3, cutoff=0.4)
    msg = f"No automation blueprint matches '{query}'."
    if close:
        msg += " Did you mean: " + ", ".join(close) + "?"
    msg += " Run /blueprint to see the catalog."
    return msg


def _manage_hint(surface: str) -> str:
    """Post-create management hint. /cron is a CLI-only slash command; on
    gateway platforms the user manages jobs by asking the agent (cronjob tool)
    or from the dashboard."""
    if surface == "cli":
        return "Manage it with /cron."
    return "Ask me to list, pause, or remove it any time."


def handle_blueprint_command(
    args: str,
    *,
    origin: Optional[Dict[str, Any]] = None,
    surface: str = "cli",
) -> BlueprintCommandResult:
    """Dispatch a ``/blueprint`` invocation.

    Returns a :class:`BlueprintCommandResult`. When ``agent_seed`` is set the
    caller must feed it to the agent as the next user turn; otherwise the
    command is fully handled and only ``text`` is shown.

    ``args`` is everything after ``/blueprint``. ``origin`` lets a directly
    created job deliver back to the chat it was set up from. ``surface``
    (``"cli"`` | ``"gateway"``) picks the right wording for follow-up hints —
    ``/cron`` only exists on the CLI.
    """
    try:
        from cron.blueprint_catalog import fill_blueprint, BlueprintFillError
    except Exception as e:  # pragma: no cover - import guard
        logger.debug("blueprint catalog import failed: %s", e)
        return BlueprintCommandResult("Automation Blueprints are unavailable in this build.")

    try:
        tokens = shlex.split(args or "")
    except ValueError:
        tokens = (args or "").split()

    # Bare -> list catalog.
    if not tokens:
        return BlueprintCommandResult(_fmt_catalog())

    query = tokens[0]
    values, _leftover = _parse_kv(tokens[1:])

    blueprint, candidates = match_blueprint(query)
    if blueprint is None:
        if candidates:
            return BlueprintCommandResult(_fmt_candidates(query, candidates))
        return BlueprintCommandResult(_fmt_no_match(query))

    # `<name>` with no inline slot values -> seed the agent to ask for them.
    if not values:
        seed = build_blueprint_seed(blueprint)
        text = (
            f"Setting up '{blueprint.title}' ({_humanize_schedule(blueprint)}). "
            "I'll ask you a couple of things…"
        )
        return BlueprintCommandResult(text, agent_seed=seed)

    # `<name> slot=val …` -> fill + create directly (deterministic shortcut).
    try:
        spec = fill_blueprint(blueprint, values, origin=_resolve_origin(origin))
    except BlueprintFillError as e:
        return BlueprintCommandResult(
            f"Can't set up '{blueprint.title}': {e}\n"
            f"Or just run /blueprint {blueprint.key} and I'll ask you for the values."
        )

    try:
        from cron.jobs import create_job

        job = create_job(**spec)
    except Exception as e:
        logger.debug("blueprint create_job failed: %s", e)
        return BlueprintCommandResult(f"Failed to create the job: {e}")

    sched = job.get("schedule_display") or spec.get("schedule", "")
    return BlueprintCommandResult(
        f"Scheduled '{blueprint.title}'"
        + (f" ({sched})" if sched else "")
        + f", delivering to {spec.get('deliver', 'origin')}. {_manage_hint(surface)}"
    )
