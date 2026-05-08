"""
Contextual first-touch onboarding hints.

Instead of blocking first-run questionnaires, show a one-time hint the *first*
time a user hits a behavior fork — message-while-running, first long-running
tool, etc.  Each hint is shown once per install (tracked in ``config.yaml`` under
``onboarding.seen.<flag>``) and then never again.

Keep this module tiny and dependency-free so both the CLI and gateway can import
it without pulling in heavy modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Flag names (stable — used as config.yaml keys under onboarding.seen)
# -------------------------------------------------------------------------

BUSY_INPUT_FLAG = "busy_input_prompt"
TOOL_PROGRESS_FLAG = "tool_progress_prompt"
OPENCLAW_RESIDUE_FLAG = "openclaw_residue_cleanup"
AGENT_GREETING_FLAG = "agent_greeting"


# -------------------------------------------------------------------------
# Hint content
# -------------------------------------------------------------------------

def busy_input_hint_gateway(mode: str) -> str:
    """Hint shown the first time a user messages while the agent is busy.

    ``mode`` is the effective busy_input_mode that was just applied, so the
    message matches reality ("I just interrupted…" vs "I just queued…").
    """
    if mode == "queue":
        return (
            "💡 First-time tip — I queued your message instead of interrupting. "
            "Send `/busy interrupt` to make new messages stop the current task "
            "immediately, or `/busy status` to check. This notice won't appear again."
        )
    if mode == "steer":
        return (
            "💡 First-time tip — I steered your message into the current run; "
            "it will arrive after the next tool call instead of interrupting. "
            "Send `/busy interrupt` or `/busy queue` to change this, or "
            "`/busy status` to check. This notice won't appear again."
        )
    return (
        "💡 First-time tip — I just interrupted my current task to answer you. "
        "Send `/busy queue` to queue follow-ups for after the current task instead, "
        "`/busy steer` to inject them mid-run without interrupting, or "
        "`/busy status` to check. This notice won't appear again."
    )


def busy_input_hint_cli(mode: str) -> str:
    """CLI version of the busy-input hint (plain text, no markdown)."""
    if mode == "queue":
        return (
            "(tip) Your message was queued for the next turn. "
            "Use /busy interrupt to make Enter stop the current run instead, "
            "or /busy steer to inject mid-run. This tip only shows once."
        )
    if mode == "steer":
        return (
            "(tip) Your message was steered into the current run; it arrives "
            "after the next tool call. Use /busy interrupt or /busy queue to "
            "change this. This tip only shows once."
        )
    return (
        "(tip) Your message interrupted the current run. "
        "Use /busy queue to queue messages for the next turn instead, "
        "or /busy steer to inject mid-run. This tip only shows once."
    )


def tool_progress_hint_gateway() -> str:
    return (
        "💡 First-time tip — that tool took a while and I'm streaming every step. "
        "If the progress messages feel noisy, send `/verbose` to cycle modes "
        "(all → new → off). This notice won't appear again."
    )


def tool_progress_hint_cli() -> str:
    return (
        "(tip) That tool ran for a while. Use /verbose to cycle tool-progress "
        "display modes (all -> new -> off -> verbose). This tip only shows once."
    )


def openclaw_residue_hint_cli() -> str:
    """Banner shown the first time Hermes starts and finds ``~/.openclaw/``.

    Points users at ``hermes claw migrate`` (non-destructive port of config,
    memory, and skills) first. ``hermes claw cleanup`` is mentioned as the
    follow-up step for users who have already migrated and want to archive
    the old directory — with a warning that archiving breaks OpenClaw.
    """
    return (
        "A legacy OpenClaw directory was detected at ~/.openclaw/.\n"
        "To port your config, memory, and skills over to Hermes, run "
        "`hermes claw migrate`.\n"
        "If you've already migrated and want to archive the old directory, "
        "run `hermes claw cleanup` (renames it to ~/.openclaw.pre-migration — "
        "OpenClaw will stop working after this).\n"
        "This tip only shows once."
    )


def detect_openclaw_residue(home: Optional[Path] = None) -> bool:
    """Return True if an OpenClaw workspace directory is present in ``$HOME``.

    Pure filesystem check — no side effects. ``home`` override exists for tests.
    """
    base = home or Path.home()
    try:
        return (base / ".openclaw").is_dir()
    except OSError:
        return False


def agent_greeting_prompt() -> str:
    """Auto-prompt greeting shown on the very first run of this install.

    The agent introduces itself by name and invites the user to start.
    This is injected as the initial user message on first run only.
    """
    try:
        from agent.prompt_builder import get_lycus_agent_name
        name = get_lycus_agent_name()
    except Exception:
        name = "this agent"
    return (
        f"I just installed. My name is {name}. "
        f"What would you like to work on?"
    )


def is_lycus_command() -> bool:
    """Check if the current command was invoked as 'lycus'."""
    import sys
    try:
        argv0 = sys.argv[0]
        return 'lycus' in argv0.lower()
    except Exception:
        return False


def get_dynamic_greeting() -> str:
    """Generate a greeting prompt for Lycus with time and weather context.

    This returns a prompt that is sent to the LLM to generate a unique greeting
    for each session. The LLM receives the current time, weather, and personality
    template and crafts a personalized greeting.
    """
    try:
        from agent.prompt_builder import get_lycus_agent_name
        name = get_lycus_agent_name()
    except Exception:
        name = "Lycus"

    # Get current time
    try:
        from datetime import datetime
        now = datetime.now()
        hour = now.hour

        # Time of day
        if 5 <= hour < 12:
            time_of_day = "early in the morning"
        elif 12 <= hour < 17:
            time_of_day = "the middle of the day"
        elif 17 <= hour < 21:
            time_of_day = "late in the evening"
        else:
            time_of_day = "late at night"

        time_str = now.strftime("%I:%M %p")
        date_str = now.strftime("%A, %B %d, %Y")
    except Exception:
        time_of_day = "some time"
        time_str = "now"
        date_str = "today"

    # Get location from IP
    try:
        import urllib.request
        import json

        with urllib.request.urlopen('https://ipapi.co/json/', timeout=5) as resp:
            geo_data = json.loads(resp.read())
            city = geo_data.get('city', 'your area')
            region = geo_data.get('region', '')
            location = f"{city}, {region}" if region else city
    except Exception:
        location = "your area"
        city = ""
        region = ""

    # Get weather
    try:
        import urllib.request
        import json

        with urllib.request.urlopen(f'https://wttr.in/{city}?format=j1', timeout=5) as resp:
            weather_data = json.loads(resp.read())
            current = weather_data.get('current_condition', [{}])[0]
            temp_c = current.get('temp_C', '--')
            temp_f = current.get('temp_F', '--')
            weather_desc = current.get('weatherDesc', [{}])[0].get('value', 'clear skies')
            feels_like = current.get('FeelsLikeC', '--')

            weather_str = f"{weather_desc.lower()} with a temperature of {temp_f}°F ({temp_c}°C)"
    except Exception:
        weather_str = "pleasant weather"

    # Load personality template
    try:
        import yaml
        from pathlib import Path
        personality_file = Path.home() / '.hermes' / 'lycus_personality.yaml'
        if personality_file.exists():
            with open(personality_file) as f:
                personality = yaml.safe_load(f)
            active = personality.get('active_template', 'default')
            template = personality.get('templates', {}).get(active, personality.get('templates', {}).get('default', ''))
        else:
            template = ""
    except Exception:
        template = ""

    # Build the prompt that will be sent to the LLM
    prompt = f"""I just started a new session. Please greet me warmly and uniquely.

Context:
- My name is {name}
- It's currently {time_str} ({time_of_day}) on {date_str}
- The weather outside {location} is: {weather_str}
"""
    if template:
        prompt += f"""- My personality/style guide:
{template}
"""
    prompt += """Based on this context, give me a warm, unique greeting. Be creative and natural — don't sound robotic. Keep it to 1-3 sentences. After greeting me, briefly ask what I'd like to work on."""

    return prompt.strip()


# -------------------------------------------------------------------------
# State read / write
# -------------------------------------------------------------------------

def _get_seen_dict(config: Mapping[str, Any]) -> Mapping[str, Any]:
    onboarding = config.get("onboarding") if isinstance(config, Mapping) else None
    if not isinstance(onboarding, Mapping):
        return {}
    seen = onboarding.get("seen")
    return seen if isinstance(seen, Mapping) else {}


def is_seen(config: Mapping[str, Any], flag: str) -> bool:
    """Return True if the user has already been shown this first-touch hint."""
    return bool(_get_seen_dict(config).get(flag))


def mark_seen(config_path: Path, flag: str) -> bool:
    """Persist ``onboarding.seen.<flag> = True`` to ``config_path``.

    Uses the atomic YAML writer so a concurrent process can't observe a
    partially-written file.  Returns True on success, False on any error
    (including the config file being absent — onboarding is best-effort).
    """
    try:
        import yaml
        from utils import atomic_yaml_write
    except Exception as e:  # pragma: no cover — dependency issue
        logger.debug("onboarding: failed to import yaml/utils: %s", e)
        return False

    try:
        cfg: dict = {}
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg.get("onboarding"), dict):
            cfg["onboarding"] = {}
        seen = cfg["onboarding"].get("seen")
        if not isinstance(seen, dict):
            seen = {}
            cfg["onboarding"]["seen"] = seen
        if seen.get(flag) is True:
            return True  # already marked — nothing to do
        seen[flag] = True
        atomic_yaml_write(config_path, cfg)
        return True
    except Exception as e:
        logger.debug("onboarding: failed to mark flag %s: %s", flag, e)
        return False


__all__ = [
    "BUSY_INPUT_FLAG",
    "TOOL_PROGRESS_FLAG",
    "OPENCLAW_RESIDUE_FLAG",
    "AGENT_GREETING_FLAG",
    "busy_input_hint_gateway",
    "busy_input_hint_cli",
    "tool_progress_hint_gateway",
    "tool_progress_hint_cli",
    "openclaw_residue_hint_cli",
    "detect_openclaw_residue",
    "agent_greeting_prompt",
    "is_lycus_command",
    "is_seen",
    "mark_seen",
]
