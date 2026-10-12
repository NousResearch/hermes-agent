"""Contextual first-touch onboarding hints.

Each hint is shown once per install the *first* time a user hits a behavior
fork, tracked in ``config.yaml`` under ``onboarding.seen.<flag>``. Kept tiny and
dependency-free so both the CLI and gateway can import it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


# Flag names (stable — used as config.yaml keys under onboarding.seen)
BUSY_INPUT_FLAG = "busy_input_prompt"
TOOL_PROGRESS_FLAG = "tool_progress_prompt"
OPENCLAW_RESIDUE_FLAG = "openclaw_residue_cleanup"
PROFILE_BUILD_FLAG = "profile_build_offered"


# Busy-input hints are keyed by the effective busy_input_mode that was just
# applied so the message matches reality; "interrupt" is the default branch.
_BUSY_INPUT_HINTS_GATEWAY = {
    "queue": (
        "💡 First-time tip — I queued your message instead of interrupting. Send `/busy interrupt` to make new messages "
        "stop the current task immediately, or `/busy status` to check. This notice won't appear again."
    ),
    "steer": (
        "💡 First-time tip — I steered your message into the current run; it will arrive after the next tool "
        "call instead of interrupting. Send `/busy interrupt` or `/busy queue` to change this, or `/busy "
        "status` to check. This notice won't appear again."
    ),
    "redirect": (
        "💡 First-time tip — I redirected the current run using your message. Completed work stays in "
        "context, and `/stop` still cancels the task. Send `/busy queue` to wait for a separate turn, or "
        "`/busy status` to check. This notice won't appear again."
    ),
}
_BUSY_INPUT_HINT_GATEWAY_DEFAULT = (
    "💡 First-time tip — I just interrupted my current task to answer you. Send `/busy queue` to queue "
    "follow-ups for after the current task instead, `/busy steer` to inject them mid-run without "
    "interrupting, or `/busy status` to check. This notice won't appear again."
)

_BUSY_INPUT_HINTS_CLI = {
    "queue": (
        "(tip) Your message was queued for the next turn. Use /busy interrupt to make Enter stop the current "
        "run instead, or /busy steer to inject mid-run. This tip only shows once."
    ),
    "steer": (
        "(tip) Your message was steered into the current run; it arrives after the next tool call. Use /busy "
        "interrupt or /busy queue to change this. This tip only shows once."
    ),
    "redirect": (
        "(tip) Your correction redirected the current run without discarding completed work. Use /stop to "
        "cancel or /busy queue to wait for a separate turn. This tip only shows once."
    ),
}
_BUSY_INPUT_HINT_CLI_DEFAULT = (
    "(tip) Your message interrupted the current run. Use /busy queue to queue messages for the next turn "
    "instead, or /busy steer to inject mid-run. This tip only shows once."
)


def busy_input_hint_gateway(mode: str) -> str:
    """Hint shown the first time a user messages while the agent is busy (markdown)."""
    return _BUSY_INPUT_HINTS_GATEWAY.get(mode, _BUSY_INPUT_HINT_GATEWAY_DEFAULT)


def busy_input_hint_cli(mode: str) -> str:
    """CLI version of the busy-input hint (plain text, no markdown)."""
    return _BUSY_INPUT_HINTS_CLI.get(mode, _BUSY_INPUT_HINT_CLI_DEFAULT)


def tool_progress_hint_gateway() -> str:
    return ("💡 First-time tip — that tool took a while and I'm streaming every step. If the progress messages "
            "feel noisy, send `/verbose` to cycle modes (all → new → off). This notice won't appear again.")


def tool_progress_hint_cli() -> str:
    return ("(tip) That tool ran for a while. Use /verbose to cycle tool-progress "
            "display modes (all -> new -> off -> verbose). This tip only shows once.")


def openclaw_residue_hint_cli() -> str:
    """Banner shown the first time Hermes finds ``~/.openclaw/``: migrate first, cleanup (which breaks OpenClaw) after."""
    return (
        "A legacy OpenClaw directory was detected at ~/.openclaw/.\n"
        "To port your config, memory, and skills over to Hermes, run `hermes claw migrate`.\n"
        "If you've already migrated and want to archive the old directory, run `hermes claw cleanup` "
        "(renames it to ~/.openclaw.pre-migration — OpenClaw will stop working after this).\n"
        "This tip only shows once."
    )


def detect_openclaw_residue(home: Optional[Path] = None) -> bool:
    """True if ``$HOME/.openclaw`` is a directory (``home`` override for tests)."""
    try:
        return ((home or Path.home()) / ".openclaw").is_dir()
    except OSError:
        return False


def _onboarding_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    onboarding = config.get("onboarding") if isinstance(config, Mapping) else None
    return onboarding if isinstance(onboarding, Mapping) else {}


def profile_build_mode(config: Mapping[str, Any]) -> str:
    """``config.onboarding.profile_build``: ``"off"`` never offers; anything else -> ``"ask"``.

    Only governs whether the offer is made.
    """
    mode = _onboarding_section(config).get("profile_build")
    return "off" if isinstance(mode, str) and mode.strip().lower() == "off" else "ask"


# Shared by both first-contact notes so a real first-message task is never
# replaced by the intro.
TASK_FIRST_CLAUSE = (
    "If this message is itself a real request or task, DO THE TASK FIRST -- call "
    "whatever tools it needs -- and only then, in the closing sentences of that same "
    "reply, do what this note asks. Never let this note replace or skip work the user "
    "actually asked for. "
)

PLAIN_INTRO_NOTE = (
    "[System note: This is the user's very first message ever. "
    + TASK_FIRST_CLAUSE
    + "What this note asks: briefly introduce yourself and mention that /help shows "
    "available commands, in one or two sentences.]"
)

SETUP_OFFER_NOTE = (
    "[System note: This is the user's very first message ever. "
    + TASK_FIRST_CLAUSE
    + "What this note asks: briefly introduce yourself, mention that /help shows available "
    "commands, and end with this one line: \"I can run a quick setup so I can help you better. "
    "Send {command} when you want it.\"]"
)


# Telegram commands take no hyphens, and Slack reaches /initiate-setup through /hermes (50-slash cap).
_SETUP_COMMAND_BY_PLATFORM = {"telegram": "/initiate_setup", "slack": "/hermes initiate-setup"}


def setup_command(platform: str) -> str:
    """How a platform's user types /initiate-setup."""
    return _SETUP_COMMAND_BY_PLATFORM.get(platform, "/initiate-setup")

def first_contact_turn_note(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    session_history_empty: bool,
    install_has_prior_sessions: bool,
    message: str,
    command: str = "/initiate-setup",
    setup_handoff: bool = False,
) -> Optional[str]:
    """Return a one-shot sidecar note for the install's first-ever message.

    Matches the gateway first-contact path: when ``profile_build`` is ``ask``
    and the offer has not been latched yet, return the opt-in
    directive and persist ``onboarding.seen.profile_build_offered``. Otherwise
    return the plain intro note. Returns ``None`` when this is not the first
    contact (non-empty session history or prior sessions exist on the install). ``setup_handoff``
    marks the task chat setup started (``start_chat``), told by the caller, never read from the text.
    """
    from agent.initiate_setup_prompt import HEADER
    from hermes_cli.profiles import SETUP_PROFILE_MARKER

    if not session_history_empty or install_has_prior_sessions or message.startswith(HEADER):
        return None
    # The task chat setup hands off to is the user's first chat in their own profile; setup already ran.
    if setup_handoff:
        return None
    # The setup chat opens with its own welcome, whatever its first message says.
    if (config_path.parent / SETUP_PROFILE_MARKER).is_file():
        return None
    try:
        if (
            profile_build_mode(config) == "ask"
            and not is_seen(config, PROFILE_BUILD_FLAG)
        ):
            mark_seen(config_path, PROFILE_BUILD_FLAG)
            from hermes_cli.anon_auth import guest_enabled

            # /initiate-setup is the offer only where the setup flow ships; elsewhere keep the memory profile offer.
            return SETUP_OFFER_NOTE.format(command=command) if guest_enabled() else profile_build_directive().strip()
        return PLAIN_INTRO_NOTE
    except Exception as e:
        logger.debug("first_contact_turn_note failed, using plain intro: %s", e)
        return PLAIN_INTRO_NOTE


def profile_build_directive() -> str:
    """System-note directive appended to the very first message ever.

    Short opt-in profile-build flow persisting to the user-profile memory store;
    phrased so the agent ASKS before any lookup and never silently reads
    connected accounts.
    """
    return (
        "\n\n"
        "[System note: This is the user's very first message ever. " + TASK_FIRST_CLAUSE
        + "What this note asks: after a one-sentence introduction (mention /help "
        "shows commands), OFFER — do not assume — to build a short profile of them so you can be more useful, and "
        "explain they can decline or do it later. If and ONLY IF they accept:\n"
        "  1. Ask for whatever they're comfortable sharing (name, what they do, how they like you to work). "
        "Volunteered facts come first.\n"
        "  2. Before ANY external lookup, say what you intend to look up and get explicit consent for that step. Never "
        "read their connected accounts (email, calendar, etc.) silently — ask each time.\n"
        "  3. With consent, you may use web_search to confirm public details (e.g. employer, public profiles) from the "
        "data points they gave.\n"
        "  4. Save each confirmed, durable fact with the memory tool using target=\"user\" — keep entries compact and "
        "high-signal.\n"
        "If they decline at any point, stop immediately and continue normally. Keep the whole exchange light and "
        "conversational, not an interrogation.]"
    )


def is_seen(config: Mapping[str, Any], flag: str) -> bool:
    """True if the user has already been shown this first-touch hint."""
    seen = _onboarding_section(config).get("seen")
    return bool(seen.get(flag)) if isinstance(seen, Mapping) else False


def mark_seen(config_path: Path, flag: str) -> bool:
    """Persist ``onboarding.seen.<flag> = True`` atomically; False on any error (best-effort)."""
    try:
        from hermes_cli.config import atomic_config_write, read_user_config_raw
    except Exception as e:  # pragma: no cover — dependency issue
        logger.debug("onboarding: failed to import config helpers: %s", e)
        return False
    try:
        cfg: dict = read_user_config_raw(config_path)
        if not isinstance(cfg.get("onboarding"), dict):
            cfg["onboarding"] = {}
        seen = cfg["onboarding"].get("seen")
        if not isinstance(seen, dict):
            seen = cfg["onboarding"]["seen"] = {}
        if seen.get(flag) is not True:
            seen[flag] = True
            atomic_config_write(config_path, cfg)
        return True
    except Exception as e:
        logger.debug("onboarding: failed to mark flag %s: %s", flag, e)
        return False


__all__ = [
    "BUSY_INPUT_FLAG",
    "OPENCLAW_RESIDUE_FLAG",
    "PLAIN_INTRO_NOTE",
    "PROFILE_BUILD_FLAG",
    "SETUP_OFFER_NOTE",
    "TOOL_PROGRESS_FLAG",
    "busy_input_hint_cli",
    "busy_input_hint_gateway",
    "detect_openclaw_residue",
    "first_contact_turn_note",
    "is_seen",
    "mark_seen",
    "openclaw_residue_hint_cli",
    "profile_build_directive",
    "profile_build_mode",
    "setup_command",
    "tool_progress_hint_cli",
    "tool_progress_hint_gateway",
]
