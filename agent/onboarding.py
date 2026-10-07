"""Contextual first-touch onboarding hints.

Each hint is shown once per install the *first* time a user hits a behavior
fork, tracked in ``config.yaml`` under ``onboarding.seen.<flag>``. Kept tiny and
dependency-free so both the CLI and gateway can import it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

from agent.i18n import t

logger = logging.getLogger(__name__)


# Flag names (stable — used as config.yaml keys under onboarding.seen)
BUSY_INPUT_FLAG = "busy_input_prompt"
TOOL_PROGRESS_FLAG = "tool_progress_prompt"
OPENCLAW_RESIDUE_FLAG = "openclaw_residue_cleanup"
PROFILE_BUILD_FLAG = "profile_build_offered"


# Busy-input hints are keyed by the effective busy_input_mode that was just
# applied so the message matches reality; "interrupt" is the default branch.
_HINT_MODES = frozenset({"queue", "steer", "redirect"})


def busy_input_hint_gateway(mode: str) -> str:
    """Hint shown the first time a user messages while the agent is busy (markdown)."""
    return t(f"tips.busy_gateway.{mode if mode in _HINT_MODES else 'interrupt'}")


def busy_input_hint_cli(mode: str) -> str:
    """CLI version of the busy-input hint (plain text, no markdown)."""
    return t(f"tips.busy_cli.{mode if mode in _HINT_MODES else 'interrupt'}")


def tool_progress_hint_gateway() -> str:
    return t("tips.tool_progress_gateway")


def tool_progress_hint_cli() -> str:
    return t("tips.tool_progress_cli")


def openclaw_residue_hint_cli() -> str:
    """Banner shown the first time Hermes finds ``~/.openclaw/``: migrate first, cleanup (which breaks OpenClaw) after."""
    return t("tips.openclaw_residue_cli")


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

    Only governs whether the offer is made; lookups inside the flow are
    consented to separately in conversation.
    """
    mode = _onboarding_section(config).get("profile_build")
    return "off" if isinstance(mode, str) and mode.strip().lower() == "off" else "ask"


# Shared by both first-contact notes so a real first-message task is never
# replaced by the intro (plain or profile-build offer).
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


def first_contact_turn_note(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    session_history_empty: bool,
    install_has_prior_sessions: bool,
) -> Optional[str]:
    """Return a one-shot sidecar note for the install's first-ever message.

    Matches the gateway first-contact path: when ``profile_build`` is ``ask``
    and the offer has not been latched yet, return the opt-in profile-build
    directive and persist ``onboarding.seen.profile_build_offered``. Otherwise
    return the plain intro note. Returns ``None`` when this is not the first
    contact (non-empty session history or prior sessions exist on the install).
    """
    if not session_history_empty or install_has_prior_sessions:
        return None
    try:
        if (
            profile_build_mode(config) == "ask"
            and not is_seen(config, PROFILE_BUILD_FLAG)
        ):
            mark_seen(config_path, PROFILE_BUILD_FLAG)
            return profile_build_directive().strip()
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
    "BUSY_INPUT_FLAG", "TOOL_PROGRESS_FLAG", "OPENCLAW_RESIDUE_FLAG", "PROFILE_BUILD_FLAG",
    "PLAIN_INTRO_NOTE", "first_contact_turn_note",
    "busy_input_hint_gateway", "busy_input_hint_cli", "tool_progress_hint_gateway", "tool_progress_hint_cli",
    "openclaw_residue_hint_cli", "detect_openclaw_residue", "profile_build_mode", "profile_build_directive",
    "is_seen", "mark_seen",
]
