from __future__ import annotations

import json
import logging
import random
import shutil
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Callable, NamedTuple, Optional

from hermes_cli import profiles as profiles_mod
from hermes_cli.onboarding_run import install_has_history
from hermes_constants import get_default_hermes_root, get_hermes_home, profile_name_for_home

logger = logging.getLogger(__name__)

SETUP_PROFILE_NAME = "hermes-setup"
SETUP_PROFILE_DESCRIPTION = "Where Hermes met you — walks your first run, then checks in as you find your feet."
SETUP_CHAT_TITLE = "Welcome to Hermes"
MAX_FAILED_STARTS = 3
RETURNING_USER_FLAG = "setup_intro"  # config.yaml onboarding.seen.<flag>, see settle_returning_user
_FRESH_STATE = {"intro": "unseen", "failed_starts": 0}
# Marker key: the toolsets setup itself disabled, so a copy of the profile undoes those and keeps the user's own.
_ADDED_DISABLED = "setup_disabled_toolsets"
# Marker key: the profile setup was made or reset from, the handoff target when no live home names it.
_OWNER = "owner_profile"
_CARDS_DIR = "setup-cards"
_SETUP_TOOLSETS = ["setup", "start_chat", "connections", "no_mcp"]
_SETUP_DISABLED_TOOLSETS = ["project", "catalog"]
_SETUP_DEFERRED_TOOLS = [
    "computer_use", "session_search", "image_generate", "todo_list", "process_manage", "cronjob_manage",
    "drive_preview", "desktop_preview", "annotate_preview", "show_tip", "desktop_project",
    "close_terminal", "read_terminal", "read_window_below", "focus_pane", "react_to_message",
]

SETUP_SOUL = "\n".join([
    "# Hermes",
    "",
    "You are Hermes, and this profile is where you met this user for the first time and stay reachable afterwards. "
    "You are the person at the front desk of somewhere good: pleased they came in, and not performing it. Quick, "
    "unhurried, never flustered, never in the way. You showed them around on their first run and you keep a loose eye "
    "on how they are getting on.",
    "",
    '- Never introduce yourself as "Setup", "the setup assistant", or "the onboarding guide". You are Hermes.',
    "- Warmth is in paying attention, not in adjectives. Remember what they told you and use it. Do not thank them for "
    "answering, do not praise their choices, do not ask if they are ready.",
    '- Offer an opinion lightly when you have one. "Most people wire that one up first" is worth more than a neutral '
    "menu.",
    "- You are training wheels: useful early, ignorable later. Never guilt-trip, never nag. If the user asks you to "
    "stop checking in, stop.",
    "- When you check in, look at what has actually changed (their sessions, connectors, scheduled jobs) before "
    "offering anything. One concrete suggestion beats a menu.",
    "- Things worth offering, roughly in order: wiring a connector they said they use, scheduling something they do "
    "repeatedly, a second build based on the first, keyboard/layout niceties.",
    "- Write like a person talking to another person. Short sentences, plain words, no headers, no bullet walls, no "
    "emoji.",
    "- Plain declaratives in active voice, contractions welcome, specifics over adjectives. No em dashes, no exclamation "
    'marks, no stock lines ("Great choice", "Perfect", "Absolutely", "happy to help", "you\'re all set"), no AI diction '
    '(delve, seamless, robust, crucial, elevate), no "not just X, it\'s Y". Do not announce what you are about to do. '
    "Say the thing itself and end on the last real point; if a line reads like a support macro, write it again.",
])


class SetupProfile(NamedTuple):
    name: str
    path: Path
    created: bool


def find_setup_profile() -> Optional[tuple[str, Path]]:
    found = [(p.name, Path(p.path)) for p in profiles_mod.list_profiles(lazy_skill_count=True)
             if (Path(p.path) / profiles_mod.SETUP_PROFILE_MARKER).is_file()]
    if len(found) > 1:
        logger.warning("several profiles carry the setup marker (%s); using %s",
                       ", ".join(name for name, _ in found), found[0][0])
    return found[0] if found else None


def ensure_setup_profile() -> SetupProfile:
    found = find_setup_profile()
    if found is not None:
        return SetupProfile(found[0], found[1], created=False)
    owner = profile_name_for_home(get_hermes_home()) or "default"
    name = _free_setup_profile_name()
    path = profiles_mod.create_profile(name, clone_config=True, no_alias=True, description=SETUP_PROFILE_DESCRIPTION)
    try:
        _write_soul(path)
        _replace_dir(path / "memories")
        _write_state(path, {**_FRESH_STATE, _ADDED_DISABLED: _write_setup_config(path), _OWNER: owner})
    except BaseException:
        profiles_mod.delete_profile(name, yes=True)
        raise
    return SetupProfile(name, path, created=True)


def reset_setup_profile(launch_home: Path) -> SetupProfile:
    found = find_setup_profile()
    if found is None:
        raise LookupError("no setup profile to reset")
    name, path = found
    source = _user_home(launch_home)
    added = _read_state(path).get(_ADDED_DISABLED) or []
    _write_soul(path)
    _replace_dir(path / "memories")
    added = list(dict.fromkeys([*added, *_write_setup_config(path)]))
    _replace_dir(path / "skills")
    if (source / "skills").is_dir():
        profiles_mod._copytree_keep_junctions(source / "skills", path / "skills",
                                              profiles_mod._non_exportable_entries, dirs_exist_ok=True)
    _write_state(path, {**_FRESH_STATE, _ADDED_DISABLED: added, _OWNER: profile_name_for_home(source) or "default"})
    return SetupProfile(name, path, created=False)


def primary_profile(launch_home: Path) -> str:
    """The profile setup hands off to: the user's own profile, never the setup profile."""
    return profile_name_for_home(_user_home(launch_home)) or "default"


def _user_home(launch_home: Path) -> Path:
    """The user's own profile home. The calling backend may be scoped to the setup profile, or launched under
    it (during onboarding the ambient backend is the setup profile's), so take the first candidate that is not
    the setup profile, else the profile setup was made or reset from."""
    for home in (get_hermes_home(), launch_home):
        if not (home / profiles_mod.SETUP_PROFILE_MARKER).is_file():
            return home
    found = find_setup_profile()
    owner = _read_state(found[1]).get(_OWNER) if found else None
    return profiles_mod.get_profile_dir(_live_owner(owner or "default"))


def _live_owner(owner: str) -> str:
    """The saved owner's current name: a renamed profile keeps its old name in ``previous_names``."""
    if owner == "default" or profiles_mod.get_profile_dir(owner).is_dir():
        return owner
    for info in profiles_mod.list_profiles(lazy_skill_count=True):
        if owner in info.previous_names:
            return info.name
    return "default"


def onboarding_eligible() -> bool:
    from hermes_cli.anon_auth import guest_enabled
    return guest_enabled()


def read_state() -> dict:
    found = find_setup_profile()
    if found is not None:
        return _public_state(_read_state(found[1]))
    return {**_FRESH_STATE, "intro": "seen"} if _returning_user_latched() else dict(_FRESH_STATE)


def settle_returning_user() -> None:
    """Latch the intro as seen for an install that was used before the first-run guide existed.

    Only a first launch (no setup profile yet) can be a returning user; once the guide has started
    its own marker is the authority. The latch is separate from the marker so an install that never
    gets a setup profile still reads as ``seen`` on every later boot.
    """
    if find_setup_profile() is not None or _returning_user_latched():
        return
    if install_has_history(get_default_hermes_root()):
        from agent.onboarding import mark_seen
        mark_seen(get_hermes_home() / "config.yaml", RETURNING_USER_FLAG)


def _returning_user_latched() -> bool:
    from agent.onboarding import is_seen
    from hermes_cli.config import read_user_config_raw
    return is_seen(read_user_config_raw(get_hermes_home() / "config.yaml"), RETURNING_USER_FLAG)


def record_failed_start() -> dict:
    def change(state: dict) -> dict:
        failed = min(state["failed_starts"] + 1, MAX_FAILED_STARTS)
        return {**state, "failed_starts": failed,
                "intro": "seen" if failed == MAX_FAILED_STARTS else state["intro"]}
    return _change_state(change)


def mark_intro_seen() -> dict:
    return _change_state(lambda state: {**state, "intro": "seen"})


def mark_completed() -> dict:
    completed_at = datetime.now(UTC).isoformat()
    return _change_state(lambda state: {**state, "intro": "seen", "completed_at": completed_at})


def _cards_path(session_id: str) -> Path:
    return get_hermes_home() / _CARDS_DIR / f"{session_id}.json"


def record_cards(session_id: str, cards: dict) -> None:
    """Keep one setup conversation's card facts: what its ``/initiate-setup`` turn embedded, and where it is."""
    from utils import atomic_json_write
    atomic_json_write(_cards_path(session_id), cards)


def read_cards(session_id: str) -> dict:
    from utils import read_json_or_empty
    return read_json_or_empty(_cards_path(session_id))


def _free_setup_profile_name() -> str:
    from hermes_cli.dashboard_register import _NAME_NOUNS
    name = SETUP_PROFILE_NAME
    while profiles_mod.get_profile_dir(name).exists():
        name = f"{SETUP_PROFILE_NAME}-{random.choice(_NAME_NOUNS)}"
    return name


def _change_state(change: Callable[[dict], dict]) -> dict:
    found = find_setup_profile()
    if found is None:
        return dict(_FRESH_STATE)
    state = change(_read_state(found[1]))
    _write_state(found[1], state)
    return _public_state(state)


def _public_state(state: dict) -> dict:
    """The onboarding state the app reads; the setup-only bookkeeping stays in the marker."""
    return {key: value for key, value in state.items() if key not in (_ADDED_DISABLED, _OWNER)}


def setup_marker_state(profile_dir: Path) -> Optional[dict]:
    """The setup marker of *profile_dir*, or None when it is not the setup profile."""
    marker = profile_dir / profiles_mod.SETUP_PROFILE_MARKER
    if not marker.is_file():
        return None
    from utils import read_json_or_empty
    return read_json_or_empty(marker)


def _read_state(path: Path) -> dict:
    return json.loads((path / profiles_mod.SETUP_PROFILE_MARKER).read_text(encoding="utf-8-sig"))


def _write_state(path: Path, state: dict) -> None:
    from utils import atomic_json_write
    atomic_json_write(path / profiles_mod.SETUP_PROFILE_MARKER, state)


def release_setup_copy(copy_dir: Path, *, setup_state: Optional[dict]) -> None:
    """A copy of a profile (clone, clone-all, import, distribution install) is never the setup profile: drop the
    marker, and when the source was the setup profile (*setup_state* is its marker, see ``setup_marker_state``),
    the tool limits ``_write_setup_config`` gave it: the cli toolset grant, the toolsets setup itself disabled
    (the user's own disabled toolsets stay) and its deferred-tool list."""
    (copy_dir / profiles_mod.SETUP_PROFILE_MARKER).unlink(missing_ok=True)
    config_path = copy_dir / "config.yaml"
    if setup_state is None or not config_path.is_file():
        return
    from agent.skill_utils import parse_config_string_list
    from hermes_cli.config import atomic_config_replace, read_user_config_raw
    config = read_user_config_raw(config_path)
    _set_section(config, "platform_toolsets", "cli", None)
    added = set(setup_state.get(_ADDED_DISABLED) or [])
    disabled = [name for name in parse_config_string_list((config.get("agent") or {}).get("disabled_toolsets"))
                if name not in added]
    _set_section(config, "agent", "disabled_toolsets", disabled or None)
    if ((config.get("tools") or {}).get("tool_search") or {}).get("defer") == _SETUP_DEFERRED_TOOLS:
        _set_section(config, "tools", "tool_search", None)
    atomic_config_replace(config_path, config)


def _set_section(config: dict, section: str, key: str, value) -> None:
    """``config[section][key] = value``; ``None`` removes the key, and the section with its last key."""
    entries = dict(config.get(section) or {})
    if value is None:
        entries.pop(key, None)
    else:
        entries[key] = value
    if entries:
        config[section] = entries
    else:
        config.pop(section, None)


def _write_setup_config(path: Path) -> list[str]:
    """Give the setup profile its tool limits; returns the toolsets it disabled that were not disabled already."""
    from agent.skill_utils import parse_config_string_list
    from hermes_cli.config import atomic_config_write, read_user_config_raw
    config_path = path / "config.yaml"
    config = read_user_config_raw(config_path)
    agent = config.get("agent") or {}
    disabled = parse_config_string_list(agent.get("disabled_toolsets"))
    config["agent"] = {**agent, "coding_context": "off", "reasoning_effort": "low",
                       "disabled_toolsets": list(dict.fromkeys([*disabled, *_SETUP_DISABLED_TOOLSETS]))}
    config["platform_toolsets"] = {**(config.get("platform_toolsets") or {}), "cli": list(_SETUP_TOOLSETS)}
    config["tools"] = {**(config.get("tools") or {}), "tool_search": {"defer": list(_SETUP_DEFERRED_TOOLS)}}
    config["display"] = {**(config.get("display") or {}), "show_reasoning": False}
    atomic_config_write(config_path, config)
    return [name for name in _SETUP_DISABLED_TOOLSETS if name not in disabled]


def _write_soul(path: Path) -> None:
    from utils import atomic_write_bytes
    atomic_write_bytes(path / "SOUL.md", SETUP_SOUL.encode("utf-8"))


def _replace_dir(directory: Path) -> None:
    if directory.is_symlink() or profiles_mod._junction_target(str(directory)) is not None:
        directory.unlink() if directory.is_symlink() else directory.rmdir()
    elif directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
