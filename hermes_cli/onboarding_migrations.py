"""One-shot release of the agentic desktop onboarding's setup profiles.

Canary/RC desktop builds made a ``hermes-setup[-noun]`` profile, marked by ``.setup-profile.json``,
and limited its tools. This turns each one into a plain profile: the marker and the setup tool
limits go; its sessions, credentials and memories stay. It runs at serve boot before any RPC is
served, under a file lock, once per install (latched in the root config).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from hermes_cli.onboarding_run import named_profiles, set_run
from hermes_constants import get_default_hermes_root, get_hermes_home

logger = logging.getLogger(__name__)

SETUP_MARKER = ".setup-profile.json"
_RELEASED_FLAG = "setup_profile_released"  # root config onboarding.seen.<flag>
_OLD_GUIDE_SKIPPED_FLAG = "setup_intro"  # written by the agentic build for a returning user
_ADDED_DISABLED_KEY = "setup_disabled_toolsets"  # marker key: toolsets setup itself disabled
# The exact cli toolset and deferred-tool lists setup wrote; a list the user changed since is theirs and stays.
_SETUP_CLI_TOOLSETS = ["setup", "start_chat", "connections", "no_mcp"]
_SETUP_DEFERRED_TOOLS = [
    "computer_use", "session_search", "image_generate", "todo_list", "process_manage", "cronjob_manage",
    "drive_preview", "desktop_preview", "annotate_preview", "show_tip", "desktop_project",
    "close_terminal", "read_terminal", "read_window_below", "focus_pane", "react_to_message",
]
# Name -> the hub identifier its optional-skills copy installed under; a same-named skill from elsewhere stays.
_RETIRED_SKILLS = {name: f"official/productivity/{name}" for name in ("initiate-setup", "first-task")}
_lock_holder = threading.local()


def release_setup_profiles() -> None:
    """Release every setup profile under the root, map the old guide's outcome onto ``onboarding.run``,
    and uninstall the two retired skills. A no-op once latched, and when this backend runs inside a
    setup profile (it never rewrites its own live home; a launch from another profile does it)."""
    from agent.onboarding import is_seen, mark_seen
    from hermes_cli.auth import _file_lock
    from hermes_cli.config import read_user_config_raw

    if (get_hermes_home() / SETUP_MARKER).is_file():
        return
    root = get_default_hermes_root()
    config_path = root / "config.yaml"
    with _file_lock(root / ".setup-profile-release.lock", _lock_holder, 5.0,
                    "another backend is releasing the setup profile"):
        if is_seen(read_user_config_raw(config_path), _RELEASED_FLAG):
            return
        marked = {path: _read_marker(path) for path in named_profiles(root) if (path / SETUP_MARKER).is_file()}
        done = _old_guide_done(marked.values(), config_path)
        if done or marked:
            # Settled before any marker goes: once released, the setup profile counts as install
            # history, so an unset run would keep the questionnaire shut for an unfinished guide.
            _settle_run(config_path, not done)
        # A profile whose identity did not reach the shared store keeps its marker and the latch stays
        # open, so the next boot retries instead of leaving the install without that identity.
        unshared = {path for path in marked if not _share_identity(path)}
        for path, marker in marked.items():
            if path not in unshared:
                _release(path, marker)
        for home in (root, *named_profiles(root)):
            _uninstall_retired_skills(home)
        if unshared:
            logger.warning("setup profile release deferred: could not share the identity of %s",
                           ", ".join(sorted(path.name for path in unshared)))
            return
        mark_seen(config_path, _RELEASED_FLAG)


def release_setup_copy(copy_dir: Path, source_dir: Path) -> None:
    """A clone, import or distribution install of a setup profile is released as it is made, since the
    boot release may already have latched (an archive from a canary build imported later). The copy
    also loses the retired onboarding skills, which the latched boot cleanup would never reach."""
    if (source_dir / SETUP_MARKER).is_file():
        _release(copy_dir, _read_marker(source_dir))
    _uninstall_retired_skills(copy_dir)


def _read_marker(path: Path) -> dict:
    from utils import read_json_or_empty
    return read_json_or_empty(path / SETUP_MARKER)


def _old_guide_done(markers, config_path: Path) -> bool:
    """The user finished or skipped the agentic guide, so the questionnaire must not open for them."""
    from agent.onboarding import is_seen
    from hermes_cli.config import read_user_config_raw

    if any(marker.get("intro") == "seen" or marker.get("completed_at") for marker in markers):
        return True
    return any(is_seen(read_user_config_raw(path), _OLD_GUIDE_SKIPPED_FLAG)
               for path in {config_path, get_hermes_home() / "config.yaml"})


def _settle_run(config_path: Path, run: bool) -> None:
    """``onboarding.run = run`` unless the root config already says what it wants."""
    from hermes_cli.config import read_user_config_raw

    section = read_user_config_raw(config_path).get("onboarding")
    if not isinstance(section, dict) or not isinstance(section.get("run"), bool):
        set_run(run)


def _share_identity(profile: Path) -> bool:
    """A guest or sign-in made inside the setup profile becomes the install's shared Nous identity
    when the shared store has none (the shared store stays the identity of record otherwise).
    False only when that write failed."""
    from hermes_cli.auth import _load_auth_store, _provider_state_in
    from hermes_cli.auth_nous import _nous_shared_store_lock, _read_shared_nous_state, _write_shared_nous_state

    state = _provider_state_in(_load_auth_store(profile / "auth.json"), "nous")
    if not state:
        return True
    with _nous_shared_store_lock():
        return _read_shared_nous_state() is not None or _write_shared_nous_state(state)


def _release(profile: Path, marker: dict) -> None:
    """Drop the marker and the tool limits setup gave the profile: its cli toolset list, the toolsets
    setup itself disabled (the user's own disabled toolsets stay) and its deferred-tool list."""
    from agent.skill_utils import parse_config_string_list
    from hermes_cli.config import atomic_config_replace, read_user_config_raw

    config_path = profile / "config.yaml"
    if config_path.is_file():
        config = read_user_config_raw(config_path)
        if (config.get("platform_toolsets") or {}).get("cli") == _SETUP_CLI_TOOLSETS:
            _set_key(config, "platform_toolsets", "cli", None)
        added = set(marker.get(_ADDED_DISABLED_KEY) or [])
        disabled = [name for name in parse_config_string_list((config.get("agent") or {}).get("disabled_toolsets"))
                    if name not in added]
        _set_key(config, "agent", "disabled_toolsets", disabled or None)
        search = dict((config.get("tools") or {}).get("tool_search") or {})
        if search.get("defer") == _SETUP_DEFERRED_TOOLS:
            search.pop("defer")  # only the list setup wrote; the user's other tool_search keys stay
            _set_key(config, "tools", "tool_search", search or None)
        atomic_config_replace(config_path, config)
    (profile / SETUP_MARKER).unlink(missing_ok=True)
    logger.info("released the setup profile %s; its chats, credentials and memories stay", profile.name)


def _set_key(config: dict, section: str, key: str, value) -> None:
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


def _uninstall_retired_skills(home: Path) -> None:
    """Hand-installed copies of the two retired onboarding skills would become live broken commands."""
    from utils import read_json_or_empty

    installed = read_json_or_empty(home / "skills" / ".hub" / "lock.json").get("installed") or {}
    retired = [name for name, identifier in _RETIRED_SKILLS.items()
               if (installed.get(name) or {}).get("source") == "official"
               and installed[name].get("identifier") == identifier]
    if not retired:
        return
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.skills_hub_install import uninstall_skill

    token = set_hermes_home_override(home)
    try:
        for name in retired:
            logger.info("%s", uninstall_skill(name)[1])
    finally:
        reset_hermes_home_override(token)
