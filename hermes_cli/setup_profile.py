from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path
from typing import NamedTuple, Optional

from hermes_cli import profiles as profiles_mod
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

SETUP_PROFILE_NAME = "hermes-setup"
SETUP_PROFILE_DESCRIPTION = "Where Hermes met you — walks your first run, then checks in as you find your feet."

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
    name = _free_setup_profile_name()
    path = profiles_mod.create_profile(name, clone_config=True, no_alias=True, description=SETUP_PROFILE_DESCRIPTION)
    try:
        _write_soul(path)
        _enable_setup_toolset(path)
        (path / profiles_mod.SETUP_PROFILE_MARKER).write_text("{}\n", encoding="utf-8")
    except BaseException:
        profiles_mod.delete_profile(name, yes=True)
        raise
    return SetupProfile(name, path, created=True)


def reset_setup_profile() -> SetupProfile:
    found = find_setup_profile()
    if found is None:
        raise LookupError("no setup profile to reset")
    name, path = found
    source = get_hermes_home()
    _write_soul(path)
    _replace_dir(path / "memories")
    for relpath in profiles_mod._CLONE_SUBDIR_FILES:
        profiles_mod._clone_file(source, path, relpath)
    _replace_dir(path / "skills")
    if (source / "skills").is_dir():
        profiles_mod._copytree_keep_junctions(source / "skills", path / "skills",
                                              profiles_mod._non_exportable_entries, dirs_exist_ok=True)
    return SetupProfile(name, path, created=False)


def _free_setup_profile_name() -> str:
    from hermes_cli.dashboard_register import _NAME_NOUNS
    name = SETUP_PROFILE_NAME
    while profiles_mod.get_profile_dir(name).exists():
        name = f"{SETUP_PROFILE_NAME}-{random.choice(_NAME_NOUNS)}"
    return name


def _enable_setup_toolset(path: Path) -> None:
    from hermes_cli.config import atomic_config_write, read_user_config_raw
    from hermes_cli.tools_config import _coerce_platform_toolsets_value, _platform_default_toolset
    config_path = path / "config.yaml"
    config = read_user_config_raw(config_path)
    platform_toolsets = config.get("platform_toolsets") or {}
    cli = _coerce_platform_toolsets_value(platform_toolsets.get("cli"), "cli")
    if not isinstance(cli, list):
        cli = [_platform_default_toolset("cli")]
    config["platform_toolsets"] = {**platform_toolsets, "cli": list(dict.fromkeys([*cli, "setup"]))}
    atomic_config_write(config_path, config)


def _write_soul(path: Path) -> None:
    from utils import atomic_write_bytes
    atomic_write_bytes(path / "SOUL.md", SETUP_SOUL.encode("utf-8"))


def _replace_dir(directory: Path) -> None:
    if directory.is_symlink() or profiles_mod._junction_target(str(directory)) is not None:
        directory.unlink() if directory.is_symlink() else directory.rmdir()
    elif directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
