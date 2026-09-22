"""The setup profile: where guided onboarding runs and the guide keeps checking in afterwards.

The backend owns it. One per home, found by ``role: setup`` in ``profile.yaml`` (the name is an
implementation detail). ``ensure_setup_profile`` creates it once and afterwards returns it as-is;
``reset_setup_profile`` restores the created state in place, keeping the directory, name and role.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import NamedTuple, Optional

from hermes_cli import profiles as profiles_mod

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
    """``(name, path)`` of the profile carrying ``role: setup``, first by name; None when absent."""
    found = [(p.name, Path(p.path)) for p in profiles_mod.list_profiles(lazy_skill_count=True)
             if p.role == profiles_mod.SETUP_ROLE]
    if len(found) > 1:
        logger.warning("several profiles carry role: setup (%s); using %s",
                       ", ".join(name for name, _ in found), found[0][0])
    return found[0] if found else None


def ensure_setup_profile() -> SetupProfile:
    """Create-or-read. A found profile is returned untouched (soul, memories, skills, config).

    A ``hermes-setup`` profile from before the role existed is adopted: it gets the role and
    nothing else, so existing installs keep their guide chat."""
    found = find_setup_profile()
    if found is not None:
        return SetupProfile(found[0], found[1], created=False)
    if profiles_mod.profile_exists(SETUP_PROFILE_NAME):
        path = profiles_mod.get_profile_dir(SETUP_PROFILE_NAME)
        profiles_mod.write_profile_meta(path, role=profiles_mod.SETUP_ROLE)
        return SetupProfile(SETUP_PROFILE_NAME, path, created=False)
    path = profiles_mod.create_profile(SETUP_PROFILE_NAME, clone_from="default", clone_config=True, no_alias=True,
                                       description=SETUP_PROFILE_DESCRIPTION)
    _write_soul(path)
    profiles_mod.write_profile_meta(path, role=profiles_mod.SETUP_ROLE)
    return SetupProfile(SETUP_PROFILE_NAME, path, created=True)


def reset_setup_profile() -> SetupProfile:
    """Restore the created state in place: soul from the template, memories and skills re-copied
    from ``default`` as create copies them. Session history is cleared by the caller, which owns
    the live sessions and the session store. Raises LookupError when no setup profile exists."""
    found = find_setup_profile()
    if found is None:
        raise LookupError("no setup profile to reset")
    name, path = found
    source = profiles_mod.get_profile_dir("default")
    _write_soul(path)
    _replace_dir(path / "memories")
    for relpath in profiles_mod._CLONE_SUBDIR_FILES:
        profiles_mod._clone_file(source, path, relpath)
    _replace_dir(path / "skills")
    if (source / "skills").is_dir():
        profiles_mod._copytree_keep_junctions(source / "skills", path / "skills",
                                              profiles_mod._non_exportable_entries, dirs_exist_ok=True)
    return SetupProfile(name, path, created=False)


def _write_soul(path: Path) -> None:
    from utils import atomic_write_text
    atomic_write_text(path / "SOUL.md", SETUP_SOUL)


def _replace_dir(directory: Path) -> None:
    """Empty *directory*. A link (symlink or NTFS junction) is removed, never followed: its
    target belongs to another profile or an external skills root."""
    if directory.is_symlink() or profiles_mod._junction_target(str(directory)) is not None:
        directory.unlink() if directory.is_symlink() else directory.rmdir()
    elif directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
