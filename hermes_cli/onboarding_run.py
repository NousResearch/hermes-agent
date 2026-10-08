"""``onboarding.run``: whether the desktop first-run questionnaire opens on the next launch.

The flag lives in the ROOT profile's ``config.yaml`` (``get_default_hermes_root()``), never in the
launch profile's: a pooled backend launched under ``-p work`` must read and write the same file as
one launched under ``default``, because the questionnaire always targets the default profile.
Unset means "only on a fresh install"; ``true`` re-runs it once; ``false`` means done.
"""

from __future__ import annotations

from pathlib import Path

from hermes_constants import (
    PROFILE_ID_RE,
    get_default_hermes_root,
    named_profile_has_identity,
    named_profile_is_deleted,
)


def _root_config() -> Path:
    return get_default_hermes_root() / "config.yaml"


def install_has_history(root: Path) -> bool:
    """A session row in the root ``state.db``, or a named profile the user made. A fresh boot
    creates neither: it leaves an empty ``state.db``, ``auth.json`` (the free-tier mint) and
    ``SOUL.md``, which is why the signal is a session row and not a file."""
    if named_profiles(root):
        return True
    db_path = root / "state.db"
    if not db_path.is_file():
        return False
    from hermes_state_registry import acquire, release_or_close
    db = acquire(db_path)
    try:
        return db.session_count_ge(1)
    finally:
        release_or_close(db)


def named_profiles(root: Path) -> list[Path]:
    """Live named profiles under *root*, with the same filter as ``profiles._iter_named_profile_dirs``."""
    profiles = root / "profiles"
    if not profiles.is_dir():
        return []
    return [entry for entry in profiles.iterdir()
            if entry.is_dir() and entry.name != "default" and PROFILE_ID_RE.match(entry.name)
            and named_profile_has_identity(entry) and not named_profile_is_deleted(entry)]


def should_run() -> bool:
    """The root config's explicit ``onboarding.run`` (managed overlay included) when it is a bool,
    else whether the install is fresh."""
    from hermes_cli.config_effective import load_user_config_effective

    # Not load_config(): it resolves the launch profile and fills the default None for an unset flag.
    section = load_user_config_effective(_root_config()).get("onboarding")
    value = section.get("run") if isinstance(section, dict) else None
    return value if isinstance(value, bool) else not install_has_history(get_default_hermes_root())


def set_run(value: bool, *, mark_profile_offered: bool = False) -> None:
    """Persist ``onboarding.run`` (and, with *mark_profile_offered*, ``onboarding.seen.profile_build_offered``)
    in the root ``config.yaml`` as one write, keeping every other key and comment. A failed write raises."""
    from agent.onboarding import PROFILE_BUILD_FLAG
    from hermes_cli import config as config_mod

    path = _root_config()
    with config_mod._CONFIG_LOCK:
        config = config_mod.read_user_config_raw(path)
        section = config_mod._ensure_dict(config, "onboarding")
        section["run"] = value
        if mark_profile_offered:
            config_mod._ensure_dict(section, "seen")[PROFILE_BUILD_FLAG] = True
        config_mod.atomic_config_write(path, config)
