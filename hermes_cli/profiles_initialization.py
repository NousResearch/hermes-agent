"""Initial profile files, written on the staging tree before publication."""
import contextlib
import os
import shutil
from pathlib import Path
from typing import Optional


def clone_staging_dir(profile_dir: Path) -> Path:
    """Fresh ``profiles/.<name>.staging-<pid>`` beside the final dir (same filesystem, so the publish
    rename is atomic). A leftover from a crashed create is discarded."""
    staging = profile_dir.parent / f".{profile_dir.name}.staging-{os.getpid()}"
    profile_dir.parent.mkdir(parents=True, exist_ok=True)
    if staging.is_symlink() or staging.is_file():
        staging.unlink()
    elif staging.is_dir():
        shutil.rmtree(staging, ignore_errors=True)
    return staging


def discard_unpublished_vault_key(ref: Optional[dict]) -> None:
    """A create that fails after a --clone-all copy got its own keychain entry (``ref``, returned by
    ``VaultStore.own_copied_key``) deletes exactly that entry; never one inferred from the source."""
    if ref is not None:
        from agent.vault_keychain import delete_key
        delete_key(ref)


def finish_profile_layout(profile_dir: Path, *, no_skills: bool, clone_all: bool,
                           description: Optional[str], soul: Optional[str] = None) -> None:
    """Seed files a fresh profile owns from day one; runs on the staging tree before publish."""
    from hermes_cli.profiles import (
        _seed_file_if_missing, _PLACEHOLDER_ENV, NO_BUNDLED_SKILLS_MARKER,
        _migrate_profile_config_if_outdated, write_profile_meta,
    )
    if soul is not None:
        (profile_dir / "SOUL.md").write_text(soul, encoding="utf-8")

    # Seed an empty .env so the profile owns a credentials file from day one. Without it,
    # profile-scoped env writes (dashboard Channels/Keys pages, `hermes -p <name> auth add`)
    # had no file until first write and the profile silently inherited shell API keys —
    # read by users as "the new profile reads the root .env". Skipped when a clone copied one.
    _seed_file_if_missing(profile_dir / ".env", _PLACEHOLDER_ENV, 0o600)

    # Default SOUL.md to customize immediately (skipped when a clone already provided one).
    with contextlib.suppress(Exception):  # best-effort — don't fail profile creation over this
        from hermes_cli.default_soul import DEFAULT_SOUL_MD
        _seed_file_if_missing(profile_dir / "SOUL.md", DEFAULT_SOUL_MD)

    # Opt-out marker read by seed_profile_skills() and `hermes update`'s all-profile sync
    # (the feature still works via the empty skills/ dir if this fails).
    if no_skills:
        _seed_file_if_missing(
            profile_dir / NO_BUNDLED_SKILLS_MARKER,
            "This profile opted out of bundled-skill seeding (`hermes profile create --no-skills`).\n"
            "Delete this file to re-enable sync on the next `hermes update`.\n",
        )

    # Migrate config-only clones now so desktop/status don't warn that a just-created
    # profile is v0/outdated; --clone-all snapshots stay byte-for-byte apart from the
    # explicit runtime/history stripping above.
    if not clone_all:
        _migrate_profile_config_if_outdated(profile_dir)

    # Description last, so a partial-create failure doesn't strand a description file.
    if description and description.strip():
        with contextlib.suppress(Exception):  # non-fatal — `hermes profile describe` works later
            write_profile_meta(profile_dir, description=description.strip(), description_auto=False)
