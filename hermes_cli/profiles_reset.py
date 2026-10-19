"""Reset a profile in place: wipe the agent's own state, keep everything it shares.

The default profile's home IS the install root: it holds ``profiles/`` (every other profile), the
source checkout and managed runtimes, and the ``state.db`` whose routing tables the multiplexed
gateway keeps for every served profile. Removing that directory is never an option, so "deleting"
the default profile means resetting it: its memories, skills, settings, persona, conversations and
cron jobs go; credentials, other profiles and the install stay.

The reset runs while the profile's runtime stays up. On Desktop the backend answering the request
is the default profile's own multiplexing gateway, which also serves every other bot, so stopping
it (``delete_profile``'s teardown) would kill the request and every other profile with it.
Every step therefore goes through the live-safe seam its store already has: sessions through
``SessionDB.delete_sessions`` with the active-write guards (a turn in flight keeps its session),
cron jobs through ``cron.jobs.remove_job`` under the jobs lock, ``config.yaml`` through the config
writer. Whole-path removals cover only per-agent files no other profile reads.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PROFILE_DELETE_REFUSAL = (
    "The default profile cannot be removed: its directory holds the Hermes install and every other "
    "profile.\nTo wipe its memories, skills, settings, chats and cron jobs, use: hermes profile reset default"
    "\nTo remove everything, use: hermes uninstall")

# Per-agent paths removed wholesale, by the category the confirmation names. Everything not listed
# is kept: credentials (``PROFILE_CREDENTIAL_PATHS``: .env, auth.json, OAuth/MCP tokens, pairing,
# browser logins, backups), ``profiles/``, the source checkout and runtimes, logs, caches,
# ``workspace/``, plugins, skins and kanban boards. ``config.yaml`` and ``SOUL.md`` are rewritten,
# not removed, and sessions are deleted row by row (``state.db`` also holds other profiles' routing).
RESET_REMOVED_PATHS: dict[str, tuple[str, ...]] = {
    "memories": ("memories", "MEMORY.md", "USER.md",
                 "memory_store.db", "memory_store.db-wal", "memory_store.db-shm"),
    "skills": ("skills", ".no-bundled-skills"),
    "persona": ("profile.yaml", "desktop.json"),
    "conversations": ("response_store.db", "response_store.db-wal", "response_store.db-shm",
                      ".hermes_history"),
}
# Recreated empty after the removal, as a fresh profile has them.
_RECREATED_DIRS = ("memories", "skills")


@dataclass
class ProfileResetResult:
    """What a reset did. ``failures`` lists paths or steps that could not be completed."""
    profile: str
    path: Path
    sessions_deleted: int = 0
    sessions_skipped: list[str] = field(default_factory=list)
    cron_jobs_removed: int = 0
    removed: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "profile": self.profile, "path": str(self.path), "sessions_deleted": self.sessions_deleted,
            "sessions_skipped": list(self.sessions_skipped), "cron_jobs_removed": self.cron_jobs_removed,
            "removed": list(self.removed), "failures": list(self.failures),
        }


def print_reset_summary(canon: str, home: Path) -> None:
    """Show what ``reset_profile`` removes and what it keeps."""
    from hermes_cli.profiles import _count_skills, _read_config_model
    model, provider = _read_config_model(home)
    print(f"\nProfile: {canon}")
    print(f"Path:    {home}")
    if model:
        print(f"Model:   {model}" + (f" ({provider})" if provider else "") + "  (kept)")
    skills = _count_skills(home)
    if skills:
        print(f"Skills:  {skills}")
    print("\nThis will permanently reset this profile:")
    print("  • Memories (built-in memory and the local memory store) are erased")
    print("  • Skills are removed; the bundled skills are installed again")
    print("  • Settings (config.yaml), SOUL.md, display name and appearance return to defaults")
    print("  • This profile's sessions and conversations are deleted")
    print("  • This profile's cron jobs are removed")
    print("\nKept:")
    print("  • API keys and logins (.env, auth.json, OAuth and platform tokens) and the model choice")
    if canon == "default":
        print("  • Every other profile (profiles/), the Hermes install, logs, caches and workspace/")
    else:
        print("  • Logs, caches, workspace/ and the profile's command alias")


def reset_profile(name: str = "default", yes: bool = False) -> ProfileResetResult | None:
    """Reset a profile to a fresh state (see the module docstring). Returns None when the user
    cancels the typed confirmation; raises ``RuntimeError`` naming every step that failed after
    attempting all of them."""
    from hermes_cli.profiles import _existing_profile_dir, normalize_profile_name
    canon, home = _existing_profile_dir(normalize_profile_name(name))
    print_reset_summary(canon, home)
    if not yes:
        print()
        try:
            confirm = input(f"Type '{canon}' to confirm: ").strip()
        except (KeyboardInterrupt, EOFError):
            confirm = None
            print()
        if confirm != canon:
            print("Cancelled.")
            return None

    result = ProfileResetResult(profile=canon, path=home)
    _delete_owned_sessions(home, canon, result)
    _remove_cron_jobs(home, result)
    for category, relpaths in RESET_REMOVED_PATHS.items():
        for relpath in relpaths:
            _remove_path(home, relpath, category, result)
    for dirname in _RECREATED_DIRS:
        (home / dirname).mkdir(parents=True, exist_ok=True)
    _reset_settings(home, result)

    from hermes_cli.profiles import seed_profile_skills
    if seed_profile_skills(home, quiet=True) is None:
        result.failures.append("bundled skill re-seed (run `hermes skills sync` to retry)")
    else:
        print("✓ Bundled skills installed again")

    if result.failures:
        raise RuntimeError(f"Profile '{canon}' was only partly reset; could not complete: "
                           + "; ".join(result.failures))
    print(f"\nProfile '{canon}' reset.")
    return result


def _delete_owned_sessions(home: Path, owner: str, result: ProfileResetResult) -> None:
    """Delete *owner*'s sessions from its store, transcripts included. Rows another profile owns
    stay, and so does a session with a turn in flight (reported, never torn mid-write)."""
    db_path = home / "state.db"
    if not db_path.exists():
        return
    from hermes_state_registry import acquire, release_or_close
    db = None
    try:
        db = acquire(db_path)
        ids = db.owned_session_ids(owner)
        skipped: list[str] = []
        result.sessions_deleted = db.delete_sessions(
            ids, sessions_dir=home / "sessions", exclude_active_write_guards=True, skipped_ids=skipped)
        result.sessions_skipped = skipped
    except Exception as exc:  # health: allow BLE001 -- every reset step runs; the failure is reported and raised at the end
        result.failures.append(f"sessions in {db_path} ({type(exc).__name__}: {exc})")
        return
    finally:
        if db is not None:
            with contextlib.suppress(Exception):
                release_or_close(db)
    print(f"✓ Deleted {result.sessions_deleted} session(s)")
    if result.sessions_skipped:
        print(f"⚠ Kept {len(result.sessions_skipped)} session(s) with a turn in progress")


def _remove_cron_jobs(home: Path, result: ProfileResetResult) -> None:
    """Remove every cron job through the jobs lock, so a live scheduler's next save cannot
    resurrect them; each removal also drops that job's output and notepad."""
    if not (home / "cron" / "jobs.json").exists():
        return
    try:
        from cron.jobs import load_jobs, remove_job, use_cron_store
        with use_cron_store(home):
            for job in load_jobs():
                if remove_job(job["id"]):
                    result.cron_jobs_removed += 1
    except Exception as exc:  # health: allow BLE001 -- every reset step runs; the failure is reported and raised at the end
        result.failures.append(f"cron jobs ({type(exc).__name__}: {exc})")
        return
    print(f"✓ Removed {result.cron_jobs_removed} cron job(s)")


def _remove_path(home: Path, relpath: str, category: str, result: ProfileResetResult) -> None:
    from hermes_cli.profiles_remove import _rmtree_make_writable, _rmtree_with_retry
    target = home / relpath
    if not (target.exists() or target.is_symlink()):
        return
    try:
        if target.is_dir() and not target.is_symlink():
            _rmtree_with_retry(target, _rmtree_make_writable)
        else:
            target.unlink()
    except OSError as exc:
        # Windows: a process still holding the file open (WinError 32) lands here.
        result.failures.append(f"{target} ({exc})")
        return
    result.removed.append(relpath)
    print(f"✓ Removed {category}: {relpath}")


def _reset_settings(home: Path, result: ProfileResetResult) -> None:
    """Rewrite ``config.yaml`` to defaults and ``SOUL.md`` to the stock persona.

    Two values survive: the model selection (exactly what a fresh profile is seeded with, so the
    kept credentials still reach a model) and an explicit ``gateway.multiplex_profiles``, which on
    the default profile decides how every OTHER profile is served."""
    try:
        from hermes_cli.config import atomic_config_replace, read_user_config_raw
        from hermes_cli.default_soul import DEFAULT_SOUL_MD
        from hermes_cli.profile_model_seed import launch_model_seed
        from hermes_cli.profiles import _migrate_profile_config_if_outdated
        config_path = home / "config.yaml"
        raw = read_user_config_raw(config_path)
        fresh = launch_model_seed(raw)
        gateway = raw.get("gateway")
        if isinstance(gateway, dict) and "multiplex_profiles" in gateway:
            fresh["gateway"] = {"multiplex_profiles": gateway["multiplex_profiles"]}
        if config_path.exists() or fresh:
            atomic_config_replace(config_path, fresh)
            _migrate_profile_config_if_outdated(home)
        (home / "SOUL.md").write_text(DEFAULT_SOUL_MD, encoding="utf-8")
    except Exception as exc:  # health: allow BLE001 -- every reset step runs; the failure is reported and raised at the end
        result.failures.append(f"settings ({type(exc).__name__}: {exc})")
        return
    print("✓ Settings and SOUL.md reset to defaults")

