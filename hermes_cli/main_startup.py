"""Scheduling for the native interactive CLI's startup work."""

from pathlib import Path
import threading


def is_native_interactive_chat(args, *, use_tui: bool) -> bool:
    """One-shot and dedicated entrypoints keep their existing startup schedule."""
    return (
        getattr(args, "command", None) in {None, "chat"}
        and not use_tui
        and not getattr(args, "query", None)
        and not getattr(args, "query_file", None)
        and not getattr(args, "oneshot", None)
    )


def prepare_chat_startup(args, *, use_tui: bool) -> None:
    """Serialize local scans before CLI imports; only the update check stays async."""
    if not is_native_interactive_chat(args, use_tui=use_tui):
        _start_chat_background_prefetch()
        return

    from hermes_cli import main
    from hermes_cli import banner

    # Competing filesystem walks and Python imports contend for the GIL/import
    # lock. Seed first, then warm exactly the caches the banner already reads.
    banner._quiet(main._sync_bundled_skills_for_startup)
    banner._available_skills_cache = None
    for warm in (banner.get_available_skills, banner.get_git_banner_state,
                 banner.get_latest_release_tag):
        banner._quiet(warm)
    if main._termux_should_prefetch_update_check():
        banner._quiet(banner.prefetch_update_check)


def _start_chat_background_prefetch() -> None:
    """Kick off the update-check/banner prefetch and the bundled-skills sync.

    Update check is opt-in on Termux (it imports rich/prompt_toolkit in the
    foreground and competes for CPU on single-core devices). The skills sync
    is idempotent and hash-gated (~120-170ms of rglob/hashing) so it normally
    runs in a daemon thread — skill loading happens at agent init, long after.
    The ONE exception is an unseeded ~/.hermes/skills: there the banner
    prefetch races the sync and caches an empty index ("No skills installed"
    on the very first launch), so the first run syncs in the foreground and
    drops the banner's skills cache.
    """
    from hermes_cli import main

    if main._termux_should_prefetch_update_check():
        try:
            from hermes_cli.banner import prefetch_banner_data, prefetch_update_check

            prefetch_update_check()
            prefetch_banner_data()  # git banner state + skills index off-thread
        except Exception:
            pass

    def _skills_dir_is_unseeded() -> bool:
        try:
            from hermes_cli.config import get_hermes_home
            skills_dir = Path(get_hermes_home()) / "skills"
            if not skills_dir.is_dir():
                return True
            return next(skills_dir.rglob("SKILL.md"), None) is None
        except Exception:
            return False

    def _skills_sync_bg() -> None:
        try:
            main._sync_bundled_skills_for_startup()
        except Exception:
            pass

    if _skills_dir_is_unseeded():
        _skills_sync_bg()
        # Drop the banner's possibly-empty skills cache so it recomputes.
        try:
            import hermes_cli.banner as _banner_mod
            _banner_mod._available_skills_cache = None
        except Exception:
            pass
    else:
        threading.Thread(
            target=_skills_sync_bg, name="bundled-skills-sync", daemon=True
        ).start()
