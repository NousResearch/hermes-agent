"""state.db housekeeping for the messaging gateway: per-profile auto-archive and auto-prune/VACUUM.

Split out of ``gateway/run.py``; scheduled at startup and on the housekeeping tick
(``_start_gateway_housekeeping``), once per served profile under that profile's scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home


def _launch_sessions_dir(config) -> Optional[tuple[Path, Path]]:
    """``(launch home, its configured transcript dir)``, or ``None`` when the gateway carries none.

    MUST be called outside any profile scope — ``get_hermes_home()`` is what identifies the launch
    home. Consumed by :func:`_profile_sessions_dir`.
    """
    sessions_dir = getattr(config, "sessions_dir", None)
    if sessions_dir is None:
        return None
    return get_hermes_home(), Path(sessions_dir)


def _profile_sessions_dir(launch: Optional[tuple[Path, Path]]) -> Path:
    """Transcript dir of the profile currently in scope.

    ``gateway.sessions_dir`` overrides the LAUNCH profile's transcript dir only; every other served
    profile keeps ``<home>/sessions``. Hardcoding ``<home>/sessions`` for the launch home too wrote
    transcripts to the configured dir while the prune unlinked under the default one, orphaning
    every pruned session's ``.json``/``.jsonl``/``request_dump_*`` forever.
    """
    home = get_hermes_home()
    if launch is not None and Path(launch[0]) == home:
        return Path(launch[1])
    return home / "sessions"


def _housekeeping_state_db_maintenance(launch: Optional[tuple[Path, Path]] = None) -> None:
    """Stale-session auto-archive plus auto-prune/VACUUM for ONE profile's state.db; both are gated
    by sessions.min_interval_hours (VACUUM additionally by its own throttles). Opens its own
    SessionDB — SQLite connections are thread-bound. The stored user images whose message left the
    DB are swept first, on every run (``agent.image_store``).

    Profile-scoped by its caller: ``acquire()``, ``get_hermes_home()`` and ``load_config()`` all
    resolve through the active scope, so an unscoped run swept only the LAUNCH profile's store with
    the LAUNCH profile's retention settings and a multiplexed secondary was never archived, pruned
    or vacuumed by anyone — the dashboard/serve trigger defers to the gateway for every profile a
    gateway owns (``web_server_sessions``). *launch* carries the launch home's configured transcript
    dir (:func:`_launch_sessions_dir`) so its override still governs its own profile."""
    from hermes_cli.config import load_config as _load_full_config
    from hermes_state_registry import acquire, release_or_close
    from agent.image_store import sweep_store
    sweep_store()  # stored user images whose message left the DB; not gated by the settings below
    _sess_cfg = (_load_full_config().get("sessions") or {})
    if not (_sess_cfg.get("auto_archive", False) or _sess_cfg.get("auto_prune", False)):
        return
    _adb = acquire()
    try:
        if _sess_cfg.get("auto_archive", False):
            _adb.maybe_auto_archive(
                idle_days=float(_sess_cfg.get("auto_archive_days", 3)),
                min_interval_hours=int(_sess_cfg.get("min_interval_hours", 24)))
        if _sess_cfg.get("auto_prune", False):
            _adb.maybe_auto_prune_and_vacuum(
                retention_days=int(_sess_cfg.get("retention_days", 90)),
                min_interval_hours=int(_sess_cfg.get("min_interval_hours", 24)),
                min_vacuum_interval_days=int(_sess_cfg.get("min_vacuum_interval_days", 30)),
                vacuum=bool(_sess_cfg.get("vacuum_after_prune", True)),
                sessions_dir=_profile_sessions_dir(launch))
    finally:
        release_or_close(_adb)
