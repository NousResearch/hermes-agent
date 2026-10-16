"""Renaming a profile releases what this process holds open under the old home, like delete does.

The rename proves no in-process ``state.db`` connection survives retirement before it moves the
home. A long-lived serve process can hold the profile's shared ``SessionDB`` through
``hermes_state_registry`` (an agent turn, the cron ticker, a dashboard read) and route the
profile's ``logs/*.log`` through ``_ProfileRoutingFileHandler``. Delete force-closed both before
its release proof; rename did not, so every rename attempt timed out with "still in use by this
Hermes process; retry rename" (PR #93508 review). Open routed log files also block the move on
Windows.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

import hermes_logging
import hermes_state_registry as registry
from hermes_cli import profile_lifecycle, profiles
from hermes_cli.sqlite_safe_read import has_live_connection
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Never inspect or stop services and processes outside this disposable fixture.
    monkeypatch.setattr(profiles, "_check_gateway_running", lambda *_: False)
    for name in ("_cleanup_gateway_service", "_maybe_unregister_gateway_service",
                 "_maybe_register_gateway_service", "_stop_profile_backends", "_stop_bot_desktop",
                 "_notify_multiplexer"):
        monkeypatch.setattr(profiles, name, lambda *_: None)
    monkeypatch.setattr(profile_lifecycle, "external_profile_file_holders", lambda *_: [])
    # A holder that is never released fails the release proof fast instead of after 5 s.
    monkeypatch.setattr(profile_lifecycle, "_PROFILE_DB_RELEASE_TIMEOUT_SECONDS", 0.2)
    try:
        yield home
    finally:
        # Keep a refused rename from leaking open handles into tempdir cleanup.
        registry.close_all_under(tmp_path)
        hermes_logging._reset_queued_handlers()
        hermes_logging._logging_initialized = False


def test_rename_force_closes_this_process_shared_session_db(profile_env):
    old_dir = profiles.create_profile("coder", no_alias=True, no_skills=True)
    db = registry.acquire(old_dir / "state.db")
    assert has_live_connection(old_dir / "state.db")

    new_dir = profiles.rename_profile("coder", "dev")

    assert new_dir == profiles.get_profile_dir("dev") and new_dir.is_dir()
    assert profiles.profile_exists("dev") and not profiles.profile_exists("coder")
    assert not has_live_connection(old_dir / "state.db")
    assert getattr(db, "_shared_registry_owned", True) is False, "the shared handle was not torn down"


def test_rename_releases_routed_log_handlers_for_the_old_home(profile_env):
    home = profile_env
    old_dir = profiles.create_profile("coder", no_alias=True, no_skills=True)
    hermes_logging.setup_logging(hermes_home=home, force=True)
    token = set_hermes_home_override(old_dir)
    try:
        assert hermes_logging.enable_profile_log_routing([home, old_dir]) is True
        logging.getLogger("agent.tests.profile-rename").error("routed record before rename")
    finally:
        reset_hermes_home_override(token)
    hermes_logging.flush_log_queue()
    resolved = old_dir.resolve()
    routers = [h for h in hermes_logging._queued_file_handlers
               if isinstance(h, hermes_logging._ProfileRoutingFileHandler)]
    held = [router._profile_handlers[resolved] for router in routers]
    assert held, "the old home's logs were never routed"

    profiles.rename_profile("coder", "dev")

    assert profiles.profile_exists("dev") and not old_dir.exists()
    assert all(resolved not in router._profile_handlers for router in routers)
    assert all(handler.stream is None for handler in held), "streams still open into the old home"
