"""Profile deletion must release the profile-routed log handlers this process holds (#112538).

A long-lived multiplex process (the Desktop's ``rabbit serve`` backend) routes records to every
served profile's ``logs/*.log`` through ``_ProfileRoutingFileHandler``, which lazily opens one
rotating handler per home. On Windows that handler is ``ConcurrentRotatingFileHandler`` and keeps
``logs/.__<name>.lock`` open, so ``delete_profile`` run inside that process fails ``rmtree`` with
``[WinError 32]``: the REST call returns 500, the Desktop drops the bot, the directory stays.
The routing seam itself is platform-neutral, so the release and the directory removal are
asserted on every OS (the lock-file symptom only reproduces on Windows).
"""

import logging
from pathlib import Path

import pytest

import rabbit_logging
from rabbit_cli import profiles
from rabbit_constants import reset_rabbit_home_override, set_rabbit_home_override


@pytest.fixture
def routed_profile(tmp_path, monkeypatch):
    """A default home logging in-process plus a created profile; host services are not touched."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".rabbit"
    home.mkdir()
    monkeypatch.setenv("RABBIT_HOME", str(home))
    monkeypatch.setattr(profiles, "_cleanup_gateway_service", lambda *_: None)
    monkeypatch.setattr(profiles, "_maybe_unregister_gateway_service", lambda *_: None)
    monkeypatch.setattr(profiles, "_stop_profile_backends", lambda *_: None)
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda *_: None)
    profile = profiles.create_profile("routed-log-delete", no_alias=True)
    rabbit_logging.setup_logging(rabbit_home=home, force=True)
    try:
        yield home, profile
    finally:
        # Keep a pre-fix failure from leaking open handles into tempdir cleanup.
        rabbit_logging._reset_queued_handlers()
        rabbit_logging._logging_initialized = False


def _log_into(profile: Path, *, adopt: bool) -> None:
    """Route one record to *profile*: explicitly (Desktop cron startup) or through the
    ``setup_logging(rabbit_home=<profile>)`` adoption path ``agent_init`` takes."""
    logger = logging.getLogger("agent.tests.profile-delete")
    token = set_rabbit_home_override(profile)
    try:
        if adopt:
            rabbit_logging.setup_logging(rabbit_home=profile)
        else:
            assert rabbit_logging.enable_profile_log_routing([Path(profile).parent.parent, profile]) is True
        logger.error("routed record before deletion")
    finally:
        reset_rabbit_home_override(token)
    rabbit_logging.flush_log_queue()
    assert "routed record before deletion" in (profile / "logs" / "agent.log").read_text(encoding="utf-8")


def _routers() -> list:
    return [h for h in rabbit_logging._queued_file_handlers
            if isinstance(h, rabbit_logging._ProfileRoutingFileHandler)]


@pytest.mark.parametrize("adopt", [False, True], ids=["explicit-routing", "setup_logging-adoption"])
def test_delete_profile_releases_every_routed_handler_for_that_home(routed_profile, adopt):
    _home, profile = routed_profile
    _log_into(profile, adopt=adopt)
    routers = _routers()
    resolved = profile.resolve()
    held = {Path(r.baseFilename).name: r._profile_handlers[resolved] for r in routers}
    assert set(held) == {"agent.log", "errors.log"}

    profiles.delete_profile("routed-log-delete", yes=True)

    for router in routers:
        assert resolved not in router._profile_handlers
        assert resolved not in router._profile_homes
    assert all(handler.stream is None for handler in held.values()), "streams still open into the deleted profile"
    assert not profile.exists()
