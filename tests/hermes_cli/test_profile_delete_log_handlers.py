"""Profile deletion must release the profile-routed log handlers this process holds.

A long-lived multiplex process (the Desktop's serve backend) routes records to every
served profile's ``logs/agent.log`` through ``_ProfileRoutingFileHandler``. On Windows the
per-home rotating handler is ``ConcurrentRotatingFileHandler``, which keeps
``logs/.__agent.lock`` open; ``hermes profile delete`` run inside that process then fails
``rmtree`` with ``[WinError 32]`` and the Desktop drops the bot while the directory stays.
"""

import logging
from pathlib import Path

import pytest

import hermes_logging
from hermes_cli import profiles
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.mark.windows_only
def test_delete_profile_after_routed_log_record(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Do not manage host services or scan unrelated developer processes in this test.
    monkeypatch.setattr(profiles, "_cleanup_gateway_service", lambda *_: None)
    monkeypatch.setattr(profiles, "_stop_profile_backends", lambda *_: None)
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda *_: None)
    profile = profiles.create_profile("routed-log-delete", no_alias=True)

    hermes_logging.setup_logging(hermes_home=home, force=True)
    try:
        assert hermes_logging.enable_profile_log_routing([home, profile]) is True
        logger = logging.getLogger("tests.profile-delete.routed-log")
        token = set_hermes_home_override(profile)
        try:
            logger.info("routed record before deletion")
        finally:
            reset_hermes_home_override(token)
        hermes_logging.flush_log_queue()
        content = (profile / "logs" / "agent.log").read_text(
            encoding="utf-8"
        )
        assert "routed record before deletion" in content

        profiles.delete_profile("routed-log-delete", yes=True)
        assert not profile.exists()
    finally:
        # Keep a pre-fix failure from leaking Windows handles into tempdir cleanup.
        hermes_logging._reset_queued_handlers()


@pytest.mark.windows_only
def test_delete_profile_after_in_process_agent_logging(tmp_path, monkeypatch):
    """The Desktop path: the serve process builds the bot's agent in-process, and
    ``agent_init`` calls ``setup_logging(hermes_home=<profile>)`` for a second home, which
    adopts it into profile routing. Deleting that profile afterwards must still succeed."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(profiles, "_cleanup_gateway_service", lambda *_: None)
    monkeypatch.setattr(profiles, "_stop_profile_backends", lambda *_: None)
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda *_: None)
    profile = profiles.create_profile("adopted-log-delete", no_alias=True)

    hermes_logging.setup_logging(hermes_home=home, force=True)
    try:
        logger = logging.getLogger("tests.profile-delete.adopted-log")
        token = set_hermes_home_override(profile)
        try:
            hermes_logging.setup_logging(hermes_home=profile)
            logger.error("adopted record before deletion")
        finally:
            reset_hermes_home_override(token)
        hermes_logging.flush_log_queue()
        content = (profile / "logs" / "agent.log").read_text(encoding="utf-8")
        assert "adopted record before deletion" in content

        profiles.delete_profile("adopted-log-delete", yes=True)
        assert not profile.exists()
    finally:
        hermes_logging._reset_queued_handlers()
