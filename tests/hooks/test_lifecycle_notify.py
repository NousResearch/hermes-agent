"""Tests for the lifecycle-notify event hook."""

import importlib
import logging
import os
from unittest import mock

import pytest


def _import_handler():
    """Import the lifecycle-notify handler module."""
    import importlib.util

    handler_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "hooks",
        "lifecycle-notify",
        "handler.py",
    )
    handler_path = os.path.normpath(handler_path)
    spec = importlib.util.spec_from_file_location("lifecycle_notify_handler", handler_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.asyncio
async def test_startup_notification_logged(caplog):
    """gateway:startup with HERMES_HOME_CHANNEL set emits a log message."""
    mod = _import_handler()

    env = {
        "HERMES_HOME_CHANNEL": "test-channel-123",
        "HERMES_AGENT_NAME": "test-agent",
        "HERMES_DEFAULT_PLATFORM": "signal",
    }
    with mock.patch.dict(os.environ, env, clear=False):
        with caplog.at_level(logging.INFO, logger="hooks.lifecycle-notify"):
            await mod.handle("gateway:startup", {"platforms": ["signal"]})

    assert any("test-agent" in r.message and "online" in r.message for r in caplog.records), (
        f"Expected log about agent online, got: {[r.message for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_skips_without_home_channel(caplog):
    """No HERMES_HOME_CHANNEL set means no log output."""
    mod = _import_handler()

    env_remove = {"HERMES_HOME_CHANNEL": ""}
    with mock.patch.dict(os.environ, env_remove, clear=False):
        # Ensure the var is actually absent
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_HOME_CHANNEL", None)
            with caplog.at_level(logging.INFO, logger="hooks.lifecycle-notify"):
                await mod.handle("gateway:startup", {})

    assert not any("online" in r.message for r in caplog.records), (
        f"Should not log when no home channel, got: {[r.message for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_skips_non_startup_events(caplog):
    """Non gateway:startup events are ignored."""
    mod = _import_handler()

    env = {
        "HERMES_HOME_CHANNEL": "test-channel-123",
        "HERMES_AGENT_NAME": "test-agent",
        "HERMES_DEFAULT_PLATFORM": "signal",
    }
    with mock.patch.dict(os.environ, env, clear=False):
        with caplog.at_level(logging.INFO, logger="hooks.lifecycle-notify"):
            await mod.handle("session:start", {"session_id": "abc"})

    assert not any("online" in r.message for r in caplog.records), (
        f"Should not log for non-startup events, got: {[r.message for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_skips_without_platform(caplog):
    """No HERMES_DEFAULT_PLATFORM means no notification."""
    mod = _import_handler()

    env = {
        "HERMES_HOME_CHANNEL": "test-channel-123",
        "HERMES_AGENT_NAME": "test-agent",
    }
    with mock.patch.dict(os.environ, env, clear=False):
        os.environ.pop("HERMES_DEFAULT_PLATFORM", None)
        with caplog.at_level(logging.INFO, logger="hooks.lifecycle-notify"):
            await mod.handle("gateway:startup", {})

    assert not any("online" in r.message for r in caplog.records), (
        f"Should not log without platform, got: {[r.message for r in caplog.records]}"
    )
