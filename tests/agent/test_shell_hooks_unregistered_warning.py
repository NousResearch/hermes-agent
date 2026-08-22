"""Tests for the "configured but never registered" shell-hook warning.

``register_from_config()`` is the only path that wires the ``hooks:`` block
onto the plugin manager, and only the CLI and gateway entry points call it.
Embedders driving ``AIAgent`` directly get a silently inert ``hooks:`` block.
``warn_if_configured_but_unregistered()`` makes that state observable.
"""

from __future__ import annotations

import logging

import pytest

from agent import shell_hooks


@pytest.fixture(autouse=True)
def _reset_registration_state():
    shell_hooks.reset_for_tests()
    yield
    shell_hooks.reset_for_tests()


def _cfg(command: str = "/tmp/hook.sh") -> dict:
    return {"hooks": {"post_tool_call": [{"command": command}]}}


def test_warns_when_hooks_configured_but_nothing_registered(caplog):
    with caplog.at_level(logging.WARNING, logger=shell_hooks.logger.name):
        assert shell_hooks.warn_if_configured_but_unregistered(_cfg()) is True

    assert "post_tool_call" in caplog.text
    # The warning must name the fix, not just the symptom.
    assert "register_from_config" in caplog.text


def test_warning_is_one_shot_per_process(caplog):
    assert shell_hooks.warn_if_configured_but_unregistered(_cfg()) is True

    # caplog accumulates across the whole test, so drop the first warning
    # before asserting the second call stays silent.
    caplog.clear()

    with caplog.at_level(logging.WARNING, logger=shell_hooks.logger.name):
        assert shell_hooks.warn_if_configured_but_unregistered(_cfg()) is False

    assert caplog.text == ""


def test_silent_when_no_hooks_configured():
    assert shell_hooks.warn_if_configured_but_unregistered({}) is False
    assert shell_hooks.warn_if_configured_but_unregistered({"hooks": {}}) is False


def test_silent_when_registration_already_happened():
    # Simulate a CLI/gateway boot having registered a hook.
    shell_hooks._registered.add(("post_tool_call", None, "/tmp/hook.sh"))

    assert shell_hooks.warn_if_configured_but_unregistered(_cfg()) is False


def test_silent_in_safe_mode(monkeypatch):
    # Safe mode skips registration deliberately; warning there is noise.
    monkeypatch.setenv("HERMES_SAFE_MODE", "1")

    assert shell_hooks.warn_if_configured_but_unregistered(_cfg()) is False


def test_never_raises_on_unloadable_config(monkeypatch):
    def _boom():
        raise RuntimeError("no config here")

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _boom)

    # cfg=None takes the config-loading path; a failure must degrade to a
    # no-op rather than break agent construction.
    assert shell_hooks.warn_if_configured_but_unregistered() is False
