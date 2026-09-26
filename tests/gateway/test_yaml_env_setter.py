"""Tests for ``gateway.platforms._shared.yaml_env_setter``.

The YAML→env bridge is first-writer-wins by design (explicit env beats YAML;
multiplex safety, #80099) — but a conflicting inherited value used to discard
the user's explicit config SILENTLY. The setter now logs a warning on conflict
(#110295) while preserving the precedence.
"""

import logging
import os
from unittest.mock import patch

import pytest

from gateway.platforms._shared import yaml_env_setter


@pytest.fixture
def setter():
    with patch("gateway.platforms._shared.profile_scoped", return_value=False):
        yield yaml_env_setter()


def test_sets_env_when_unset(setter, monkeypatch):
    monkeypatch.delenv("HERMES_TEST_BRIDGE_VAR", raising=False)
    setter("HERMES_TEST_BRIDGE_VAR", "false")
    assert os.environ["HERMES_TEST_BRIDGE_VAR"] == "false"


def test_explicit_env_wins_but_warns_on_conflict(setter, monkeypatch, caplog):
    # The issue's repro: TELEGRAM_REACTIONS=true inherited, config says false.
    monkeypatch.setenv("HERMES_TEST_BRIDGE_VAR", "true")
    with caplog.at_level(logging.WARNING, logger="gateway.platforms._shared"):
        setter("HERMES_TEST_BRIDGE_VAR", "false")
    assert os.environ["HERMES_TEST_BRIDGE_VAR"] == "true"  # precedence preserved
    assert any(
        "HERMES_TEST_BRIDGE_VAR" in r.getMessage() and "already" in r.getMessage()
        for r in caplog.records
    ), "expected a skip warning naming the variable"


def test_no_warning_when_values_agree(setter, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_TEST_BRIDGE_VAR", "false")
    with caplog.at_level(logging.WARNING, logger="gateway.platforms._shared"):
        setter("HERMES_TEST_BRIDGE_VAR", "false")
    assert os.environ["HERMES_TEST_BRIDGE_VAR"] == "false"
    assert not [r for r in caplog.records if "HERMES_TEST_BRIDGE_VAR" in r.getMessage()]


def test_none_value_skipped_silently(setter, monkeypatch, caplog):
    monkeypatch.delenv("HERMES_TEST_BRIDGE_VAR", raising=False)
    with caplog.at_level(logging.WARNING, logger="gateway.platforms._shared"):
        setter("HERMES_TEST_BRIDGE_VAR", None)
    assert "HERMES_TEST_BRIDGE_VAR" not in os.environ
    assert not caplog.records


def test_list_values_comma_joined(setter, monkeypatch):
    monkeypatch.delenv("HERMES_TEST_BRIDGE_VAR", raising=False)
    setter("HERMES_TEST_BRIDGE_VAR", ["a", "b"])
    assert os.environ["HERMES_TEST_BRIDGE_VAR"] == "a,b"


def test_multiplex_scope_skips_write(setter, monkeypatch, caplog):
    # Under a secondary profile's scope the setter must not touch process env at all.
    monkeypatch.delenv("HERMES_TEST_BRIDGE_VAR", raising=False)
    with patch("gateway.platforms._shared.profile_scoped", return_value=True):
        scoped = yaml_env_setter()
    with caplog.at_level(logging.WARNING, logger="gateway.platforms._shared"):
        scoped("HERMES_TEST_BRIDGE_VAR", "false")
    assert "HERMES_TEST_BRIDGE_VAR" not in os.environ
    assert not caplog.records
