"""`hermes vault sources --enable` must persist vault.<manager>.enabled = True.

Regression for the pop-as-enable path: DEFAULT_CONFIG["vault"][bitwarden|onepassword]
["enabled"] is False, so deleting the key and reloading merges the default back.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.config import load_config
from hermes_cli.vault import _cmd_sources


@pytest.fixture
def isolated_home(tmp_path):
    """Point HERMES_HOME at a temp dir so tests never touch real config."""
    (tmp_path / ".env").touch()
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


@pytest.fixture
def mock_console():
    console = MagicMock()
    with patch("hermes_cli.vault._console", return_value=console):
        yield console


def test_enable_bitwarden_persists_true_after_reload(isolated_home, mock_console):
    _cmd_sources(SimpleNamespace(enable="bitwarden", disable=None))

    enabled = load_config()["vault"]["bitwarden"]["enabled"]
    assert enabled is True


def test_disable_bitwarden_persists_false_after_reload(isolated_home, mock_console):
    _cmd_sources(SimpleNamespace(enable=None, disable="bitwarden"))

    enabled = load_config()["vault"]["bitwarden"]["enabled"]
    assert enabled is False


def test_enable_unknown_does_not_persist_manager(isolated_home, mock_console):
    _cmd_sources(SimpleNamespace(enable="unknown", disable=None))

    vault = load_config().get("vault") or {}
    assert "unknown" not in vault
    assert vault.get("bitwarden", {}).get("enabled") is not True
    assert vault.get("onepassword", {}).get("enabled") is not True
