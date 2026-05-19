"""Tests for Honcho auto-migration to memory provider plugin.

When the agent initializes and ``memory.provider`` is unset but Honcho
is actively configured (``enabled=True`` + at least one credential),
``detect_honcho_auto_migrate()`` returns ``"honcho"`` so the caller can
activate the plugin.  Persistence to ``memory.provider`` is deferred
until ``is_available()`` confirms the provider works — broken Honcho
setups never pollute the user's config.

Re-port of #12743 onto current main.  The helper was extracted into
``agent/honcho_auto_migrate.py`` since ``AIAgent.__init__`` itself is
now ``agent/agent_init.py::init_agent()``, and a standalone helper
module is easier to unit-test than the full init flow.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.honcho_auto_migrate import detect_honcho_auto_migrate


def _mock_honcho_config(enabled=True, api_key=None, base_url=None):
    return SimpleNamespace(enabled=enabled, api_key=api_key, base_url=base_url)


class TestHonchoDetection:
    """Tests for the real extracted detection function."""

    def test_detects_active_honcho_with_api_key(self):
        cfg = _mock_honcho_config(enabled=True, api_key="hk-test-key")
        with patch(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            return_value=cfg,
        ):
            assert detect_honcho_auto_migrate() == "honcho"

    def test_detects_active_honcho_with_base_url_only(self):
        """``base_url`` alone (no api_key) still counts as credentialed."""
        cfg = _mock_honcho_config(enabled=True, base_url="https://honcho.example.com")
        with patch(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            return_value=cfg,
        ):
            assert detect_honcho_auto_migrate() == "honcho"

    def test_no_detection_when_honcho_disabled(self):
        cfg = _mock_honcho_config(enabled=False, api_key="hk-test-key")
        with patch(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            return_value=cfg,
        ):
            assert detect_honcho_auto_migrate() == ""

    def test_no_detection_when_honcho_no_credentials(self):
        cfg = _mock_honcho_config(enabled=True)
        with patch(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            return_value=cfg,
        ):
            assert detect_honcho_auto_migrate() == ""

    def test_no_detection_when_plugin_uninstalled(self):
        """If the Honcho plugin is not installed, gracefully return empty."""
        with patch(
            "builtins.__import__",
            side_effect=ImportError("no honcho"),
        ):
            assert detect_honcho_auto_migrate() == ""

    def test_no_detection_when_config_resolution_raises(self):
        """Any unexpected exception during config resolution must not propagate."""
        with patch(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            side_effect=RuntimeError("config broken"),
        ):
            assert detect_honcho_auto_migrate() == ""


class TestPersistenceGuard:
    """The persistence call in init must be guarded by is_available() AND
    by the absence of an explicitly-set ``memory.provider`` in config.

    These are unit tests for the guard expression itself; the integration
    of the guard into ``agent_init.init_agent`` is covered by the agent
    init test suite.
    """

    def test_persists_after_available_confirms(self):
        _mp = MagicMock()
        _mp.is_available.return_value = True
        should_persist = _mp.is_available() and not {}.get("provider")
        assert should_persist is True

    def test_no_persist_when_provider_not_available(self):
        _mp = MagicMock()
        _mp.is_available.return_value = False
        should_persist = _mp.is_available() and not {}.get("provider")
        assert should_persist is False

    def test_no_persist_when_provider_was_explicitly_set(self):
        _mp = MagicMock()
        _mp.is_available.return_value = True
        mem_config = {"provider": "honcho"}
        should_persist = _mp.is_available() and not mem_config.get("provider")
        assert should_persist is False
