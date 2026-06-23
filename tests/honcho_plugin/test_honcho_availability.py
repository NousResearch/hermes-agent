"""Availability checks for the Honcho memory provider."""

from __future__ import annotations

import builtins
from unittest.mock import MagicMock, patch

from plugins.memory.honcho import HonchoMemoryProvider


def test_is_available_false_when_honcho_dependency_missing():
    provider = HonchoMemoryProvider()
    cfg = MagicMock(enabled=True, api_key="honcho-test-key", base_url="")
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "honcho":
            raise ImportError("No module named 'honcho'")
        return real_import(name, globals, locals, fromlist, level)

    with patch(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        return_value=cfg,
    ), patch("builtins.__import__", side_effect=fake_import):
        assert provider.is_available() is False
