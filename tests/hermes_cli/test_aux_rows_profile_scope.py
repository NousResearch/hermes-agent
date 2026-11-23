"""``GET /api/model/auxiliary?profile=B`` judges each pin with profile B's provider plugins.

``local_endpoint`` tells a free local pin from a paid one, and an external-process provider is never
local. The process-provider plugin here is installed only in profile ``b``: reading the launch
profile's registry would mark ``b``'s ``process://`` pin local and hide the paid-pin warning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hermes_cli.plugins as plugins_mod
from hermes_cli.plugins import PluginManager
from hermes_cli.web_routers.models import get_auxiliary_models

PROVIDER = "scope-fixture"
PIN = {"provider": PROVIDER, "model": "m", "base_url": "process://scope-fixture"}


@pytest.fixture
def plugin_only_in_b(tmp_path, monkeypatch):
    home_a = tmp_path / ".hermes"
    home_b = home_a / "profiles" / "b"
    plugin = home_b / "plugins" / PROVIDER
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(json.dumps(
        {"name": PROVIDER, "kind": "model-provider", "version": "0.1.0", "description": "fixture"}))
    (plugin / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        f"register_provider(ProviderProfile(name='{PROVIDER}', auth_type='external_process', "
        "base_url='process://scope-fixture'))\n")
    for home in (home_a, home_b):
        (home / "config.yaml").write_text(json.dumps({"auxiliary": {"compression": PIN}}))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: tmp_path / "empty-bundled")
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])
    plugins_mod._reset_plugin_managers_for_tests()
    yield
    plugins_mod._reset_plugin_managers_for_tests()


def _compression_row(profile):
    return next(row for row in get_auxiliary_models(profile=profile)["tasks"] if row["task"] == "compression")


def test_pin_is_judged_with_the_selected_profiles_plugins(plugin_only_in_b):
    assert _compression_row("b")["local_endpoint"] is False
    # Control: the launch profile has no such provider, so its dot-less URL still reads as a LAN host.
    assert _compression_row(None)["local_endpoint"] is True
