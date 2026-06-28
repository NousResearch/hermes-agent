"""Tests for the freellmapi integration plugin (CLI/core)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_DIR = _REPO_ROOT / "plugins" / "freellmapi"


def _load_core():
    module_name = "freellmapi_plugin_core_test"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name,
        _PLUGIN_DIR / "core.py",
        submodule_search_locations=[str(_PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def core_module():
    return _load_core()


def test_setup_enables_plugin_and_fallback(tmp_path, monkeypatch, core_module):
    home = tmp_path / ".hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_text("model:\n  provider: opencode-zen\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(core_module, "_read_env_key", lambda _name: "freellmapi-test-key")

    with patch.object(core_module, "_probe_models", return_value={"ok": True, "model_count": 3}):
        payload = core_module.setup(apply_model=False)

    saved = config_path.read_text(encoding="utf-8")
    assert "freellmapi" in saved
    assert payload["changes"]
    assert payload["status"]["plugin_enabled"] is True


def test_doctor_fails_without_api_key(tmp_path, monkeypatch, core_module):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  enabled:\n    - freellmapi\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(core_module, "_read_env_key", lambda _name: "")

    payload = core_module.doctor()
    assert payload["ok"] is False
    names = [check["name"] for check in payload["checks"]]
    assert "api_key" in names


def test_doctor_passes_with_mock_probe(tmp_path, monkeypatch, core_module):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - freellmapi\nmodel:\n  provider: freellmapi\n  default: auto\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(core_module, "_read_env_key", lambda _name: "freellmapi-deadbeef")

    with patch.object(
        core_module,
        "_probe_models",
        return_value={"ok": True, "model_count": 5, "models": ["auto", "gemini-2.5-flash"]},
    ):
        payload = core_module.doctor()

    assert payload["ok"] is True
    probe_check = next(c for c in payload["checks"] if c["name"] == "models_probe")
    assert probe_check["ok"] is True


def test_tailscale_base_url_from_env(tmp_path, monkeypatch, core_module):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").write_text("TAILSCALE_DNS_NAME=downl.taile4f666.ts.net\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert core_module._resolve_base_url() == "https://downl.taile4f666.ts.net/freellmapi/v1"


def test_register_exposes_cli():
    module_name = "freellmapi_plugin_init_test"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name,
        _PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(_PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    class Ctx:
        def __init__(self):
            self.cli_commands = []

        def register_cli_command(self, **kwargs):
            self.cli_commands.append(kwargs)

    ctx = Ctx()
    module.register(ctx)
    assert ctx.cli_commands[0]["name"] == "freellmapi"
