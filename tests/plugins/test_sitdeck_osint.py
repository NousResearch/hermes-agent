"""Tests for sitdeck-osint plugin (no live browser)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "sitdeck-osint"


def _load_module(name: str):
    pkg = "sitdeck_osint_test_pkg"
    if pkg not in sys.modules:
        import types

        package = types.ModuleType(pkg)
        package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
        sys.modules[pkg] = package
        for stem in ("credentials", "browser_crawl", "core", "stack", "cli"):
            mod_name = f"{pkg}.{stem}"
            spec = importlib.util.spec_from_file_location(mod_name, PLUGIN_DIR / f"{stem}.py")
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = pkg
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
    return sys.modules[f"{pkg}.{name}"]


def test_normalize_email_gmail_local_part():
    creds = _load_module("credentials")
    assert creds._normalize_email("Mine0119") == "Mine0119@gmail.com"
    assert creds._normalize_email("user@example.com") == "user@example.com"


def test_credential_status_without_env(monkeypatch):
    monkeypatch.delenv("SITDECK_EMAIL", raising=False)
    monkeypatch.delenv("SITDECK_PASSWORD", raising=False)
    creds = _load_module("credentials")
    status = creds.credential_status()
    assert status["email_configured"] is False
    assert status["password_configured"] is False


def test_check_available_requires_both(monkeypatch):
    monkeypatch.setenv("SITDECK_EMAIL", "test@gmail.com")
    monkeypatch.delenv("SITDECK_PASSWORD", raising=False)
    core = _load_module("core")
    assert core.check_available() is False
    monkeypatch.setenv("SITDECK_PASSWORD", "secret")
    assert core.check_available() is True


def test_handle_status_json():
    core = _load_module("core")
    payload = json.loads(core.handle_status({}))
    assert payload["success"] is True
    assert payload["replacement_for"] == "worldmonitor_pro_mcp"


def test_build_digest_failure():
    crawl = _load_module("browser_crawl")
    text = crawl.build_digest({"success": False, "error": "missing_credentials"})
    assert "missing_credentials" in text


@patch("hermes_cli.config.load_config", return_value={})
@patch("hermes_cli.tools_config._get_platform_tools", return_value=[])
@patch("hermes_cli.tools_config._save_platform_tools")
@patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
@patch("hermes_cli.plugins_cmd._resolve_plugin_key", return_value="sitdeck-osint")
@patch("hermes_cli.plugins_cmd._save_enabled_set")
def test_setup_dry_run(
    _save_en,
    _resolve,
    _get_en,
    _save_tools,
    _get_tools,
    _load,
    monkeypatch,
):
    stack = _load_module("stack")

    def _fake_disable(*, dry_run=False):
        return {"success": True, "status": "would_disable", "dry_run": dry_run}

    monkeypatch.setattr(stack, "disable_worldmonitor_mcp", _fake_disable)
    result = stack.setup_sitdeck_stack(email="Mine0119", dry_run=True, write_env=True)
    assert result["success"] is True
    assert result["env"]["would_set"]["SITDECK_EMAIL"] == "Mine0119@gmail.com"
