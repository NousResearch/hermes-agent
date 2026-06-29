from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "ai-partner-os"


def load_core():
    package_name = "ai_partner_os_test_plugin"
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]

    package = types.ModuleType(package_name)
    package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.core",
        PLUGIN_DIR / "core.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name
    sys.modules[f"{package_name}.core"] = module
    spec.loader.exec_module(module)
    return module


def test_bridge_start_rejects_public_host_without_confirmation():
    core = load_core()

    payload = core.start_bridge({"host": "0.0.0.0"})

    assert payload["ok"] is False
    assert payload["confirmation_required"] is True
    assert "noauth WebSocket" in payload["reason"]


def test_bridge_start_rejects_tailscale_bind_without_confirmation():
    core = load_core()

    payload = core.start_bridge({"tailscale": True})

    assert payload["ok"] is False
    assert payload["confirmation_required"] is True
    assert payload["host"] == "0.0.0.0"
