"""Test helper for loading Olympus' dashboard backend module by file path."""
from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "plugin_api.py"
spec = importlib.util.spec_from_file_location("olympus_plugin_api", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load Olympus plugin API from {MODULE_PATH}")

plugin_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin_api)
