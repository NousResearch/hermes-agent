"""Regression test for issue #40836: Scalar gateway config crashes streaming fallback."""
import importlib.util
import sys
from pathlib import Path

import pytest

# Load config module directly from the repo
_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "gateway.config", str(_REPO / "gateway" / "config.py")
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["gateway.config"] = _mod
try:
    _spec.loader.exec_module(_mod)
except Exception:
    pytest.skip("Could not load gateway.config (missing deps), skipping", allow_module_level=True)

from gateway.config import GatewayConfig, load_gateway_config  # noqa: E402


class TestIssue40836:
    """Scalar gateway value (e.g. `gateway: disabled`) should not crash
    the streaming-fallback lookup."""

    def test_load_gateway_config_with_scalar_gateway(self, tmp_path, monkeypatch):
        """`gateway: disabled` in YAML must not raise AttributeError."""
        import os
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg_dir = tmp_path
        cfg_file = cfg_dir / "config.yaml"
        cfg_file.write_text("gateway: disabled\n")
        # Should not raise
        gw = load_gateway_config()
        assert isinstance(gw, GatewayConfig)

    def test_from_dict_scalar_gateway_value(self):
        """from_dict with gateway scalar should produce a valid GatewayConfig."""
        data = {"gateway": "disabled"}
        gw = GatewayConfig.from_dict(data)
        assert isinstance(gw, GatewayConfig)

    def test_from_dict_none_gateway(self):
        """from_dict with None gateway should work."""
        data = {"gateway": None}
        gw = GatewayConfig.from_dict(data)
        assert isinstance(gw, GatewayConfig)

    def test_from_dict_string_gateway_streaming(self):
        """Scalar gateway string with streaming key should not crash."""
        data = {"gateway": {"streaming": "not-a-dict"}}
        gw = GatewayConfig.from_dict(data)
        assert isinstance(gw, GatewayConfig)
