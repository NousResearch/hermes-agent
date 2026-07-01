"""Regression tests: NVIDIA NIM curated model list includes all reported models.

Issue #48627: minimax-m3 and minimax-m2.7 were missing from the Desktop model
selector because they weren't in the nvidia curated list in hermes_cli/models.py.
"""

from hermes_cli.models import _PROVIDER_MODELS


class TestNvidiaNimCuratedModels:
    """Verify the NIM curated list includes third-party agentic models."""

    def test_minimax_m3_in_nvidia_curated(self):
        """minimaxai/minimax-m3 must appear in the NIM curated list."""
        assert "minimaxai/minimax-m3" in _PROVIDER_MODELS["nvidia"]

    def test_minimax_m27_in_nvidia_curated(self):
        """minimaxai/minimax-m2.7 must appear in the NIM curated list."""
        assert "minimaxai/minimax-m2.7" in _PROVIDER_MODELS["nvidia"]

    def test_minimax_m25_in_nvidia_curated(self):
        """minimaxai/minimax-m2.5 must still appear (existing entry)."""
        assert "minimaxai/minimax-m2.5" in _PROVIDER_MODELS["nvidia"]

    def test_nvidia_curated_has_third_party_section(self):
        """NIM curated list should include third-party models like qwen, deepseek."""
        nvidia = _PROVIDER_MODELS["nvidia"]
        assert any(m.startswith("qwen/") for m in nvidia)
        assert any(m.startswith("deepseek-ai/") for m in nvidia)
        assert any(m.startswith("minimaxai/") for m in nvidia)
