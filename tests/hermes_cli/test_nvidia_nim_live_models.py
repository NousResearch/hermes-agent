"""Regression tests for #47977 — NVIDIA NIM picker must prefer the live
``/v1/models`` catalog over the models.dev registry snapshot.

Before this fix, NVIDIA NIM had no entry in ``_SPECIAL_MODEL_LISTS``
(``hermes_cli/model_setup_flows.py``), so it fell into the generic
resolution order that tries models.dev first whenever models.dev returns
anything — regardless of whether those entries are still live on NIM. NIM
retires models (HTTP 410) and returns HTTP 404 for absent ones frequently
enough that this produced a picker where the majority of entries were
dead, with an EOL model selected as the default (row 0).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

from hermes_cli.config import load_config


@pytest.fixture(autouse=True)
def _clear_nvidia_env(monkeypatch):
    for key in ("NVIDIA_API_KEY", "NVIDIA_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


class TestNvidiaSpecialModelList:
    def test_registered_in_special_model_lists(self):
        from hermes_cli.model_setup_flows import _SPECIAL_MODEL_LISTS, _nvidia_models

        assert _SPECIAL_MODEL_LISTS["nvidia"] is _nvidia_models

    def test_live_catalog_used_even_when_models_dev_has_entries(self):
        """models.dev returning results must NOT preempt a successful live
        probe — this is the exact defect from #47977."""
        from hermes_cli.model_setup_flows import _nvidia_models
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY["nvidia"]
        live_models = ["nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3-super-120b-a12b"]

        with patch(
            "hermes_cli.models.fetch_api_models",
            return_value=live_models,
        ) as fetch_mock, patch(
            "hermes_cli.model_setup_flows._models_dev_merged",
        ) as mdev_mock:
            result = _nvidia_models(
                pconfig,
                ["nvidia/llama-3.3-nemotron-super-49b-v1"],  # stale curated floor
                "nvidia-test-key",
                "https://integrate.api.nvidia.com/v1",
            )

        fetch_mock.assert_called_once()
        mdev_mock.assert_not_called()
        assert result == live_models

    def test_falls_back_to_models_dev_when_live_probe_fails(self):
        from hermes_cli.model_setup_flows import _nvidia_models
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY["nvidia"]

        with patch(
            "hermes_cli.models.fetch_api_models",
            return_value=None,
        ), patch(
            "hermes_cli.model_setup_flows._models_dev_merged",
            return_value=["nvidia/nemotron-3-ultra-550b-a55b"],
        ) as mdev_mock:
            result = _nvidia_models(pconfig, [], "nvidia-test-key", "https://integrate.api.nvidia.com/v1")

        mdev_mock.assert_called_once()
        assert result == ["nvidia/nemotron-3-ultra-550b-a55b"]

    def test_falls_back_to_curated_when_live_and_models_dev_both_fail(self):
        from hermes_cli.model_setup_flows import _nvidia_models
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY["nvidia"]
        curated = ["nvidia/nemotron-3-ultra-550b-a55b"]

        with patch(
            "hermes_cli.models.fetch_api_models",
            return_value=None,
        ), patch(
            "hermes_cli.model_setup_flows._models_dev_merged",
            return_value=[],
        ):
            result = _nvidia_models(pconfig, curated, "nvidia-test-key", "https://integrate.api.nvidia.com/v1")

        assert result == curated


class TestNvidiaSetupFlowPersistsLiveSelection:
    def test_model_flow_api_key_provider_persists_nvidia_selection(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-test-key")

        with patch(
            "hermes_cli.models.fetch_api_models",
            return_value=["nvidia/nemotron-3-ultra-550b-a55b"],
        ), patch(
            "hermes_cli.auth._prompt_model_selection",
            return_value="nvidia/nemotron-3-ultra-550b-a55b",
        ), patch(
            "hermes_cli.auth.deactivate_provider",
        ), patch(
            "builtins.input",
            return_value="",
        ):
            from hermes_cli.model_setup_flows import _model_flow_api_key_provider

            _model_flow_api_key_provider(load_config(), "nvidia", "old-model")

        import yaml
        from hermes_constants import get_hermes_home

        config = yaml.safe_load((get_hermes_home() / "config.yaml").read_text()) or {}
        model_cfg = config.get("model")
        assert isinstance(model_cfg, dict)
        assert model_cfg["provider"] == "nvidia"
        assert model_cfg["default"] == "nvidia/nemotron-3-ultra-550b-a55b"
