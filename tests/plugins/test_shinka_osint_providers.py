from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "shinka-osint"


def load_providers():
    name = "shinka_osint_providers_test"
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PLUGIN_DIR / "providers.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_blocks_china_region_providers():
    providers = load_providers()
    with pytest.raises(ValueError, match="blocks China-region"):
        providers.assert_provider_allowed("deepseek")
    with pytest.raises(ValueError, match="blocks China-region"):
        providers.assert_provider_allowed("custom", "https://api.deepseek.com/v1")


def test_allows_western_host_for_chinese_model_slug():
    providers = load_providers()
    providers.assert_provider_allowed("openrouter", "https://openrouter.ai/api/v1")
    assert providers.host_is_western("https://integrate.api.nvidia.com/v1")


def test_build_env_overlay_without_auth():
    providers = load_providers()
    with patch.object(providers, "resolve_llm", return_value=None):
        env = providers.build_env_overlay()
    assert env["SHINKA_DISABLE_GEMINI_EMBEDDING"] == "1"
    assert env["SHINKA_LLM_AVAILABLE"] == "0"


def test_build_env_overlay_openai_codex():
    providers = load_providers()
    resolved = providers.ResolvedLLM(
        provider_id="openai-codex",
        model="gpt-5.2",
        api_key="tok",
        base_url="https://chatgpt.com/backend-api/codex",
        source="test",
        western_host=True,
    )
    with patch.object(providers, "resolve_llm", return_value=resolved):
        env = providers.build_env_overlay()
    assert env["OPENAI_API_KEY"] == "tok"
    assert env["SHINKA_HERMES_LLM_PROVIDER"] == "openai-codex"
    assert env["SHINKA_LLM_AVAILABLE"] == "1"


def test_provider_status_milspec_no_gemini_required():
    providers = load_providers()
    status = providers.provider_status()
    assert status["milspec_requires_gemini"] is False
    assert "blocked_provider_ids" in status["policy"]
