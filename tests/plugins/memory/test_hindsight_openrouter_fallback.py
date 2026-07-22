"""Regression tests for embedded Hindsight OpenRouter credential handling."""

import stat
import sys
from types import SimpleNamespace

from plugins.memory.hindsight import (
    HindsightMemoryProvider,
    _build_embedded_profile_env,
    _materialize_embedded_profile_env,
)


def test_embedded_profile_env_uses_openrouter_key_when_dedicated_key_is_missing(
    monkeypatch,
):
    monkeypatch.delenv("HINDSIGHT_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")

    env = _build_embedded_profile_env({
        "llm_provider": "openrouter",
        "llm_model": "deepseek/deepseek-v4-flash",
    })

    assert env["HINDSIGHT_API_LLM_API_KEY"] == "openrouter-test-key"


def test_embedded_profile_env_uses_deepseek_key_for_deepseek_compatible_url(
    monkeypatch,
):
    monkeypatch.delenv("HINDSIGHT_LLM_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")

    env = _build_embedded_profile_env({
        "llm_provider": "openai_compatible",
        "llm_base_url": "https://api.deepseek.com/v1",
        "llm_model": "deepseek-v4-flash",
    })

    assert env["HINDSIGHT_API_LLM_API_KEY"] == "deepseek-test-key"


def test_materialized_embedded_profile_env_is_owner_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    profile_env = _materialize_embedded_profile_env({
        "profile": "hermes-test",
        "llm_provider": "openrouter",
        "llm_api_key": "test-key",
        "llm_model": "deepseek/deepseek-v4-flash",
    })

    assert stat.S_IMODE(profile_env.stat().st_mode) == 0o600


def test_get_client_uses_openrouter_key_when_dedicated_key_is_missing(monkeypatch):
    captured = {}

    class FakeHindsightEmbedded:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "hindsight",
        SimpleNamespace(HindsightEmbedded=FakeHindsightEmbedded),
    )
    monkeypatch.setattr(
        "plugins.memory.hindsight._check_local_runtime",
        lambda: (True, ""),
    )
    monkeypatch.delenv("HINDSIGHT_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")

    provider = HindsightMemoryProvider()
    provider._mode = "local_embedded"
    provider._config = {
        "profile": "hermes",
        "llm_provider": "openrouter",
        "llm_model": "deepseek/deepseek-v4-flash",
        "idle_timeout": 0,
    }
    provider._llm_base_url = "https://openrouter.ai/api/v1"

    provider._get_client()

    assert captured["llm_api_key"] == "openrouter-test-key"
