"""An explicitly requested provider is not rerouted by the main model's local endpoint.

``model.provider: auto`` + a local ``model.base_url`` (Ollama, LM Studio, a vLLM box) is the main
model's route. Every caller that names a provider of its own (``--provider``, delegation, a
fallback entry, a cron job, an auxiliary pin) must reach that provider, not OpenRouter.
"""

import pytest

from hermes_cli.runtime_provider import resolve_runtime_provider

LOCAL_URL = "http://localhost:11434/v1"


@pytest.fixture
def local_main_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"model:\n  provider: auto\n  default: qwen3:8b\n  base_url: {LOCAL_URL}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key")
    return home


def test_explicit_provider_not_hijacked_by_local_config_base_url(local_main_home):
    resolved = resolve_runtime_provider(requested="gemini", target_model="gemini-2.5-flash")

    assert resolved["provider"] == "gemini"
    assert "generativelanguage.googleapis.com" in resolved["base_url"]
    assert resolved["api_key"] == "AIza-test-key"


def test_unset_provider_still_honors_local_config_base_url(local_main_home):
    resolved = resolve_runtime_provider(requested=None)

    assert resolved["base_url"] == LOCAL_URL
    assert resolved["api_key"] != "sk-or-test-key"
