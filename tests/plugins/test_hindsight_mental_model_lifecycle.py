"""Mental model initialization ordering and stable prompt inclusion."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import plugins.memory.hindsight as hindsight


@pytest.fixture
def setup_provider(monkeypatch):
    monkeypatch.setattr(hindsight, "_maybe_upgrade_client", lambda: None)
    monkeypatch.setattr(hindsight, "_check_local_runtime", lambda: (True, None))
    monkeypatch.setattr(hindsight, "_cloud_api_key", lambda cfg: "test-key")
    monkeypatch.setattr(hindsight, "_export_port_health_grace_timeout", lambda cfg: None)

    def make(mode="cloud", model_id="test-model", content="Profile content", memory_mode="hybrid"):
        config = {"mode": mode, "mental_model_id": model_id, "bank_id": "test-bank", "memory_mode": memory_mode}
        monkeypatch.setattr(hindsight, "_load_config", lambda: config)
        provider = hindsight.HindsightMemoryProvider()
        events = []
        async def get_model(*args, **kwargs):
            events.append("fetch")
            return SimpleNamespace(content=content)
        sdk = Mock(side_effect=get_model)
        client = SimpleNamespace(mental_models=SimpleNamespace(get_mental_model=sdk))
        monkeypatch.setattr(provider, "_get_client", lambda: client)
        monkeypatch.setattr(provider, "_start_embedded_daemon", lambda: events.append("start"))
        return provider, sdk, events
    return make


@pytest.mark.parametrize("mode", ["cloud", "local_external", "local_embedded"])
@pytest.mark.parametrize("memory_mode", ["context", "tools", "hybrid"])
def test_initialize_fetches_before_return_and_prompt_is_stable(setup_provider, mode, memory_mode):
    provider, sdk, events = setup_provider(mode=mode, memory_mode=memory_mode)
    provider.initialize("test-session")
    assert events == (["start", "fetch"] if mode == "local_embedded" else ["fetch"])
    sdk.assert_called_once_with("test-bank", "test-model", detail="content", _request_timeout=5.0)
    prompt = provider.system_prompt_block()
    assert "Profile content" in prompt and "<memory-context>" in prompt
    assert provider.system_prompt_block() == prompt
    assert sdk.call_count == 1


@pytest.mark.parametrize("model_id", ["", None, "   "])
def test_unconfigured_model_skips_fetch(setup_provider, model_id):
    provider, sdk, events = setup_provider(model_id=model_id)
    provider.initialize("test-session")
    sdk.assert_not_called()
    assert "<memory-context>" not in provider.system_prompt_block()


@pytest.mark.parametrize("content", [None, ""])
def test_empty_content_is_not_injected(setup_provider, content):
    provider, sdk, events = setup_provider(content=content)
    provider.initialize("test-session")
    assert sdk.call_count == 1
    assert "<memory-context>" not in provider.system_prompt_block()


def test_fetch_failure_does_not_fail_initialization(setup_provider):
    provider, sdk, events = setup_provider()
    sdk.side_effect = RuntimeError("test fetch failure")
    provider.initialize("test-session")
    assert provider._mental_model_content == ""
    assert "<memory-context>" not in provider.system_prompt_block()


def test_disabled_embedded_start_skips_fetch(setup_provider, monkeypatch):
    provider, sdk, events = setup_provider(mode="local_embedded")
    def disable():
        provider._mode = "disabled"
    monkeypatch.setattr(provider, "_start_embedded_daemon", disable)
    provider.initialize("test-session")
    sdk.assert_not_called()
    assert "<memory-context>" not in provider.system_prompt_block()
