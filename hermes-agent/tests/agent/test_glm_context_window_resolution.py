"""GLM context windows: hyphenated relay slugs, provider-level custom overrides, and the aux feasibility
check inherit the right window instead of the ``glm`` 202,752 catch-all (#97398, #98387, #89500)."""
from types import SimpleNamespace

from agent import conversation_compression as cc
from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS, get_model_context_length
from hermes_cli.config_providers import get_custom_provider_context_length


def test_hyphenated_relay_slug_resolves_to_specific_catalog_entry():
    """``z-ai-glm-5-3`` must hit ``glm-5.3``, not the shorter ``glm`` family entry."""
    assert get_model_context_length("z-ai-glm-5-3", provider="custom") == DEFAULT_CONTEXT_LENGTHS["glm-5.3"]
    assert DEFAULT_CONTEXT_LENGTHS["glm-5.3"] > DEFAULT_CONTEXT_LENGTHS["glm"]


def test_entry_level_custom_provider_context_length_backs_models_without_per_model_override():
    providers = [{"name": "relay", "base_url": "https://relay.example/v1", "context_length": 1_000_000,
                  "models": {"pinned": {"context_length": 65536}}}]
    assert get_custom_provider_context_length("pinned", "https://relay.example/v1", providers) == 65536
    assert get_custom_provider_context_length("z-ai-glm-5-3", "https://relay.example/v1", providers) == 1_000_000
    assert get_custom_provider_context_length("x", "https://other.example/v1", providers) is None


def test_feasibility_check_inherits_main_window_when_aux_is_the_main_model(monkeypatch):
    """Main pinned at 1M via model.context_length; aux compression = same model on the same route → the
    threshold must NOT be auto-lowered to a re-resolved catalog value."""
    lowered = []
    monkeypatch.setattr(cc, "_lower_threshold_to_aux_context", lambda *a, **k: lowered.append(k))
    import agent.auxiliary_client as aux
    monkeypatch.setattr(aux, "_resolve_task_provider_model", lambda task: ("auto", None, None, None, None))
    client = SimpleNamespace(base_url="https://relay.example/v1/", api_key="k")
    monkeypatch.setattr(aux, "get_text_auxiliary_client", lambda task, main_runtime=None: (client, "uncatalogued-relay-model"))
    import agent.model_metadata as mm
    monkeypatch.setattr(mm, "get_model_context_length", lambda *a, **k: 128_000)
    agent = SimpleNamespace(
        compression_enabled=True, model="uncatalogued-relay-model", base_url="https://relay.example/v1", provider="custom",
        _custom_providers=None, _aux_compression_context_length_config=None,
        _current_main_runtime=lambda: None, status_callback=None, _emit_status=lambda m: None,
        context_compressor=SimpleNamespace(context_length=1_000_000, threshold_tokens=500_000),
    )
    cc.check_compression_model_feasibility(agent)
    assert lowered == []
