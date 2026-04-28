"""Regression tests for /model support of config.yaml custom_providers.

The terminal `hermes model` flow already exposes `custom_providers`, but the
shared slash-command pipeline (`/model` in CLI/gateway/Telegram) historically
only looked at `providers:`.
"""

import hermes_cli.providers as providers_mod
from hermes_cli.model_switch import list_authenticated_providers, switch_model
from hermes_cli.providers import resolve_provider_full


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def test_list_authenticated_providers_includes_custom_providers(monkeypatch):
    """No-args /model menus should include saved custom_providers entries."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="openai-codex",
        user_providers={},
        custom_providers=[
            {
                "name": "Local (127.0.0.1:4141)",
                "base_url": "http://127.0.0.1:4141/v1",
                "model": "rotator-openrouter-coding",
            }
        ],
        max_models=50,
    )

    assert any(
        p["slug"] == "custom:local-(127.0.0.1:4141)"
        and p["name"] == "Local (127.0.0.1:4141)"
        and p["models"] == ["rotator-openrouter-coding"]
        and p["api_url"] == "http://127.0.0.1:4141/v1"
        for p in providers
    )


def test_resolve_provider_full_finds_named_custom_provider():
    """Explicit /model --provider should resolve saved custom_providers entries."""
    resolved = resolve_provider_full(
        "custom:local-(127.0.0.1:4141)",
        user_providers={},
        custom_providers=[
            {
                "name": "Local (127.0.0.1:4141)",
                "base_url": "http://127.0.0.1:4141/v1",
            }
        ],
    )

    assert resolved is not None
    assert resolved.id == "custom:local-(127.0.0.1:4141)"
    assert resolved.name == "Local (127.0.0.1:4141)"
    assert resolved.base_url == "http://127.0.0.1:4141/v1"
    assert resolved.source == "user-config"


def test_switch_model_accepts_explicit_named_custom_provider(monkeypatch):
    """Shared /model switch pipeline should accept --provider for custom_providers."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "no-key-required",
            "base_url": "http://127.0.0.1:4141/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)

    result = switch_model(
        raw_input="rotator-openrouter-coding",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        current_base_url="https://chatgpt.com/backend-api/codex",
        current_api_key="",
        explicit_provider="custom:local-(127.0.0.1:4141)",
        user_providers={},
        custom_providers=[
            {
                "name": "Local (127.0.0.1:4141)",
                "base_url": "http://127.0.0.1:4141/v1",
                "model": "rotator-openrouter-coding",
            }
        ],
    )

    assert result.success is True
    assert result.target_provider == "custom:local-(127.0.0.1:4141)"
    assert result.provider_label == "Local (127.0.0.1:4141)"
    assert result.new_model == "rotator-openrouter-coding"
    assert result.base_url == "http://127.0.0.1:4141/v1"
    assert result.api_key == "no-key-required"


def test_list_groups_same_name_custom_providers_into_one_row(monkeypatch):
    """Multiple custom_providers entries sharing a name should produce one row
    with all models collected, not N duplicate rows."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="openrouter",
        user_providers={},
        custom_providers=[
            {"name": "Ollama Cloud", "base_url": "https://ollama.com/v1", "model": "qwen3-coder:480b-cloud"},
            {"name": "Ollama Cloud", "base_url": "https://ollama.com/v1", "model": "glm-5.1:cloud"},
            {"name": "Ollama Cloud", "base_url": "https://ollama.com/v1", "model": "kimi-k2.5"},
            {"name": "Ollama Cloud", "base_url": "https://ollama.com/v1", "model": "minimax-m2.7:cloud"},
            {"name": "Moonshot", "base_url": "https://api.moonshot.ai/v1", "model": "kimi-k2-thinking"},
        ],
        max_models=50,
    )

    ollama_rows = [p for p in providers if p["name"] == "Ollama Cloud"]
    assert len(ollama_rows) == 1, f"Expected 1 Ollama Cloud row, got {len(ollama_rows)}"
    assert ollama_rows[0]["models"] == [
        "qwen3-coder:480b-cloud", "glm-5.1:cloud", "kimi-k2.5", "minimax-m2.7:cloud"
    ]
    assert ollama_rows[0]["total_models"] == 4

    moonshot_rows = [p for p in providers if p["name"] == "Moonshot"]
    assert len(moonshot_rows) == 1
    assert moonshot_rows[0]["models"] == ["kimi-k2-thinking"]


def test_list_deduplicates_same_model_in_group(monkeypatch):
    """Duplicate model entries under the same provider name should not produce
    duplicate entries in the models list."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="openrouter",
        user_providers={},
        custom_providers=[
            {"name": "MyProvider", "base_url": "http://localhost:11434/v1", "model": "llama3"},
            {"name": "MyProvider", "base_url": "http://localhost:11434/v1", "model": "llama3"},
            {"name": "MyProvider", "base_url": "http://localhost:11434/v1", "model": "mistral"},
        ],
        max_models=50,
    )

    my_rows = [p for p in providers if p["name"] == "MyProvider"]
    assert len(my_rows) == 1
    assert my_rows[0]["models"] == ["llama3", "mistral"]
    assert my_rows[0]["total_models"] == 2


def test_list_enumerates_dict_format_models_alongside_default(monkeypatch):
    """custom_providers entry with dict-format ``models:`` plus singular
    ``model:`` should surface the default and every dict key.

    Regression: Hermes's own writer stores configured models as a dict
    keyed by model id, but the /model picker previously only honored the
    singular ``model:`` field, so multi-model custom providers appeared
    to have only the active model.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="openai-codex",
        user_providers={},
        custom_providers=[
            {
                "name": "DeepSeek",
                "base_url": "https://api.deepseek.com",
                "api_mode": "chat_completions",
                "model": "deepseek-chat",
                "models": {
                    "deepseek-chat": {"context_length": 128000},
                    "deepseek-reasoner": {"context_length": 128000},
                },
            }
        ],
        max_models=50,
    )

    ds_rows = [p for p in providers if p["name"] == "DeepSeek"]
    assert len(ds_rows) == 1
    assert ds_rows[0]["models"] == ["deepseek-chat", "deepseek-reasoner"]
    assert ds_rows[0]["total_models"] == 2


def test_list_enumerates_dict_format_models_without_singular_model(monkeypatch):
    """Dict-format ``models:`` with no singular ``model:`` should still
    enumerate every dict key (previously the picker reported 0 models)."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="openai-codex",
        user_providers={},
        custom_providers=[
            {
                "name": "Thor",
                "base_url": "http://thor.lab:8337/v1",
                "models": {
                    "gemma-4-26B-A4B-it-MXFP4_MOE": {"context_length": 262144},
                    "Qwen3.5-35B-A3B-MXFP4_MOE": {"context_length": 262144},
                    "gemma-4-31B-it-Q4_K_M": {"context_length": 262144},
                },
            }
        ],
        max_models=50,
    )

    thor_rows = [p for p in providers if p["name"] == "Thor"]
    assert len(thor_rows) == 1
    assert set(thor_rows[0]["models"]) == {
        "gemma-4-26B-A4B-it-MXFP4_MOE",
        "Qwen3.5-35B-A3B-MXFP4_MOE",
        "gemma-4-31B-it-Q4_K_M",
    }
    assert thor_rows[0]["total_models"] == 3


def test_list_dedupes_dict_model_matching_singular_default(monkeypatch):
    """When the singular ``model:`` is also a key in the ``models:`` dict,
    it must appear exactly once in the picker."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="openai-codex",
        user_providers={},
        custom_providers=[
            {
                "name": "DeepSeek",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "models": {
                    "deepseek-chat": {"context_length": 128000},
                    "deepseek-reasoner": {"context_length": 128000},
                },
            }
        ],
        max_models=50,
    )

    ds_rows = [p for p in providers if p["name"] == "DeepSeek"]
    assert ds_rows[0]["models"].count("deepseek-chat") == 1
    assert ds_rows[0]["models"] == ["deepseek-chat", "deepseek-reasoner"]



# ─────────────────────────────────────────────────────────────────────────────
# #9210: group custom_providers by (base_url, api_key) in /model picker
# ─────────────────────────────────────────────────────────────────────────────

def test_list_authenticated_providers_groups_same_endpoint(monkeypatch):
    """Multiple custom_providers entries sharing a base_url+api_key must be
    returned as a single picker row with all their models merged."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom",
        current_base_url="http://localhost:11434/v1",
        user_providers={},
        custom_providers=[
            {"name": "Ollama — MiniMax M2.7", "base_url": "http://localhost:11434/v1",
             "api_key": "ollama", "model": "minimax-m2.7"},
            {"name": "Ollama — GLM 5.1",      "base_url": "http://localhost:11434/v1",
             "api_key": "ollama", "model": "glm-5.1"},
            {"name": "Ollama — Qwen3-coder", "base_url": "http://localhost:11434/v1",
             "api_key": "ollama", "model": "qwen3-coder"},
        ],
        max_models=50,
    )

    custom_groups = [p for p in providers if p.get("is_user_defined")]
    assert len(custom_groups) == 1, (
        "Expected 1 group for shared endpoint, got "
        f"{[p['slug'] for p in custom_groups]}"
    )
    group = custom_groups[0]
    assert set(group["models"]) == {"minimax-m2.7", "glm-5.1", "qwen3-coder"}
    assert group["total_models"] == 3
    # Per-model suffix stripped from display name
    assert group["name"] == "Ollama"


def test_list_authenticated_providers_current_endpoint_uses_current_slug(monkeypatch):
    """When current_base_url matches the grouped endpoint, the slug must
    equal current_provider so picker selection routes through the live
    credential pipeline."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom",
        current_base_url="http://localhost:11434/v1",
        user_providers={},
        custom_providers=[
            {"name": "Ollama — GLM 5.1", "base_url": "http://localhost:11434/v1",
             "api_key": "ollama", "model": "glm-5.1"},
        ],
        max_models=50,
    )

    matches = [p for p in providers if p.get("is_user_defined")]
    assert len(matches) == 1
    group = matches[0]
    assert group["slug"] == "custom"
    assert group["is_current"] is True


def test_list_authenticated_providers_distinct_endpoints_stay_separate(monkeypatch):
    """Entries with different base_urls must produce separate picker rows
    even if some display names happen to be similar."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        user_providers={},
        custom_providers=[
            {"name": "Ollama — GLM 5.1", "base_url": "http://localhost:11434/v1",
             "api_key": "ollama", "model": "glm-5.1"},
            {"name": "Moonshot", "base_url": "https://api.moonshot.cn/v1",
             "api_key": "sk-m", "model": "moonshot-v1"},
            {"name": "Ollama — Qwen3-coder", "base_url": "http://localhost:11434/v1",
             "api_key": "ollama", "model": "qwen3-coder"},
        ],
        max_models=50,
    )

    custom_groups = [p for p in providers if p.get("is_user_defined")]
    assert len(custom_groups) == 2
    # Ollama endpoint collapses to one row with both models
    ollama = next(p for p in custom_groups if p["name"] == "Ollama")
    assert set(ollama["models"]) == {"glm-5.1", "qwen3-coder"}
    moonshot = next(p for p in custom_groups if p["name"] == "Moonshot")
    assert moonshot["models"] == ["moonshot-v1"]


def test_list_authenticated_providers_same_url_different_keys_disambiguated(monkeypatch):
    """Two custom_providers entries with the same base_url but different
    api_keys (and identical cleaned names) must both stay visible in the
    picker — slug is suffixed to disambiguate."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        user_providers={},
        custom_providers=[
            {"name": "OpenAI — key A", "base_url": "https://api.openai.com/v1",
             "api_key": "sk-AAA", "model": "gpt-5.4"},
            {"name": "OpenAI — key B", "base_url": "https://api.openai.com/v1",
             "api_key": "sk-BBB", "model": "gpt-4.6"},
        ],
        max_models=50,
    )

    custom_groups = [p for p in providers if p.get("is_user_defined")]
    assert len(custom_groups) == 2
    slugs = sorted(p["slug"] for p in custom_groups)
    # First group keeps the base slug, second gets a numeric suffix
    assert slugs == ["custom:openai", "custom:openai-2"]
    # Each row has a distinct model
    models = {p["slug"]: p["models"] for p in custom_groups}
    assert models["custom:openai"] == ["gpt-5.4"]
    assert models["custom:openai-2"] == ["gpt-4.6"]


def test_list_authenticated_providers_total_models_reflects_grouped_count(monkeypatch):
    """After grouping six entries into one row, total_models must reflect
    the full count, and every grouped model appears in the list."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    entries = [
        {"name": f"Ollama \u2014 Model {i}", "base_url": "http://localhost:11434/v1",
         "api_key": "ollama", "model": f"model-{i}"}
        for i in range(6)
    ]
    providers = list_authenticated_providers(
        user_providers={},
        custom_providers=entries,
        max_models=4,
    )

    groups = [p for p in providers if p.get("is_user_defined")]
    assert len(groups) == 1
    group = groups[0]
    assert group["total_models"] == 6
    # All six models are preserved in the grouped row.
    assert sorted(group["models"]) == sorted(f"model-{i}" for i in range(6))
