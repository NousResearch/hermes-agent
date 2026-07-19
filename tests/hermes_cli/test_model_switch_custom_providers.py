"""Regression tests for /model support of config.yaml custom_providers.

The terminal `hermes model` flow already exposes `custom_providers`, but the
shared slash-command pipeline (`/model` in CLI/gateway/Telegram) historically
only looked at `providers:`.
"""

import hermes_cli.providers as providers_mod
import pytest
from hermes_cli.model_switch import list_authenticated_providers, switch_model
from hermes_cli.providers import resolve_provider_full


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


@pytest.fixture(autouse=True)
def _disable_live_custom_provider_model_probe(monkeypatch):
    """Keep custom-provider picker fixtures independent of local model servers."""
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *_a, **_kw: None)


def test_list_authenticated_providers_includes_custom_providers(monkeypatch):
    """No-args /model menus should include saved custom_providers entries."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a, **k: [])

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


def test_list_authenticated_providers_can_skip_custom_provider_live_probe(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    fetch = lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected probe"))
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    providers = list_authenticated_providers(
        user_providers={},
        custom_providers=[
            {
                "name": "Slow Local",
                "base_url": "http://127.0.0.1:8080/v1",
                "api_key": "sk-local",
                "model": "local-model",
            }
        ],
        probe_custom_providers=False,
    )

    row = next(p for p in providers if p["slug"] == "custom:slow-local")
    assert row["models"] == ["local-model"]
    assert row["total_models"] == 1


def test_list_authenticated_providers_can_probe_only_current_custom_provider(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    calls = []

    def fetch(api_key, api_url, **kwargs):
        calls.append(api_url)
        if api_url == "http://active.local/v1":
            return ["active-a", "active-b"]
        raise AssertionError(f"unexpected probe for {api_url}")

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    providers = list_authenticated_providers(
        current_provider="custom:active-proxy",
        user_providers={},
        custom_providers=[
            {
                "name": "Active Proxy",
                "base_url": "http://active.local/v1",
                "api_key": "sk-active",
                "model": "configured-active",
            },
            {
                "name": "Offline Proxy",
                "base_url": "http://offline.local/v1",
                "api_key": "sk-offline",
                "model": "configured-offline",
            },
        ],
        probe_custom_providers=False,
        probe_current_custom_provider=True,
    )

    active = next(p for p in providers if p["slug"] == "custom:active-proxy")
    offline = next(p for p in providers if p["slug"] == "custom:offline-proxy")
    assert calls == ["http://active.local/v1"]
    assert active["is_current"] is True
    assert active["models"] == ["active-a", "active-b"]
    assert offline["models"] == ["configured-offline"]


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


def test_list_authenticated_providers_includes_active_bare_custom_endpoint(monkeypatch):
    """Bare model.provider=custom + model.base_url should still populate /model.

    Users can configure a one-off OpenAI-compatible endpoint directly under
    ``model:`` without a named ``providers:`` or ``custom_providers:`` row.
    The gateway picker receives only the current model/base_url slice, so it
    must surface that active endpoint rather than looking like config was
    ignored.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom",
        current_base_url="https://www.ccsub.net/v1",
        current_model="gpt-4o",
        user_providers={},
        custom_providers=[],
        max_models=50,
    )

    bare_custom = next((p for p in providers if p["slug"] == "custom"), None)
    assert bare_custom is not None
    assert bare_custom["name"] == "Custom endpoint"
    assert bare_custom["is_current"] is True
    assert bare_custom["is_user_defined"] is True
    assert bare_custom["models"] == ["gpt-4o"]
    assert bare_custom["api_url"] == "https://www.ccsub.net/v1"


def test_list_authenticated_providers_can_probe_active_bare_custom_endpoint(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr(
        "hermes_cli.models.fetch_api_models",
        lambda api_key, api_url, **kwargs: ["gpt-4o", "gpt-4o-mini"],
    )

    providers = list_authenticated_providers(
        current_provider="custom",
        current_base_url="https://www.ccsub.net/v1",
        current_model="gpt-4o",
        user_providers={},
        custom_providers=[],
        probe_custom_providers=False,
        probe_current_custom_provider=True,
    )

    bare_custom = next(p for p in providers if p["slug"] == "custom")
    assert bare_custom["is_current"] is True
    assert bare_custom["models"] == ["gpt-4o", "gpt-4o-mini"]


def test_switch_model_accepts_explicit_bare_custom_current_endpoint(monkeypatch):
    """Picker selections for bare custom endpoints should route to current base_url."""
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)

    result = switch_model(
        raw_input="gpt-4o-mini",
        current_provider="custom",
        current_model="gpt-4o",
        current_base_url="https://www.ccsub.net/v1",
        current_api_key="sk-test",
        explicit_provider="custom",
        user_providers={},
        custom_providers=[],
    )

    assert result.success is True
    assert result.target_provider == "custom"
    assert result.provider_label == "Custom endpoint"
    assert result.new_model == "gpt-4o-mini"
    assert result.base_url == "https://www.ccsub.net/v1"
    assert result.api_key == "sk-test"


def test_is_aggregator_recognizes_named_custom_provider():
    assert providers_mod.is_aggregator("custom:hpc-ai") is True
    assert providers_mod.is_aggregator("custom:litellm") is True


def test_is_aggregator_leaves_unknown_provider_non_aggregator():
    assert providers_mod.is_aggregator("not-a-provider") is False


def test_is_routing_aggregator_excludes_flat_namespace_resellers():
    """opencode-go / opencode-zen stay ``is_aggregator=True`` (model-switch
    relies on it to search their flat bare-name catalog), but they are NOT
    routing aggregators — their models are first-party, so the picker dedup
    must not strip them. (#47077)"""
    # Still aggregators for model-switch flat-catalog resolution.
    assert providers_mod.is_aggregator("opencode-go") is True
    assert providers_mod.is_aggregator("opencode-zen") is True
    # But NOT routing aggregators for picker-dedup purposes.
    assert providers_mod.is_routing_aggregator("opencode-go") is False
    assert providers_mod.is_routing_aggregator("opencode-zen") is False
    # True routers and custom proxies remain routing aggregators.
    assert providers_mod.is_routing_aggregator("openrouter") is True
    assert providers_mod.is_routing_aggregator("custom:litellm") is True
    assert providers_mod.is_routing_aggregator("not-a-provider") is False


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
    fetch = lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected probe"))
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

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
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a, **k: [])

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


def test_custom_provider_no_key_singular_model_still_probes_live_models(monkeypatch):
    """A singular ``model:`` is the active selection, not an explicit catalog.

    No-key local endpoints such as Ollama and llama.cpp should still be probed
    so /model matches the terminal ``hermes model`` flow.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    calls = []

    def fake_fetch_api_models(api_key, base_url, **kwargs):
        calls.append((api_key, base_url, kwargs))
        return ["llama3", "mistral", "qwen3-coder"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)

    providers = list_authenticated_providers(
        current_provider="openai-codex",
        user_providers={},
        custom_providers=[
            {
                "name": "Local Ollama",
                "base_url": "http://localhost:11434/v1",
                "model": "llama3",
            }
        ],
        max_models=50,
    )

    assert calls == [("", "http://localhost:11434/v1", {"headers": None})]
    row = next(p for p in providers if p["name"] == "Local Ollama")
    assert row["models"] == ["llama3", "mistral", "qwen3-coder"]
    assert row["total_models"] == 3


def test_custom_provider_explicit_model_matching_default_skips_probe(monkeypatch):
    """Explicitness comes from ``models:``, even when dedup adds no item."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    calls = []

    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return ["unexpected-live-model"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    providers = list_authenticated_providers(
        current_provider="custom:local-ollama",
        user_providers={},
        custom_providers=[
            {
                "name": "Local Ollama",
                "base_url": "http://localhost:11434/v1",
                "model": "llama3",
                "models": {"llama3": {}},
            }
        ],
    )

    row = next(p for p in providers if p["name"] == "Local Ollama")
    assert calls == []
    assert row["models"] == ["llama3"]


def test_custom_provider_group_explicit_duplicate_skips_probe(monkeypatch):
    """A later grouped entry can explicitly narrow to an existing model."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    calls = []

    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return ["unexpected-live-model"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    providers = list_authenticated_providers(
        current_provider="custom:local-ollama",
        user_providers={},
        custom_providers=[
            {
                "name": "Local Ollama",
                "base_url": "http://localhost:11434/v1",
                "model": "llama3",
            },
            {
                "name": "Local Ollama",
                "base_url": "http://localhost:11434/v1",
                "models": ["llama3"],
            },
        ],
    )

    row = next(p for p in providers if p["name"] == "Local Ollama")
    assert calls == []
    assert row["models"] == ["llama3"]


def test_custom_provider_current_only_probe_respects_explicit_catalog(monkeypatch):
    """Normal GUI opens probe only the active singular-only provider."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    calls = []

    def fetch(api_key, base_url, **kwargs):
        calls.append((api_key, base_url, kwargs))
        return ["live-a", "live-b"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    providers = list_authenticated_providers(
        current_provider="custom:active",
        current_base_url="http://active.local/v1",
        user_providers={},
        custom_providers=[
            {
                "name": "Active",
                "base_url": "http://active.local/v1",
                "model": "seed",
            },
            {
                "name": "Offline",
                "base_url": "http://offline.local/v1",
                "model": "offline-seed",
            },
            {
                "name": "Static",
                "base_url": "http://static.local/v1",
                "model": "only",
                "models": ["only"],
            },
        ],
        probe_custom_providers=False,
        probe_current_custom_provider=True,
    )

    assert calls == [("", "http://active.local/v1", {"headers": None})]
    rows = {row["name"]: row for row in providers if row.get("is_user_defined")}
    assert rows["Active"]["models"] == ["live-a", "live-b"]
    assert rows["Offline"]["models"] == ["offline-seed"]
    assert rows["Static"]["models"] == ["only"]


def test_custom_provider_current_explicit_catalog_skips_probe(monkeypatch):
    """Current-only GUI probing must still honor an explicit catalog."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    calls = []

    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return ["unexpected-live-model"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    providers = list_authenticated_providers(
        current_provider="custom:static",
        current_base_url="http://static.local/v1",
        user_providers={},
        custom_providers=[
            {
                "name": "Static",
                "base_url": "http://static.local/v1",
                "model": "only",
                "models": ["only"],
            }
        ],
        probe_custom_providers=False,
        probe_current_custom_provider=True,
    )

    assert calls == []
    row = next(p for p in providers if p["name"] == "Static")
    assert row["is_current"] is True
    assert row["models"] == ["only"]


def test_custom_provider_empty_explicit_list_allows_probe(monkeypatch):
    """An empty ``models:`` declaration is not an explicit catalog."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    calls = []

    def fetch(api_key, base_url, **kwargs):
        calls.append((api_key, base_url, kwargs))
        return ["live-a", "live-b"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    providers = list_authenticated_providers(
        current_provider="custom:local",
        user_providers={},
        custom_providers=[
            {
                "name": "Local",
                "base_url": "http://local.test/v1",
                "model": "seed",
                "models": [],
            }
        ],
    )

    assert calls == [("", "http://local.test/v1", {"headers": None})]
    row = next(p for p in providers if p["name"] == "Local")
    assert row["models"] == ["live-a", "live-b"]


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
        probe_custom_providers=False,
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
    credential pipeline — provided current_provider is a real slug, not
    the corrupt bare "custom" (see #17478)."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom:ollama",
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
    assert group["slug"] == "custom:ollama"
    assert group["is_current"] is True


def test_list_authenticated_providers_bare_custom_slug_recovers(monkeypatch):
    """Regression for #17478: when a prior failed switch left the bare
    literal "custom" in model.provider, the picker must NOT propagate
    that broken slug. It must fall back to the canonical
    ``custom:<name>`` form so the picker stays usable."""
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
    # Canonical slug, NOT the bare "custom" that caused #17478
    assert group["slug"] == "custom:ollama"
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
        probe_custom_providers=False,
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


def test_list_authenticated_providers_same_url_different_key_env_and_api_mode_stay_separate(monkeypatch):
    """Same gateway host but different key_env/api_mode entries are distinct providers."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom:gpt",
        current_base_url="https://gateway.example.com",
        user_providers={},
        custom_providers=[
            {
                "name": "gpt",
                "base_url": "https://gateway.example.com",
                "key_env": "GPT_KEY",
                "api_mode": "codex_responses",
                "model": "gpt-5.5",
            },
            {
                "name": "claude",
                "base_url": "https://gateway.example.com",
                "key_env": "CLAUDE_KEY",
                "api_mode": "anthropic_messages",
                "model": "claude-opus-4-8",
            },
        ],
        max_models=50,
    )

    custom = [p for p in providers if p.get("is_user_defined")]
    by_slug = {p["slug"]: p for p in custom}

    assert set(by_slug) == {"custom:gpt", "custom:claude"}
    assert by_slug["custom:gpt"]["models"] == ["gpt-5.5"]
    assert by_slug["custom:claude"]["models"] == ["claude-opus-4-8"]
    assert by_slug["custom:gpt"]["is_current"] is True
    assert by_slug["custom:claude"]["is_current"] is False


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
        probe_custom_providers=False,
    )

    groups = [p for p in providers if p.get("is_user_defined")]
    assert len(groups) == 1
    group = groups[0]
    assert group["total_models"] == 6
    # All six models are preserved in the grouped row.
    assert sorted(group["models"]) == sorted(f"model-{i}" for i in range(6))


def test_lmstudio_picker_probes_active_config_base_url(monkeypatch):
    """When `provider: lmstudio` is saved with a remote base_url and no
    LM_BASE_URL env var, the picker must probe the saved base_url — not
    127.0.0.1. Regression: prior behavior always probed localhost, so users
    with LM Studio on a lab box saw the wrong (or empty) model list.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.delenv("LM_BASE_URL", raising=False)
    monkeypatch.delenv("LM_API_KEY", raising=False)

    captured: dict = {}

    def _fake_fetch(api_key=None, base_url=None, timeout=5.0):
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        return ["qwen/qwen3-coder-30b"]

    monkeypatch.setattr("hermes_cli.models.fetch_lmstudio_models", _fake_fetch)

    list_authenticated_providers(
        current_provider="lmstudio",
        current_base_url="http://192.168.1.10:1234/v1",
        current_model="qwen/qwen3-coder-30b",
    )

    assert captured["base_url"] == "http://192.168.1.10:1234/v1"


def test_lmstudio_picker_lm_base_url_env_wins_over_active_config(monkeypatch):
    """LM_BASE_URL env var must still take precedence over the saved
    base_url so users can temporarily redirect the picker without editing
    config.yaml.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setenv("LM_BASE_URL", "http://override.local:9999/v1")
    monkeypatch.delenv("LM_API_KEY", raising=False)

    captured: dict = {}

    def _fake_fetch(api_key=None, base_url=None, timeout=5.0):
        captured["base_url"] = base_url
        return []

    monkeypatch.setattr("hermes_cli.models.fetch_lmstudio_models", _fake_fetch)

    list_authenticated_providers(
        current_provider="lmstudio",
        current_base_url="http://192.168.1.10:1234/v1",
    )

    assert captured["base_url"] == "http://override.local:9999/v1"


def test_lmstudio_picker_skips_probe_when_not_configured(monkeypatch):
    """If the user has never configured LM Studio (no LM_API_KEY / LM_BASE_URL
    and not on lmstudio), the picker must not pay the localhost probe cost
    just to discover LM Studio is unavailable.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.delenv("LM_BASE_URL", raising=False)
    monkeypatch.delenv("LM_API_KEY", raising=False)

    captured: dict = {}

    def _fake_fetch(api_key=None, base_url=None, timeout=5.0):
        captured["base_url"] = base_url
        return []

    monkeypatch.setattr("hermes_cli.models.fetch_lmstudio_models", _fake_fetch)

    list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
    )

    assert "base_url" not in captured


def test_custom_providers_uses_live_models_for_multi_model_endpoint(monkeypatch):
    """Custom providers with api_key + base_url should prefer live /models.

    Custom providers (section 4 of list_authenticated_providers) point at
    gateways like Bifrost that expose hundreds of models.  Reading only the
    static ``models:`` dict from config.yaml leaves the /model picker with
    a stale subset.  Live discovery fills the picker with all available
    models from the endpoint.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    calls = []

    def fake_fetch_api_models(api_key, base_url, **kwargs):
        calls.append((api_key, base_url, kwargs))
        return ["gateway-model-a", "gateway-model-b", "gateway-model-c"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)

    custom_providers = [
        {
            "name": "my-gateway",
            "api_key": "sk-gateway-key",
            "base_url": "https://gateway.example.com/v1",
            "model": "gateway-model-a",
            "models": {
                "gateway-model-a": {"context_length": 128000},
                "gateway-model-b": {"context_length": 128000},
            },
        }
    ]

    providers = list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
        custom_providers=custom_providers,
        max_models=50,
    )

    gateway_prov = next(
        (
            p
            for p in providers
            if p.get("api_url") == "https://gateway.example.com/v1"
        ),
        None,
    )

    assert gateway_prov is not None, "Custom provider group not found in results"
    assert calls == [
        ("sk-gateway-key", "https://gateway.example.com/v1", {"headers": None})
    ], "fetch_api_models must be called with the custom provider's credentials"
    assert gateway_prov["models"] == [
        "gateway-model-a",
        "gateway-model-b",
        "gateway-model-c",
    ], "Live models must replace the static subset"
    assert gateway_prov["total_models"] == 3


def test_custom_provider_live_model_probe_uses_extra_headers(monkeypatch):
    """custom_providers[].extra_headers must apply to live /models probes."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    calls = []

    def fake_fetch_api_models(api_key, base_url, **kwargs):
        calls.append((api_key, base_url, kwargs))
        return ["gateway-model"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)

    providers = list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
        custom_providers=[
            {
                "name": "LLM Proxy",
                "api_key": "local-key",
                "base_url": "http://localhost:8081/v1",
                "extra_headers": {
                    "sleeve-harness": "hermes",
                    "sleeve-base-url": "http://localhost:8081/v1",
                },
            }
        ],
        max_models=50,
    )

    gateway_prov = next(
        (
            p
            for p in providers
            if p.get("api_url") == "http://localhost:8081/v1"
        ),
        None,
    )

    assert gateway_prov is not None
    assert calls == [
        (
            "local-key",
            "http://localhost:8081/v1",
            {
                "headers": {
                    "sleeve-harness": "hermes",
                    "sleeve-base-url": "http://localhost:8081/v1",
                }
            },
        )
    ]
    assert gateway_prov["models"] == ["gateway-model"]


def test_same_endpoint_different_extra_headers_not_collapsed(monkeypatch):
    """Entries sharing (api_url, credential, api_mode) but declaring different
    extra_headers must NOT collapse into one picker row — each is a distinct
    header-authenticated endpoint (e.g. per-tenant routing behind one proxy)
    and must probe /models with its own headers."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    calls = []

    def fake_fetch_api_models(api_key, base_url, **kwargs):
        calls.append((api_key, base_url, kwargs.get("headers")))
        # Return a per-tenant model list keyed by the routing header so we can
        # assert each row got its OWN probe rather than a shared one.
        tenant = (kwargs.get("headers") or {}).get("X-Tenant", "none")
        return [f"model-{tenant}"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)

    providers = list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
        custom_providers=[
            {
                "name": "Proxy Tenant A",
                "api_key": "shared-key",
                "base_url": "http://localhost:8081/v1",
                "extra_headers": {"X-Tenant": "a"},
            },
            {
                "name": "Proxy Tenant B",
                "api_key": "shared-key",
                "base_url": "http://localhost:8081/v1",
                "extra_headers": {"X-Tenant": "b"},
            },
        ],
        max_models=50,
    )

    rows = [
        p for p in providers if p.get("api_url") == "http://localhost:8081/v1"
    ]
    # Two distinct rows, not one collapsed row.
    assert len(rows) == 2, f"expected 2 rows, got {len(rows)}: {rows}"

    # Each tenant was probed with its OWN header set (order-independent).
    assert ("shared-key", "http://localhost:8081/v1", {"X-Tenant": "a"}) in calls
    assert ("shared-key", "http://localhost:8081/v1", {"X-Tenant": "b"}) in calls

    # Each row surfaces the model list its own headers unlocked.
    models_by_row = {tuple(r["models"]) for r in rows}
    assert models_by_row == {("model-a",), ("model-b",)}


def test_custom_providers_discover_models_false_keeps_explicit_subset(monkeypatch):
    """Custom providers (section 4) with ``discover_models: false`` must keep
    their explicit ``models:`` subset instead of replacing it with live
    /models, even when an api_key is present.

    This mirrors section 3 (user ``providers:``) behaviour and supports
    endpoints that expose a full aggregator catalog via /models but only
    serve a configured subset.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    calls = []

    def fake_fetch_api_models(api_key, base_url, **kwargs):
        calls.append((api_key, base_url, kwargs))
        return ["gateway-model-a", "gateway-model-b", "gateway-model-c"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)

    custom_providers = [
        {
            "name": "my-gateway",
            "api_key": "***",
            "base_url": "https://gateway.example.com/v1",
            "discover_models": False,
            "model": "gateway-model-a",
            "models": {
                "gateway-model-a": {"context_length": 128000},
                "gateway-model-b": {"context_length": 128000},
            },
        }
    ]

    providers = list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
        custom_providers=custom_providers,
        max_models=50,
    )

    gateway_prov = next(
        (
            p
            for p in providers
            if p.get("api_url") == "https://gateway.example.com/v1"
        ),
        None,
    )

    assert gateway_prov is not None, "Custom provider group not found in results"
    assert calls == [], (
        "fetch_api_models must NOT be called when discover_models is false"
    )
    assert gateway_prov["models"] == [
        "gateway-model-a",
        "gateway-model-b",
    ], "Explicit models: subset must be preserved when discovery is disabled"
    assert gateway_prov["total_models"] == 2


def test_custom_providers_discover_models_false_string_is_normalised(monkeypatch):
    """String ``discover_models: "false"`` (hand-edited / env-style configs)
    must be treated as a disable, same as the boolean ``False`` and section 3.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    calls = []

    def fake_fetch_api_models(api_key, base_url, **kwargs):
        calls.append((api_key, base_url, kwargs))
        return ["live-a", "live-b"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)

    custom_providers = [
        {
            "name": "my-gateway",
            "api_key": "***",
            "base_url": "https://gateway.example.com/v1",
            "discover_models": "false",
            "model": "only-model",
            "models": {"only-model": {"context_length": 128000}},
        }
    ]

    providers = list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
        custom_providers=custom_providers,
        max_models=50,
    )

    gateway_prov = next(
        (p for p in providers if p.get("api_url") == "https://gateway.example.com/v1"),
        None,
    )

    assert gateway_prov is not None
    assert calls == [], "string 'false' must disable live discovery"
    assert gateway_prov["models"] == ["only-model"]


def test_custom_providers_discover_models_false_list_of_dict_ids(monkeypatch):
    """List-of-dicts ``models: [{id: ...}]`` must be preserved as configured
    model IDs when discovery is disabled."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    calls = []

    def fake_fetch_api_models(api_key, base_url, **kwargs):
        calls.append((api_key, base_url, kwargs))
        return ["live-a", "live-b"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)

    custom_providers = [
        {
            "name": "static-gateway",
            "api_key": "***",
            "base_url": "https://router.example.com/v1",
            "discover_models": False,
            "model": "claude-3-7-sonnet",
            "models": [
                {"id": "claude-3-7-sonnet"},
                {"id": "claude-sonnet-4"},
            ],
        }
    ]

    providers = list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
        custom_providers=custom_providers,
        max_models=50,
    )

    gateway_prov = next(
        (p for p in providers if p.get("api_url") == "https://router.example.com/v1"),
        None,
    )

    assert gateway_prov is not None
    assert calls == [], "discover_models: false must skip live discovery"
    assert gateway_prov["models"] == ["claude-3-7-sonnet", "claude-sonnet-4"]
    assert gateway_prov["total_models"] == 2


def test_list_of_dict_models_prefers_id_over_label(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
        custom_providers=[
            {
                "name": "static-gateway",
                "base_url": "https://router.example.com/v1",
                "discover_models": False,
                "models": [{"id": "real-model-id", "name": "Friendly Label"}],
            }
        ],
        max_models=50,
    )

    gateway_prov = next(
        (p for p in providers if p.get("api_url") == "https://router.example.com/v1"),
        None,
    )

    assert gateway_prov is not None
    assert gateway_prov["models"] == ["real-model-id"]


def test_resolve_custom_provider_passes_key_env():
    """resolve_custom_provider should propagate key_env into api_key_env_vars.

    Regression: previously api_key_env_vars was always (), silently dropping
    the configured env var and causing 401s on every request.
    """
    from hermes_cli.providers import resolve_custom_provider

    resolved = resolve_custom_provider(
        "custom:token-plan",
        custom_providers=[
            {
                "name": "token-plan",
                "base_url": "https://token-plan-sgp.xiaomimimo.com/v1",
                "key_env": "XIAOMI_MIMO_API_KEY",
                "model": "mimo-v2-pro",
            }
        ],
    )

    assert resolved is not None
    assert resolved.api_key_env_vars == ("XIAOMI_MIMO_API_KEY",)
    assert resolved.base_url == "https://token-plan-sgp.xiaomimimo.com/v1"


def test_resolve_custom_provider_bare_custom_self_heal_passes_key_env():
    """The bare-'custom' self-heal path must also propagate key_env.

    A corrupt stored provider of the bare string 'custom' falls back to the
    first valid entry; that fallback previously hardcoded api_key_env_vars=(),
    dropping the env var just like the named-match path did.
    """
    from hermes_cli.providers import resolve_custom_provider

    resolved = resolve_custom_provider(
        "custom",
        custom_providers=[
            {
                "name": "token-plan",
                "base_url": "https://token-plan-sgp.xiaomimimo.com/v1",
                "key_env": "XIAOMI_MIMO_API_KEY",
            }
        ],
    )

    assert resolved is not None
    assert resolved.api_key_env_vars == ("XIAOMI_MIMO_API_KEY",)


def test_discovered_models_auto_saved_to_cache(monkeypatch):
    """Discovered models are persisted to config so ``discover_models: false``
    has a populated cache on the next read (#65652).

    When a successful probe returns live models, ``_save_discovered_models_to_config``
    must be called with the provider's base_url and the discovered model list.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    save_calls = []

    def fake_fetch_api_models(api_key, base_url, **kwargs):
        return ["discovered-a", "discovered-b", "discovered-c"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)
    monkeypatch.setattr(
        "hermes_cli.model_switch._save_discovered_models_to_config",
        lambda api_url, model_ids: save_calls.append((api_url, model_ids)),
    )

    custom_providers = [
        {
            "name": "my-gateway",
            "api_key": "***",
            "base_url": "https://gateway.example.com/v1",
            "discover_models": True,
            "model": "only-model",
            "models": {"only-model": {"context_length": 128000}},
        }
    ]

    providers = list_authenticated_providers(
        current_provider="my-gateway",
        current_base_url="https://gateway.example.com/v1",
        custom_providers=custom_providers,
        max_models=50,
        probe_custom_providers=True,
    )

    assert len(save_calls) == 1, (
        "_save_discovered_models_to_config must be called after a successful probe"
    )
    assert save_calls[0][0] == "https://gateway.example.com/v1"
    assert save_calls[0][1] == ["discovered-a", "discovered-b", "discovered-c"]

    gateway_prov = next(
        (p for p in providers if p.get("api_url") == "https://gateway.example.com/v1"),
        None,
    )
    assert gateway_prov is not None
    assert gateway_prov["models"] == ["discovered-a", "discovered-b", "discovered-c"]


def test_discovered_models_not_saved_on_empty_probe(monkeypatch):
    """When a probe returns an empty list, no auto-save must happen (#65652)."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    save_calls = []

    def fake_fetch_api_models(api_key, base_url, **kwargs):
        return []

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)
    monkeypatch.setattr(
        "hermes_cli.model_switch._save_discovered_models_to_config",
        lambda api_url, model_ids: save_calls.append((api_url, model_ids)),
    )

    custom_providers = [
        {
            "name": "my-gateway",
            "api_key": "***",
            "base_url": "https://gateway.example.com/v1",
            "discover_models": True,
            "model": "only-model",
        }
    ]

    list_authenticated_providers(
        current_provider="my-gateway",
        current_base_url="https://gateway.example.com/v1",
        custom_providers=custom_providers,
        max_models=50,
        probe_custom_providers=True,
    )

    assert save_calls == [], "Empty probe must not trigger a save"


def test_save_discovered_models_skips_unchanged(monkeypatch):
    """``_save_discovered_models_to_config`` must not write config when the
    model list hasn't changed (#65652)."""
    from hermes_cli.model_switch import _save_discovered_models_to_config

    save_calls = []

    def fake_save(config):
        save_calls.append(dict(config))

    monkeypatch.setattr("hermes_cli.config.save_config", fake_save)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "custom_providers": [
                {
                    "name": "my-gateway",
                    "base_url": "https://gateway.example.com/v1",
                    "models": ["model-a", "model-b"],
                }
            ]
        },
    )

    # Same list — no write
    _save_discovered_models_to_config(
        "https://gateway.example.com/v1",
        ["model-a", "model-b"],
    )
    assert save_calls == [], "Unchanged models must not trigger config write"

    # Changed list — write
    _save_discovered_models_to_config(
        "https://gateway.example.com/v1",
        ["model-a", "model-b", "model-c"],
    )
    assert len(save_calls) == 1, "Changed models must trigger config write"
    updated = save_calls[0]["custom_providers"][0]
    assert updated["models"] == ["model-a", "model-b", "model-c"]


def test_save_discovered_models_noop_on_empty_args(monkeypatch):
    """``_save_discovered_models_to_config`` is a no-op when api_url or
    model_ids are blank (#65652)."""
    from hermes_cli.model_switch import _save_discovered_models_to_config

    load_calls = 0

    def fake_load():
        nonlocal load_calls
        load_calls += 1
        return {"custom_providers": []}

    monkeypatch.setattr("hermes_cli.config.load_config", fake_load)

    _save_discovered_models_to_config("", ["a"])
    _save_discovered_models_to_config("https://x.com", [])
    _save_discovered_models_to_config("", [])

    assert load_calls == 0, "load_config must not be called for empty args"


def test_save_discovered_models_preserves_dict_form(monkeypatch):
    """``_save_discovered_models_to_config`` must not replace a dict-form
    ``models`` mapping (per-model metadata like ``context_length``) with
    a flat list of strings (#67841)."""
    from hermes_cli.model_switch import _save_discovered_models_to_config

    save_calls = []

    def fake_save(config):
        save_calls.append(dict(config))

    monkeypatch.setattr("hermes_cli.config.save_config", fake_save)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "custom_providers": [
                {
                    "name": "my-gateway",
                    "base_url": "https://gateway.example.com/v1",
                    "models": {
                        "configured-model": {"context_length": 8192},
                    },
                }
            ]
        },
    )

    # Dict-form models must NOT be overwritten by discovered models
    _save_discovered_models_to_config(
        "https://gateway.example.com/v1",
        ["configured-model", "discovered-model"],
    )
    assert save_calls == [], (
        "Dict-form models must not be replaced with a flat list"
    )


def test_save_discovered_models_preserves_list_of_dicts_form(monkeypatch):
    """``_save_discovered_models_to_config`` must not replace a list-of-dicts
    ``models`` form (per-model metadata via ``[{id: ...}]``) with a flat list
    of strings (#67841 sibling site)."""
    from hermes_cli.model_switch import _save_discovered_models_to_config

    save_calls = []

    def fake_save(config):
        save_calls.append(dict(config))

    monkeypatch.setattr("hermes_cli.config.save_config", fake_save)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "custom_providers": [
                {
                    "name": "my-gateway",
                    "base_url": "https://gateway.example.com/v1",
                    "models": [
                        {"id": "configured-model", "context_length": 8192},
                        {"id": "other-model", "context_length": 4096},
                    ],
                }
            ]
        },
    )

    # List-of-dicts models must NOT be overwritten by discovered models
    _save_discovered_models_to_config(
        "https://gateway.example.com/v1",
        ["configured-model", "discovered-model"],
    )
    assert save_calls == [], (
        "List-of-dicts models must not be replaced with a flat list"
    )


def test_shared_url_different_display_names_are_separate_rows(monkeypatch):
    """Multiple custom_providers entries sharing base_url + api_key + api_mode
    but with *different* display-name prefixes (e.g. a proxy fronting
    cerebras, groq and perplexity at one URL) must each get their own picker
    row, not collapse into one."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    # Stub live discovery so the test is deterministic regardless of network.
    monkeypatch.setattr(
        "hermes_cli.models.fetch_api_models",
        lambda api_key, base_url, **kwargs: [],
    )

    providers = list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
        user_providers={},
        custom_providers=[
            {"name": "Cerebras", "base_url": "https://proxy.example.com/v1",
             "api_key": "proxy-key", "model": "llama-4-scout"},
            {"name": "Groq", "base_url": "https://proxy.example.com/v1",
             "api_key": "proxy-key", "model": "llama-4-scout"},
            {"name": "Perplexity", "base_url": "https://proxy.example.com/v1",
             "api_key": "proxy-key", "model": "sonar-pro"},
        ],
        max_models=50,
    )

    custom = [p for p in providers if p.get("is_user_defined")]
    names = sorted(p["name"] for p in custom)
    assert names == ["Cerebras", "Groq", "Perplexity"], (
        f"expected three separate rows, got {names}"
    )
    # Each row carries only its own model (no cross-contamination).
    by_name = {p["name"]: p["models"] for p in custom}
    assert by_name["Cerebras"] == ["llama-4-scout"]
    assert by_name["Groq"] == ["llama-4-scout"]
    assert by_name["Perplexity"] == ["sonar-pro"]


def test_shared_url_per_model_suffix_still_collapses(monkeypatch):
    """Per-model suffix entries sharing the same display-name prefix (e.g.
    "Ollama — A", "Ollama — B") must still collapse into one row even with
    the display-prefix grouping dimension."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    # Stub live discovery so a locally-running Ollama cannot override the
    # static configured models and make the assertion flaky.
    monkeypatch.setattr(
        "hermes_cli.models.fetch_api_models",
        lambda api_key, base_url, **kwargs: [],
    )

    providers = list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
        user_providers={},
        custom_providers=[
            {"name": "Ollama \u2014 GLM 5.1", "base_url": "http://localhost:11434/v1",
             "api_key": "ollama", "model": "glm-5.1"},
            {"name": "Ollama \u2014 Qwen3-coder", "base_url": "http://localhost:11434/v1",
             "api_key": "ollama", "model": "qwen3-coder"},
        ],
        max_models=50,
    )

    custom = [p for p in providers if p.get("is_user_defined")]
    assert len(custom) == 1, (
        f"expected one collapsed row, got {[p['name'] for p in custom]}"
    )
    assert custom[0]["name"] == "Ollama"
    assert set(custom[0]["models"]) == {"glm-5.1", "qwen3-coder"}


def test_excluded_providers_hides_builtin_row(monkeypatch):
    """``excluded_providers`` must hide a built-in provider row that would
    otherwise surface when its credentials are present."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    baseline = list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
        user_providers={},
        custom_providers=[],
        max_models=50,
    )
    assert any(p["slug"] == "openrouter" for p in baseline), (
        "sanity: openrouter row must appear when OPENROUTER_API_KEY is set"
    )

    filtered = list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
        user_providers={},
        custom_providers=[],
        max_models=50,
        excluded_providers=["openrouter"],
    )
    assert not any(p["slug"] == "openrouter" for p in filtered), (
        "excluded_providers=['openrouter'] must hide the openrouter row"
    )


def test_excluded_providers_empty_is_noop(monkeypatch):
    """An empty ``excluded_providers`` list must not change picker output."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    a = list_authenticated_providers(
        current_provider="openrouter",
        user_providers={},
        custom_providers=[],
        max_models=50,
    )
    b = list_authenticated_providers(
        current_provider="openrouter",
        user_providers={},
        custom_providers=[],
        max_models=50,
        excluded_providers=[],
    )
    assert [p["slug"] for p in a] == [p["slug"] for p in b]
