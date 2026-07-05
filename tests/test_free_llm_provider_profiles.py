import importlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from hermes_cli.models import get_default_model_for_provider
from providers import get_provider_profile


def test_free_llm_profiles_resolve_by_canonical_name_and_aliases():
    expected = {
        "groq": ("groq-cloud", "groqcloud"),
        "mistral": ("mistralai", "mistral-ai", "la-plateforme"),
        "cerebras": ("cerebras-ai", "cerebras-cloud", "cerebras-inference"),
    }

    for canonical, aliases in expected.items():
        profile = get_provider_profile(canonical)
        assert profile is not None
        assert profile.name == canonical
        for alias in aliases:
            assert get_provider_profile(alias) is profile


def test_groq_profile_fields():
    profile = get_provider_profile("groq")

    assert profile.display_name == "Groq"
    assert profile.description == "Groq Cloud — fast OpenAI-compatible inference"
    assert profile.signup_url == "https://console.groq.com/keys"
    assert profile.env_vars == ("GROQ_API_KEY",)
    assert profile.base_url == "https://api.groq.com/openai/v1"
    assert profile.default_aux_model == "llama-3.1-8b-instant"
    assert profile.fallback_models == (
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
    )


def test_mistral_profile_fields():
    profile = get_provider_profile("mistral")

    assert profile.display_name == "Mistral AI"
    assert profile.description == "Mistral AI — direct OpenAI-compatible API"
    assert profile.signup_url == "https://console.mistral.ai/api-keys/"
    assert profile.env_vars == ("MISTRAL_API_KEY",)
    assert profile.base_url == "https://api.mistral.ai/v1"
    assert profile.default_aux_model == "mistral-small-latest"
    assert profile.fallback_models == (
        "mistral-medium-latest",
        "mistral-small-latest",
        "mistral-large-latest",
    )


def test_cerebras_profile_fields():
    profile = get_provider_profile("cerebras")

    assert profile.display_name == "Cerebras"
    assert (
        profile.description
        == "Cerebras Inference — high-speed OpenAI-compatible inference"
    )
    assert profile.signup_url == "https://cloud.cerebras.ai/"
    assert profile.env_vars == ("CEREBRAS_API_KEY",)
    assert profile.base_url == "https://api.cerebras.ai/v1"
    assert profile.models_url == "https://api.cerebras.ai/public/v1/models"
    assert profile.default_aux_model == "gpt-oss-120b"
    assert profile.fallback_models == (
        "gpt-oss-120b",
        "gemma-4-31b",
        "zai-glm-4.7",
    )


def test_api_key_profiles_are_auto_included_in_canonical_providers():
    models = importlib.import_module("hermes_cli.models")
    canonical = {provider.slug: provider for provider in models.CANONICAL_PROVIDERS}

    assert canonical["groq"].label == "Groq"
    assert canonical["groq"].tui_desc == (
        "Groq Cloud — fast OpenAI-compatible inference"
    )
    assert canonical["mistral"].label == "Mistral AI"
    assert canonical["mistral"].tui_desc == (
        "Mistral AI — direct OpenAI-compatible API"
    )
    assert canonical["cerebras"].label == "Cerebras"
    assert canonical["cerebras"].tui_desc == (
        "Cerebras Inference — high-speed OpenAI-compatible inference"
    )

    static_region = open(models.__file__, encoding="utf-8").read().split(
        "# Auto-extend CANONICAL_PROVIDERS", 1
    )[0]
    assert 'ProviderEntry("groq"' not in static_region
    assert 'ProviderEntry("mistral"' not in static_region
    assert 'ProviderEntry("cerebras"' not in static_region


def test_profile_fallback_models_power_non_interactive_defaults():
    assert get_default_model_for_provider("groq") == "openai/gpt-oss-120b"
    assert get_default_model_for_provider("groq-cloud") == "openai/gpt-oss-120b"
    assert get_default_model_for_provider("groqcloud") == "openai/gpt-oss-120b"
    assert get_default_model_for_provider("mistral") == "mistral-medium-latest"
    assert get_default_model_for_provider("mistralai") == "mistral-medium-latest"
    assert get_default_model_for_provider("la-plateforme") == "mistral-medium-latest"
    assert get_default_model_for_provider("cerebras") == "gpt-oss-120b"
    assert get_default_model_for_provider("cerebras-ai") == "gpt-oss-120b"
    assert get_default_model_for_provider("cerebras-cloud") == "gpt-oss-120b"


def test_auth_add_stores_profile_alias_credentials_under_canonical_provider(monkeypatch):
    auth_commands = importlib.import_module("hermes_cli.auth_commands")
    auth_mod = importlib.import_module("hermes_cli.auth")

    assert auth_commands._normalize_provider("groqcloud") == "groq"
    assert auth_commands._normalize_provider("la-plateforme") == "mistral"
    assert auth_commands._normalize_provider("cerebras-cloud") == "cerebras"

    seen_pools = []
    seen_entries = []

    class DummyPool:
        def __init__(self):
            self._entries = []

        def entries(self):
            return list(self._entries)

        def add_entry(self, entry):
            self._entries.append(entry)
            seen_entries.append(entry)

    def fake_load_pool(provider):
        seen_pools.append(provider)
        return DummyPool()

    monkeypatch.setattr(auth_commands, "load_pool", fake_load_pool)
    monkeypatch.setattr(auth_mod, "_load_auth_store", lambda: {"suppressed_sources": {}})
    monkeypatch.setattr(auth_mod, "unsuppress_credential_source", lambda *_args: None)

    auth_commands.auth_add_command(
        SimpleNamespace(
            provider="groqcloud",
            auth_type="api_key",
            api_key="sk-test-value",
            label="test",
        )
    )

    assert seen_pools == ["groq"]
    assert len(seen_entries) == 1
    assert seen_entries[0].provider == "groq"
    assert seen_entries[0].base_url == "https://api.groq.com/openai/v1"


def test_auth_list_uses_canonical_pool_for_profile_alias(monkeypatch):
    auth_commands = importlib.import_module("hermes_cli.auth_commands")

    seen_pools = []

    class EmptyPool:
        def entries(self):
            return []

    def fake_load_pool(provider):
        seen_pools.append(provider)
        return EmptyPool()

    monkeypatch.setattr(auth_commands, "load_pool", fake_load_pool)

    auth_commands.auth_list_command(SimpleNamespace(provider="la-plateforme"))

    assert seen_pools == ["mistral"]


def test_credential_pool_reads_legacy_alias_pool_for_canonical_provider(monkeypatch):
    auth_mod = importlib.import_module("hermes_cli.auth")
    legacy_entry = {
        "id": "legacy1",
        "label": "legacy",
        "auth_type": "api_key",
        "source": "manual",
        "access_token": "sk-test-value",
    }
    store = {"credential_pool": {"groqcloud": [legacy_entry]}}

    monkeypatch.setattr(auth_mod, "_load_auth_store", lambda: store)
    monkeypatch.setattr(auth_mod, "_load_global_auth_store", lambda: None)

    assert auth_mod.read_credential_pool("groq") == [legacy_entry]


def test_credential_pool_reads_canonical_pool_for_alias_provider(monkeypatch):
    auth_mod = importlib.import_module("hermes_cli.auth")
    canonical_entry = {
        "id": "canonical1",
        "label": "canonical",
        "auth_type": "api_key",
        "source": "manual",
        "access_token": "sk-test-value",
    }
    store = {"credential_pool": {"groq": [canonical_entry]}}

    monkeypatch.setattr(auth_mod, "_load_auth_store", lambda: store)
    monkeypatch.setattr(auth_mod, "_load_global_auth_store", lambda: None)

    assert auth_mod.read_credential_pool("groqcloud") == [canonical_entry]


def test_credential_pool_write_migrates_legacy_alias_entries(monkeypatch):
    auth_mod = importlib.import_module("hermes_cli.auth")
    legacy_entry = {
        "id": "legacy1",
        "label": "legacy",
        "auth_type": "api_key",
        "source": "manual",
        "access_token": "sk-test-value",
    }
    store = {"credential_pool": {"groqcloud": [legacy_entry.copy()]}}

    monkeypatch.setattr(auth_mod, "_auth_store_lock", lambda: nullcontext())
    monkeypatch.setattr(auth_mod, "_load_auth_store", lambda: store)
    monkeypatch.setattr(auth_mod, "_save_auth_store", lambda _store: Path("/tmp/auth.json"))

    auth_mod.write_credential_pool("groq", [legacy_entry])

    assert store["credential_pool"]["groq"] == [legacy_entry]
    assert "groqcloud" not in store["credential_pool"]
