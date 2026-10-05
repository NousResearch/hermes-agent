"""A bare ``provider: custom`` main model whose ``model.base_url`` is on openrouter.ai must use the
key configured beside that base_url (``model.api_key``, literal or ``${VAR}``, or ``model.key_env``),
exactly as a bare custom model on any other host does. Before the fix the OpenRouter branch of the
terminal resolver only looked at OPENROUTER_API_KEY / OPENAI_API_KEY, so such a config raised
AuthError ("resolved without credentials") with no env key set."""

import json

import pytest

from hermes_cli import config as _cfg
from hermes_cli import runtime_provider as rp

OR_URL = "https://openrouter.ai/api/v1"


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {}}))
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENROUTER_BASE_URL", "CUSTOM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)

    def write(model_yaml):
        (hermes_home / "config.yaml").write_text("model:\n" + model_yaml)
        _cfg._LOAD_CONFIG_CACHE.clear()
        _cfg._RAW_CONFIG_CACHE.clear()

    yield write
    _cfg._LOAD_CONFIG_CACHE.clear()
    _cfg._RAW_CONFIG_CACHE.clear()


def test_literal_model_api_key_resolves_on_openrouter(home):
    home(f"  provider: custom\n  default: openai/gpt-4o\n  base_url: {OR_URL}\n  api_key: sk-or-literal-test\n")

    resolved = rp.resolve_runtime_provider()

    assert resolved["provider"] == "custom"
    assert resolved["base_url"] == OR_URL
    assert resolved["api_key"] == "sk-or-literal-test"


def test_var_model_api_key_resolves_on_openrouter(home, monkeypatch):
    monkeypatch.setenv("MY_OR_TEST_KEY", "sk-or-from-var-test")
    home(f"  provider: custom\n  default: openai/gpt-4o\n  base_url: {OR_URL}\n  api_key: ${{MY_OR_TEST_KEY}}\n")

    resolved = rp.resolve_runtime_provider()

    assert resolved["provider"] == "custom"
    assert resolved["api_key"] == "sk-or-from-var-test"


def test_model_key_env_resolves_on_openrouter(home, monkeypatch):
    monkeypatch.setenv("MY_OR_KEY_ENV_TEST", "sk-or-key-env-test")
    home(f"  provider: custom\n  default: openai/gpt-4o\n  base_url: {OR_URL}\n  key_env: MY_OR_KEY_ENV_TEST\n")

    resolved = rp.resolve_runtime_provider()

    assert resolved["api_key"] == "sk-or-key-env-test"


def test_env_openrouter_key_still_used_without_config_key(home, monkeypatch):
    """Unchanged from before the fix: with nothing configured, OPENROUTER_API_KEY backs openrouter.ai."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-test")
    home(f"  provider: custom\n  default: openai/gpt-4o\n  base_url: {OR_URL}\n")

    resolved = rp.resolve_runtime_provider()

    assert resolved["provider"] == "custom"
    assert resolved["api_key"] == "sk-or-env-test"


def test_config_key_beats_env_openrouter_key_for_bare_custom(home, monkeypatch):
    """Same precedence as a bare custom model on any other host: the key declared beside
    model.base_url wins over the ambient env key."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-test")
    home(f"  provider: custom\n  default: openai/gpt-4o\n  base_url: {OR_URL}\n  api_key: sk-or-literal-test\n")

    resolved = rp.resolve_runtime_provider()

    assert resolved["api_key"] == "sk-or-literal-test"


def test_config_key_not_sent_to_custom_base_url_override(home, monkeypatch):
    """model.api_key was declared for model.base_url; CUSTOM_BASE_URL pointing at another
    openrouter.ai path must not inherit it."""
    monkeypatch.setenv("CUSTOM_BASE_URL", "https://openrouter.ai/api/other")
    home(f"  provider: custom\n  default: openai/gpt-4o\n  base_url: {OR_URL}\n  api_key: sk-or-literal-test\n")

    with pytest.raises(rp.AuthError):
        rp.resolve_runtime_provider()


def test_openrouter_provider_ignores_model_api_key_as_before(home):
    """Scope guard: only bare custom changes; ``provider: openrouter`` keeps its env-only keys."""
    home(f"  provider: openrouter\n  default: openai/gpt-4o\n  base_url: {OR_URL}\n  api_key: sk-or-literal-test\n")
    from hermes_cli.runtime_provider_backends import _resolve_openrouter_runtime

    resolved = _resolve_openrouter_runtime(requested_provider="openrouter")

    assert resolved["api_key"] == ""


def test_local_custom_endpoint_unchanged(home, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-test")
    home("  provider: custom\n  default: qwen\n  base_url: http://127.0.0.1:8080/v1\n  api_key: sk-local-test\n")

    resolved = rp.resolve_runtime_provider()

    assert resolved["provider"] == "custom"
    assert resolved["base_url"] == "http://127.0.0.1:8080/v1"
    assert resolved["api_key"] == "sk-local-test"


def test_unresolved_var_model_api_key_falls_back_to_env_key(home, monkeypatch):
    # load_config leaves an unset ${VAR} verbatim; it must not be sent as the bearer.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-test")
    home(f"  provider: custom\n  default: openai/gpt-4o\n  base_url: {OR_URL}\n  api_key: ${{NOPE_UNSET_OR_TEST_KEY}}\n")

    resolved = rp.resolve_runtime_provider()

    assert resolved["api_key"] == "sk-or-env-test"


def test_unresolved_var_model_api_key_without_env_key_still_raises(home):
    home(f"  provider: custom\n  default: openai/gpt-4o\n  base_url: {OR_URL}\n  api_key: ${{NOPE_UNSET_OR_TEST_KEY}}\n")

    with pytest.raises(rp.AuthError):
        rp.resolve_runtime_provider()


def test_requested_custom_over_stale_openrouter_provider_does_not_use_model_api_key(home):
    # ``--provider custom`` over a config still saying ``provider: openrouter``: model.base_url is not
    # the trusted custom endpoint then, so the key beside it stays unused (the use_config_base_url gate).
    home(f"  provider: openrouter\n  default: openai/gpt-4o\n  base_url: {OR_URL}\n  api_key: sk-or-literal-test\n")

    try:
        resolved = rp.resolve_runtime_provider(requested="custom")
    except rp.AuthError:
        return
    assert resolved.get("api_key") != "sk-or-literal-test"
