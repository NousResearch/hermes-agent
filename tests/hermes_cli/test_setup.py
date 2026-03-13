import json

from hermes_cli.auth import _update_config_for_provider, get_active_provider
from hermes_cli.config import load_config, save_config
from hermes_cli.setup import (
    setup_model_provider,
    _discover_working_profile_providers,
    _configure_model_profiles_interactive,
    _ensure_model_profiles_config,
    _ordered_model_profile_names,
    _reset_model_profiles_config,
)


def _clear_provider_env(monkeypatch):
    for key in (
        "NOUS_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "LLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)



def test_nous_oauth_setup_keeps_current_model_when_syncing_disk_provider(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()

    prompt_choices = iter([0, 2])
    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice",
        lambda *args, **kwargs: next(prompt_choices),
    )
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: "")

    def _fake_login_nous(*args, **kwargs):
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps({"active_provider": "nous", "providers": {}}))
        _update_config_for_provider("nous", "https://inference.example.com/v1")

    monkeypatch.setattr("hermes_cli.auth._login_nous", _fake_login_nous)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_runtime_credentials",
        lambda *args, **kwargs: {
            "base_url": "https://inference.example.com/v1",
            "api_key": "nous-key",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.auth.fetch_nous_models",
        lambda *args, **kwargs: ["gemini-3-flash"],
    )

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()

    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "nous"
    assert reloaded["model"]["base_url"] == "https://inference.example.com/v1"
    assert reloaded["model"]["default"] == "anthropic/claude-opus-4.6"


def test_custom_setup_clears_active_oauth_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({"active_provider": "nous", "providers": {}}))

    config = load_config()

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", lambda *args, **kwargs: 3)

    prompt_values = iter(
        [
            "https://custom.example/v1",
            "custom-api-key",
            "custom/model",
            "",
        ]
    )
    monkeypatch.setattr(
        "hermes_cli.setup.prompt",
        lambda *args, **kwargs: next(prompt_values),
    )

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()

    assert get_active_provider() is None
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "custom"
    assert reloaded["model"]["base_url"] == "https://custom.example/v1"
    assert reloaded["model"]["default"] == "custom/model"


def test_discover_working_profile_providers_filters_failing(monkeypatch):
    def _fake_resolve(requested=None, **_kwargs):
        if requested in {"openrouter", "openai-codex"}:
            return {
                "provider": requested,
                "base_url": "https://example.com/v1",
                "api_key": "k",
                "api_mode": "chat_completions",
            }
        raise RuntimeError("no creds")

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _fake_resolve)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    working, errors = _discover_working_profile_providers()

    ids = [pid for pid, _ in working]
    assert "openrouter" in ids
    assert "openai-codex" in ids
    assert all(pid not in ids for pid in ("nous", "zai", "custom"))
    assert errors


def test_configure_model_profiles_uses_only_working_provider_choices(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = load_config()
    _ensure_model_profiles_config(cfg)

    # yes: configure wizard, yes: configure chat, no: coding/planning/research/delegation/custom add
    yes_no_answers = iter([True, True, False, False, False, False, False])
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: next(yes_no_answers))

    # provider pick -> first working provider (openrouter), model pick -> first live suggestion
    prompt_choices = iter([1, 0])
    monkeypatch.setattr("hermes_cli.setup.prompt_choice", lambda *args, **kwargs: next(prompt_choices))
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: "")

    monkeypatch.setattr(
        "hermes_cli.setup._discover_working_profile_providers",
        lambda: ([
            ("openrouter", {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "test-key"}),
        ], []),
    )
    monkeypatch.setattr(
        "hermes_cli.setup._live_model_suggestions_for_runtime",
        lambda _runtime: ["anthropic/claude-sonnet-4"],
    )

    _configure_model_profiles_interactive(cfg)

    chat = cfg["model_profiles"]["chat"]
    assert chat["provider"] == "openrouter"
    assert chat["model"] == "anthropic/claude-sonnet-4"


def test_custom_profiles_are_preserved_and_ordered(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = load_config()
    cfg["model_profiles"] = {
        "ops": {"model": "gpt-4.1-mini", "provider": "openrouter"},
        "chat": {"model": "anthropic/claude-sonnet-4", "provider": "openrouter"},
    }

    profiles = _ensure_model_profiles_config(cfg)

    assert "ops" in profiles
    ordered = _ordered_model_profile_names(profiles)
    assert ordered[:5] == ["chat", "coding", "planning", "research", "delegation"]
    assert ordered[-1] == "ops"


def test_reset_model_profiles_config_keeps_defaults_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = load_config()
    cfg["model_profiles"] = {"ops": {"model": "x", "provider": "custom"}}
    cfg["model_routing"] = {"rules": [{"profile": "ops", "if_goal_matches": ["deploy"]}]}

    profiles = _reset_model_profiles_config(cfg)

    assert set(profiles.keys()) == {"chat", "coding", "planning", "research", "delegation"}
    assert cfg["model_routing"] == {"rules": []}
