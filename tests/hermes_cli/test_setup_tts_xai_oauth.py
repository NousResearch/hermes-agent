"""Regression: TTS/setup xAI OAuth must not hijack the active chat provider."""

import json

import pytest

import hermes_yaml as yaml


def test_run_xai_oauth_login_from_setup_does_not_hijack_active_provider(
    tmp_path, monkeypatch
):
    """TTS/setup OAuth must save tokens without switching chat inference routing.

    Regression: `_run_xai_oauth_login_from_setup` used to call
    `_update_config_for_provider("xai-oauth")` (and token save flipped
    `active_provider`), so `hermes setup tts` OAuth login hijacked the main
    chat provider.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    auth_path = hermes_home / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "version": 1,
                "active_provider": "openrouter",
                "providers": {},
            }
        ),
        encoding="utf-8",
    )
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "provider": "openrouter",
                    "default": "anthropic/claude-sonnet-4",
                    "base_url": "https://openrouter.ai/api/v1",
                },
                "tts": {"provider": "edge"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "hermes_cli.auth._xai_oauth_device_code_login",
        lambda **kwargs: {
            "tokens": {
                "access_token": "tts-xai-access",
                "refresh_token": "tts-xai-refresh",
                "id_token": "",
                "token_type": "Bearer",
            },
            "discovery": {"token_endpoint": "https://auth.x.ai/oauth2/token"},
            "redirect_uri": "",
            "base_url": "https://api.x.ai/v1",
            "last_refresh": "2026-07-25T12:00:00Z",
        },
    )
    monkeypatch.setattr("hermes_cli.auth._is_remote_session", lambda: True)

    from hermes_cli.auth import is_source_suppressed, suppress_credential_source
    from hermes_cli.setup_tts import _run_xai_oauth_login_from_setup

    suppress_credential_source("xai-oauth", "device_code")
    assert is_source_suppressed("xai-oauth", "device_code") is True

    assert _run_xai_oauth_login_from_setup() is True

    assert is_source_suppressed("xai-oauth", "device_code") is False

    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    assert auth["active_provider"] == "openrouter"
    xai_state = auth["providers"]["xai-oauth"]
    assert xai_state["tokens"]["access_token"] == "tts-xai-access"
    assert xai_state["tokens"]["refresh_token"] == "tts-xai-refresh"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["model"]["provider"] == "openrouter"
    assert config["model"]["base_url"] == "https://openrouter.ai/api/v1"
    assert config["model"]["default"] == "anthropic/claude-sonnet-4"


def test_tts_xai_step_prefers_existing_api_key_over_oauth(monkeypatch):
    """Wizard copy must match runtime: an explicit XAI_API_KEY wins over stored OAuth
    tokens (the subscription bearer 403s on metered /v1/tts — #87045, #113727)."""
    import hermes_cli.setup_tts as setup_tts

    messages = []
    monkeypatch.setattr(setup_tts, "_xai_oauth_logged_in_for_setup", lambda: True)
    monkeypatch.setattr(setup_tts._setup, "get_env_value", lambda key: "xai-key" if key == "XAI_API_KEY" else "")
    monkeypatch.setattr(setup_tts._setup, "print_success", lambda msg: messages.append(msg))
    monkeypatch.setattr(setup_tts._setup, "prompt", lambda *a, **k: "")
    monkeypatch.setattr(setup_tts._setup, "prompt_choice", lambda *a, **k: pytest.fail("no prompt expected"))

    config = {}
    assert setup_tts._tts_xai_step(config) == "xai"
    assert len(messages) == 1 and "XAI_API_KEY" in messages[0]
    assert "OAuth credentials" not in messages[0]
    assert "XAI_API_KEY" in setup_tts._tts_xai_step.__doc__.split(">")[0]
