import sys
from unittest.mock import patch

from hermes_cli.models import normalize_provider, provider_model_ids
from hermes_cli.runtime_provider import resolve_runtime_provider


def test_normalize_provider_maps_chatgpt_aliases():
    assert normalize_provider("chatgpt") == "chatgpt-web"
    assert normalize_provider("chatgpt-web") == "chatgpt-web"
    assert normalize_provider("chatgpt.com") == "chatgpt-web"


def test_provider_model_ids_chatgpt_web_prefers_live_catalog(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.chatgpt_web.resolve_chatgpt_web_runtime_credentials",
        lambda **kwargs: {"api_key": "chatgpt-web-token"},
    )
    monkeypatch.setattr(
        "hermes_cli.chatgpt_web.fetch_chatgpt_web_model_ids",
        lambda access_token=None, **kwargs: ["gpt-5-thinking", "gpt-5-instant", "gpt-5"],
    )

    assert provider_model_ids("chatgpt-web") == ["gpt-5-thinking", "gpt-5-instant", "gpt-5"]


def test_resolve_runtime_provider_chatgpt_web_uses_chatgpt_web_mode(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_chatgpt_web_runtime_credentials",
        lambda **kwargs: {
            "provider": "chatgpt-web",
            "api_key": "chatgpt-web-token",
            "base_url": "https://chatgpt.com/backend-api/f",
            "source": "codex-oauth",
        },
    )

    runtime = resolve_runtime_provider(requested="chatgpt-web")

    assert runtime["provider"] == "chatgpt-web"
    assert runtime["api_mode"] == "chatgpt_web"
    assert runtime["api_key"] == "chatgpt-web-token"
    assert runtime["base_url"] == "https://chatgpt.com/backend-api/f"


def test_model_flow_chatgpt_web_uses_runtime_access_token_for_model_list(monkeypatch):
    from hermes_cli.main import _model_flow_chatgpt_web

    captured = {}

    monkeypatch.setattr(
        "hermes_cli.auth.get_chatgpt_web_auth_status",
        lambda: {"logged_in": True},
    )
    monkeypatch.setattr(
        "hermes_cli.chatgpt_web.resolve_chatgpt_web_runtime_credentials",
        lambda **kwargs: {"api_key": "chatgpt-web-token"},
    )

    def _fake_fetch(access_token=None, **kwargs):
        captured["access_token"] = access_token
        return ["gpt-5-thinking", "gpt-5-instant"]

    monkeypatch.setattr(
        "hermes_cli.chatgpt_web.fetch_chatgpt_web_model_ids",
        _fake_fetch,
    )
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection",
        lambda model_ids, current_model="": None,
    )

    _model_flow_chatgpt_web({}, current_model="gpt-5-thinking")

    assert captured["access_token"] == "chatgpt-web-token"


def test_resolve_chatgpt_web_runtime_credentials_prefers_session_token_exchange(monkeypatch):
    from hermes_cli.chatgpt_web import DEFAULT_CHATGPT_WEB_BASE_URL, resolve_chatgpt_web_runtime_credentials

    monkeypatch.setenv("CHATGPT_WEB_SESSION_TOKEN", "session-token")
    monkeypatch.delenv("CHATGPT_WEB_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(
        "hermes_cli.chatgpt_web._fetch_chatgpt_web_access_token_from_session",
        lambda session_token, **kwargs: "access-from-session",
    )

    creds = resolve_chatgpt_web_runtime_credentials()

    assert creds["provider"] == "chatgpt-web"
    assert creds["api_key"] == "access-from-session"
    assert creds["base_url"] == DEFAULT_CHATGPT_WEB_BASE_URL
    assert creds["source"] == "session-token"


def test_select_provider_and_model_lists_chatgpt_web_in_top_menu(monkeypatch):
    from hermes_cli import main as hermes_main

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "gpt-5", "provider": "openrouter"}},
    )
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda key: "")
    monkeypatch.setattr("hermes_cli.auth.resolve_provider", lambda requested, **kwargs: requested)

    prompts = []
    dispatched = {}

    def _fake_prompt_provider_choice(choices, **kwargs):
        prompts.append(list(choices))
        return next(idx for idx, label in enumerate(choices) if "ChatGPT Web" in label)

    monkeypatch.setattr(hermes_main, "_prompt_provider_choice", _fake_prompt_provider_choice)
    monkeypatch.setattr(
        hermes_main,
        "_model_flow_chatgpt_web",
        lambda config, current_model="": dispatched.setdefault("args", (config, current_model)),
    )

    hermes_main.select_provider_and_model()

    assert prompts
    assert any("ChatGPT Web" in label for label in prompts[0])
    assert dispatched["args"][1] == "gpt-5"


def test_main_accepts_chatgpt_web_for_login_and_logout(monkeypatch):
    from hermes_cli import main as hermes_main

    seen = []
    monkeypatch.setattr(hermes_main, "cmd_login", lambda args: seen.append(("login", args.provider)))
    monkeypatch.setattr(hermes_main, "cmd_logout", lambda args: seen.append(("logout", args.provider)))

    monkeypatch.setattr(sys, "argv", ["hermes", "login", "--provider", "chatgpt-web"])
    hermes_main.main()

    monkeypatch.setattr(sys, "argv", ["hermes", "logout", "--provider", "chatgpt-web"])
    hermes_main.main()

    assert seen == [("login", "chatgpt-web"), ("logout", "chatgpt-web")]
