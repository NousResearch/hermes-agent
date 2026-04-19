import json
import sys
from unittest.mock import patch

import httpx

from hermes_cli.models import normalize_provider, provider_model_ids
from hermes_cli.providers import get_label
from hermes_cli.runtime_provider import resolve_runtime_provider


def test_normalize_provider_maps_chatgpt_aliases():
    assert normalize_provider("chatgpt") == "chatgpt-web"
    assert normalize_provider("chatgpt-web") == "chatgpt-web"
    assert normalize_provider("chatgpt.com") == "chatgpt-web"


def test_provider_model_ids_chatgpt_web_prefers_live_catalog(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.chatgpt_web.resolve_chatgpt_web_runtime_credentials",
        lambda **kwargs: {"api_key": "***"},
    )
    monkeypatch.setattr(
        "hermes_cli.chatgpt_web.fetch_chatgpt_web_model_ids",
        lambda access_token=None, **kwargs: ["gpt-5-thinking", "gpt-5-instant", "gpt-5"],
    )

    models = provider_model_ids("chatgpt-web")

    assert models[:3] == ["gpt-5-thinking", "gpt-5-instant", "gpt-5"]
    assert "gpt-4o" in models


def test_provider_model_ids_chatgpt_web_keeps_legacy_aliases_alongside_live_catalog(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.chatgpt_web.resolve_chatgpt_web_runtime_credentials",
        lambda **kwargs: {"api_key": "***"},
    )
    monkeypatch.setattr(
        "hermes_cli.chatgpt_web.fetch_chatgpt_web_model_ids",
        lambda access_token=None, **kwargs: ["gpt-5-4-thinking", "gpt-5-4-instant", "gpt-5"],
    )

    models = provider_model_ids("chatgpt-web")

    assert "gpt-5-4-thinking" in models
    assert "gpt-5-4-instant" in models
    assert "gpt-5-thinking" in models
    assert "gpt-5-instant" in models


def test_resolve_runtime_provider_chatgpt_web_uses_chatgpt_web_mode(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_chatgpt_web_runtime_credentials",
        lambda **kwargs: {
            "provider": "chatgpt-web",
            "api_key": "chatgpt-web-token",
            "base_url": "https://chatgpt.com/backend-api/f",
            "source": "codex-oauth",
            "session_token": "session-cookie",
            "cookie_header": "cf_clearance=cf-cookie",
            "browser_cookies": [{"name": "cf_clearance", "value": "cf-cookie"}],
            "user_agent": "Mozilla/Test",
            "device_id": "device-123",
        },
    )

    runtime = resolve_runtime_provider(requested="chatgpt-web")

    assert runtime["provider"] == "chatgpt-web"
    assert runtime["api_mode"] == "chatgpt_web"
    assert runtime["api_key"] == "chatgpt-web-token"
    assert runtime["base_url"] == "https://chatgpt.com/backend-api/f"
    assert runtime["session_token"] == "session-cookie"
    assert runtime["cookie_header"] == "cf_clearance=cf-cookie"
    assert runtime["browser_cookies"] == [{"name": "cf_clearance", "value": "cf-cookie"}]
    assert runtime["user_agent"] == "Mozilla/Test"
    assert runtime["device_id"] == "device-123"


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
    monkeypatch.setenv("CHATGPT_WEB_COOKIE_HEADER", "cf_clearance=cf-cookie; oai-did=device-cookie")
    monkeypatch.setenv("CHATGPT_WEB_USER_AGENT", "Mozilla/Test")
    monkeypatch.setenv("CHATGPT_WEB_DEVICE_ID", "device-cookie")
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
    assert creds["cookie_header"] == "cf_clearance=cf-cookie; oai-did=device-cookie"
    assert creds["user_agent"] == "Mozilla/Test"
    assert creds["device_id"] == "device-cookie"


def test_format_initial_message_keeps_developer_instructions_on_remote_thread():
    from hermes_cli.chatgpt_web import _format_initial_message

    prompt = _format_initial_message(
        instructions="You are Hermes Agent. Use tools before answering.",
        messages=[
            {"role": "assistant", "content": "Earlier answer."},
            {"role": "user", "content": "Continue from the latest step."},
        ],
        has_remote_thread=True,
    )

    assert "Developer instructions" in prompt
    assert "You are Hermes Agent. Use tools before answering." in prompt
    assert "Latest user request:\nContinue from the latest step." in prompt


def test_format_initial_message_keeps_latest_tool_context_on_remote_thread():
    from hermes_cli.chatgpt_web import _format_initial_message

    prompt = _format_initial_message(
        instructions="Use tools before answering.",
        messages=[
            {"role": "user", "content": "Find the branch and then inspect Wikipedia."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command": "git rev-parse --abbrev-ref HEAD"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": '{"output":"main","exit_code":0,"error":null}'},
        ],
        has_remote_thread=True,
    )

    assert "Latest user request:\nFind the branch and then inspect Wikipedia." in prompt
    assert "Local Hermes context after that request" in prompt
    assert "<tool_call>" in prompt
    assert "\"name\": \"terminal\"" in prompt
    assert "<tool_response>" in prompt
    assert "\"output\":\"main\"" in prompt
    assert "same task" in prompt
    assert "not a new user request" in prompt


def test_format_initial_message_renders_multimodal_user_text_without_image_noise():
    from hermes_cli.chatgpt_web import _format_initial_message

    prompt = _format_initial_message(
        instructions="Follow Hermes rules.",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the attached image briefly."},
                    {"type": "input_image", "image_url": "file:///tmp/red-square.png"},
                ],
            },
        ],
        has_remote_thread=False,
    )

    assert "Describe the attached image briefly." in prompt
    assert "file:///tmp/red-square.png" not in prompt


def test_select_chatgpt_web_browser_response_text_falls_back_to_article_text():
    from hermes_cli.chatgpt_web import _select_chatgpt_web_browser_response_text

    text, model = _select_chatgpt_web_browser_response_text(
        {
            "assistant": [],
            "articles": [
                {"text": "Describe the attached image briefly.", "author": "user", "model": ""},
                {"text": "Describe the attached image briefly. The image is a red square.", "author": "assistant", "model": "gpt-5-4-thinking"},
            ],
            "mainText": "Describe the attached image briefly. The image is a red square.",
        },
        "Describe the attached image briefly.",
    )

    assert text == "The image is a red square."
    assert model == "gpt-5-4-thinking"


def test_select_chatgpt_web_browser_response_text_ignores_prompt_echo_page_chrome():
    from hermes_cli.chatgpt_web import _select_chatgpt_web_browser_response_text

    text, model = _select_chatgpt_web_browser_response_text(
        {
            "assistant": [],
            "articles": [],
            "mainText": (
                "Developer instructions: ... Conversation so far: User: Look at this local image. "
                "Reading documents Thinking ChatGPT can make mistakes. See Cookie Preferences."
            ),
        },
        "Look at this local image. Answer only with the dominant color and shape.",
    )

    assert text == ""
    assert model == ""


def test_materialize_chatgpt_web_browser_image_accepts_data_urls():
    from hermes_cli.chatgpt_web import _materialize_chatgpt_web_browser_image

    png_data_url = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0xQAAAAASUVORK5CYII="
    )

    materialized_path, cleanup_path = _materialize_chatgpt_web_browser_image(png_data_url)

    try:
        assert materialized_path == cleanup_path
        assert materialized_path.endswith(".png")
        with open(materialized_path, "rb") as handle:
            assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
    finally:
        if cleanup_path:
            import os
            if os.path.exists(cleanup_path):
                os.unlink(cleanup_path)


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


def test_main_accepts_chatgpt_web_for_auth_browser(monkeypatch):
    from hermes_cli import main as hermes_main

    seen = []
    monkeypatch.setattr(
        hermes_main,
        "cmd_auth",
        lambda args: seen.append((args.auth_action, args.provider, args.label, args.timeout, args.debug_port, args.keep_open)),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "auth",
            "browser",
            "chatgpt-web",
            "--label",
            "termux-x11-browser",
            "--timeout",
            "42",
            "--debug-port",
            "9333",
            "--keep-open",
        ],
    )
    hermes_main.main()

    assert seen == [("browser", "chatgpt-web", "termux-x11-browser", 42, 9333, True)]


def test_main_accepts_chatgpt_web_for_chat_provider(monkeypatch):
    from hermes_cli import main as hermes_main

    seen = []
    monkeypatch.setattr(hermes_main, "cmd_chat", lambda args: seen.append(args.provider))

    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "chat", "--provider", "chatgpt-web", "-q", "hello"],
    )
    hermes_main.main()

    assert seen == ["chatgpt-web"]


def test_provider_label_chatgpt_web_is_human_readable():
    assert get_label("chatgpt-web") == "ChatGPT Web"


def test_get_chatgpt_web_auth_status_prefers_chatgpt_web_pool(tmp_path, monkeypatch):
    from hermes_cli.auth import get_chatgpt_web_auth_status
    from hermes_cli.chatgpt_web import DEFAULT_CHATGPT_WEB_BASE_URL

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("CHATGPT_WEB_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CHATGPT_WEB_SESSION_TOKEN", raising=False)

    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "chatgpt-web": [
                        {
                            "id": "cred-web",
                            "label": "web-account",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "manual:device_code",
                            "access_token": "chatgpt-web-token",
                            "base_url": DEFAULT_CHATGPT_WEB_BASE_URL,
                            "last_refresh": "2026-03-23T10:00:00Z",
                        }
                    ]
                },
            }
        )
    )

    status = get_chatgpt_web_auth_status()

    assert status["logged_in"] is True
    assert status["auth_mode"] == "oauth"
    assert status["source"] == "pool:web-account"
    assert status["api_key"] == "chatgpt-web-token"


def test_get_chatgpt_web_auth_status_accepts_pool_session_token(tmp_path, monkeypatch):
    from hermes_cli.auth import get_chatgpt_web_auth_status
    from hermes_cli.chatgpt_web import DEFAULT_CHATGPT_WEB_BASE_URL

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("CHATGPT_WEB_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CHATGPT_WEB_SESSION_TOKEN", raising=False)

    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "chatgpt-web": [
                        {
                            "id": "cred-web",
                            "label": "session-cookie",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "manual:session_token",
                            "access_token": "",
                            "session_token": "session-cookie-token",
                            "base_url": DEFAULT_CHATGPT_WEB_BASE_URL,
                        }
                    ]
                },
            }
        )
    )

    status = get_chatgpt_web_auth_status()

    assert status["logged_in"] is True
    assert status["auth_mode"] == "session_token"
    assert status["source"] == "pool:session-cookie"
    assert status["api_key"] == ""


def test_resolve_chatgpt_web_runtime_credentials_prefers_pool_entry(tmp_path, monkeypatch):
    from hermes_cli.chatgpt_web import DEFAULT_CHATGPT_WEB_BASE_URL, resolve_chatgpt_web_runtime_credentials

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("CHATGPT_WEB_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CHATGPT_WEB_SESSION_TOKEN", raising=False)

    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "chatgpt-web": [
                        {
                            "id": "cred-web",
                            "label": "web-account",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "manual:device_code",
                            "access_token": "chatgpt-web-token",
                            "base_url": DEFAULT_CHATGPT_WEB_BASE_URL,
                            "last_refresh": "2026-03-23T10:00:00Z",
                        }
                    ]
                },
            }
        )
    )

    creds = resolve_chatgpt_web_runtime_credentials()

    assert creds["provider"] == "chatgpt-web"
    assert creds["api_key"] == "chatgpt-web-token"
    assert creds["base_url"] == DEFAULT_CHATGPT_WEB_BASE_URL
    assert creds["source"] == "pool:web-account"


def test_resolve_chatgpt_web_runtime_credentials_refreshes_pool_session_token(tmp_path, monkeypatch):
    from hermes_cli.chatgpt_web import DEFAULT_CHATGPT_WEB_BASE_URL, resolve_chatgpt_web_runtime_credentials

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("CHATGPT_WEB_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CHATGPT_WEB_SESSION_TOKEN", raising=False)
    monkeypatch.setattr(
        "hermes_cli.chatgpt_web._fetch_chatgpt_web_access_token_from_session",
        lambda session_token, **kwargs: "refreshed-web-access-token",
    )

    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "chatgpt-web": [
                        {
                            "id": "cred-web",
                            "label": "session-cookie",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "manual:session_token",
                            "access_token": "",
                            "session_token": "session-cookie-token",
                            "cookie_header": "cf_clearance=cf-cookie; oai-did=device-cookie",
                            "browser_cookies": [{"name": "cf_clearance", "value": "cf-cookie"}],
                            "device_id": "device-cookie",
                            "user_agent": "Mozilla/Test",
                            "base_url": DEFAULT_CHATGPT_WEB_BASE_URL,
                        }
                    ]
                },
            }
        )
    )

    creds = resolve_chatgpt_web_runtime_credentials()

    assert creds["provider"] == "chatgpt-web"
    assert creds["api_key"] == "refreshed-web-access-token"
    assert creds["base_url"] == DEFAULT_CHATGPT_WEB_BASE_URL
    assert creds["source"] == "pool:session-cookie"
    assert creds["session_token"] == "session-cookie-token"
    assert creds["cookie_header"] == "cf_clearance=cf-cookie; oai-did=device-cookie"
    assert creds["browser_cookies"] == [{"name": "cf_clearance", "value": "cf-cookie"}]
    assert creds["device_id"] == "device-cookie"
    assert creds["user_agent"] == "Mozilla/Test"


def test_resolve_chatgpt_web_runtime_credentials_force_refreshes_exhausted_pool_entry(tmp_path, monkeypatch):
    from hermes_cli.chatgpt_web import DEFAULT_CHATGPT_WEB_BASE_URL, resolve_chatgpt_web_runtime_credentials

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("CHATGPT_WEB_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CHATGPT_WEB_SESSION_TOKEN", raising=False)
    monkeypatch.setattr(
        "hermes_cli.chatgpt_web._fetch_chatgpt_web_access_token_from_session",
        lambda session_token, **kwargs: "fresh-access-token",
    )
    auth_path = hermes_home / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "chatgpt-web": [
                        {
                            "id": "cred-web",
                            "label": "desktop-browser",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "manual:session_token",
                            "access_token": "stale-access-token",
                            "session_token": "session-cookie-token",
                            "last_status": "exhausted",
                            "last_status_at": 123.0,
                            "last_error_code": 401,
                            "last_error_message": "expired",
                            "base_url": DEFAULT_CHATGPT_WEB_BASE_URL,
                        }
                    ]
                },
            }
        )
    )

    creds = resolve_chatgpt_web_runtime_credentials(force_refresh=True)

    assert creds["provider"] == "chatgpt-web"
    assert creds["api_key"] == "fresh-access-token"
    assert creds["source"] == "pool:desktop-browser"

    saved = json.loads(auth_path.read_text())
    refreshed_entry = saved["credential_pool"]["chatgpt-web"][0]
    assert refreshed_entry["access_token"] == "fresh-access-token"
    assert refreshed_entry.get("last_status") is None
    assert refreshed_entry.get("last_error_code") is None


def test_build_chatgpt_web_headers_merge_browser_cookie_state():
    from hermes_cli.chatgpt_web import _build_chatgpt_web_headers

    headers = _build_chatgpt_web_headers(
        access_token="chatgpt-web-token",
        session_token="session-cookie",
        cookie_header="cf_clearance=cf-cookie",
        browser_cookies=[{"name": "extra_cookie", "value": "extra-value"}],
        device_id="device-cookie",
        user_agent="Mozilla/Test",
        accept="text/event-stream",
    )

    assert headers["Authorization"] == "Bearer chatgpt-web-token"
    assert headers["User-Agent"] == "Mozilla/Test"
    assert headers["Accept"] == "text/event-stream"
    assert "cf_clearance=cf-cookie" in headers["Cookie"]
    assert "extra_cookie=extra-value" in headers["Cookie"]
    assert "__Secure-next-auth.session-token=session-cookie" in headers["Cookie"]
    assert "oai-did=device-cookie" in headers["Cookie"]


def test_format_initial_message_includes_tool_calls_and_tool_responses():
    from hermes_cli.chatgpt_web import _format_initial_message

    prompt = _format_initial_message(
        instructions="Use tools when needed.",
        has_remote_thread=False,
        messages=[
            {"role": "user", "content": "Find the file."},
            {
                "role": "assistant",
                "content": "I will inspect the repo.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_files",
                            "arguments": '{"pattern": "chatgpt_web.py"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": '{"matches":["hermes_cli/chatgpt_web.py"]}'},
            {"role": "user", "content": "Continue."},
        ],
    )

    assert "Developer instructions (higher priority than the conversation below):" in prompt
    assert "<tool_call>" in prompt
    assert "search_files" in prompt
    assert "<tool_response>" in prompt
    assert "hermes_cli/chatgpt_web.py" in prompt
    assert "Hermes continuation hint" in prompt
    assert "Do not reply with progress narration like 'I will continue'." in prompt



def test_stream_chatgpt_web_completion_parses_patch_events(monkeypatch):
    from hermes_cli.chatgpt_web import stream_chatgpt_web_completion

    deltas = []

    class _JSONResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("boom", request=None, response=None)

        def json(self):
            return self._payload

    class _StreamResponse:
        def __init__(self, lines):
            self._lines = lines
            self.status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield from self._lines

    class _Client:
        def post(self, url, headers=None, json=None):
            if url.endswith("/conversation/prepare"):
                return _JSONResponse({"conduit_token": "conduit-123"})
            if url.endswith("/sentinel/chat-requirements"):
                return _JSONResponse({"token": "req-token", "proofofwork": {}})
            raise AssertionError(f"unexpected POST {url}")

        def stream(self, method, url, headers=None, json=None):
            assert method == "POST"
            assert url.endswith("/conversation")
            return _StreamResponse([
                'data: "v1"',
                'data: {"conversation_id":"conv_123","type":"resume_conversation_token"}',
                'data: {"v":{"message":{"id":"msg_123","author":{"role":"assistant"},"content":{"content_type":"text","parts":[""]},"status":"in_progress","metadata":{}}}}',
                'data: {"o":"patch","v":[{"p":"/message/content/parts/0","o":"append","v":"Hel"},{"p":"/message/content/parts/0","o":"append","v":"lo"},{"p":"/message/status","o":"replace","v":"finished_successfully"}]}',
                'data: {"type":"message_stream_complete","conversation_id":"conv_123"}',
                'data: [DONE]',
            ])

    result = stream_chatgpt_web_completion(
        access_token="chatgpt-web-token",
        model="gpt-5-thinking",
        messages=[{"role": "user", "content": "hello"}],
        on_delta=deltas.append,
        client=_Client(),
        history_and_training_disabled=True,
    )

    assert result["content"] == "Hello"
    assert result["conversation_id"] == "conv_123"
    assert result["message_id"] == "msg_123"
    assert deltas == ["Hel", "lo"]



def test_stream_chatgpt_web_completion_parses_patch_events_without_outer_op(monkeypatch):
    from hermes_cli.chatgpt_web import stream_chatgpt_web_completion

    deltas = []

    class _JSONResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("boom", request=None, response=None)

        def json(self):
            return self._payload

    class _StreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'data: "v1"'
            yield 'data: {"conversation_id":"conv_789","type":"resume_conversation_token"}'
            yield 'data: {"v":{"message":{"id":"msg_789","author":{"role":"assistant"},"content":{"content_type":"text","parts":[""]},"status":"in_progress","metadata":{}}}}'
            yield 'data: {"o":"patch","v":[{"p":"/message/content/parts/0","o":"append","v":"Yes,"}]}'
            yield 'data: {"v":[{"p":"/message/content/parts/0","o":"append","v":" tools are active"}]}'
            yield 'data: {"v":[{"p":"/message/content/parts/0","o":"append","v":" and ready."},{"p":"/message/status","o":"replace","v":"finished_successfully"}]}'
            yield 'data: {"type":"message_stream_complete","conversation_id":"conv_789"}'
            yield 'data: [DONE]'

    class _Client:
        def post(self, url, headers=None, json=None):
            if url.endswith("/conversation/prepare"):
                return _JSONResponse({"conduit_token": "conduit-789"})
            if url.endswith("/sentinel/chat-requirements"):
                return _JSONResponse({"token": "***", "proofofwork": {}})
            raise AssertionError(f"unexpected POST {url}")

        def stream(self, method, url, headers=None, json=None):
            assert method == "POST"
            assert url.endswith("/conversation")
            return _StreamResponse()

    result = stream_chatgpt_web_completion(
        access_token="chatgpt-web-token",
        model="gpt-5-thinking",
        messages=[{"role": "user", "content": "hello"}],
        on_delta=deltas.append,
        client=_Client(),
        history_and_training_disabled=True,
    )

    assert result["content"] == "Yes, tools are active and ready."
    assert result["conversation_id"] == "conv_789"
    assert result["message_id"] == "msg_789"
    assert deltas == ["Yes,", " tools are active", " and ready."]


def test_stream_chatgpt_web_completion_prefers_http_transport_for_text_turns(monkeypatch):
    from hermes_cli import chatgpt_web as chatgpt_web_mod

    monkeypatch.setattr(chatgpt_web_mod, "_chatgpt_web_debug_base", lambda: "http://127.0.0.1:9225")
    monkeypatch.delenv("CHATGPT_WEB_FORCE_BROWSER_FETCH", raising=False)
    monkeypatch.setattr(
        chatgpt_web_mod,
        "_chatgpt_web_browser_fetch_sync",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("browser transport should stay disabled for text turns")),
    )

    class _JSONResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("boom", request=None, response=None)

        def json(self):
            return self._payload

    class _StreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'data: {"conversation_id":"conv_http","type":"resume_conversation_token"}'
            yield 'data: {"v":{"message":{"id":"msg_http","author":{"role":"assistant"},"content":{"content_type":"text","parts":["OK over HTTP"]},"status":"finished_successfully","metadata":{}}}}'
            yield 'data: {"type":"message_stream_complete","conversation_id":"conv_http"}'
            yield 'data: [DONE]'

    class _Client:
        def post(self, url, headers=None, json=None):
            if url.endswith("/conversation/prepare"):
                return _JSONResponse({"conduit_token": "conduit-http"})
            if url.endswith("/sentinel/chat-requirements"):
                return _JSONResponse({"token": "***", "proofofwork": {}})
            raise AssertionError(f"unexpected POST {url}")

        def stream(self, method, url, headers=None, json=None):
            assert method == "POST"
            assert url.endswith("/conversation")
            return _StreamResponse()

    result = chatgpt_web_mod.stream_chatgpt_web_completion(
        access_token="chatgpt-web-token",
        model="gpt-5-thinking",
        messages=[{"role": "user", "content": "hello"}],
        client=_Client(),
        history_and_training_disabled=True,
    )

    assert result["content"] == "OK over HTTP"
    assert result["conversation_id"] == "conv_http"
    assert result["message_id"] == "msg_http"



def test_stream_chatgpt_web_completion_auto_uses_browser_transport_for_browser_backed_session(monkeypatch):
    from hermes_cli import chatgpt_web as chatgpt_web_mod

    browser_calls = []

    monkeypatch.setattr(chatgpt_web_mod, "_chatgpt_web_debug_base", lambda: "http://127.0.0.1:9227")
    monkeypatch.delenv("CHATGPT_WEB_FORCE_BROWSER_FETCH", raising=False)

    def _fake_browser_fetch_sync(**kwargs):
        browser_calls.append((kwargs["method"], kwargs["url"]))
        url = kwargs["url"]
        if url.endswith("/conversation/prepare"):
            return {"status": 200, "text": '{"conduit_token":"conduit-browser"}'}
        if url.endswith("/sentinel/chat-requirements"):
            return {"status": 200, "text": '{"token":"req-browser","proofofwork":{}}'}
        if url.endswith("/conversation"):
            return {
                "status": 200,
                "text": "\n".join([
                    'data: {"conversation_id":"conv_browser_text","type":"resume_conversation_token"}',
                    'data: {"v":{"message":{"id":"msg_browser_text","author":{"role":"assistant"},"content":{"content_type":"text","parts":["READY"]},"status":"finished_successfully","metadata":{}}}}',
                    'data: {"type":"message_stream_complete","conversation_id":"conv_browser_text"}',
                    "data: [DONE]",
                ]),
            }
        raise AssertionError(f"unexpected browser fetch {url}")

    monkeypatch.setattr(chatgpt_web_mod, "_chatgpt_web_browser_fetch_sync", _fake_browser_fetch_sync)

    class _Client:
        def post(self, url, headers=None, json=None):
            raise AssertionError("HTTP POST transport should stay disabled for browser-backed sessions")

        def stream(self, method, url, headers=None, json=None):
            raise AssertionError("HTTP stream transport should stay disabled for browser-backed sessions")

    result = chatgpt_web_mod.stream_chatgpt_web_completion(
        access_token="chatgpt-web-token",
        model="gpt-5-thinking",
        messages=[{"role": "user", "content": "hello"}],
        client=_Client(),
        history_and_training_disabled=True,
        session_token="session-123",
    )

    assert result["content"] == "READY"
    assert result["conversation_id"] == "conv_browser_text"
    assert result["message_id"] == "msg_browser_text"
    assert browser_calls == [
        ("POST", "https://chatgpt.com/backend-api/f/conversation/prepare"),
        ("POST", "https://chatgpt.com/backend-api/sentinel/chat-requirements"),
        ("POST", "https://chatgpt.com/backend-api/f/conversation"),
    ]


def test_stream_chatgpt_web_completion_tolerates_protocol_close_after_completion(monkeypatch):
    from hermes_cli.chatgpt_web import stream_chatgpt_web_completion

    class _JSONResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("boom", request=None, response=None)

        def json(self):
            return self._payload

    class _StreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'data: {"conversation_id":"conv_456","type":"resume_conversation_token"}'
            yield 'data: {"v":{"message":{"id":"msg_456","author":{"role":"assistant"},"content":{"content_type":"text","parts":[""]},"status":"in_progress","metadata":{}}}}'
            yield 'data: {"o":"patch","v":[{"p":"/message/content/parts/0","o":"append","v":"OK"}]}'
            yield 'data: {"type":"message_stream_complete","conversation_id":"conv_456"}'
            raise httpx.RemoteProtocolError("incomplete chunked read")

    class _Client:
        def post(self, url, headers=None, json=None):
            if url.endswith("/conversation/prepare"):
                return _JSONResponse({"conduit_token": "conduit-456"})
            if url.endswith("/sentinel/chat-requirements"):
                return _JSONResponse({"token": "***", "proofofwork": {}})
            raise AssertionError(f"unexpected POST {url}")

        def stream(self, method, url, headers=None, json=None):
            assert method == "POST"
            assert url.endswith("/conversation")
            return _StreamResponse()

    result = stream_chatgpt_web_completion(
        access_token="chatgpt-web-token",
        model="gpt-5-thinking",
        messages=[{"role": "user", "content": "hello"}],
        client=_Client(),
        history_and_training_disabled=True,
    )

    assert result["content"] == "OK"
    assert result["conversation_id"] == "conv_456"
    assert result["message_id"] == "msg_456"


def test_stream_chatgpt_web_completion_routes_multimodal_turns_through_browser(monkeypatch):
    from hermes_cli import chatgpt_web as chatgpt_web_mod

    captured = {}

    async def _fake_browser_multimodal_completion(**kwargs):
        captured.update(kwargs)
        return {
            "content": "red square",
            "conversation_id": "conv_browser",
            "message_id": "msg_browser",
            "parent_message_id": "msg_browser",
            "model": "gpt-5-thinking",
            "finish_reason": "stop",
            "images": [],
        }

    monkeypatch.setattr(chatgpt_web_mod, "_chatgpt_web_debug_base", lambda: "http://127.0.0.1:9225")
    monkeypatch.setattr(chatgpt_web_mod, "_chatgpt_web_browser_multimodal_completion", _fake_browser_multimodal_completion)

    result = chatgpt_web_mod.stream_chatgpt_web_completion(
        access_token="chatgpt-web-token",
        model="gpt-5-thinking",
        session_token="session-123",
        browser_cookies=[{"name": "oai-did", "value": "device-1", "domain": "chatgpt.com"}],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the attached image."},
                    {"type": "input_image", "image_url": "file:///tmp/red-square.png"},
                ],
            }
        ],
        timeout=15,
    )

    assert result["content"] == "red square"
    assert captured["debug_base"] == "http://127.0.0.1:9225"
    assert captured["model"] == "gpt-5-thinking"
    assert captured["prompt_text"].startswith("Conversation so far:")
    assert captured["image_sources"] == ["file:///tmp/red-square.png"]
    assert captured["session_token"] == "session-123"
    assert captured["browser_cookies"] == [{"name": "oai-did", "value": "device-1", "domain": "chatgpt.com"}]


def test_chatgpt_web_browser_auth_cookies_adds_session_cookie():
    from hermes_cli.chatgpt_web import _chatgpt_web_browser_auth_cookies

    cookies = _chatgpt_web_browser_auth_cookies(
        session_token="session-abc",
        browser_cookies=[{"name": "oai-did", "value": "device-1", "domain": "chatgpt.com"}],
    )

    assert any(item["name"] == "oai-did" for item in cookies)
    assert any(
        item["name"] == "__Secure-next-auth.session-token"
        and item["value"] == "session-abc"
        and item["domain"] == "chatgpt.com"
        for item in cookies
    )


def test_chatgpt_web_browser_fetch_opens_chatgpt_target_when_no_tab_exists(monkeypatch):
    from hermes_cli import chatgpt_web as chatgpt_web_mod

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps([
                {"id": "wiki", "type": "page", "url": "https://en.wikipedia.org/wiki/OpenAI"},
            ]).encode("utf-8")

    class _FakeWebSocket:
        def __init__(self):
            self._messages = [
                json.dumps({"id": 1, "result": {}}),
                json.dumps(
                    {
                        "id": 2,
                        "result": {},
                    }
                ),
                json.dumps(
                    {
                        "id": 3,
                        "result": {},
                    }
                ),
                json.dumps(
                    {
                        "id": 4,
                        "result": {},
                    }
                ),
                json.dumps(
                    {
                        "id": 5,
                        "result": {
                            "result": {
                                "value": json.dumps({"status": 200, "ok": True, "text": '{"ok":true}'})
                            }
                        },
                    }
                ),
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def send(self, payload):
            return None

        async def recv(self):
            return self._messages.pop(0)

    class _FakeWebSockets:
        def connect(self, ws_url, max_size=None):
            assert ws_url == "ws://127.0.0.1:9225/devtools/page/chatgpt"
            return _FakeWebSocket()

    created = {}

    monkeypatch.setattr(chatgpt_web_mod, "websockets", _FakeWebSockets())
    monkeypatch.setattr(chatgpt_web_mod.urllib.request, "urlopen", lambda *args, **kwargs: _Response())
    async def _fake_wait_for_location(ws, next_id, **kwargs):
        return {"href": "https://chatgpt.com/", "readyState": "complete"}

    monkeypatch.setattr(chatgpt_web_mod, "_chatgpt_web_browser_wait_for_location", _fake_wait_for_location)

    async def _fake_create_target(debug_base, url):
        created["debug_base"] = debug_base
        created["url"] = url
        return "chatgpt-target"

    monkeypatch.setattr(chatgpt_web_mod, "_chatgpt_web_browser_create_target", _fake_create_target)
    monkeypatch.setattr(
        chatgpt_web_mod,
        "_chatgpt_web_browser_page_target",
        lambda debug_base, target_id: {
            "id": target_id,
            "type": "page",
            "url": "https://chatgpt.com/",
            "webSocketDebuggerUrl": "ws://127.0.0.1:9225/devtools/page/chatgpt",
        },
    )

    result = chatgpt_web_mod._chatgpt_web_browser_fetch_sync(
        debug_base="http://127.0.0.1:9225",
        url="https://chatgpt.com/backend-api/f/conversation/prepare",
    )

    assert created == {
        "debug_base": "http://127.0.0.1:9225",
        "url": "https://chatgpt.com/",
    }
    assert result == {"status": 200, "ok": True, "text": '{"ok":true}'}


def test_stream_chatgpt_web_completion_resolves_async_generated_images(monkeypatch):
    from hermes_cli.chatgpt_web import stream_chatgpt_web_completion

    class _JSONResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("boom", request=None, response=None)

        def json(self):
            return self._payload

    class _StreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'data: {"conversation_id":"conv_img","type":"resume_conversation_token"}'
            yield 'data: {"v":{"message":{"id":"msg_json","author":{"role":"assistant"},"content":{"content_type":"text","parts":[""]},"status":"in_progress","metadata":{"image_gen_multi_stream":true}}}}'
            yield 'data: {"o":"patch","v":[{"p":"/message/content/parts/0","o":"append","v":"{\\n  \\\"prompt\\\": \\\"A rainforest\\\",\\n  \\\"size\\\": \\\"1024x1024\\\"\\n}"}]}'
            yield 'data: {"v":{"message":{"id":"msg_image","author":{"role":"tool","name":"image_tool"},"content":{"content_type":"text","parts":["Processing image"]},"status":"in_progress","metadata":{"image_gen_task_id":"task_123"}}}}'
            yield 'data: {"type":"server_ste_metadata","conversation_id":"conv_img","metadata":{"tool_name":"ImageGenToolTemporal","turn_use_case":"image gen"}}'
            yield 'data: {"type":"message_stream_complete","conversation_id":"conv_img"}'
            yield 'data: [DONE]'

    class _Client:
        def post(self, url, headers=None, json=None):
            if url.endswith("/conversation/prepare"):
                return _JSONResponse({"conduit_token": "conduit-img"})
            if url.endswith("/sentinel/chat-requirements"):
                return _JSONResponse({"token": "***", "proofofwork": {}})
            raise AssertionError(f"unexpected POST {url}")

        def get(self, url, headers=None, params=None):
            if url.endswith("/conversation/conv_img"):
                return _JSONResponse({
                    "conversation_id": "conv_img",
                    "current_node": "msg_image",
                    "mapping": {
                        "msg_image": {
                            "id": "msg_image",
                            "message": {
                                "id": "msg_image",
                                "author": {"role": "tool", "name": "image_tool"},
                                "content": {
                                    "content_type": "multimodal_text",
                                    "parts": [{
                                        "content_type": "image_asset_pointer",
                                        "asset_pointer": "sediment://file_abc123",
                                        "width": 1024,
                                        "height": 1536,
                                        "size_bytes": 123,
                                        "metadata": {"generation": {"orientation": "portrait"}},
                                    }],
                                },
                                "metadata": {"async_task_id": "task_123"},
                            },
                            "parent": None,
                        }
                    },
                })
            if url.endswith("/files/download/file_abc123"):
                assert params == {"inline": "false", "conversation_id": "conv_img"}
                return _JSONResponse({
                    "status": "success",
                    "download_url": "https://chatgpt.com/backend-api/estuary/content?id=file_abc123&sig=xyz",
                    "file_name": "rainforest.png",
                    "mime_type": "image/png",
                    "file_size_bytes": 123,
                })
            raise AssertionError(f"unexpected GET {url}")

        def stream(self, method, url, headers=None, json=None):
            assert method == "POST"
            assert url.endswith("/conversation")
            return _StreamResponse()

    result = stream_chatgpt_web_completion(
        access_token="chatgpt-web-token",
        model="gpt-5-3-instant",
        messages=[{"role": "user", "content": "Generate a rainforest image"}],
        client=_Client(),
        history_and_training_disabled=False,
        timeout=30,
    )

    assert result["content"] == "https://chatgpt.com/backend-api/estuary/content?id=file_abc123&sig=xyz"
    assert result["conversation_id"] == "conv_img"
    assert result["message_id"] == "msg_image"
    assert result["images"][0]["file_id"] == "file_abc123"
    assert result["images"][0]["file_name"] == "rainforest.png"


def test_stream_chatgpt_web_completion_does_not_extend_timeout_for_image_polling(monkeypatch):
    from hermes_cli import chatgpt_web as chatgpt_web_mod

    captured = {}

    class _JSONResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("boom", request=None, response=None)

        def json(self):
            return self._payload

    class _StreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'data: {"conversation_id":"conv_budget","type":"resume_conversation_token"}'
            yield 'data: {"v":{"message":{"id":"msg_budget","author":{"role":"assistant"},"content":{"content_type":"text","parts":[""]},"status":"in_progress","metadata":{"image_gen_multi_stream":true}}}}'
            yield 'data: {"o":"patch","v":[{"p":"/message/content/parts/0","o":"append","v":"{\\n  \\\"prompt\\\": \\\"A rainforest\\\",\\n  \\\"size\\\": \\\"1024x1024\\\"\\n}"}]}'
            yield 'data: {"type":"message_stream_complete","conversation_id":"conv_budget"}'
            yield 'data: [DONE]'

    class _Client:
        def post(self, url, headers=None, json=None):
            if url.endswith("/conversation/prepare"):
                return _JSONResponse({"conduit_token": "conduit-budget"})
            if url.endswith("/sentinel/chat-requirements"):
                return _JSONResponse({"token": "***", "proofofwork": {}})
            raise AssertionError(f"unexpected POST {url}")

        def stream(self, method, url, headers=None, json=None):
            return _StreamResponse()

    monotonic_values = iter([100.0, 102.0])
    monkeypatch.setattr(chatgpt_web_mod.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        chatgpt_web_mod,
        "_resolve_chatgpt_web_generated_images",
        lambda *args, **kwargs: captured.setdefault("timeout", kwargs["timeout"]) or [],
    )

    result = chatgpt_web_mod.stream_chatgpt_web_completion(
        access_token="chatgpt-web-token",
        model="gpt-5-3-instant",
        messages=[{"role": "user", "content": "Generate a rainforest image"}],
        client=_Client(),
        history_and_training_disabled=False,
        timeout=1.0,
    )

    assert captured == {}
    assert result["images"] == []
    assert "prompt" in result["content"]
