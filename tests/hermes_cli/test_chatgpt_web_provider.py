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
                return _JSONResponse({"token": "req-token", "proofofwork": {}})
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
