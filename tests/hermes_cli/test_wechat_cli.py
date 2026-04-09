from argparse import Namespace

import json
from pathlib import Path

from gateway.platforms.wechat import WeChatSessionStore
from hermes_cli.wechat import (
    _extract_base_url,
    _extract_token,
    _poll_bind_status,
    _resolve_base_url,
    wechat_command,
)


def _make_args(**kwargs):
    defaults = {
        "wechat_action": None,
        "base_url": "",
        "listen_host": "127.0.0.1",
        "listen_port": 0,
        "timeout": 10,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_extract_token_and_base_url_nested():
    payload = {
        "bind": {
            "auth": {
                "access_token": "token-123",
                "baseUrl": "http://127.0.0.1:8787",
            }
        }
    }
    assert _extract_token(payload) == "token-123"
    assert _extract_base_url(payload) == "http://127.0.0.1:8787"


def test_extract_token_prefers_bot_token_over_generic_token():
    payload = {
        "status": "confirmed",
        "token": "temporary-token",
        "bot_token": "real-bot-token",
    }
    assert _extract_token(payload) == "real-bot-token"


def test_resolve_base_url_rejects_non_localhost(monkeypatch):
    monkeypatch.setenv("WECHAT_ILINK_URL", "https://example.com")
    assert _resolve_base_url("") == ""


def test_resolve_base_url_accepts_ilink_https(monkeypatch):
    monkeypatch.setenv("WECHAT_ILINK_URL", "https://ilinkai.weixin.qq.com")
    assert _resolve_base_url("") == "https://ilinkai.weixin.qq.com"


def test_resolve_base_url_rejects_ilink_http(monkeypatch):
    monkeypatch.setenv("WECHAT_ILINK_URL", "http://ilinkai.weixin.qq.com")
    assert _resolve_base_url("") == ""


def test_wechat_bind_persists_token_when_validation_passes(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("WECHAT_ILINK_URL", "https://ilinkai.weixin.qq.com")

    bind_status_payload = {
        "bot_token": "bind-token",
        "baseurl": "https://ilinkai.weixin.qq.com",
        "context_tokens": {"chat": "ctx"},
    }
    monkeypatch.setattr(
        "hermes_cli.wechat._request_bind_qrcode",
        lambda *_: (True, {"qrcode": "scan-me", "qrcode_img_content": "https://example.com/qr"}),
    )
    monkeypatch.setattr(
        "hermes_cli.wechat._poll_bind_status",
        lambda **_: (True, bind_status_payload),
    )
    monkeypatch.setattr("hermes_cli.wechat._validate_session_config", lambda *_: (True, "ok"))
    monkeypatch.setattr("hermes_cli.wechat._print_qr", lambda *_: None)

    wechat_command(_make_args(wechat_action="bind"))

    out = capsys.readouterr().out
    assert "WeChat binding completed." in out
    assert "QR URL:" in out
    assert "https://example.com/qr" in out
    state = WeChatSessionStore().load()
    assert state.bearer_token == "bind-token"
    assert state.base_url == "https://ilinkai.weixin.qq.com"
    assert state.context_tokens.get("chat") == "ctx"


def test_wechat_bind_clears_stale_cursor_on_fresh_bind(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("WECHAT_ILINK_URL", "https://ilinkai.weixin.qq.com")

    stale_state = WeChatSessionStore().load()
    stale_state.bearer_token = "old-token"
    stale_state.base_url = "https://ilinkai.weixin.qq.com"
    stale_state.cursor = "stale-cursor"
    stale_state.context_tokens = {"old": "context"}
    WeChatSessionStore().save(stale_state)

    bind_status_payload = {
        "bot_token": "new-token",
        "baseurl": "https://ilinkai.weixin.qq.com",
        "context_tokens": {"chat": "ctx"},
    }
    monkeypatch.setattr(
        "hermes_cli.wechat._request_bind_qrcode",
        lambda *_: (True, {"qrcode": "scan-me", "qrcode_img_content": "https://example.com/qr"}),
    )
    monkeypatch.setattr(
        "hermes_cli.wechat._poll_bind_status",
        lambda **_: (True, bind_status_payload),
    )
    monkeypatch.setattr("hermes_cli.wechat._validate_session_config", lambda *_: (True, "ok"))
    monkeypatch.setattr("hermes_cli.wechat._print_qr", lambda *_: None)

    wechat_command(_make_args(wechat_action="bind"))

    out = capsys.readouterr().out
    assert "WeChat binding completed." in out
    state = WeChatSessionStore().load()
    assert state.bearer_token == "new-token"
    assert state.context_tokens == {"chat": "ctx"}
    assert state.cursor is None


def test_wechat_bind_timeout(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("WECHAT_ILINK_URL", "https://ilinkai.weixin.qq.com")
    monkeypatch.setattr(
        "hermes_cli.wechat._request_bind_qrcode",
        lambda *_: (True, {"qrcode": "scan-me"}),
    )
    monkeypatch.setattr(
        "hermes_cli.wechat._poll_bind_status",
        lambda **_: (False, "Timed out waiting for QR confirmation."),
    )

    wechat_command(_make_args(wechat_action="bind", timeout=1))

    out = capsys.readouterr().out
    assert "Timed out waiting for QR confirmation." in out


def test_poll_bind_status_uses_remaining_deadline_for_http_timeout(monkeypatch):
    calls = []

    def fake_http_get_json(url, *, headers, timeout=15):
        calls.append(timeout)
        return True, {"status": "confirmed", "bot_token": "bind-token"}

    monkeypatch.setattr("hermes_cli.wechat._http_get_json", fake_http_get_json)

    ok, payload = _poll_bind_status(
        base_url="https://ilinkai.weixin.qq.com",
        qrcode="scan-me",
        timeout=30,
        poll_interval_seconds=2,
    )

    assert ok is True
    assert payload["status"] == "confirmed"
    assert len(calls) == 1
    assert 39 <= calls[0] <= 40


def test_poll_bind_status_treats_socket_timeout_as_wait(monkeypatch):
    calls = []

    def fake_http_get_json(url, *, headers, timeout=15):
        calls.append(timeout)
        if len(calls) == 1:
            return False, "The read operation timed out"
        return True, {"status": "confirmed", "bot_token": "bind-token"}

    monkeypatch.setattr("hermes_cli.wechat._http_get_json", fake_http_get_json)
    monkeypatch.setattr("hermes_cli.wechat.time.sleep", lambda *_: None)

    ok, payload = _poll_bind_status(
        base_url="https://ilinkai.weixin.qq.com",
        qrcode="scan-me",
        timeout=30,
        poll_interval_seconds=2,
    )

    assert ok is True
    assert payload["status"] == "confirmed"
    assert len(calls) == 2


def test_wechat_status_reports_runtime_and_session(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("WECHAT_ILINK_URL", "http://127.0.0.1:8787")

    state = WeChatSessionStore().load()
    state.bearer_token = "bind-token"
    WeChatSessionStore().save(state)

    (tmp_path / "gateway_state.json").write_text(
        json.dumps(
            {
                "gateway_state": "running",
                "platforms": {
                    "wechat": {
                        "state": "connected",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    wechat_command(_make_args(wechat_action="status"))

    out = capsys.readouterr().out
    assert "WeChat Status" in out
    assert "Base URL: configured" in out
    assert "Auth: configured" in out
    assert "Gateway: running" in out
    assert "Adapter state: connected" in out


def test_wechat_status_uses_profile_scoped_session_store(monkeypatch, tmp_path, capsys):
    profile_home = tmp_path / "profiles" / "owl"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("WECHAT_ILINK_URL", "http://127.0.0.1:8787")

    state = WeChatSessionStore().load()
    state.bearer_token = "profile-token"
    WeChatSessionStore().save(state)

    wechat_command(_make_args(wechat_action="status"))

    out = capsys.readouterr().out
    assert str(profile_home / ".wechat_session") in out
    assert "Auth: configured" in out
