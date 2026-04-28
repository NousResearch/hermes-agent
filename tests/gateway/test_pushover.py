"""Unit tests for the outbound-only Pushover platform adapter."""

from __future__ import annotations

import json

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms import pushover as pushover_module
from gateway.platforms.pushover import PushoverAdapter, _redacted_error, _redact_values


def _config(**extra):
    return PlatformConfig(
        enabled=True,
        token="tok",
        home_channel=HomeChannel(platform=Platform.PUSHOVER, chat_id="home", name="Pushover"),
        extra={"user": "usr", **extra},
    )


def test_build_payload_uses_explicit_device_not_chat_id():
    adapter = PushoverAdapter(_config(title="Hermes", device="iphone", priority=0, sound="pushover"))

    payload = adapter._build_payload(
        "arbitrary-channel-name",
        "hello",
        {"title": "Override", "priority": 1, "url": "https://example.test", "url_title": "Example"},
    )

    assert payload == {
        "token": "tok",
        "user": "usr",
        "message": "hello",
        "title": "Override",
        "device": "iphone",
        "priority": "1",
        "sound": "pushover",
        "url": "https://example.test",
        "url_title": "Example",
    }


def test_build_payload_metadata_device_overrides_default():
    adapter = PushoverAdapter(_config(device="iphone"))

    payload = adapter._build_payload("home", "hello", {"device": "ipad"})

    assert payload["device"] == "ipad"


def test_resolve_credentials_from_app_label_file(tmp_path):
    cred_file = tmp_path / "pushover.json"
    cred_file.write_text(
        json.dumps({"user": "fileusr", "apps": {"claude_code": "filetok"}}),
        encoding="utf-8",
    )
    adapter = PushoverAdapter(
        PlatformConfig(
            enabled=True,
            extra={"credentials_file": str(cred_file), "app": "claude_code"},
        )
    )

    assert adapter.app_label == "claude_code"
    assert adapter._token == "filetok"
    assert adapter._user == "fileusr"


def test_resolve_credentials_prefers_direct_config_over_file(tmp_path):
    cred_file = tmp_path / "pushover.json"
    cred_file.write_text(
        json.dumps({"user": "fileusr", "apps": {"claude_code": "filetok"}}),
        encoding="utf-8",
    )

    adapter = PushoverAdapter(_config(credentials_file=str(cred_file), app="claude_code"))

    assert adapter._token == "tok"
    assert adapter._user == "usr"


def test_resolve_credentials_malformed_json_degrades_gracefully(tmp_path):
    cred_file = tmp_path / "pushover.json"
    cred_file.write_text("not-json", encoding="utf-8")

    adapter = PushoverAdapter(PlatformConfig(enabled=True, extra={"credentials_file": str(cred_file)}))

    assert adapter._token == ""
    assert adapter._user == ""


def test_resolve_credentials_missing_app_falls_back_to_top_level_token(tmp_path):
    cred_file = tmp_path / "pushover.json"
    cred_file.write_text(
        json.dumps({"user": "fileusr", "token": "fallbacktok", "apps": {"other": "othertok"}}),
        encoding="utf-8",
    )

    adapter = PushoverAdapter(
        PlatformConfig(enabled=True, extra={"credentials_file": str(cred_file), "app": "missing"})
    )

    assert adapter._token == "fallbacktok"
    assert adapter._user == "fileusr"


def test_redaction_helpers_keep_context_and_remove_credentials():
    error = _redacted_error("HTTP 400 token=tok123&user=usr456 retry later")

    assert "HTTP 400" in error
    assert "retry later" in error
    assert "tok123" not in error
    assert "usr456" not in error
    assert _redact_values("failed for tok and usr", "tok", "usr") == "failed for [REDACTED] and [REDACTED]"


@pytest.mark.asyncio
async def test_connect_false_when_aiohttp_missing(monkeypatch):
    monkeypatch.setattr(pushover_module, "AIOHTTP_AVAILABLE", False)
    adapter = PushoverAdapter(_config())

    assert await adapter.connect() is False
    assert adapter.fatal_error_code == "pushover_dependency_missing"


@pytest.mark.asyncio
async def test_send_false_when_aiohttp_missing(monkeypatch):
    monkeypatch.setattr(pushover_module, "AIOHTTP_AVAILABLE", False)
    result = await PushoverAdapter(_config()).send("home", "hello")

    assert result.success is False
    assert result.retryable is False
    assert "aiohttp" in result.error.lower()


@pytest.mark.asyncio
async def test_connect_false_when_credentials_missing():
    adapter = PushoverAdapter(PlatformConfig(enabled=True, extra={"credentials_file": "/no/such/file"}))

    assert await adapter.connect() is False
    assert adapter.fatal_error_code == "pushover_credentials_missing"


@pytest.mark.asyncio
async def test_send_rejects_empty_before_session(monkeypatch):
    def fail_session(*args, **kwargs):
        raise AssertionError("session should not be opened")

    monkeypatch.setattr(pushover_module.aiohttp, "ClientSession", fail_session)
    result = await PushoverAdapter(_config()).send("home", "   ")

    assert result.success is False
    assert "empty" in result.error.lower()


@pytest.mark.asyncio
async def test_send_rejects_missing_credentials_before_session(monkeypatch):
    def fail_session(*args, **kwargs):
        raise AssertionError("session should not be opened")

    monkeypatch.setattr(pushover_module.aiohttp, "ClientSession", fail_session)
    adapter = PushoverAdapter(PlatformConfig(enabled=True, extra={"credentials_file": "/no/such/file"}))

    result = await adapter.send("home", "hello")

    assert result.success is False
    assert "credentials" in result.error.lower()


class _FakeResponse:
    def __init__(self, status: int, body: dict):
        self.status = status
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        if isinstance(self._body, str):
            return self._body
        return json.dumps(self._body)


class _FakeSession:
    posts = []
    responses = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, data):
        self.posts.append((url, data.copy()))
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_send_success_uses_one_session_and_reports_chunks(monkeypatch):
    _FakeSession.posts = []
    _FakeSession.responses = [_FakeResponse(200, {"status": 1, "request": "req-1"})]
    monkeypatch.setattr(pushover_module.aiohttp, "ClientSession", _FakeSession)

    result = await PushoverAdapter(_config()).send("home", "hello")

    assert result.success is True
    assert result.message_id == "req-1"
    assert result.raw_response["request"] == "req-1"
    assert result.raw_response["delivered_chunks"] == 1
    assert result.raw_response["requests"] == ["req-1"]
    assert _FakeSession.posts[0][1]["message"] == "hello"


@pytest.mark.asyncio
async def test_send_success_reports_all_chunk_request_ids(monkeypatch):
    _FakeSession.posts = []
    _FakeSession.responses = [
        _FakeResponse(200, {"status": 1, "request": "req-1"}),
        _FakeResponse(200, {"status": 1, "request": "req-2"}),
        _FakeResponse(200, {"status": 1, "request": "req-3"}),
    ]
    monkeypatch.setattr(pushover_module.aiohttp, "ClientSession", _FakeSession)
    adapter = PushoverAdapter(_config())
    adapter.truncate_message = lambda content, limit: ["hello", " ", "world"]

    result = await adapter.send("home", "hello world")

    assert result.success is True
    assert result.message_id == "req-3"
    assert result.raw_response["request"] == "req-3"
    assert result.raw_response["requests"] == ["req-1", "req-2", "req-3"]
    assert result.raw_response["delivered_chunks"] == 3


@pytest.mark.asyncio
async def test_send_failure_reports_partial_delivery(monkeypatch):
    _FakeSession.posts = []
    _FakeSession.responses = [
        _FakeResponse(200, {"status": 1, "request": "req-1"}),
        _FakeResponse(500, {"status": 0, "errors": ["server down"], "request": "req-2"}),
    ]
    monkeypatch.setattr(pushover_module.aiohttp, "ClientSession", _FakeSession)
    adapter = PushoverAdapter(_config())
    adapter.MAX_MESSAGE_LENGTH = 5

    result = await adapter.send("home", "hello world")

    assert result.success is False
    assert result.retryable is True
    assert result.raw_response["delivered_chunks"] == 1
    assert result.raw_response["total_chunks"] >= 2
    assert result.raw_response["request"] == "req-2"
    assert result.raw_response["requests"] == ["req-1", "req-2"]


@pytest.mark.asyncio
async def test_validate_success_sanitizes_response(monkeypatch):
    _FakeSession.posts = []
    _FakeSession.responses = [_FakeResponse(200, {"status": 1, "devices": ["iphone"], "licenses": ["iOS"]})]
    monkeypatch.setattr(pushover_module.aiohttp, "ClientSession", _FakeSession)

    result = await PushoverAdapter(_config()).validate()

    assert result.success is True
    assert result.raw_response == {"status": 1, "devices_count": 1, "licenses": ["iOS"]}


def test_apply_env_overrides_enables_pushover_from_credential_file(monkeypatch):
    monkeypatch.setenv("PUSHOVER_CREDENTIALS_FILE", "~/.hermes/credentials/pushover.json")
    monkeypatch.setenv("PUSHOVER_APP", "claude_code")
    monkeypatch.setenv("PUSHOVER_HOME_CHANNEL", "home")
    monkeypatch.setenv("PUSHOVER_HOME_CHANNEL_NAME", "Pushover")
    monkeypatch.delenv("PUSHOVER_TOKEN", raising=False)
    monkeypatch.delenv("PUSHOVER_USER", raising=False)

    config = GatewayConfig()
    _apply_env_overrides(config)
    pcfg = config.platforms[Platform.PUSHOVER]

    assert pcfg.enabled is True
    assert pcfg.extra["credentials_file"] == "~/.hermes/credentials/pushover.json"
    assert pcfg.extra["app"] == "claude_code"
    assert pcfg.home_channel.chat_id == "home"
