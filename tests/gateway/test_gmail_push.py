"""Tests for the native Gmail push gateway adapter."""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import gateway.platforms.gmail_push as gmail_mod
from gateway.config import Platform, PlatformConfig
from gateway.platforms.gmail_push import GmailPushAdapter, check_gmail_push_requirements
from gateway.run import GatewayRunner


def _make_config(tmp_path, **overrides) -> PlatformConfig:
    base_dir = tmp_path / "gmail-push"
    extra = {
        "account": "reader@example.com",
        "topic": "projects/demo/topics/hermes-gmail-push",
        "subscription": "hermes-gmail-push",
        "endpoint": {
            "host": "127.0.0.1",
            "port": 0,
            "path": "/gmail-push",
            "public_url": "https://example.com/gmail-push",
        },
        "oauth": {
            "client_secret_path": str(base_dir / "client_secret.json"),
            "token_path": str(base_dir / "token.json"),
        },
        "watch": {
            "label_ids": ["INBOX"],
            "label_filter_behavior": "INCLUDE",
            "renew_every_hours": 24,
        },
        "push_auth": {
            "service_account_email": "push-auth@example.iam.gserviceaccount.com",
            "audience": "https://example.com/gmail-push",
        },
        "processing": {
            "history_types": ["messageAdded"],
            "fetch_format": "full",
            "include_headers": ["From", "Subject", "Precedence"],
            "include_html": False,
            "max_body_chars": 20_000,
        },
        "state": {
            "path": str(base_dir / "state.json"),
        },
    }
    for key, value in overrides.items():
        extra[key] = value
    return PlatformConfig(enabled=True, extra=extra)


def _pubsub_envelope(
    *,
    message_id: str = "pubsub-1",
    account: str = "reader@example.com",
    history_id: str = "901",
) -> dict:
    payload = {"emailAddress": account, "historyId": history_id}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii").rstrip("=")
    return {
        "message": {
            "messageId": message_id,
            "publishTime": "2026-04-15T12:00:00Z",
            "data": encoded,
        }
    }


def _gmail_message(message_id: str = "msg-1") -> dict:
    body = base64.urlsafe_b64encode(b"hello from the newsletter").decode("ascii").rstrip("=")
    return {
        "id": message_id,
        "threadId": "thread-1",
        "labelIds": ["INBOX", "CATEGORY_UPDATES"],
        "internalDate": "1760581000000",
        "snippet": "Short Gmail snippet",
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [
                {"name": "From", "value": "Weekly Digest <digest@example.com>"},
                {"name": "Subject", "value": "The week in AI"},
                {"name": "Precedence", "value": "bulk"},
            ],
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": body},
                }
            ],
        },
    }


def _create_app(adapter: GmailPushAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post(adapter._path, adapter._handle_push)
    return app


class Fake404Error(Exception):
    status_code = 404


class Fake401Error(Exception):
    status_code = 401


def test_connect_stores_watch_baseline(tmp_path, monkeypatch):
    async def _run():
        monkeypatch.setattr(gmail_mod, "AIOHTTP_AVAILABLE", True)
        monkeypatch.setattr(gmail_mod, "GOOGLE_CLIENT_AVAILABLE", True)
        adapter = GmailPushAdapter(_make_config(tmp_path))
        monkeypatch.setattr(adapter, "_load_credentials", MagicMock(return_value=object()))
        monkeypatch.setattr(
            adapter,
            "_watch_mailbox",
            MagicMock(return_value={"historyId": "12345", "expiration": "67890"}),
        )

        connected = await adapter.connect()
        try:
            assert connected is True
            assert adapter._state["last_history_id"] == "12345"
            assert adapter._state["watch_expiration_ms"] == 67890
            assert adapter._state["last_watch_renewed_at"]
        finally:
            await adapter.disconnect()

    asyncio.run(_run())


def test_check_gmail_push_requirements(monkeypatch):
    monkeypatch.setattr(gmail_mod, "AIOHTTP_AVAILABLE", True)
    monkeypatch.setattr(gmail_mod, "GOOGLE_CLIENT_AVAILABLE", True)
    assert check_gmail_push_requirements() is True
    monkeypatch.setattr(gmail_mod, "GOOGLE_CLIENT_AVAILABLE", False)
    assert check_gmail_push_requirements() is False


def test_verify_pubsub_bearer_token_rejects_wrong_service_account(tmp_path, monkeypatch):
    adapter = GmailPushAdapter(_make_config(tmp_path))
    monkeypatch.setattr(gmail_mod, "GOOGLE_CLIENT_AVAILABLE", True)
    monkeypatch.setattr(gmail_mod, "GoogleRequest", lambda: object())
    monkeypatch.setattr(gmail_mod, "google_id_token", SimpleNamespace())
    monkeypatch.setattr(
        gmail_mod.google_id_token,
        "verify_oauth2_token",
        lambda token, request, audience: {
            "email": "wrong@example.iam.gserviceaccount.com",
            "email_verified": True,
        },
        raising=False,
    )

    with pytest.raises(PermissionError, match="Unexpected Pub/Sub service account"):
        adapter._verify_pubsub_bearer_token("token")


def test_parse_pubsub_envelope_handles_base64url(tmp_path):
    adapter = GmailPushAdapter(_make_config(tmp_path))
    parsed = adapter._parse_pubsub_envelope(_pubsub_envelope(message_id="pubsub-42", history_id="777"))
    assert parsed["email_address"] == "reader@example.com"
    assert parsed["history_id"] == "777"
    assert parsed["pubsub_message_id"] == "pubsub-42"


def test_relative_path_cannot_escape_hermes_home(tmp_path):
    config = _make_config(
        tmp_path,
        oauth={"token_path": "../../../etc/passwd"},
    )

    with pytest.raises(ValueError, match="escapes Hermes home"):
        GmailPushAdapter(config)


def test_duplicate_pubsub_message_id_is_ignored(tmp_path, monkeypatch):
    async def _run():
        adapter = GmailPushAdapter(_make_config(tmp_path))
        adapter._recent_delivery_ids["pubsub-1"] = 1.0
        monkeypatch.setattr(adapter, "_verify_pubsub_bearer_token", lambda token: {"ok": True})
        reconcile = AsyncMock(return_value=[])
        monkeypatch.setattr(adapter, "_reconcile_notification", reconcile)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                adapter._path,
                json=_pubsub_envelope(),
                headers={"Authorization": "Bearer token"},
            )

        assert resp.status == 204
        reconcile.assert_not_awaited()

    asyncio.run(_run())


def test_auth_failures_return_generic_error(tmp_path, monkeypatch):
    async def _run():
        adapter = GmailPushAdapter(_make_config(tmp_path))
        monkeypatch.setattr(
            adapter,
            "_verify_pubsub_bearer_token",
            MagicMock(side_effect=PermissionError("Unexpected Pub/Sub service account")),
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                adapter._path,
                json=_pubsub_envelope(),
                headers={"Authorization": "Bearer token"},
            )
            payload = await resp.json()

        assert resp.status == 401
        assert payload == {"error": "Authentication failed"}

    asyncio.run(_run())


def test_push_for_unexpected_account_is_acknowledged(tmp_path, monkeypatch):
    async def _run():
        adapter = GmailPushAdapter(_make_config(tmp_path))
        monkeypatch.setattr(adapter, "_verify_pubsub_bearer_token", lambda token: {"ok": True})
        reconcile = AsyncMock(return_value=[])
        monkeypatch.setattr(adapter, "_reconcile_notification", reconcile)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                adapter._path,
                json=_pubsub_envelope(account="other@example.com"),
                headers={"Authorization": "Bearer token"},
            )

        assert resp.status == 204
        reconcile.assert_not_awaited()

    asyncio.run(_run())


def test_push_dispatches_message_event_for_bulk_mail(tmp_path, monkeypatch):
    async def _run():
        adapter = GmailPushAdapter(_make_config(tmp_path))
        adapter._state["last_history_id"] = "100"
        adapter.set_message_handler(AsyncMock(return_value=None))
        monkeypatch.setattr(adapter, "_verify_pubsub_bearer_token", lambda token: {"ok": True})
        monkeypatch.setattr(
            adapter,
            "_fetch_history_pages",
            MagicMock(
                return_value=[
                    {
                        "historyId": "200",
                        "history": [
                            {
                                "messagesAdded": [
                                    {"message": {"id": "msg-1"}},
                                ]
                            }
                        ],
                    }
                ]
            ),
        )
        monkeypatch.setattr(adapter, "_get_message", MagicMock(return_value=_gmail_message()))

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                adapter._path,
                json=_pubsub_envelope(history_id="200"),
                headers={"Authorization": "Bearer token"},
            )
            assert resp.status == 204
            await asyncio.sleep(0)

        adapter._message_handler.assert_awaited_once()
        event = adapter._message_handler.await_args.args[0]
        assert event.source.platform == Platform.GMAIL_PUSH
        assert event.message_id == "msg-1"
        assert "The week in AI" in event.text
        assert event.raw_message["normalized"]["gmail_message"]["headers"]["Precedence"] == "bulk"
        assert adapter._state["last_history_id"] == "200"

    asyncio.run(_run())


def test_stale_history_cursor_rebaselines_and_marks_degraded(tmp_path, monkeypatch):
    async def _run():
        adapter = GmailPushAdapter(_make_config(tmp_path))
        adapter._state["last_history_id"] = "stale"
        monkeypatch.setattr(adapter, "_verify_pubsub_bearer_token", lambda token: {"ok": True})
        monkeypatch.setattr(adapter, "_fetch_history_pages", MagicMock(side_effect=Fake404Error("stale")))
        monkeypatch.setattr(
            adapter,
            "_watch_mailbox",
            MagicMock(return_value={"historyId": "fresh-900", "expiration": "5000"}),
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                adapter._path,
                json=_pubsub_envelope(history_id="901"),
                headers={"Authorization": "Bearer token"},
            )

        assert resp.status == 204
        assert adapter._state["last_history_id"] == "fresh-900"
        assert adapter._state["degraded"] is True
        assert "historical backfill skipped" in adapter._state["last_error"]

    asyncio.run(_run())


def test_send_logs_response_instead_of_email(tmp_path):
    async def _run():
        adapter = GmailPushAdapter(_make_config(tmp_path))
        result = await adapter.send("gmail_push:reader@example.com:msg-1", "Digest stored")
        assert result.success is True
        assert adapter._response_log[0]["content"] == "Digest stored"

    asyncio.run(_run())


def test_gmail_request_rebuilds_service_after_401(tmp_path, monkeypatch):
    class _Execute:
        def __init__(self, result=None, error=None):
            self._result = result
            self._error = error

        def execute(self):
            if self._error is not None:
                raise self._error
            return self._result

    class _Messages:
        def __init__(self, request):
            self._request = request

        def get(self, **kwargs):
            return self._request

    class _Users:
        def __init__(self, request):
            self._messages = _Messages(request)

        def messages(self):
            return self._messages

    class _Service:
        def __init__(self, request):
            self._users = _Users(request)

        def users(self):
            return self._users

    adapter = GmailPushAdapter(_make_config(tmp_path))
    monkeypatch.setattr(gmail_mod, "HttpError", Fake401Error)
    monkeypatch.setattr(adapter, "_load_credentials", MagicMock(return_value=object()))
    monkeypatch.setattr(
        gmail_mod,
        "google_build",
        MagicMock(
            side_effect=[
                _Service(_Execute(error=Fake401Error("expired"))),
                _Service(_Execute(result={"id": "msg-1"})),
            ]
        ),
    )

    result = adapter._get_message("msg-1")

    assert result["id"] == "msg-1"
    assert gmail_mod.google_build.call_count == 2


def test_gateway_runner_creates_gmail_push_adapter(tmp_path, monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(group_sessions_per_user=True, thread_sessions_per_user=False)
    config = _make_config(tmp_path)
    monkeypatch.setattr(
        "gateway.platforms.gmail_push.check_gmail_push_requirements",
        lambda: True,
    )
    adapter = GatewayRunner._create_adapter(runner, Platform.GMAIL_PUSH, config)
    assert isinstance(adapter, GmailPushAdapter)


def test_gateway_runner_authorizes_gmail_push_internal_events():
    runner = object.__new__(GatewayRunner)
    source = SimpleNamespace(platform=Platform.GMAIL_PUSH, user_id="", user_name="", chat_type="channel")
    assert GatewayRunner._is_user_authorized(runner, source) is True
