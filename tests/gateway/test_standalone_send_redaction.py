"""Invariant: a standalone sender's failure envelope never carries vendor secrets to the model.

Every ``plugins/platforms/*/adapter.py::_standalone_send`` used to build ``{"error": f"... {e}"}``
by hand; a token or signed URL inside an httpx/aiohttp exception went straight into the tool
result. They now share ``gateway.platforms._shared.send_error`` (the ``send_message`` redactor).
This drives two real senders end-to-end with a transport that raises a secret-bearing error.
"""

import pytest

import agent.redact as _redact
from gateway.config import PlatformConfig
from gateway.platforms._shared import send_error
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_FAKE_TOKEN = "hermes-test-bearer-credential-ABCDEFGHIJKLMNOPQRSTUVWX"
_FAKE_URL_SECRET = "https://hooks.example/send?access_token=sk_live_ABCDEF0123456789"


@pytest.fixture(autouse=True)
def _redaction_on(monkeypatch):
    # The switch is snapshotted at import; a developer shell with HERMES_REDACT_SECRETS=false must not
    # turn this contract test into a no-op.
    monkeypatch.setattr(_redact, "_REDACT_ENABLED", True)


def test_send_error_redacts_token_and_signed_url():
    out = send_error(f"upstream said 401 for Authorization: Bearer {_FAKE_TOKEN} at {_FAKE_URL_SECRET}")
    assert set(out) == {"error"}
    assert _FAKE_TOKEN not in out["error"]
    assert "sk_live_ABCDEF0123456789" not in out["error"]
    assert "401" in out["error"]  # the diagnostic shape survives


@pytest.mark.asyncio
async def test_ntfy_standalone_failure_is_redacted(monkeypatch):
    ntfy = load_plugin_adapter("ntfy")
    monkeypatch.setenv("NTFY_TOPIC", "hermes-test")

    class _Boom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            raise RuntimeError(f"connect failed: Authorization: Bearer {_FAKE_TOKEN} {_FAKE_URL_SECRET}")

    monkeypatch.setattr(ntfy, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(ntfy.httpx, "AsyncClient", _Boom)
    result = await ntfy._standalone_send(PlatformConfig(enabled=True, extra={}), "hermes-test", "hi")
    assert "error" in result
    assert _FAKE_TOKEN not in result["error"] and "sk_live_ABCDEF0123456789" not in result["error"]
    assert "connect failed" in result["error"]


@pytest.mark.asyncio
async def test_irc_standalone_failure_is_redacted(monkeypatch):
    irc = load_plugin_adapter("irc")

    async def _boom(*a, **k):
        raise OSError(f"refused; proxy Authorization: Basic {_FAKE_TOKEN}")

    monkeypatch.setattr(irc.asyncio, "open_connection", _boom)
    result = await irc._standalone_send(
        PlatformConfig(enabled=True, extra={"server": "irc.example", "channel": "#x", "nickname": "h"}), "#x", "hi")
    assert "error" in result
    assert _FAKE_TOKEN not in result["error"]
    assert "refused" in result["error"]


@pytest.mark.parametrize("failure_stage", ["upload", "session_enter", "session_exit"])
@pytest.mark.asyncio
async def test_mattermost_standalone_failure_preserves_safe_receipt(monkeypatch, tmp_path, failure_stage):
    import aiohttp
    import tools.send_message_tool as send_tool
    import gateway.platforms.base as platform_base

    mattermost = load_plugin_adapter("mattermost")
    monkeypatch.setattr(send_tool, "_live_adapter", lambda platform: (None, None))
    monkeypatch.setattr(mattermost, "_url_and_token", lambda config: ("https://mattermost.example", "test-token"))
    monkeypatch.setattr(platform_base, "resolve_proxy_url", lambda **kwargs: None)
    media = tmp_path / "attachment.png"
    media.write_bytes(b"test-image")
    error_text = "connect failed: " + "Author" + "ization" + ": " + "Bearer " + _FAKE_TOKEN + " at " + _FAKE_URL_SECRET
    calls = []

    class Response:
        status = 201

        def __init__(self, upload):
            self.upload = upload

        async def __aenter__(self):
            if self.upload and failure_stage == "upload":
                raise RuntimeError(error_text)
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return {"file_infos": [{"id": "file-id"}]} if self.upload else {"id": "post-id"}

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            if failure_stage == "session_enter":
                raise RuntimeError(error_text)
            return self

        async def __aexit__(self, *args):
            if failure_stage == "session_exit":
                raise RuntimeError(error_text)
            return False

        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return Response(url.endswith("/files"))

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    result = await mattermost._standalone_send(
        PlatformConfig(enabled=True, extra={}), "test-channel", "caption", media_files=[str(media)]
    )
    assert _FAKE_TOKEN not in result["error"]
    assert _FAKE_URL_SECRET not in result["error"]
    assert "connect failed" in result["error"]
    assert result["success"] is False
    assert result["platform"] == "mattermost" and result["chat_id"] == "test-channel"
    assert result["total_media"] == 1
    delivered = int(failure_stage == "session_exit")
    assert result["delivered_media"] == delivered and result["failed_media"] == 1 - delivered
    expected_ids = [] if failure_stage == "session_enter" else ["post-id"]
    assert result["message_ids"] == expected_ids
    assert result["message_id"] == (expected_ids[-1] if expected_ids else None)
    assert result["partial_failure"] == bool(expected_ids)
    assert len(calls) == (0 if failure_stage == "session_enter" else 2)
