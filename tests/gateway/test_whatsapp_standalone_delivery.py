"""Tests for standalone (out-of-process cron) WhatsApp delivery.

Task 4 of the delivery-reliability plan: the standalone sender must reuse
the same bounded retry classifier and dead-letter ledger as the live
gateway adapter rather than duplicating single-attempt send logic.

- 401 is permanent (single attempt, no retry).
- 503 is retried up to 3 attempts and reaches the dead-letter ledger.
- A later 200 (e.g. 429 then 200) never reaches the dead-letter ledger.
"""

import json

import pytest
from unittest.mock import MagicMock

from plugins.platforms.whatsapp import delivery_ledger
from plugins.platforms.whatsapp.adapter import _standalone_send
from plugins.platforms.whatsapp.delivery_reliability import set_delivery_policy_hook


@pytest.fixture(autouse=True)
def _no_real_backoff_wait(monkeypatch):
    """Retry backoff still runs but with a zero delay, so tests stay fast."""
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.delivery_reliability.retry_backoff_delay",
        lambda *a, **k: 0,
    )


@pytest.fixture(autouse=True)
def _clear_policy_hook():
    set_delivery_policy_hook(None)
    yield
    set_delivery_policy_hook(None)


class _FakeResponse:
    def __init__(self, status, json_payload=None, text_payload=""):
        self.status = status
        self._json = json_payload if json_payload is not None else {}
        self._text = text_payload

    async def json(self):
        return self._json

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, *, json, headers, timeout):
        self.calls.append({"url": url, "json": json, "headers": dict(headers)})
        item = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _patch_session(monkeypatch, responses):
    session = _FakeSession(responses)
    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **k: session)
    return session


def _pconfig():
    cfg = MagicMock()
    cfg.extra = {"bridge_port": 3000}
    return cfg


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.delenv(delivery_ledger._LEDGER_ENABLED_ENV, raising=False)
    monkeypatch.setenv(delivery_ledger._LEDGER_PATH_ENV, str(tmp_path / "ledger.jsonl"))
    yield


@pytest.mark.asyncio
class TestStandaloneSendPermanentFailure:
    async def test_401_is_permanent_single_attempt(self, monkeypatch):
        session = _patch_session(monkeypatch, [_FakeResponse(401, text_payload="unauthorized")])

        result = await _standalone_send(_pconfig(), "5511999999999", "hello")

        assert "error" in result
        assert result["error_category"] == "http_401"
        assert result["attempts"] == 1
        assert len(session.calls) == 1


@pytest.mark.asyncio
class TestStandaloneSendDeadLetter:
    async def test_503_exhausts_three_attempts_and_dead_letters(self, tmp_path, monkeypatch):
        path = tmp_path / "ledger.jsonl"
        monkeypatch.setenv(delivery_ledger._LEDGER_ENABLED_ENV, "1")
        monkeypatch.setenv(delivery_ledger._LEDGER_PATH_ENV, str(path))
        session = _patch_session(monkeypatch, [_FakeResponse(503, text_payload="unavailable")])

        result = await _standalone_send(_pconfig(), "5511999999999", "hello")

        assert result["error_category"] == "http_503"
        assert result["attempts"] == 3
        assert result["dead_letter_ref"]
        assert len(session.calls) == 3

        entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(entries) == 1
        assert entries[0]["route"] == "/send"
        assert entries[0]["attempts"] == 3
        assert entries[0]["category"] == "http_503"

    async def test_later_200_never_reaches_dead_letter(self, tmp_path, monkeypatch):
        path = tmp_path / "ledger.jsonl"
        monkeypatch.setenv(delivery_ledger._LEDGER_ENABLED_ENV, "1")
        monkeypatch.setenv(delivery_ledger._LEDGER_PATH_ENV, str(path))
        _patch_session(monkeypatch, [
            _FakeResponse(429, text_payload="slow down"),
            _FakeResponse(200, json_payload={"messageId": "m1"}),
        ])

        result = await _standalone_send(_pconfig(), "5511999999999", "hello")

        assert result["success"] is True
        assert result["message_id"] == "m1"
        assert not path.exists()


@pytest.mark.asyncio
class TestStandaloneSendMedia:
    async def test_media_send_retries_connection_refused(self, tmp_path):
        async def _raise(*a, **k):
            raise ConnectionRefusedError()

        media = tmp_path / "pic.png"
        media.write_bytes(b"png")

        class _RefusingSession:
            def __init__(self):
                self.calls = 0

            def post(self, *a, **k):
                self.calls += 1
                raise ConnectionRefusedError()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        session = _RefusingSession()
        import aiohttp as _aiohttp
        import unittest.mock as _mock
        with _mock.patch.object(_aiohttp, "ClientSession", lambda *a, **k: session):
            result = await _standalone_send(
                _pconfig(), "5511999999999", "", media_files=[(str(media), False)],
            )

        assert result["error_category"] == "connection_refused"
        assert result["attempts"] == 3
        assert session.calls == 3
