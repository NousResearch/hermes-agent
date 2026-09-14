"""Signal plugin standalone sender — chunking + HTTP boundary (owned by the plugin)."""

import asyncio
from types import SimpleNamespace

import pytest

from plugins.platforms.signal.signal_rate_limit import _reset_scheduler
from plugins.platforms.signal.standalone import _send_signal


@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    _reset_scheduler()
    yield
    _reset_scheduler()


class _FakeSignalHttp:
    """Stand-in for httpx.AsyncClient used as an async context manager.

    Pops a response from the queue per `post` call. Each entry is either
    a dict (returned from .json()) or an exception instance (raised).
    Captures (url, payload) per call.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, *_a, **_kw):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def post(self, url, json=None):
        self.calls.append({"url": url, "payload": json})
        if not self.responses:
            raise AssertionError("Unexpected extra POST")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        resp = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda data=item: data,
        )
        return resp


def _install_signal_http(monkeypatch, fake):
    """Patch httpx.AsyncClient so the lazy import in ``_send_signal`` picks it up."""
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)


class TestSendSignalChunking:
    def test_text_only_single_rpc(self, monkeypatch):
        fake = _FakeSignalHttp([{"result": {"timestamp": 1}}])
        _install_signal_http(monkeypatch, fake)

        result = asyncio.run(
            _send_signal(
                {"http_url": "http://localhost:8080", "account": "+15551234567"},
                "+15557654321",
                "hello",
            )
        )

        assert result["success"] is True
        assert result["platform"] == "signal"
        assert result["chat_id"].endswith("4321")
        assert len(fake.calls) == 1
        params = fake.calls[0]["payload"]["params"]
        assert params["message"] == "hello"
        assert "attachments" not in params
        assert "textStyle" not in params
        assert "textStyles" not in params

    def test_skipped_missing_files_reported_in_warnings(self, tmp_path, monkeypatch):
        good = tmp_path / "ok.png"
        good.write_bytes(b"\x89PNG" + b"\x00" * 16)

        fake = _FakeSignalHttp([{"result": {"timestamp": 1}}])
        _install_signal_http(monkeypatch, fake)

        result = asyncio.run(
            _send_signal(
                {"http_url": "http://localhost:8080", "account": "+15551234567"},
                "+15557654321",
                "msg",
                media_files=[(str(good), False), (str(tmp_path / "missing.png"), False)],
            )
        )

        assert result["success"] is True
        assert "warnings" in result
        params = fake.calls[0]["payload"]["params"]
        assert len(params["attachments"]) == 1
