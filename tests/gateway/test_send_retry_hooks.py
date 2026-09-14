"""Invariants for the shared send-retry path.

Photon used to fork ``BasePlatformAdapter._send_with_retry`` and so missed three landed fixes
(server ``retry_after`` honoured, the inline wait cap, no failure notice inside a flood penalty).
It now overrides two hooks. Slack and Discord parsed ``Retry-After`` by hand and only understood
the numeric form; both now go through ``agent.retry_utils.parse_retry_after_seconds``.
"""

from email.utils import format_datetime
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class _Adapter(BasePlatformAdapter):
    """Sends fail with a scripted list of results; records every call."""

    def __init__(self, results):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self._results = list(results)
        self.sent: list = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return self._results.pop(0) if self._results else SendResult(success=True, message_id="ok")


class _FinalOnRefusal(_Adapter):
    def _send_retry_is_final(self, result: SendResult) -> bool:
        return isinstance(result.raw_response, dict) and result.raw_response.get("error_class") == "target_not_allowed"


@pytest.mark.asyncio
async def test_final_failure_skips_retry_and_plain_text_fallback(monkeypatch):
    """A structured refusal must come back untouched: no backoff sleep, no second send."""
    slept = []

    async def _sleep(d):
        slept.append(d)
    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", _sleep)
    refusal = SendResult(success=False, error="Target not allowed", retryable=True,
                         raw_response={"error_class": "target_not_allowed", "retryable": False})
    adapter = _FinalOnRefusal([refusal])
    result = await adapter._send_with_retry("c", "hello")
    assert result is refusal
    assert adapter.sent == ["hello"] and slept == []


@pytest.mark.asyncio
async def test_server_retry_after_is_honoured_by_every_adapter(monkeypatch):
    """Photon's fork slept base_delay*2**n regardless of the server hint; via the base every
    adapter waits what the server asked (plus jitter) — the fix that landed once now reaches all."""
    slept = []

    async def _sleep(d):
        slept.append(d)
    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", _sleep)
    monkeypatch.setattr("gateway.platforms.base.random.uniform", lambda a, b: 0.0)
    adapter = _FinalOnRefusal([SendResult(success=False, error="429 rate limited", retryable=True, retry_after=7.0)])
    result = await adapter._send_with_retry("c", "hello", base_delay=2.0)
    assert result.success and adapter.sent == ["hello", "hello"]
    assert slept == [7.0]


def _http_date_in(seconds: int) -> str:
    return format_datetime(datetime.now(timezone.utc) + timedelta(seconds=seconds), usegmt=True)


def test_slack_retry_after_understands_http_date():
    from plugins.platforms.slack.adapter import SlackAdapter
    exc = SimpleNamespace(response=SimpleNamespace(headers={"Retry-After": _http_date_in(90)}))
    seconds = SlackAdapter._retry_after_from_exc(exc)
    assert seconds is not None and 80 <= seconds <= 90
    assert SlackAdapter._retry_after_from_exc(SimpleNamespace(response=None)) is None


def test_discord_retry_after_understands_http_date_and_reset_after():
    from plugins.platforms.discord.adapter import DiscordAdapter
    extract = DiscordAdapter._extract_discord_retry_after
    dated = SimpleNamespace(response=SimpleNamespace(headers={"retry-after": _http_date_in(90)}))
    seconds = extract(dated)
    assert seconds is not None and 80 <= seconds <= 90
    reset = SimpleNamespace(response=SimpleNamespace(headers={"X-RateLimit-Reset-After": "0.25"}))
    assert extract(reset) == 1.0  # floored: a sub-second hint must not hot-loop
    assert extract(SimpleNamespace(retry_after=12)) == 12.0
    assert extract(SimpleNamespace()) is None
