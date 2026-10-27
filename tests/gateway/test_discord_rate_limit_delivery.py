"""A Discord rate-limit refusal must reach the delivery ledger as a replayable failure.

Discord states its wait in a header (``Retry-After`` / ``X-RateLimit-Reset-After``), never in the body, so
a refusal's own wording carries no number. Both outbound boundaries used to treat a 429 like any other
non-200 response: the adapter's ``send()`` returned a plain error string with no retryability, and the
standalone HTTP sender returned the same generic envelope. The consequence was a real delivery gap — a cron
job completed its task, the Discord POST answered 429, and the payload was reported as a delivery error and
dropped. Nothing had been accepted, so a replay was safe, but no row existed to replay.

The fix carries the server's wait in the canonical ``flood_control:<seconds>`` marker the ledger already
understands, at both boundaries, and lets cron's standalone lane keep the payload when the refusal is a
rate limit rather than only when it is reconnect-only.
"""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import importlib

import pytest

import cron.scheduler_delivery as sd
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.delivery_ledger import flood_wait_seconds, is_flood_error
from gateway.platforms.base import SendResult, classify_send_error

# Package import, not the loader: the Discord adapter resolves its sibling modules through its own
# package, which ``load_plugin_adapter`` (a file-location spec with no package) cannot provide.
discord = importlib.import_module("plugins.platforms.discord.adapter")


@pytest.fixture(autouse=True)
def _fresh_ledger(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from gateway import delivery_ledger as dl
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: True)
    monkeypatch.setattr(sd, "_maybe_mirror_cron_delivery", lambda *a, **k: None)
    yield dl


def _adapter():
    return discord.DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))


def _rate_limit_exception(headers):
    """A discord.py-shaped 429: HTTPException status 429 with the wait in the headers."""
    exc = Exception("429 Too Many Requests")
    setattr(exc, "response", SimpleNamespace(status=429, headers=headers))
    return exc


def _client_that_raises(exc):
    channel = SimpleNamespace(send=AsyncMock(side_effect=exc))
    adapter = _adapter()
    adapter._client = SimpleNamespace(get_channel=MagicMock(return_value=channel))
    return adapter


# ---------------------------------------------------------------------------
# Boundary 1: the adapter's outbound send()


@pytest.mark.asyncio
async def test_send_rate_limit_carries_the_servers_wait_and_is_replayable():
    adapter = _client_that_raises(_rate_limit_exception({"Retry-After": "97"}))
    result = await adapter.send("123", "the report")

    assert not result.success
    assert result.retryable is True
    assert result.error_kind == "rate_limited"
    assert result.retry_after == 97.0
    # The marker is what the ledger reads back; the diagnostic wording must survive alongside it.
    assert is_flood_error(result.error)
    assert flood_wait_seconds(result.error) == 97.0
    assert "429" in result.error


@pytest.mark.asyncio
async def test_send_rate_limit_without_a_header_wait_stays_replayable():
    adapter = _client_that_raises(_rate_limit_exception({}))
    result = await adapter.send("123", "the report")

    assert not result.success and result.retryable is True
    assert result.retry_after is None
    # No number to honour, so the ledger falls back to its default rather than mis-reading one.
    assert not is_flood_error(result.error)
    assert classify_send_error(result.error) == "rate_limited"


@pytest.mark.asyncio
async def test_non_rate_limit_send_failure_is_unchanged():
    adapter = _client_that_raises(RuntimeError("channel not found"))
    result = await adapter.send("123", "the report")

    assert not result.success
    assert result.error == "channel not found"
    assert result.retry_after is None and result.error_kind is None
    assert not is_flood_error(result.error)


# ---------------------------------------------------------------------------
# Boundary 2: the standalone HTTP sender (the path that failed in production)


class _Response:
    """Aiohttp-shaped response: status, headers, and a body readable through ``text()``."""

    def __init__(self, status, body, headers=None):
        self.status = status
        self._body = body
        self.headers = headers or {}

    async def text(self):
        return self._body


@pytest.mark.asyncio
async def test_standalone_429_envelope_carries_the_canonical_marker():
    resp = _Response(429, "You are being rate limited.", {"Retry-After": "0.3"})
    data, err = await discord._standalone_response_json_or_error(resp, "Discord API error")

    assert data is None
    assert set(err) == {"error"}
    assert is_flood_error(err["error"])
    assert flood_wait_seconds(err["error"]) == 0.3
    assert "Discord API error (429)" in err["error"]
    assert "You are being rate limited." in err["error"]


@pytest.mark.asyncio
async def test_standalone_429_falls_back_to_the_discord_reset_header():
    resp = _Response(429, "slow down", {"X-RateLimit-Reset-After": "12"})
    _, err = await discord._standalone_response_json_or_error(resp, "Discord API error")
    assert flood_wait_seconds(err["error"]) == 12.0


@pytest.mark.asyncio
async def test_other_standalone_statuses_are_not_flood_marked():
    resp = _Response(403, "Missing Permissions", {"Retry-After": "5"})
    _, err = await discord._standalone_response_json_or_error(resp, "Discord API error")
    assert not is_flood_error(err["error"])
    assert "Missing Permissions" in err["error"]


# ---------------------------------------------------------------------------
# Boundary 3: cron's standalone lane must keep the payload for redelivery


def _target(loop):
    fields = {name: None for name in sd._TargetDelivery.__dataclass_fields__}
    fields.update(job={"id": "job-1"}, platform=Platform.DISCORD, platform_name="discord",
                  chat_id="941804866093850685", thread_id="1558285615710085282",
                  transport=SimpleNamespace(is_relay=False, adapter=None),
                  config=GatewayConfig(), loop=loop, target_adapters={}, mirror_text="", origin={})
    return sd._TargetDelivery(**fields)


@pytest.fixture
def gateway_loop():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)


def test_standalone_rate_limit_queues_the_payload_for_the_sweep(monkeypatch, gateway_loop, _fresh_ledger):
    dl = _fresh_ledger
    t = _target(gateway_loop)
    # Cron's standalone sender returns ``(None, error_string)`` — the envelope's error text.
    envelope = (None, "delivery error: Discord API error (429) flood_control:0.3: "
                       "You are being rate limited. (target discord:941804866093850685:1558285615710085282)")
    monkeypatch.setattr(sd, "_standalone_send", lambda t, content, media: envelope)

    target_errors, delivery_errors = [], []
    sd._deliver_standalone(t, "the report", [], target_errors, delivery_errors)

    assert any("You are being rate limited." in e for e in target_errors)
    assert any("queued text for" in e and "rate-limit wait" in e for e in delivery_errors)

    # The row is due once the server's wait has passed, and the sweep claims it.
    claimed = dl.sweep_failed_for_runtime("discord", now=time.time() + 1)
    assert [(row["chat_id"], row["thread_id"], row["content"]) for row in claimed] == [
        ("941804866093850685", "1558285615710085282", "the report")]


def test_standalone_rate_limit_is_not_claimed_before_the_wait_passes(monkeypatch, gateway_loop, _fresh_ledger):
    dl = _fresh_ledger
    t = _target(gateway_loop)
    envelope = (None, "delivery error: Discord API error (429) flood_control:90: "
                      "You are being rate limited. (target x)")
    monkeypatch.setattr(sd, "_standalone_send", lambda t, content, media: envelope)
    sd._deliver_standalone(t, "the report", [], [], [])

    # Still inside the server's wait: the sweep must leave it alone rather than hot-loop.
    assert dl.sweep_failed_for_runtime("discord", now=time.time()) == []
    assert dl.sweep_failed_for_runtime("discord", now=time.time() + 91)


def test_standalone_non_rate_limit_failure_still_queues_nothing(monkeypatch, gateway_loop, _fresh_ledger):
    dl = _fresh_ledger
    t = _target(gateway_loop)
    envelope = (None, "delivery error: Discord API error (403): Missing Permissions (target x)")
    monkeypatch.setattr(sd, "_standalone_send", lambda t, content, media: envelope)

    target_errors, delivery_errors = [], []
    sd._deliver_standalone(t, "the report", [], target_errors, delivery_errors)

    assert any("Missing Permissions" in e for e in target_errors)
    assert not any("queued" in e for e in delivery_errors)
    assert dl.sweep_failed_for_runtime("discord", now=time.time() + 1) == []
