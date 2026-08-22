"""Tests for per-route webhook event coalescing.

Covers:
- Startup validation of the ``coalesce`` route block
- Group-key rendering (bare field vs template, unresolved fields)
- Debounce behavior: latest event wins, timer re-arms per event
- max_wait cap: a steady stream cannot starve dispatch
- Non-coalescing routes are unaffected (immediate dispatch)
- deliver_only + coalesce rejected at startup
- Flush-on-disconnect dispatches pending groups
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


def _make_adapter(routes=None, **extra_overrides):
    extra = {
        "host": "127.0.0.1",
        "port": 0,
        "routes": routes or {},
        "rate_limit": 100,
    }
    extra.update(extra_overrides)
    return WebhookAdapter(PlatformConfig(enabled=True, extra=extra))


def _coalesce_route(**coalesce):
    return {
        "secret": _INSECURE_NO_AUTH,
        "prompt": "PR {pull_request.number}: {action}",
        "coalesce": coalesce or {"key": "pull_request.number"},
    }


def _mock_request(body: bytes, route_name: str = "pr", delivery_id: str = ""):
    req = MagicMock()
    headers = {}
    if delivery_id:
        headers["X-GitHub-Delivery"] = delivery_id
    req.headers = headers
    req.content_length = len(body)
    req.match_info = {"route_name": route_name}
    req.method = "POST"

    async def _read():
        return body

    req.read = _read
    return req


def _payload(pr_number: int, action: str = "synchronize") -> bytes:
    return json.dumps(
        {"action": action, "pull_request": {"number": pr_number}}
    ).encode()


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------


class TestCoalesceValidation:
    @pytest.mark.asyncio
    async def test_missing_key_rejected(self):
        adapter = _make_adapter(
            routes={"pr": _coalesce_route(window_seconds=5)}
        )
        with pytest.raises(ValueError, match="coalesce block"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_non_dict_coalesce_rejected(self):
        route = _coalesce_route()
        route["coalesce"] = "pull_request.number"
        adapter = _make_adapter(routes={"pr": route})
        with pytest.raises(ValueError, match="coalesce block"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_nonpositive_window_rejected(self):
        adapter = _make_adapter(
            routes={
                "pr": _coalesce_route(key="pull_request.number", window_seconds=0)
            }
        )
        with pytest.raises(ValueError, match="window_seconds"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_bool_max_wait_rejected(self):
        adapter = _make_adapter(
            routes={
                "pr": _coalesce_route(
                    key="pull_request.number", max_wait_seconds=True
                )
            }
        )
        with pytest.raises(ValueError, match="max_wait_seconds"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_deliver_only_with_coalesce_rejected(self):
        route = _coalesce_route(key="pull_request.number")
        route["deliver_only"] = True
        route["deliver"] = "telegram"
        adapter = _make_adapter(routes={"pr": route})
        with pytest.raises(ValueError, match="deliver_only"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_valid_coalesce_config_connects(self):
        adapter = _make_adapter(
            routes={
                "pr": _coalesce_route(
                    key="pull_request.number",
                    window_seconds=10,
                    max_wait_seconds=60,
                )
            }
        )
        assert await adapter.connect() is True
        await adapter.disconnect()


# ---------------------------------------------------------------------------
# Group-key rendering
# ---------------------------------------------------------------------------


class TestGroupKey:
    def test_bare_field(self):
        adapter = _make_adapter()
        key = adapter._coalesce_group_key(
            "pr", "pull_request.number", {"pull_request": {"number": 7}}, "pull_request"
        )
        assert key == "pr|7"

    def test_template(self):
        adapter = _make_adapter()
        key = adapter._coalesce_group_key(
            "pr",
            "{repository.full_name}#{pull_request.number}",
            {
                "repository": {"full_name": "acme/widgets"},
                "pull_request": {"number": 7},
            },
            "pull_request",
        )
        assert key == "pr|acme/widgets#7"

    def test_unresolved_field_stays_literal(self):
        adapter = _make_adapter()
        key = adapter._coalesce_group_key("pr", "missing.field", {}, "x")
        assert key == "pr|{missing.field}"

    def test_route_scoped(self):
        adapter = _make_adapter()
        k1 = adapter._coalesce_group_key("a", "id", {"id": "same"}, "e")
        k2 = adapter._coalesce_group_key("b", "id", {"id": "same"}, "e")
        assert k1 != k2


# ---------------------------------------------------------------------------
# Debounce behavior (through the real handler)
# ---------------------------------------------------------------------------


class TestCoalescingBehavior:
    @pytest.mark.asyncio
    async def test_rapid_events_dispatch_once_with_latest(self):
        adapter = _make_adapter(
            routes={
                "pr": _coalesce_route(
                    key="pull_request.number", window_seconds=0.05
                )
            }
        )
        adapter.handle_message = AsyncMock()

        for i, action in enumerate(["opened", "synchronize", "synchronize"]):
            resp = await adapter._handle_webhook(
                _mock_request(
                    _payload(7, action), delivery_id=f"d{i}"
                )
            )
            assert resp.status == 202
            assert json.loads(resp.text)["status"] == "coalesced"

        # Nothing dispatched during the window
        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.15)

        assert adapter.handle_message.call_count == 1
        event = adapter.handle_message.call_args[0][0]
        # Latest event's delivery id and prompt won
        assert event.message_id == "d2"
        assert "PR 7: synchronize" in event.text
        assert "3 webhook events" in event.text
        assert adapter._coalesce_pending == {}
        assert adapter._coalesce_tasks == {}

    @pytest.mark.asyncio
    async def test_distinct_groups_dispatch_independently(self):
        adapter = _make_adapter(
            routes={
                "pr": _coalesce_route(
                    key="pull_request.number", window_seconds=0.05
                )
            }
        )
        adapter.handle_message = AsyncMock()

        await adapter._handle_webhook(_mock_request(_payload(1), delivery_id="a"))
        await adapter._handle_webhook(_mock_request(_payload(2), delivery_id="b"))
        await asyncio.sleep(0.15)

        assert adapter.handle_message.call_count == 2
        ids = {c.args[0].message_id for c in adapter.handle_message.call_args_list}
        assert ids == {"a", "b"}

    @pytest.mark.asyncio
    async def test_single_event_has_no_coalesce_note(self):
        adapter = _make_adapter(
            routes={
                "pr": _coalesce_route(
                    key="pull_request.number", window_seconds=0.05
                )
            }
        )
        adapter.handle_message = AsyncMock()
        await adapter._handle_webhook(_mock_request(_payload(9), delivery_id="x"))
        await asyncio.sleep(0.15)
        event = adapter.handle_message.call_args[0][0]
        assert "coalesced" not in event.text

    @pytest.mark.asyncio
    async def test_max_wait_caps_starvation(self):
        adapter = _make_adapter(
            routes={
                "pr": _coalesce_route(
                    key="pull_request.number",
                    window_seconds=10,  # would debounce forever under stream
                    max_wait_seconds=0.1,
                )
            }
        )
        adapter.handle_message = AsyncMock()

        # A steady stream faster than the window
        for i in range(3):
            await adapter._handle_webhook(
                _mock_request(_payload(5), delivery_id=f"s{i}")
            )
            await asyncio.sleep(0.05)

        # max_wait (0.1s) elapsed during the stream → dispatch happened
        await asyncio.sleep(0.1)
        assert adapter.handle_message.call_count >= 1

    @pytest.mark.asyncio
    async def test_route_without_coalesce_dispatches_immediately(self):
        adapter = _make_adapter(
            routes={
                "pr": {
                    "secret": _INSECURE_NO_AUTH,
                    "prompt": "PR {pull_request.number}",
                }
            }
        )
        adapter.handle_message = AsyncMock()
        resp = await adapter._handle_webhook(
            _mock_request(_payload(3), delivery_id="imm")
        )
        assert json.loads(resp.text)["status"] == "accepted"
        await asyncio.sleep(0)
        assert adapter.handle_message.call_count == 1

    @pytest.mark.asyncio
    async def test_duplicate_delivery_still_suppressed_before_coalescing(self):
        adapter = _make_adapter(
            routes={
                "pr": _coalesce_route(
                    key="pull_request.number", window_seconds=0.05
                )
            }
        )
        adapter.handle_message = AsyncMock()
        r1 = await adapter._handle_webhook(
            _mock_request(_payload(7), delivery_id="dup")
        )
        r2 = await adapter._handle_webhook(
            _mock_request(_payload(7), delivery_id="dup")
        )
        assert json.loads(r1.text)["status"] == "coalesced"
        assert json.loads(r2.text)["status"] == "duplicate"
        await asyncio.sleep(0.15)
        assert adapter.handle_message.call_count == 1


# ---------------------------------------------------------------------------
# Flush on disconnect
# ---------------------------------------------------------------------------


class TestFlushOnDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_flushes_pending_groups(self):
        adapter = _make_adapter(
            routes={
                "pr": _coalesce_route(
                    key="pull_request.number", window_seconds=60
                )
            }
        )
        adapter.handle_message = AsyncMock()
        await adapter._handle_webhook(
            _mock_request(_payload(4), delivery_id="pend")
        )
        adapter.handle_message.assert_not_called()

        await adapter.disconnect()
        await asyncio.sleep(0)

        assert adapter.handle_message.call_count == 1
        assert adapter.handle_message.call_args[0][0].message_id == "pend"
        assert adapter._coalesce_pending == {}
