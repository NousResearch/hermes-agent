"""Tests for the webhook adapter's multi-deliver feature.

Covers:
- Legacy string format backward compatibility (single target)
- Multi-target list format (concurrent delivery)
- Best-effort semantics (partial failure still returns success)
- All-fail returns failure
- Template rendering in multi-target deliver_extra fields
- deliver_only with multi-target
- Startup validation for multi-target deliver_only
- _normalize_deliver_targets() normalization logic
"""

import asyncio
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(routes, **extra_kw) -> WebhookAdapter:
    extra = {"host": "127.0.0.1", "port": 0, "routes": routes}
    extra.update(extra_kw)
    config = PlatformConfig(enabled=True, extra=extra)
    return WebhookAdapter(config)


def _create_app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _wire_mock_targets(adapter: WebhookAdapter, platforms: list):
    """Attach a gateway_runner with mocked target adapters for multiple platforms."""
    mock_adapters = {}
    for p in platforms:
        mock_target = AsyncMock()
        mock_target.send = AsyncMock(return_value=SendResult(success=True))
        mock_adapters[Platform(p)] = mock_target

    mock_runner = MagicMock()
    mock_runner.adapters = mock_adapters
    mock_runner.config.get_home_channel.return_value = None

    adapter.gateway_runner = mock_runner
    return mock_adapters


def _wire_mock_target(adapter: WebhookAdapter, platform_name: str = "telegram"):
    """Attach a gateway_runner with a single mocked target adapter."""
    adapters = _wire_mock_targets(adapter, [platform_name])
    return adapters[Platform(platform_name)]


# ===================================================================
# _normalize_deliver_targets()
# ===================================================================

class TestNormalizeDeliverTargets:
    """Unit tests for the target normalization helper."""

    def test_legacy_string_format(self):
        adapter = _make_adapter({})
        delivery = {"deliver": "telegram", "deliver_extra": {"chat_id": "123"}}
        targets = adapter._normalize_deliver_targets(delivery)
        assert targets == [{"type": "telegram", "chat_id": "123"}]

    def test_legacy_string_default_log(self):
        adapter = _make_adapter({})
        delivery = {}
        targets = adapter._normalize_deliver_targets(delivery)
        assert targets == [{"type": "log"}]

    def test_multi_target_list_format(self):
        adapter = _make_adapter({})
        delivery = {
            "deliver": [
                {"type": "telegram", "chat_id": "111"},
                {"type": "discord", "chat_id": "222"},
            ]
        }
        targets = adapter._normalize_deliver_targets(delivery)
        assert len(targets) == 2
        assert targets[0] == {"type": "telegram", "chat_id": "111"}
        assert targets[1] == {"type": "discord", "chat_id": "222"}

    def test_multi_target_with_payload_rendering(self):
        adapter = _make_adapter({})
        delivery = {
            "deliver": [
                {"type": "telegram", "chat_id": "{user.telegram_id}"},
                {"type": "github_comment", "repo": "{repo.name}", "pr_number": "{pr.number}"},
            ]
        }
        payload = {
            "user": {"telegram_id": "99999"},
            "repo": {"name": "org/project"},
            "pr": {"number": "42"},
        }
        targets = adapter._normalize_deliver_targets(delivery, payload)
        assert targets[0] == {"type": "telegram", "chat_id": "99999"}
        assert targets[1] == {"type": "github_comment", "repo": "org/project", "pr_number": "42"}

    def test_empty_list_falls_back_to_log(self):
        adapter = _make_adapter({})
        delivery = {"deliver": []}
        targets = adapter._normalize_deliver_targets(delivery)
        assert targets == [{"type": "log"}]

    def test_invalid_items_skipped(self):
        adapter = _make_adapter({})
        delivery = {
            "deliver": [
                {"type": "telegram", "chat_id": "111"},
                "invalid_string",  # should be skipped
                {"no_type_key": True},  # should be skipped
            ]
        }
        targets = adapter._normalize_deliver_targets(delivery)
        assert len(targets) == 1
        assert targets[0]["type"] == "telegram"


# ===================================================================
# send() with multi-target
# ===================================================================

class TestSendMultiTarget:
    """Test the send() method with multi-target delivery."""

    @pytest.mark.asyncio
    async def test_send_single_legacy_target(self):
        """Legacy format still works — backward compatible."""
        adapter = _make_adapter({})
        mock_target = _wire_mock_target(adapter, "telegram")

        chat_id = "webhook:test:delivery-1"
        adapter._delivery_info[chat_id] = {
            "deliver": "telegram",
            "deliver_extra": {"chat_id": "123"},
            "payload": {},
        }

        result = await adapter.send(chat_id, "Hello world")
        assert result.success
        mock_target.send.assert_called_once_with("123", "Hello world", metadata=None)

    @pytest.mark.asyncio
    async def test_send_multi_target_all_succeed(self):
        """Multi-target: all succeed → success=True."""
        adapter = _make_adapter({})
        adapters = _wire_mock_targets(adapter, ["telegram", "discord"])

        chat_id = "webhook:test:delivery-2"
        adapter._delivery_info[chat_id] = {
            "deliver": [
                {"type": "telegram", "chat_id": "111"},
                {"type": "discord", "chat_id": "222"},
            ],
            "payload": {},
        }

        result = await adapter.send(chat_id, "Hello multi")
        assert result.success
        adapters[Platform("telegram")].send.assert_called_once_with(
            "111", "Hello multi", metadata=None
        )
        adapters[Platform("discord")].send.assert_called_once_with(
            "222", "Hello multi", metadata=None
        )

    @pytest.mark.asyncio
    async def test_send_multi_target_partial_failure(self):
        """Multi-target: one fails → success=True (best-effort), error describes failure."""
        adapter = _make_adapter({})
        adapters = _wire_mock_targets(adapter, ["telegram", "discord"])
        # Make discord fail
        adapters[Platform("discord")].send = AsyncMock(
            return_value=SendResult(success=False, error="Connection refused")
        )

        chat_id = "webhook:test:delivery-3"
        adapter._delivery_info[chat_id] = {
            "deliver": [
                {"type": "telegram", "chat_id": "111"},
                {"type": "discord", "chat_id": "222"},
            ],
            "payload": {},
        }

        result = await adapter.send(chat_id, "Hello partial")
        assert result.success  # best-effort: at least one succeeded
        assert "discord" in result.error

    @pytest.mark.asyncio
    async def test_send_multi_target_all_fail(self):
        """Multi-target: all fail → success=False."""
        adapter = _make_adapter({})
        adapters = _wire_mock_targets(adapter, ["telegram", "discord"])
        adapters[Platform("telegram")].send = AsyncMock(
            return_value=SendResult(success=False, error="Timeout")
        )
        adapters[Platform("discord")].send = AsyncMock(
            return_value=SendResult(success=False, error="Connection refused")
        )

        chat_id = "webhook:test:delivery-4"
        adapter._delivery_info[chat_id] = {
            "deliver": [
                {"type": "telegram", "chat_id": "111"},
                {"type": "discord", "chat_id": "222"},
            ],
            "payload": {},
        }

        result = await adapter.send(chat_id, "Hello fail")
        assert not result.success
        assert "telegram" in result.error
        assert "discord" in result.error

    @pytest.mark.asyncio
    async def test_send_log_target(self):
        """Log target always succeeds and doesn't need gateway_runner."""
        adapter = _make_adapter({})

        chat_id = "webhook:test:delivery-5"
        adapter._delivery_info[chat_id] = {
            "deliver": "log",
            "deliver_extra": {},
            "payload": {},
        }

        result = await adapter.send(chat_id, "Hello log")
        assert result.success


# ===================================================================
# deliver_only with multi-target (via HTTP POST)
# ===================================================================

class TestDeliverOnlyMultiTarget:
    """Test deliver_only routes with multi-target delivery."""

    @pytest.mark.asyncio
    async def test_direct_deliver_multi_target(self):
        routes = {
            "multi-push": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": [
                    {"type": "telegram", "chat_id": "111"},
                    {"type": "discord", "chat_id": "222"},
                ],
                "deliver_only": True,
                "prompt": "Alert: {message}",
            }
        }
        adapter = _make_adapter(routes)
        adapters = _wire_mock_targets(adapter, ["telegram", "discord"])
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as client:
            payload = json.dumps({"message": "Server is down"})
            resp = await client.post(
                "/webhooks/multi-push",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "alert",
                },
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["status"] == "delivered"

        adapters[Platform("telegram")].send.assert_called_once()
        adapters[Platform("discord")].send.assert_called_once()

        # Verify the content was rendered correctly
        tg_call = adapters[Platform("telegram")].send.call_args
        assert "Server is down" in tg_call[0][1]

    @pytest.mark.asyncio
    async def test_direct_deliver_multi_target_with_template_extras(self):
        """Template values in deliver target extras get rendered with payload."""
        routes = {
            "pr-multi": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": [
                    {"type": "telegram", "chat_id": "{notify.telegram_id}"},
                    {"type": "github_comment", "repo": "{repo.full_name}", "pr_number": "{pr.number}"},
                ],
                "deliver_only": True,
                "prompt": "PR reviewed",
            }
        }
        adapter = _make_adapter(routes)
        adapters = _wire_mock_targets(adapter, ["telegram"])
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as client:
            payload = json.dumps({
                "notify": {"telegram_id": "99999"},
                "repo": {"full_name": "org/repo"},
                "pr": {"number": "42"},
            })
            resp = await client.post(
                "/webhooks/pr-multi",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "test",
                },
            )
            assert resp.status == 200

        # Telegram should be called with the rendered chat_id
        adapters[Platform("telegram")].send.assert_called_once()
        tg_call = adapters[Platform("telegram")].send.call_args
        assert tg_call[0][0] == "99999"  # chat_id rendered from payload


# ===================================================================
# Startup validation for multi-target deliver_only
# ===================================================================

class TestStartupValidationMultiTarget:
    """connect() should validate multi-target deliver_only configs."""

    @pytest.mark.asyncio
    async def test_deliver_only_list_with_log_rejected(self):
        """A deliver_only route with a 'log' type in the list is rejected."""
        routes = {
            "bad-route": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": [
                    {"type": "telegram", "chat_id": "111"},
                    {"type": "log"},
                ],
                "deliver_only": True,
            }
        }
        adapter = _make_adapter(routes)
        with pytest.raises(ValueError, match="deliver_only.*log"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_deliver_only_empty_list_rejected(self):
        """A deliver_only route with an empty deliver list is rejected."""
        routes = {
            "empty-route": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": [],
                "deliver_only": True,
            }
        }
        adapter = _make_adapter(routes)
        with pytest.raises(ValueError, match="deliver_only.*empty"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_deliver_only_list_missing_type_rejected(self):
        """A deliver_only route with items missing 'type' is rejected."""
        routes = {
            "bad-item": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": [
                    {"chat_id": "111"},  # missing "type"
                ],
                "deliver_only": True,
            }
        }
        adapter = _make_adapter(routes)
        with pytest.raises(ValueError, match="must be a dict with a 'type' key"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_deliver_only_valid_multi_target_accepted(self):
        """A properly configured multi-target deliver_only route passes validation."""
        routes = {
            "good-multi": {
                "secret": _INSECURE_NO_AUTH,
                "deliver": [
                    {"type": "telegram", "chat_id": "111"},
                    {"type": "discord", "chat_id": "222"},
                ],
                "deliver_only": True,
            }
        }
        adapter = _make_adapter(routes)
        # Should not raise — just test it doesn't crash during validation.
        # (connect() will try to bind a port which may fail in test env,
        #  so we only test that ValueError is not raised for the validation part)
        try:
            await adapter.connect()
        except (OSError, ValueError) as e:
            # OSError for port binding is fine; ValueError means validation failed
            if isinstance(e, ValueError):
                pytest.fail(f"Validation unexpectedly failed: {e}")
        finally:
            await adapter.disconnect()


# ===================================================================
# Backward compatibility
# ===================================================================

class TestBackwardCompatibility:
    """Ensure existing single-deliver configs still work unchanged."""

    @pytest.mark.asyncio
    async def test_legacy_string_deliver_with_deliver_extra(self):
        """Old-style deliver + deliver_extra still works."""
        adapter = _make_adapter({})
        mock_target = _wire_mock_target(adapter, "telegram")

        chat_id = "webhook:compat:delivery-1"
        adapter._delivery_info[chat_id] = {
            "deliver": "telegram",
            "deliver_extra": {"chat_id": "99999", "message_thread_id": "42"},
            "payload": {},
        }

        result = await adapter.send(chat_id, "Backward compat test")
        assert result.success
        mock_target.send.assert_called_once_with(
            "99999", "Backward compat test", metadata={"thread_id": "42"}
        )

    @pytest.mark.asyncio
    async def test_legacy_github_comment_deliver(self):
        """Old-style github_comment delivery still works."""
        adapter = _make_adapter({})

        chat_id = "webhook:compat:delivery-2"
        adapter._delivery_info[chat_id] = {
            "deliver": "github_comment",
            "deliver_extra": {"repo": "org/repo", "pr_number": "42"},
            "payload": {},
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = await adapter.send(chat_id, "LGTM!")

        assert result.success
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "org/repo" in call_args
        assert "42" in call_args
