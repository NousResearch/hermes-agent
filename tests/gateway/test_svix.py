"""Unit tests for the Svix polling platform adapter.

Scope is deliberately narrow: the render → deliver pipeline (``send``,
``_render_prompt``, ``_inject_skill``, github/cross-platform delivery) lives
in ``WebhookDeliveryMixin`` and is already exercised by
``test_webhook_integration.py``. These tests cover only the svix-*specific*
glue that has no webhook equivalent:

1. ``_parse_polling_url``  — URL → (app_id, sink_id, server_url) + errors
2. ``_resolve_route_token`` — literal > env-var > global precedence + raises
3. ``connect``            — config validation surfaces as a *non-retryable*
   fatal error and returns False (the gateway treats a raised exception as a
   transient retry, which is wrong for a config typo)
4. ``_process_message``   — event-type extraction (regression guard for the
   ``eventType`` vs ``event_type`` SDK attribute), event filtering, in-process
   dedup, and the ``deliver_only`` bypass

The ``_process_message`` message objects are real ``PollingEndpointMessageOut``
instances, not mocks, so a future getattr against a non-existent attribute is
caught rather than silently falling back to a default.
"""

import asyncio
import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The svix SDK is a lazy/runtime dependency (LAZY_DEPS), not part of the
# `[all]` extra that CI installs — so it is absent unless lazy-installed.
# Skip the whole module cleanly rather than erroring at collection time.
pytest.importorskip("svix")

from gateway.config import Platform, PlatformConfig
from gateway.platforms import svix as svix_mod
from gateway.platforms.svix import SvixAdapter, _parse_polling_url
from svix.models.polling_endpoint_message_out import PollingEndpointMessageOut


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(routes, **extra_kw) -> SvixAdapter:
    extra = {"routes": routes}
    extra.update(extra_kw)
    config = PlatformConfig(enabled=True, extra=extra)
    return SvixAdapter(config)


def _make_message(
    *, event_type: str, msg_id: str = "msg_1", payload: dict | None = None
) -> PollingEndpointMessageOut:
    """A real Svix poll message — uses snake_case ``event_type`` like the SDK."""
    return PollingEndpointMessageOut(
        event_type=event_type,
        id=msg_id,
        payload=payload or {},
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )


# ===================================================================
# _parse_polling_url
# ===================================================================

class TestParsePollingURL:

    def test_standard_url(self):
        app_id, sink_id, server_url = _parse_polling_url(
            "https://api.svix.com/api/v1/app/app_xxx/poller/poll_yyy/"
        )
        assert app_id == "app_xxx"
        assert sink_id == "poll_yyy"
        assert server_url == "https://api.svix.com"

    def test_region_host_preserved(self):
        # EU / self-hosted hosts must flow through to SvixOptions(server_url=...)
        _, _, server_url = _parse_polling_url(
            "https://api.eu.svix.com/api/v1/app/app_1/poller/poll_2/"
        )
        assert server_url == "https://api.eu.svix.com"

    def test_trailing_slash_optional(self):
        app_id, sink_id, _ = _parse_polling_url(
            "https://api.svix.com/api/v1/app/app_1/poller/poll_2"
        )
        assert (app_id, sink_id) == ("app_1", "poll_2")

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Invalid Svix polling URL"):
            _parse_polling_url("https://example.com/not/a/poller")


# ===================================================================
# _resolve_route_token
# ===================================================================

class TestResolveRouteToken:

    def test_literal_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("SOME_ENV", "env_tok")
        adapter = _make_adapter({})
        token = adapter._resolve_route_token(
            "r", {"auth_token": "literal_tok", "auth_token_env": "SOME_ENV"}
        )
        assert token == "literal_tok"

    def test_env_var_lookup(self, monkeypatch):
        monkeypatch.setenv("ROUTE_TOK", "from_env")
        adapter = _make_adapter({})
        token = adapter._resolve_route_token("r", {"auth_token_env": "ROUTE_TOK"})
        assert token == "from_env"

    def test_empty_configured_env_raises(self, monkeypatch):
        monkeypatch.delenv("MISSING_TOK", raising=False)
        adapter = _make_adapter({})
        with pytest.raises(ValueError, match="auth_token_env=MISSING_TOK"):
            adapter._resolve_route_token("r", {"auth_token_env": "MISSING_TOK"})

    def test_no_token_returns_empty(self):
        adapter = _make_adapter({})
        assert adapter._resolve_route_token("r", {}) == ""


# ===================================================================
# connect() — config validation contract
# ===================================================================

_VALID_URL = "https://api.svix.com/api/v1/app/app_1/poller/poll_2/"


class TestConnectContract:
    """connect() must return False (never raise) on permanent misconfig, and
    flag it as a *non-retryable* fatal error so the gateway stops retrying."""

    @pytest.mark.asyncio
    async def test_missing_sdk_is_nonretryable_fatal(self):
        adapter = _make_adapter({"r": {"url": _VALID_URL, "auth_token": "t"}})
        with patch.object(svix_mod, "SVIX_AVAILABLE", False):
            assert await adapter.connect() is False
        assert adapter.has_fatal_error
        assert adapter.fatal_error_retryable is False
        assert adapter.fatal_error_code == "svix_missing_dependency"

    @pytest.mark.asyncio
    async def test_missing_url_is_nonretryable_fatal(self):
        adapter = _make_adapter({"r": {"auth_token": "t"}})
        assert await adapter.connect() is False
        assert adapter.has_fatal_error
        assert adapter.fatal_error_retryable is False
        assert adapter.fatal_error_code == "svix_invalid_config"

    @pytest.mark.asyncio
    async def test_bad_url_is_nonretryable_fatal(self):
        adapter = _make_adapter(
            {"r": {"url": "https://x/no/poller", "auth_token": "t"}}
        )
        assert await adapter.connect() is False
        assert adapter.fatal_error_code == "svix_invalid_config"

    @pytest.mark.asyncio
    async def test_missing_token_is_nonretryable_fatal(self):
        adapter = _make_adapter({"r": {"url": _VALID_URL}})
        assert await adapter.connect() is False
        assert adapter.fatal_error_code == "svix_invalid_config"

    @pytest.mark.asyncio
    async def test_deliver_only_with_log_is_nonretryable_fatal(self):
        adapter = _make_adapter(
            {"r": {"url": _VALID_URL, "auth_token": "t",
                   "deliver_only": True, "deliver": "log"}}
        )
        assert await adapter.connect() is False
        assert adapter.fatal_error_code == "svix_invalid_config"

    @pytest.mark.asyncio
    async def test_valid_config_starts_poll_tasks(self):
        adapter = _make_adapter(
            {"r1": {"url": _VALID_URL, "auth_token": "t"}}
        )
        # Stub the poll loop so connect() doesn't hit the network.
        adapter._poll_route = AsyncMock()
        try:
            assert await adapter.connect() is True
            assert not adapter.has_fatal_error
            assert set(adapter._poll_tasks) == {"r1"}
            assert "r1" in adapter._clients
            # Deterministic default consumer ID so restarts reuse the cursor.
            assert adapter._consumer_ids["r1"] == "hermes-r1"
        finally:
            await adapter.disconnect()


# ===================================================================
# _process_message
# ===================================================================

class TestProcessMessage:

    @pytest.mark.asyncio
    async def test_event_type_is_read_and_passes_filter(self):
        """Regression guard: event_type must come from message.event_type.

        If it were read from a non-existent attribute (the original
        ``eventType`` bug), every event would become "unknown" and this
        filtered route would drop the message — handle_message never fires.
        """
        adapter = _make_adapter(
            {"r": {"events": ["issues.opened"], "prompt": "{__event__}",
                   "deliver": "log"}}
        )
        adapter.handle_message = AsyncMock()
        msg = _make_message(event_type="issues.opened", msg_id="m1")

        await adapter._process_message("r", msg)

        adapter.handle_message.assert_called_once()
        event = adapter.handle_message.call_args.args[0]
        # __event__ rendered the real event type, not "unknown".
        assert event.text == "issues.opened"
        assert event.source.chat_id == "svix:r:m1"
        assert adapter._delivery_info["svix:r:m1"]["deliver"] == "log"

    @pytest.mark.asyncio
    async def test_event_outside_filter_is_dropped(self):
        adapter = _make_adapter(
            {"r": {"events": ["issues.opened"], "deliver": "log"}}
        )
        adapter.handle_message = AsyncMock()

        await adapter._process_message(
            "r", _make_message(event_type="issues.closed")
        )

        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_duplicate_message_id_processed_once(self):
        adapter = _make_adapter({"r": {"deliver": "log"}})
        adapter.handle_message = AsyncMock()
        msg = _make_message(event_type="x", msg_id="dup")

        await adapter._process_message("r", msg)
        await adapter._process_message("r", msg)

        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_deliver_only_bypasses_agent(self):
        adapter = _make_adapter(
            {"r": {"deliver_only": True, "deliver": "telegram",
                   "prompt": "hello {name}"}}
        )
        adapter.handle_message = AsyncMock()
        adapter._direct_deliver = AsyncMock()

        await adapter._process_message(
            "r", _make_message(event_type="x", payload={"name": "world"})
        )

        adapter.handle_message.assert_not_called()
        adapter._direct_deliver.assert_called_once()
        # The rendered template (not the agent) is the message body.
        content, delivery = adapter._direct_deliver.call_args.args
        assert content == "hello world"
        assert delivery["deliver"] == "telegram"


# ===================================================================
# _poll_route — the actual Svix SDK call, iterator advancement, done handling
# ===================================================================

class TestPollRoute:
    """Exercises the poll loop itself (the rest of the suite stubs it out).

    Guards the riskiest unverified surface: the exact ``consumer_poll(app_id,
    sink_id, consumer_id, options)`` call shape, that the first request omits
    the iterator (server resumes from the tracked consumer position), that a
    returned iterator is carried into the next request, and that ``done`` is
    honored (``False`` → drain immediately, ``True`` → wait then poll again).
    """

    def _adapter_with_fake_client(self, consumer_poll: AsyncMock) -> SvixAdapter:
        adapter = _make_adapter({"r": {"deliver": "log"}}, poll_interval=0.01)
        client = MagicMock()
        client.message.poller.consumer_poll = consumer_poll
        adapter._clients["r"] = client
        adapter._client_endpoints["r"] = ("app_1", "poll_2")
        adapter._consumer_ids["r"] = "hermes-r"
        adapter._process_message = AsyncMock()
        return adapter

    @pytest.mark.asyncio
    async def test_drains_then_advances_iterator_and_waits_when_caught_up(self):
        msg1 = _make_message(event_type="issues.opened", msg_id="m1")
        # Page 1: has data, more pages pending (done=False) → loop immediately.
        # Page 2: empty + caught up (done=True) → sleep, which we hijack to
        # break the otherwise-infinite loop.
        page1 = SimpleNamespace(data=[msg1], iterator="it1", done=False)
        page2 = SimpleNamespace(data=[], iterator="it1", done=True)
        consumer_poll = AsyncMock(side_effect=[page1, page2])
        adapter = self._adapter_with_fake_client(consumer_poll)

        with patch.object(
            svix_mod.asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError)
        ):
            with pytest.raises(asyncio.CancelledError):
                await adapter._poll_route("r")

        # Two polls: page1 (drain immediately), page2 (caught up → sleep → cancel).
        assert consumer_poll.call_count == 2

        # Call shape: positional (app_id, sink_id, consumer_id, options).
        first_args = consumer_poll.call_args_list[0].args
        assert first_args[:3] == ("app_1", "poll_2", "hermes-r")
        # First request omits the iterator so the server resumes from the
        # consumer's tracked position.
        assert first_args[3].iterator is None
        assert first_args[3].limit == adapter._poll_limit

        # Second request carries the iterator returned by page1.
        second_args = consumer_poll.call_args_list[1].args
        assert second_args[3].iterator == "it1"

        # The single message on page1 was dispatched, and the in-memory cursor
        # advanced.
        adapter._process_message.assert_called_once_with("r", msg1)
        assert adapter._iterators["r"] == "it1"

    @pytest.mark.asyncio
    async def test_poll_error_backs_off_then_retries(self):
        # First poll raises (transient), second is caught-up → sleep → cancel.
        # Proves the loop doesn't die on an error and retries after backoff.
        page = SimpleNamespace(data=[], iterator=None, done=True)
        consumer_poll = AsyncMock(side_effect=[RuntimeError("boom"), page])
        adapter = self._adapter_with_fake_client(consumer_poll)

        sleeps: list[float] = []

        async def _record_sleep(delay):
            sleeps.append(delay)
            # Let the backoff sleep pass; cancel on the caught-up wait so the
            # loop terminates after the successful retry.
            if len(sleeps) >= 2:
                raise asyncio.CancelledError

        with patch.object(svix_mod.asyncio, "sleep", _record_sleep):
            with pytest.raises(asyncio.CancelledError):
                await adapter._poll_route("r")

        assert consumer_poll.call_count == 2
        # First sleep is the post-error backoff (1.0s), second is the caught-up
        # poll_interval wait.
        assert sleeps[0] == 1.0
