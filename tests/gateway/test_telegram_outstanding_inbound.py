"""Telegram inbound updates held by the platform are OBSERVABLE, never silently dropped.

Kanban t_1915c9a3, finding C3.

On 2026-10-23 the gateway's getUpdates consumer wedged for ~19 minutes. Five of the
operator's DMs landed in that window. Telegram HELD them server-side
(``get_webhook_info().pending_update_count > 0``) — they were never handed to the
gateway at all, so:

* no ``_pending_messages`` entry existed (the gateway never saw the message);
* no ``delivery_obligations`` row was created;
* no "response ready" line was logged for the chat.

The only evidence was one log line about a non-zero ``pending_update_count``, which
nobody was reading. The messages were not lost — Telegram had them — but from the
operator's side the gateway had gone deaf with no acknowledgement of any kind. The
silent discard is the failure mode that cost the hour.

These tests pin the fix: a held inbound must be visible in the platform's runtime
status (``needs_attention`` + an explicit message, so ``hermes gateway status`` and
the desktop Channels page show it) and in the journal (an explicit "held by the
platform" line naming the count and the episode age).

See also tests/gateway/test_telegram_pending_update_probe.py, which covers the
WEDGE DETECTION that already existed; this file covers the OBSERVABILITY that did not.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _make_adapter(*, pending: int) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._webhook_mode = False
    adapter._app = MagicMock()
    adapter._app.updater.running = True
    bot = MagicMock()
    bot.get_webhook_info = AsyncMock(
        return_value=MagicMock(pending_update_count=pending)
    )
    adapter._app.bot = bot
    adapter._bot = bot
    return adapter


@pytest.mark.asyncio
async def test_a_held_inbound_is_published_to_the_platform_runtime_status():
    """One probe seeing a queued update flags the platform as needing attention.

    The message is not lost (Telegram has it), but the gateway has not been handed it,
    so nothing local would say so. The runtime status is where an operator looks.
    """
    adapter = _make_adapter(pending=3)
    with patch("gateway.status.publish_runtime_status") as publish:
        await adapter._probe_pending_updates(adapter._app.bot, 5)
    publish.assert_called_once()
    kwargs = publish.call_args.kwargs
    assert kwargs["platform"] == adapter.platform.value
    assert kwargs["needs_attention"] is True
    assert kwargs["error_code"] == "inbound_updates_outstanding"
    # The message must say the messages are HELD, not lost, and how many.
    assert "3 inbound update(s)" in kwargs["error_message"]
    assert "not lost" in kwargs["error_message"]
    # The platform's own state must NOT be touched: the send path is healthy and the
    # platform is up — it is the INBOUND queue that is not draining. Marking it
    # "retrying" would misreport a connected platform as reconnecting.
    assert "platform_state" not in kwargs


@pytest.mark.asyncio
async def test_the_journal_records_that_the_platform_is_holding_the_messages():
    """The journal must distinguish "held by the platform" from "never arrived"."""
    adapter = _make_adapter(pending=2)
    with patch("gateway.status.publish_runtime_status"), \
            patch("plugins.platforms.telegram.adapter.logger") as log:
        await adapter._probe_pending_updates(adapter._app.bot, 5)
    warnings = [c for c in log.warning.call_args_list if c.args]
    assert warnings, "a held inbound must produce a warning line"
    # The format string is the first arg; the rendered text is in the args after it.
    texts = []
    for call in warnings:
        template = str(call.args[0])
        params = call.args[1:]
        try:
            texts.append(template % params if params else template)
        except TypeError:
            texts.append(template)
    text = " ".join(texts)
    assert "holding" in text
    assert "NOT lost" in text


@pytest.mark.asyncio
async def test_the_outstanding_count_is_tracked_on_the_adapter():
    """The adapter keeps the current held count and the episode's first-seen time."""
    adapter = _make_adapter(pending=4)
    with patch("gateway.status.publish_runtime_status"):
        await adapter._probe_pending_updates(adapter._app.bot, 5)
    assert adapter._outstanding_updates_count == 4
    assert adapter._outstanding_updates_since is not None
    assert adapter._outstanding_updates_peak == 4


@pytest.mark.asyncio
async def test_an_empty_queue_clears_the_outstanding_state_and_the_attention_flag():
    """When the queue drains the flag is cleared — a drained queue is not an incident.

    The episode timer is cleared too, so a LATER episode measures its own age instead of
    inheriting the age of a previous one.
    """
    adapter = _make_adapter(pending=0)
    adapter._outstanding_updates_count = 5
    adapter._outstanding_updates_since = 1000.0
    with patch("gateway.status.publish_runtime_status") as publish:
        await adapter._probe_pending_updates(adapter._app.bot, 5)
    assert adapter._outstanding_updates_count == 0
    assert adapter._outstanding_updates_since is None
    kwargs = publish.call_args.kwargs
    assert kwargs["needs_attention"] is False
    assert kwargs["error_code"] is None


@pytest.mark.asyncio
async def test_a_second_episode_measures_its_own_age():
    """Each episode's age starts at its own first observation."""
    adapter = _make_adapter(pending=1)
    with patch("gateway.status.publish_runtime_status"):
        await adapter._probe_pending_updates(adapter._app.bot, 5)
        first_since = adapter._outstanding_updates_since
        assert first_since is not None
        # The queue drains, then a new update is held.
        adapter._app.bot.get_webhook_info = AsyncMock(
            return_value=MagicMock(pending_update_count=0))
        await adapter._probe_pending_updates(adapter._app.bot, 5)
        assert adapter._outstanding_updates_since is None
        adapter._app.bot.get_webhook_info = AsyncMock(
            return_value=MagicMock(pending_update_count=2))
        await adapter._probe_pending_updates(adapter._app.bot, 5)
        assert adapter._outstanding_updates_since is not None
        assert adapter._outstanding_updates_since >= first_since


@pytest.mark.asyncio
async def test_an_unchanged_count_does_not_republish_or_relog():
    """A steady held count is one incident, not a new one every probe.

    The peak is kept, so a queue that grew and then settled still shows the high-water
    mark to whoever reads the status.
    """
    adapter = _make_adapter(pending=2)
    with patch("gateway.status.publish_runtime_status") as publish:
        await adapter._probe_pending_updates(adapter._app.bot, 5)
        assert publish.call_count == 1
        await adapter._probe_pending_updates(adapter._app.bot, 5)
        assert publish.call_count == 1  # unchanged -> no second publish
    assert adapter._outstanding_updates_count == 2


@pytest.mark.asyncio
async def test_a_growing_count_is_published_again():
    """A queue that grows is a new observation and must be published again."""
    adapter = _make_adapter(pending=1)
    with patch("gateway.status.publish_runtime_status") as publish:
        await adapter._probe_pending_updates(adapter._app.bot, 5)
        adapter._app.bot.get_webhook_info = AsyncMock(
            return_value=MagicMock(pending_update_count=6))
        await adapter._probe_pending_updates(adapter._app.bot, 5)
    assert publish.call_count == 2
    assert adapter._outstanding_updates_count == 6
    assert adapter._outstanding_updates_peak == 6


@pytest.mark.asyncio
async def test_a_status_publish_failure_never_breaks_the_probe():
    """Best-effort: a broken status write must not stop the wedge detection."""
    adapter = _make_adapter(pending=2)
    with patch("gateway.status.publish_runtime_status", side_effect=RuntimeError("boom")):
        # Must not raise.
        await adapter._probe_pending_updates(adapter._app.bot, 5)
    # The count is still tracked locally, and the wedge counter still advanced.
    assert adapter._outstanding_updates_count == 2
    assert adapter._polling_pending_stuck_count == 1


@pytest.mark.asyncio
async def test_the_wedge_escalation_still_works_with_the_new_observability():
    """The observability must not displace the recovery ladder.

    Two consecutive stuck probes escalate to a polling restart — that is what actually
    recovers the held messages. The new status publishing rides alongside it.
    """
    adapter = _make_adapter(pending=3)
    recovery = AsyncMock()
    with patch("gateway.status.publish_runtime_status"):
        with patch.object(adapter, "_handle_polling_network_error", new=recovery):
            await adapter._probe_pending_updates(adapter._app.bot, 5)
            assert adapter._polling_error_task is None
            await adapter._probe_pending_updates(adapter._app.bot, 5)
            task = adapter._polling_error_task
            assert task is not None
            await task
    recovery.assert_awaited_once()
