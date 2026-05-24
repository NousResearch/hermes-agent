"""Dedup persistence contract test — ensures message IDs survive a restart.

SDK ``Deduper.check_and_mark`` drives dedup, backed by
``JsonFileDedupStore`` injected in ``FeishuAdapter.connect()``. The
cross-restart contract is exercised end-to-end through the persistence
path: dispatch an event, flush dedup state on shutdown, build a second
adapter on the same state file, dispatch the same event, and assert the
second dispatch is deduped.
"""

import asyncio

import pytest

pytest.importorskip("lark_oapi.channel")

from .conftest import dispatch_inbound_event


def test_duplicate_message_dropped_on_second_dispatch(adapter_harness):
    event = {
        "header": {"event_id": "evt_dup", "event_type": "im.message.receive_v1",
                   "create_time": "1714200000000", "token": "t", "app_id": "cli_test_app"},
        "event": {
            "sender": {"sender_id": {"open_id": "ou_alice", "user_id": "u_alice"}, "sender_type": "user"},
            "message": {
                "message_id": "om_dup_test", "chat_id": "p2p_alice", "chat_type": "p2p",
                "message_type": "text", "content": '{"text":"x"}',
                "create_time": "1714200000000", "mentions": [],
            }
        }
    }

    async def _run():
        await dispatch_inbound_event(adapter_harness, event)
        await dispatch_inbound_event(adapter_harness, event)

    asyncio.run(_run())
    assert len(adapter_harness.captured_inbound) == 1, (
        "Second dispatch of same message_id must be deduped"
    )


def test_dedup_state_persists_across_restart(tmp_path):
    """Cross-restart dedup contract — exercised through SDK Deduper +
    JsonFileDedupStore.

    Build adapter1, dispatch event A, shutdown (flush). Build adapter2 with
    the same dedup state file, dispatch SAME event A, verify second
    dispatch is deduped. Exercises the persistence path end-to-end with no
    Hermes-internal helper calls; breaks of the dedup state file schema or
    the SDK Deduper interaction land here.
    """
    from .conftest import _build_capturing_lark_client, _build_platform_config

    state_path = tmp_path / "feishu_seen_message_ids.json"
    captured_sends: list = []

    event = {
        "header": {"event_id": "evt_persist", "event_type": "im.message.receive_v1",
                   "create_time": "1714200000000", "token": "t", "app_id": "cli_test_app"},
        "event": {
            "sender": {"sender_id": {"open_id": "ou_alice", "user_id": "u_alice"},
                       "sender_type": "user"},
            "message": {
                "message_id": "om_persist_test", "chat_id": "p2p_alice",
                "chat_type": "p2p", "message_type": "text",
                "content": '{"text":"hello"}',
                "create_time": "1714200000000", "mentions": [],
            }
        }
    }

    def _build():
        from gateway.platforms.feishu import FeishuAdapter, JsonFileDedupStore

        config = _build_platform_config()
        adapter = FeishuAdapter(config)
        adapter._dedup_state_path = state_path
        adapter._dedup_store = JsonFileDedupStore(
            path=state_path,
            max_entries=256,
            account_id=adapter._app_id,
        )
        captured_inbound: list = []

        async def _capture(event):
            captured_inbound.append(event)

        adapter.set_message_handler(_capture)
        adapter._client = _build_capturing_lark_client(captured_sends)
        adapter._bot_open_id = "ou_hermes_bot"
        adapter._bot_user_id = "u_hermes_bot"
        adapter._bot_name = "HermesBot"
        adapter._group_policy = "open"
        adapter._default_group_policy = "open"

        # Build a minimal AdapterHarness-compatible wrapper for
        # dispatch_inbound_event (it only reads ``adapter`` + writes to
        # ``captured_inbound``). We set a mock channel with the same
        # bot_identity used by ``to_message_event``.
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        mock_channel = MagicMock(name="feishu_channel")
        mock_channel.bot_identity = SimpleNamespace(
            open_id="ou_hermes_bot", user_id="u_hermes_bot", name="HermesBot",
        )

        async def _noop_send(*a, **k):
            return SimpleNamespace(success=True, message_id="om_x", error=None)

        async def _noop_get_chat_info(chat_id):
            inferred = "p2p" if str(chat_id).startswith("p2p") else "group"
            return SimpleNamespace(chat_id=chat_id, name="Test", chat_type=inferred)

        async def _noop_download(*a, **k):
            raise RuntimeError("mock channel: skip download")

        async def _noop_fetch(message_id):
            return {"data": {"items": []}}

        mock_channel.send = _noop_send
        mock_channel.get_chat_info = _noop_get_chat_info
        mock_channel.download_resource_to_file = _noop_download
        mock_channel.fetch_message = _noop_fetch
        adapter._channel = mock_channel

        from .conftest import AdapterHarness
        return AdapterHarness(
            adapter=adapter,
            captured_inbound=captured_inbound,
            captured_sends=captured_sends,
        )

    async def _run():
        # First adapter: dispatch event, flush dedup state on shutdown.
        h1 = _build()
        await dispatch_inbound_event(h1, event)
        assert len(h1.captured_inbound) == 1, (
            "First dispatch must succeed (not yet seen)"
        )
        # Flush the JsonFileDedupStore directly — disconnect() does this
        # via channel.disconnect path which we don't fully wire in tests.
        h1.adapter._dedup_store.flush()

        # Second adapter on the same state file: same event must be deduped.
        h2 = _build()
        await dispatch_inbound_event(h2, event)
        assert len(h2.captured_inbound) == 0, (
            "Second adapter must dedup persisted message_id"
        )

    asyncio.run(_run())
