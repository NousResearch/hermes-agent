"""Focused checks around the real SDK policy/config boundary."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("lark_oapi.channel")


def _policy_decision(adapter, msg):
    from lark_oapi.channel.safety.policy_gate import PolicyGate

    cfg = adapter._build_sdk_policy_config(adapter._settings)
    return PolicyGate(cfg).evaluate(msg)


def _message(
    *,
    sender,
    mentioned_all=False,
    mentions=None,
    content_text="hello",
    message_id="om_sdk_pipeline",
):
    from lark_oapi.channel import Conversation, InboundMessage
    from lark_oapi.channel.types import TextContent

    return InboundMessage(
        id=message_id,
        create_time=1,
        conversation=Conversation(chat_id="oc_team", chat_type="group"),
        sender=sender,
        content=TextContent(text=content_text),
        content_text=content_text,
        mentions=list(mentions or []),
        mentioned_all=mentioned_all,
        resources=[],
        raw={},
    )


def _sdk_event_data(
    *,
    event_id,
    message_id,
    sender_open_id,
    sender_user_id=None,
    sender_type="user",
    text="hello",
):
    return SimpleNamespace(
        header=SimpleNamespace(event_id=event_id),
        event=SimpleNamespace(
            sender=SimpleNamespace(
                sender_id=SimpleNamespace(
                    open_id=sender_open_id,
                    user_id=sender_user_id or sender_open_id.replace("ou_", "u_"),
                ),
                sender_type=sender_type,
            ),
            message=SimpleNamespace(
                message_id=message_id,
                chat_id="oc_team",
                chat_type="group",
                message_type="text",
                content=json.dumps({"text": text}),
                create_time="1714200000000",
                mentions=[],
            ),
        ),
    )


async def _dispatch_through_sdk_pipeline(adapter, data):
    channel = adapter._build_sdk_channel(register_handlers=False)
    delivered = []
    channel.on("message", lambda msg: delivered.append(msg))

    await channel._handle_message_event(data)
    for _ in range(20):
        if delivered:
            break
        await asyncio.sleep(0.05)
    return delivered


async def _drain_adapter_tasks():
    for _ in range(20):
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_sdk_message_handoff_keeps_adapter_tasks_on_adapter_loop(adapter_harness):
    """SDK bg-loop callbacks must not create Hermes background tasks on that loop."""
    from lark_oapi.channel import Identity

    adapter = adapter_harness.adapter
    adapter._loop = asyncio.get_running_loop()
    adapter._allow_bots = "all"
    adapter._require_mention = False

    handler_started = threading.Event()
    release_handler = threading.Event()

    async def blocking_handler(_event):
        handler_started.set()
        await asyncio.to_thread(release_handler.wait)

    adapter.set_message_handler(blocking_handler)

    bg_loop_ready = threading.Event()
    bg_error = []

    def run_on_sdk_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bg_loop_ready.set()

        async def invoke_handler():
            try:
                await adapter._on_sdk_message(
                    _message(
                        sender=Identity(
                            open_id="ou_peer_bot",
                            user_id="u_peer_bot",
                            is_bot=True,
                        ),
                    )
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                bg_error.append(exc)
            finally:
                loop.call_soon(loop.stop)

        loop.create_task(invoke_handler())
        loop.run_forever()
        loop.close()

    thread = threading.Thread(target=run_on_sdk_loop, name="fake-sdk-bg-loop")
    thread.start()

    assert bg_loop_ready.wait(timeout=2.0)
    assert await asyncio.to_thread(handler_started.wait, timeout=2.0)

    background_tasks = list(adapter._background_tasks)
    assert len(background_tasks) == 1
    assert background_tasks[0].get_loop() is adapter._loop

    await adapter.cancel_background_tasks()
    release_handler.set()
    thread.join(timeout=2.0)

    assert bg_error == []


@pytest.mark.asyncio
async def test_on_sdk_message_keeps_admin_group_messages_mention_gated(adapter_harness):
    from lark_oapi.channel import Identity, Mention

    adapter = adapter_harness.adapter
    adapter._admins = {"ou_admin"}
    adapter._allow_bots = "none"
    adapter._require_mention = True

    await adapter._on_sdk_message(
        _message(
            sender=Identity(open_id="ou_admin", user_id="u_admin"),
            message_id="om_admin_no_mention",
        )
    )
    await _drain_adapter_tasks()
    assert adapter_harness.captured_inbound == []

    await adapter._on_sdk_message(
        _message(
            sender=Identity(open_id="ou_admin", user_id="u_admin"),
            mentions=[
                Mention(
                    key="@_user_1",
                    open_id="ou_hermes_bot",
                    user_id="u_hermes_bot",
                    name="HermesBot",
                )
            ],
            content_text="@HermesBot hello",
            message_id="om_admin_with_mention",
        )
    )
    await _drain_adapter_tasks()
    assert len(adapter_harness.captured_inbound) == 1


def test_sdk_policy_admits_peer_bot_before_hermes_bot_gate(monkeypatch):
    from lark_oapi.channel import Identity
    from tests.gateway.feishu.test_settings_projection import _make_adapter

    adapter = _make_adapter(
        monkeypatch,
        FEISHU_GROUP_POLICY="allowlist",
        FEISHU_ALLOWED_USERS="u_human",
        FEISHU_ALLOW_BOTS="all",
        FEISHU_REQUIRE_MENTION="false",
    )
    msg = _message(
        sender=Identity(open_id="ou_peer_bot", user_id="u_peer_bot", is_bot=True),
    )

    assert _policy_decision(adapter, msg).allowed is True


def test_sdk_policy_admits_mention_all_when_mentions_are_required(monkeypatch):
    from lark_oapi.channel import Identity
    from tests.gateway.feishu.test_settings_projection import _make_adapter

    adapter = _make_adapter(
        monkeypatch,
        FEISHU_GROUP_POLICY="open",
        FEISHU_REQUIRE_MENTION="true",
    )
    msg = _message(
        sender=Identity(open_id="ou_human"),
        mentioned_all=True,
        content_text="@_all hello",
    )

    assert _policy_decision(adapter, msg).allowed is True


def test_reaction_config_does_not_depend_on_sdk_sent_cache(adapter_harness):
    channel = adapter_harness.adapter._build_sdk_channel(register_handlers=False)

    assert channel._config.inbound.reaction_notifications == "all"


@pytest.mark.asyncio
async def test_sdk_message_pipeline_admits_mention_all(monkeypatch):
    from tests.gateway.feishu.test_settings_projection import _make_adapter

    adapter = _make_adapter(
        monkeypatch,
        FEISHU_GROUP_POLICY="open",
        FEISHU_REQUIRE_MENTION="true",
    )
    delivered = await _dispatch_through_sdk_pipeline(
        adapter,
        _sdk_event_data(
            event_id="evt_sdk_all",
            message_id="om_sdk_all",
            sender_open_id="ou_human",
            text="@_all hello",
        ),
    )

    assert len(delivered) == 1
    assert delivered[0].mentioned_all is True


@pytest.mark.asyncio
async def test_sdk_message_pipeline_admits_peer_bot_before_hermes_gate(monkeypatch):
    from tests.gateway.feishu.test_settings_projection import _make_adapter

    adapter = _make_adapter(
        monkeypatch,
        FEISHU_GROUP_POLICY="allowlist",
        FEISHU_ALLOWED_USERS="u_human",
        FEISHU_ALLOW_BOTS="all",
        FEISHU_REQUIRE_MENTION="false",
    )
    delivered = await _dispatch_through_sdk_pipeline(
        adapter,
        _sdk_event_data(
            event_id="evt_sdk_peer_bot",
            message_id="om_sdk_peer_bot",
            sender_open_id="ou_peer_bot",
            sender_user_id="u_peer_bot",
            sender_type="bot",
            text="hello",
        ),
    )

    assert len(delivered) == 1
    assert delivered[0].sender.is_bot is True


@pytest.mark.asyncio
async def test_sdk_message_pipeline_treats_app_sender_as_peer_bot(monkeypatch):
    from tests.gateway.feishu.test_settings_projection import _make_adapter

    adapter = _make_adapter(
        monkeypatch,
        FEISHU_GROUP_POLICY="allowlist",
        FEISHU_ALLOWED_USERS="u_human",
        FEISHU_ALLOW_BOTS="all",
        FEISHU_REQUIRE_MENTION="false",
    )
    delivered = await _dispatch_through_sdk_pipeline(
        adapter,
        _sdk_event_data(
            event_id="evt_sdk_peer_app",
            message_id="om_sdk_peer_app",
            sender_open_id="ou_peer_bot",
            sender_user_id="u_peer_bot",
            sender_type="app",
            text="hello",
        ),
    )

    assert len(delivered) == 1
    assert delivered[0].sender.is_bot is True


@pytest.mark.asyncio
async def test_on_sdk_message_drops_peer_bot_when_disabled(adapter_harness):
    from lark_oapi.channel import Identity

    adapter = adapter_harness.adapter
    adapter._allow_bots = "none"

    await adapter._on_sdk_message(
        _message(
            sender=Identity(open_id="ou_peer_bot", user_id="u_peer_bot", is_bot=True),
        )
    )
    await _drain_adapter_tasks()

    assert adapter_harness.captured_inbound == []


@pytest.mark.asyncio
async def test_on_sdk_message_admits_peer_bot_when_all_allowed(adapter_harness):
    from lark_oapi.channel import Identity

    adapter = adapter_harness.adapter
    adapter._allow_bots = "all"
    adapter._require_mention = False

    await adapter._on_sdk_message(
        _message(
            sender=Identity(open_id="ou_peer_bot", user_id="u_peer_bot", is_bot=True),
        )
    )
    await _drain_adapter_tasks()

    assert len(adapter_harness.captured_inbound) == 1
    assert adapter_harness.captured_inbound[0].source.is_bot is True


@pytest.mark.asyncio
async def test_on_sdk_message_requires_self_mention_for_peer_bot_mentions_mode(adapter_harness):
    from lark_oapi.channel import Identity, Mention

    adapter = adapter_harness.adapter
    adapter._allow_bots = "mentions"

    await adapter._on_sdk_message(
        _message(
            sender=Identity(open_id="ou_peer_bot", user_id="u_peer_bot", is_bot=True),
            message_id="om_peer_bot_unmentioned",
        )
    )
    await _drain_adapter_tasks()
    assert adapter_harness.captured_inbound == []

    await adapter._on_sdk_message(
        _message(
            sender=Identity(open_id="ou_peer_bot", user_id="u_peer_bot", is_bot=True),
            mentions=[Mention(key="@_user_1", open_id="ou_hermes_bot")],
            content_text="@_user_1 ping",
            message_id="om_peer_bot_mentioned",
        )
    )
    await _drain_adapter_tasks()
    assert len(adapter_harness.captured_inbound) == 1


@pytest.mark.asyncio
async def test_on_sdk_message_rechecks_human_allowlist_when_sdk_fallback_needed(adapter_harness):
    from lark_oapi.channel import Identity

    adapter = adapter_harness.adapter
    adapter._allow_bots = "all"
    adapter._allow_all_users = False
    adapter._group_policy = "allowlist"
    adapter._default_group_policy = "allowlist"
    adapter._allowed_group_users = {"u_human"}
    adapter._require_mention = False

    await adapter._on_sdk_message(
        _message(
            sender=Identity(open_id="ou_denied", user_id="u_denied"),
            message_id="om_denied_human",
        )
    )
    await _drain_adapter_tasks()
    assert adapter_harness.captured_inbound == []

    await adapter._on_sdk_message(
        _message(
            sender=Identity(open_id="ou_human", user_id="u_human"),
            message_id="om_allowed_human",
        )
    )
    await _drain_adapter_tasks()
    assert len(adapter_harness.captured_inbound) == 1


@pytest.mark.asyncio
async def test_on_sdk_message_rejects_empty_human_allowlist(adapter_harness):
    from lark_oapi.channel import Identity

    adapter = adapter_harness.adapter
    adapter._allow_bots = "none"
    adapter._allow_all_users = False
    adapter._group_policy = "allowlist"
    adapter._default_group_policy = "allowlist"
    adapter._allowed_group_users = set()
    adapter._require_mention = False

    await adapter._on_sdk_message(
        _message(
            sender=Identity(open_id="ou_denied", user_id="u_denied"),
            message_id="om_empty_allowlist_human",
        )
    )
    await _drain_adapter_tasks()

    assert adapter_harness.captured_inbound == []


class TestDispatchDoesNotBlockSdkCallback:
    """The SDK callback must return promptly after scheduling handle_message
    on the adapter loop. Blocking on the handler (e.g. via await wrap_future)
    serializes inbound dispatch and defeats SDK per-chat parallelism.
    """

    @pytest.mark.asyncio
    async def test_dispatch_returns_before_handle_message_completes(self):
        from gateway.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter.__new__(FeishuAdapter)
        adapter._loop = asyncio.get_running_loop()
        adapter._channel = None
        # _submit_on_loop pulls these via self.
        adapter._log_background_failure = lambda fut: None

        handler_started = asyncio.Event()
        handler_release = asyncio.Event()
        handler_completed = asyncio.Event()

        async def fake_handle_message(event):
            handler_started.set()
            await handler_release.wait()
            handler_completed.set()

        adapter.handle_message = fake_handle_message

        # Build a minimal event using the same pattern as existing tests in
        # test_adapter_contract.py -- pass a SimpleNamespace so we don't have
        # to satisfy the full MessageEvent constructor.
        event = SimpleNamespace(platform="feishu", text="x")

        # Dispatch must return promptly even though handler is blocked.
        await asyncio.wait_for(
            adapter._dispatch_handle_message_on_adapter_loop(event, label="message"),
            timeout=0.5,
        )
        # Handler is scheduled but not yet complete.
        await asyncio.wait_for(handler_started.wait(), timeout=0.5)
        assert not handler_completed.is_set(), (
            "dispatch returned but handler also already completed -- "
            "did it really run async?"
        )

        # Let the handler finish; cleanup.
        handler_release.set()
        await asyncio.wait_for(handler_completed.wait(), timeout=0.5)
