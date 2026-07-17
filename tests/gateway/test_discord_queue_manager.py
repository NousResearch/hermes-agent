"""Behavior tests for Discord's private owner-scoped queue manager."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import utf16_len
from plugins.platforms.discord.adapter import DiscordAdapter, QueueManagerView


class StubQueueRunner:
    """Gateway-handler fake plus direct APIs that must never be used by the UI."""

    def __init__(self, items=None):
        self.items = [dict(item) for item in (items or [])]
        self.handler_calls: list = []
        self.refusal: str | None = None
        self.session_key = "session:456"
        self.snapshot_id = ""
        self._snapshot_counter = 0
        self._session_key_for_source = MagicMock(return_value="session:456")
        self.list_explicit_queue_items = MagicMock(side_effect=self._direct_list)
        self.remove_explicit_queue_item = MagicMock(side_effect=self._direct_remove)
        self.clear_explicit_queue_items = MagicMock(side_effect=self._direct_clear)

    def _profile_name_for_source(self, *_args, **_kwargs):
        return None

    def _direct_list(self, _session_key, _owner_user_id, *, adapter=None):
        del adapter
        return [dict(item) for item in self.items]

    def _direct_remove(self, _session_key, _owner_user_id, queue_id, *, adapter=None):
        del adapter
        for index, item in enumerate(self.items):
            if (item.get("queue_id") or item.get("id")) == queue_id:
                self.items.pop(index)
                return True
        return False

    def _direct_clear(self, _session_key, _owner_user_id, *, adapter=None):
        del adapter
        removed = len(self.items)
        self.items.clear()
        return removed

    def _fresh_snapshot(self, action: str, **extra):
        self._snapshot_counter += 1
        self.snapshot_id = f"snapshot-{self._snapshot_counter}"
        return {
            "type": "queue_management",
            "action": action,
            "ok": True,
            "session_key": self.session_key,
            "snapshot_id": self.snapshot_id,
            "items": [
                {
                    "id": item.get("queue_id") or item.get("id"),
                    "position": index,
                    "created_at": item.get("created_at"),
                    "origin": "explicit",
                    "preview": item.get("preview", ""),
                    "has_media": bool(item.get("has_media")),
                }
                for index, item in enumerate(self.items, start=1)
            ],
            **extra,
        }

    async def handle(self, event):
        """Model GatewayRunner's snapshot-bound native-manager contract."""
        self.handler_calls.append(event)
        if self.refusal is not None:
            return self.refusal

        request = event.metadata["_hermes_native_discord_queue_management"]
        action = request["action"]
        session_key = request.get("session_key")
        if session_key is not None and session_key != self.session_key:
            return {
                "type": "queue_management",
                "action": action,
                "ok": False,
                "error": "session_changed",
            }
        snapshot_id = request.get("snapshot_id")
        if session_key is not None and snapshot_id != self.snapshot_id:
            return {
                "type": "queue_management",
                "action": action,
                "ok": False,
                "error": "snapshot_stale",
            }
        if action == "list":
            return self._fresh_snapshot("list")
        if action == "remove":
            queue_id = request.get("queue_id")
            for index, item in enumerate(self.items):
                if (item.get("queue_id") or item.get("id")) == queue_id:
                    self.items.pop(index)
                    return self._fresh_snapshot(
                        "remove", removed=True, queue_id=queue_id
                    )
            return self._fresh_snapshot(
                "remove",
                removed=False,
                queue_id=queue_id,
                error="not_found",
            )
        if action == "clear":
            selected_ids = {str(queue_id) for queue_id in request.get("queue_ids") or []}
            before = len(self.items)
            self.items = [
                item
                for item in self.items
                if (item.get("queue_id") or item.get("id")) not in selected_ids
            ]
            return self._fresh_snapshot("clear", removed_count=before - len(self.items))
        raise AssertionError(f"unexpected queue-manager action: {action}")

    def assert_no_direct_calls(self):
        self._session_key_for_source.assert_not_called()
        self.list_explicit_queue_items.assert_not_called()
        self.remove_explicit_queue_item.assert_not_called()
        self.clear_explicit_queue_items.assert_not_called()


def _queue_item(index: int, preview: str | None = None, *, has_media=False):
    return {
        "queue_id": f"opaque-{index}",
        "preview": preview if preview is not None else f"queued prompt {index}",
        "has_media": has_media,
        "created_at": f"2026-07-17T00:{index:02d}:00+00:00",
    }


@pytest.fixture
def adapter():
    result = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    result._check_slash_authorization = AsyncMock(return_value=True)
    _install_runner(result, StubQueueRunner())
    result.handle_message = AsyncMock()
    result.send = AsyncMock()
    return result


def _install_runner(adapter, runner):
    adapter.gateway_runner = runner
    adapter._message_handler = AsyncMock(side_effect=runner.handle)
    return runner


def _interaction(user_id=123, *, data=None, defer_side_effect=None):
    channel = SimpleNamespace(
        id=456,
        name="general",
        guild=SimpleNamespace(id=789, name="TestGuild"),
        topic=None,
    )
    return SimpleNamespace(
        user=SimpleNamespace(
            id=user_id,
            name=f"user-{user_id}",
            display_name=f"User {user_id}",
            roles=[],
        ),
        channel=channel,
        channel_id=channel.id,
        guild=channel.guild,
        guild_id=channel.guild.id,
        data=data or {},
        response=SimpleNamespace(
            defer=AsyncMock(side_effect=defer_side_effect),
            edit_message=AsyncMock(),
            send_message=AsyncMock(),
        ),
        edit_original_response=AsyncMock(),
    )


def _component(view, custom_id):
    return next(child for child in view.children if child.custom_id == custom_id)


def _edited_view(interaction):
    return interaction.edit_original_response.await_args.kwargs["view"]


def _assert_management_event(
    event,
    interaction,
    typed_command,
    action,
    queue_id=None,
    queue_ids=None,
    session_key=None,
    snapshot_id=None,
):
    assert event.text == typed_command
    assert event.raw_message is interaction
    assert event.source.user_id == str(interaction.user.id)
    assert event.source.chat_id == str(interaction.channel_id)
    marker = {"action": action}
    if queue_id is not None:
        marker["queue_id"] = queue_id
    if queue_ids is not None:
        marker["queue_ids"] = queue_ids
    actual_marker = dict(event.metadata["_hermes_native_discord_queue_management"])
    if session_key is None:
        actual_marker.pop("session_key", None)
    else:
        marker["session_key"] = session_key
    if snapshot_id is None:
        actual_marker.pop("snapshot_id", None)
    else:
        marker["snapshot_id"] = snapshot_id
    assert actual_marker == marker


def _assert_no_public_delivery(adapter, runner):
    runner.assert_no_direct_calls()
    adapter.handle_message.assert_not_awaited()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_open_manager_authorizes_then_defers_before_gateway_dispatch(adapter):
    events: list[str] = []
    runner = _install_runner(adapter, StubQueueRunner([_queue_item(1)]))

    async def authorize(_interaction, command):
        assert command == "/queue"
        events.append("auth")
        return True

    interaction = _interaction()

    async def defer(**kwargs):
        assert kwargs == {"ephemeral": True}
        events.append("defer")

    adapter._check_slash_authorization = AsyncMock(side_effect=authorize)
    interaction.response.defer = AsyncMock(side_effect=defer)

    await adapter._open_queue_manager_slash(interaction, "/queue")

    assert events == ["auth", "defer"]
    adapter._message_handler.assert_awaited_once()
    assert runner.handler_calls == [adapter._message_handler.await_args.args[0]]
    _assert_management_event(
        runner.handler_calls[0], interaction, "/queue", "list"
    )
    interaction.edit_original_response.assert_awaited_once()
    kwargs = interaction.edit_original_response.await_args.kwargs
    assert kwargs["view"].owner_user_id == "123"
    assert "queued prompt 1" in kwargs["content"]
    assert "allowed_mentions" in kwargs
    _assert_no_public_delivery(adapter, runner)


@pytest.mark.asyncio
async def test_open_manager_rejection_stops_before_defer_and_gateway_dispatch(adapter):
    runner = adapter.gateway_runner
    adapter._check_slash_authorization = AsyncMock(return_value=False)
    interaction = _interaction()

    await adapter._open_queue_manager_slash(interaction, "/q")

    adapter._check_slash_authorization.assert_awaited_once_with(interaction, "/q")
    interaction.response.defer.assert_not_awaited()
    interaction.edit_original_response.assert_not_awaited()
    adapter._message_handler.assert_not_awaited()
    _assert_no_public_delivery(adapter, runner)


@pytest.mark.asyncio
async def test_unknown_interaction_on_defer_performs_zero_gateway_handler_calls(adapter):
    class UnknownInteraction(Exception):
        status = 404
        code = 10062

    runner = adapter.gateway_runner
    interaction = _interaction(
        defer_side_effect=UnknownInteraction("Unknown interaction")
    )

    await adapter._open_queue_manager_slash(interaction, "/queue")

    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    interaction.edit_original_response.assert_not_awaited()
    adapter._message_handler.assert_not_awaited()
    _assert_no_public_delivery(adapter, runner)


@pytest.mark.asyncio
async def test_open_manager_slash_access_denial_stays_private_without_view(adapter):
    denial = "⛔ /queue is admin-only here."
    runner = _install_runner(adapter, StubQueueRunner([_queue_item(1)]))
    runner.refusal = denial
    interaction = _interaction()

    await adapter._open_queue_manager_slash(interaction, "/q")

    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    adapter._message_handler.assert_awaited_once()
    _assert_management_event(runner.handler_calls[0], interaction, "/q", "list")
    interaction.edit_original_response.assert_awaited_once_with(
        content=denial,
        view=None,
        allowed_mentions=interaction.edit_original_response.await_args.kwargs[
            "allowed_mentions"
        ],
    )
    _assert_no_public_delivery(adapter, runner)


@pytest.mark.asyncio
async def test_queue_manager_delete_clear_and_refresh_dispatch_through_gateway(adapter):
    runner = _install_runner(adapter, StubQueueRunner([_queue_item(1), _queue_item(2)]))
    opening = _interaction()
    await adapter._open_queue_manager_slash(opening, "/q")
    view = _edited_view(opening)

    selecting = _interaction(data={"values": ["opaque-2"]})
    await _component(view, "queue_manager_select").callback(selecting)
    assert view.selected_queue_id == "opaque-2"

    deleting = _interaction()
    await _component(view, "queue_manager_delete").callback(deleting)
    assert [item["queue_id"] for item in view._items] == ["opaque-1"]
    assert view.selected_queue_id is None

    starting_clear = _interaction()
    await _component(view, "queue_manager_clear").callback(starting_clear)
    assert view.confirming_clear is True

    confirming = _interaction()
    await _component(view, "queue_manager_clear_confirm").callback(confirming)
    assert runner.items == []
    assert view.confirming_clear is False

    refreshing = _interaction()
    await _component(view, "queue_manager_refresh").callback(refreshing)

    expected = [
        (opening, "list", None, None, None, None),
        (selecting, "list", None, None, "session:456", "snapshot-1"),
        (deleting, "remove", "opaque-2", None, "session:456", "snapshot-2"),
        (starting_clear, "list", None, None, "session:456", "snapshot-3"),
        (confirming, "clear", None, ["opaque-1"], "session:456", "snapshot-4"),
        (refreshing, "list", None, None, "session:456", "snapshot-5"),
    ]
    assert len(runner.handler_calls) == len(expected)
    for event, (interaction, action, queue_id, queue_ids, session_key, snapshot_id) in zip(
        runner.handler_calls, expected
    ):
        _assert_management_event(
            event,
            interaction,
            "/q",
            action,
            queue_id,
            queue_ids,
            session_key=session_key,
            snapshot_id=snapshot_id,
        )
    assert adapter._message_handler.await_count == len(expected)
    _assert_no_public_delivery(adapter, runner)


@pytest.mark.asyncio
async def test_component_gateway_denial_is_neutral_and_has_zero_ui_mutation(adapter):
    runner = _install_runner(adapter, StubQueueRunner([_queue_item(1)]))
    opening = _interaction()
    await adapter._open_queue_manager_slash(opening, "/queue")
    view = _edited_view(opening)
    prior_items = list(view._items)

    runner.refusal = "⛔ /queue is admin-only here."
    denied = _interaction(data={"values": ["opaque-1"]})
    await _component(view, "queue_manager_select").callback(denied)

    assert view.selected_queue_id is None
    assert view._items == prior_items
    denied.response.edit_message.assert_not_awaited()
    denied.response.send_message.assert_awaited_once()
    args, kwargs = denied.response.send_message.await_args
    assert "not permitted" in args[0].lower()
    assert kwargs["ephemeral"] is True
    assert "view" not in kwargs
    assert "allowed_mentions" in kwargs
    _assert_management_event(runner.handler_calls[-1], denied, "/queue", "list")
    _assert_no_public_delivery(adapter, runner)


@pytest.mark.asyncio
async def test_queue_manager_requires_owner_and_live_adapter_authorization(adapter):
    runner = _install_runner(adapter, StubQueueRunner([_queue_item(1)]))
    opening = _interaction()
    await adapter._open_queue_manager_slash(opening, "/queue")
    view = _edited_view(opening)

    foreign = _interaction(user_id=999, data={"values": ["opaque-1"]})
    await _component(view, "queue_manager_select").callback(foreign)
    foreign.response.send_message.assert_awaited_once()
    assert foreign.response.send_message.await_args.kwargs["ephemeral"] is True
    assert view.selected_queue_id is None
    assert len(runner.handler_calls) == 1

    adapter._check_slash_authorization = AsyncMock(return_value=False)
    owner = _interaction(data={"values": ["opaque-1"]})
    await _component(view, "queue_manager_select").callback(owner)
    adapter._check_slash_authorization.assert_awaited_once_with(owner, "/queue")
    owner.response.edit_message.assert_not_awaited()
    assert view.selected_queue_id is None
    assert len(runner.handler_calls) == 1
    _assert_no_public_delivery(adapter, runner)


@pytest.mark.asyncio
async def test_queue_manager_rechecks_live_authorization_on_every_component(adapter):
    cases = [
        ("queue_manager_select", {"data": {"values": ["opaque-1"]}}),
        ("queue_manager_previous", {"page": 1}),
        ("queue_manager_next", {}),
        ("queue_manager_delete", {"selected_queue_id": "opaque-1"}),
        ("queue_manager_clear", {}),
        ("queue_manager_clear_confirm", {"confirming_clear": True}),
        ("queue_manager_clear_cancel", {"confirming_clear": True}),
        ("queue_manager_refresh", {}),
    ]

    for custom_id, setup in cases:
        runner = _install_runner(
            adapter, StubQueueRunner([_queue_item(index) for index in range(12)])
        )
        view = QueueManagerView(
            adapter=adapter,
            owner_user_id="123",
            typed_command="/queue",
            items=runner.items,
        )
        view.page = setup.get("page", view.page)
        view.selected_queue_id = setup.get(
            "selected_queue_id", view.selected_queue_id
        )
        view.confirming_clear = setup.get(
            "confirming_clear", view.confirming_clear
        )
        view._build_components()
        interaction = _interaction(data=setup.get("data", {}))
        await _component(view, custom_id).callback(interaction)
        assert runner.handler_calls
        _assert_no_public_delivery(adapter, runner)

    commands = [
        call.args[1] for call in adapter._check_slash_authorization.await_args_list
    ]
    assert commands == ["/queue"] * len(cases)


@pytest.mark.asyncio
async def test_queue_manager_consumed_delete_refreshes_view_without_removing_another_item(adapter):
    runner = _install_runner(adapter, StubQueueRunner([_queue_item(1)]))
    opening = _interaction()
    await adapter._open_queue_manager_slash(opening, "/queue")
    view = _edited_view(opening)
    view.selected_queue_id = "opaque-1"
    view._build_components()
    runner.items.clear()

    stale = _interaction()
    await _component(view, "queue_manager_delete").callback(stale)

    assert view.selected_queue_id is None
    assert view._items == []
    stale.response.edit_message.assert_awaited_once()
    assert "no other queued turn was removed" in (
        stale.response.edit_message.await_args.kwargs["content"].lower()
    )
    stale.response.send_message.assert_not_awaited()
    _assert_management_event(
        runner.handler_calls[-1],
        stale,
        "/queue",
        "remove",
        "opaque-1",
        session_key="session:456",
        snapshot_id="snapshot-1",
    )
    _assert_no_public_delivery(adapter, runner)


@pytest.mark.asyncio
async def test_queue_manager_consumed_clear_refreshes_view_without_removing_new_arrivals(adapter):
    runner = _install_runner(adapter, StubQueueRunner([_queue_item(1)]))
    opening = _interaction()
    await adapter._open_queue_manager_slash(opening, "/queue")
    view = _edited_view(opening)

    await _component(view, "queue_manager_clear").callback(_interaction())
    runner.items.clear()
    runner.items.append(_queue_item(2))
    confirm = _interaction()
    await _component(view, "queue_manager_clear_confirm").callback(confirm)

    assert view.confirming_clear is False
    assert [item["queue_id"] for item in view._items] == ["opaque-2"]
    confirm.response.edit_message.assert_awaited_once()
    assert "no other queued turn was removed" in (
        confirm.response.edit_message.await_args.kwargs["content"].lower()
    )
    confirm.response.send_message.assert_not_awaited()
    _assert_no_public_delivery(adapter, runner)


@pytest.mark.asyncio
async def test_queue_manager_clear_requires_confirmation_and_cancel_has_no_queue_side_effect(adapter):
    runner = _install_runner(adapter, StubQueueRunner([_queue_item(1), _queue_item(2)]))
    opening = _interaction()
    await adapter._open_queue_manager_slash(opening, "/queue")
    view = _edited_view(opening)

    start = _interaction()
    await _component(view, "queue_manager_clear").callback(start)
    assert view.confirming_clear is True
    assert len(runner.items) == 2

    cancel = _interaction()
    await _component(view, "queue_manager_clear_cancel").callback(cancel)
    assert view.confirming_clear is False
    assert len(runner.items) == 2
    assert all(
        event.metadata["_hermes_native_discord_queue_management"]["action"] != "clear"
        for event in runner.handler_calls
    )
    _assert_no_public_delivery(adapter, runner)


@pytest.mark.asyncio
async def test_queue_manager_clear_confirm_applies_returned_snapshot(adapter):
    runner = _install_runner(adapter, StubQueueRunner([_queue_item(1), _queue_item(2)]))
    opening = _interaction()
    await adapter._open_queue_manager_slash(opening, "/queue")
    view = _edited_view(opening)

    await _component(view, "queue_manager_clear").callback(_interaction())
    confirm = _interaction()
    await _component(view, "queue_manager_clear_confirm").callback(confirm)

    assert runner.items == []
    assert view.confirming_clear is False
    assert "empty" in view.render_content().lower()
    _assert_management_event(
        runner.handler_calls[-1],
        confirm,
        "/queue",
        "clear",
        queue_ids=["opaque-1", "opaque-2"],
        session_key="session:456",
        snapshot_id="snapshot-2",
    )
    assert len(runner.handler_calls) == 3
    _assert_no_public_delivery(adapter, runner)


@pytest.mark.asyncio
async def test_queue_manager_clear_preserves_items_added_after_confirmation(adapter):
    runner = _install_runner(adapter, StubQueueRunner([_queue_item(1), _queue_item(2)]))
    opening = _interaction()
    await adapter._open_queue_manager_slash(opening, "/queue")
    view = _edited_view(opening)

    await _component(view, "queue_manager_clear").callback(_interaction())
    runner.items.append(_queue_item(3))

    confirm = _interaction()
    await _component(view, "queue_manager_clear_confirm").callback(confirm)

    assert [item["queue_id"] for item in runner.items] == ["opaque-3"]
    assert [item["queue_id"] for item in view._items] == ["opaque-3"]
    _assert_management_event(
        runner.handler_calls[-1],
        confirm,
        "/queue",
        "clear",
        queue_ids=["opaque-1", "opaque-2"],
        session_key="session:456",
        snapshot_id="snapshot-2",
    )
    assert len(runner.handler_calls) == 3
    _assert_no_public_delivery(adapter, runner)


@pytest.mark.asyncio
async def test_queue_manager_session_change_blocks_delete_without_ui_mutation(adapter):
    runner = _install_runner(adapter, StubQueueRunner([_queue_item(1)]))
    opening = _interaction()
    await adapter._open_queue_manager_slash(opening, "/queue")
    view = _edited_view(opening)
    view.selected_queue_id = "opaque-1"
    view._build_components()
    prior_items = list(view._items)

    runner.session_key = "session:changed"
    stale = _interaction()
    await _component(view, "queue_manager_delete").callback(stale)

    assert [item["queue_id"] for item in runner.items] == ["opaque-1"]
    assert view._items == prior_items
    assert view.selected_queue_id == "opaque-1"
    stale.response.edit_message.assert_not_awaited()
    stale.response.send_message.assert_awaited_once()
    _assert_management_event(
        runner.handler_calls[-1],
        stale,
        "/queue",
        "remove",
        "opaque-1",
        session_key="session:456",
    )
    _assert_no_public_delivery(adapter, runner)



def test_queue_manager_paginates_at_ten_items():
    adapter = SimpleNamespace(_check_slash_authorization=AsyncMock(return_value=True))
    view = QueueManagerView(
        adapter=adapter,
        owner_user_id="123",
        typed_command="/queue",
        items=[_queue_item(index) for index in range(23)],
    )

    assert len(_component(view, "queue_manager_select").options) == 10
    assert _component(view, "queue_manager_previous").disabled is True
    assert _component(view, "queue_manager_next").disabled is False


@pytest.mark.asyncio
async def test_queue_manager_next_previous_and_refresh_edit_ephemeral_message(adapter):
    runner = _install_runner(adapter, StubQueueRunner([_queue_item(index) for index in range(23)]))
    view = QueueManagerView(
        adapter=adapter,
        owner_user_id="123",
        typed_command="/queue",
        items=runner.items,
    )

    next_interaction = _interaction()
    await _component(view, "queue_manager_next").callback(next_interaction)
    assert view.page == 1
    assert len(_component(view, "queue_manager_select").options) == 10
    assert next_interaction.response.edit_message.await_args.kwargs["view"] is view

    previous_interaction = _interaction()
    await _component(view, "queue_manager_previous").callback(previous_interaction)
    assert view.page == 0

    refresh_interaction = _interaction()
    await _component(view, "queue_manager_refresh").callback(refresh_interaction)
    assert refresh_interaction.response.edit_message.await_args.kwargs["view"] is view
    for event, interaction in zip(
        runner.handler_calls,
        (next_interaction, previous_interaction, refresh_interaction),
    ):
        _assert_management_event(event, interaction, "/queue", "list")
    _assert_no_public_delivery(adapter, runner)


def test_queue_manager_neutralizes_mentions_and_limits_utf16_fields():
    raw_preview = "  hello\n\t@everyone   <@123>  " + ("😀" * 80)
    view = QueueManagerView(
        adapter=SimpleNamespace(_check_slash_authorization=AsyncMock(return_value=True)),
        owner_user_id="123",
        typed_command="/queue",
        items=[
            {
                **_queue_item(1, raw_preview, has_media=True),
                "media_path": "/tmp/secret-photo.png",
                "reply_text": "private reply",
                "owner_user_id": "owner-secret",
            }
        ],
    )

    content = view.render_content()
    option = _component(view, "queue_manager_select").options[0]
    assert "hello @\u200beveryone <@\u200b123>" in content
    assert "@everyone" not in content
    assert "<@123>" not in content
    assert utf16_len(option.label) <= 100
    assert "/tmp/secret-photo.png" not in content
    assert "private reply" not in content
    assert "owner-secret" not in content
    assert "📎" in content


@pytest.mark.asyncio
async def test_queue_manager_timeout_disables_components_and_edits_original(adapter):
    original = _interaction()
    view = QueueManagerView(
        adapter=adapter,
        owner_user_id="123",
        typed_command="/queue",
        items=[_queue_item(1)],
        original_interaction=original,
    )

    await view.on_timeout()

    assert all(child.disabled for child in view.children)
    assert original.edit_original_response.await_args.kwargs["view"] is view
    assert "allowed_mentions" in original.edit_original_response.await_args.kwargs
    assert "expired" in view.render_content().lower()
