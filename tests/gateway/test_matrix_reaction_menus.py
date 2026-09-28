"""Reaction menus keep a requester's choice in its original conversation."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.session import SessionSource
from hermes_cli.tools_config import _get_platform_tools
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.mark.parametrize("platform,configured,expected", [
    ("matrix", False, False), ("matrix", True, True),
    ("cli", True, False), ("telegram", True, False), ("api_server", True, False),
])
def test_menu_toolset_requires_matrix_opt_in(platform, configured, expected):
    from toolsets import resolve_toolset

    config = {"platform_toolsets": {platform: ["reaction_menu"]}} if configured else {}
    tools = {tool for toolset in _get_platform_tools(config, platform) for tool in resolve_toolset(toolset)}
    assert ("present_menu" in tools) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("rejection", ["actor", "unauthorized", "revoked", "room", "target", "key", "removed", "expired", "session", "approval", "picker"])
async def test_menu_choice_is_scoped_and_consumed_once(monkeypatch, rejection):
    ReactionEvent = pytest.importorskip("mautrix.types").ReactionEvent
    from gateway.run_turn_runner_menu import MenuDelivery
    from tools.reaction_menu_model import ReactionMenu

    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="token", extra={"homeserver": "https://matrix.test"}))
    adapter._user_id = "@bot:matrix.test"
    adapter._allowed_user_ids = {"@alice:matrix.test", "@bob:matrix.test"}
    adapter._approval_require_sender = False
    adapter._client = SimpleNamespace()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="$menu"))
    adapter._send_reaction = AsyncMock(return_value="$seed")
    adapter._send_invalid_reaction_feedback = AsyncMock()
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("plugins.platforms.matrix.adapter.time", SimpleNamespace(monotonic=lambda: clock.now))

    source = SessionSource(platform=Platform.MATRIX, chat_id="!room:matrix.test", chat_type="group",
                           user_id="@alice:matrix.test", thread_id="$thread", profile="secondary")
    entry = SimpleNamespace(session_id="conversation")
    runner = SimpleNamespace(session_store=SimpleNamespace(lookup_by_session_key=lambda key: entry),
                             _is_user_authorized_for_source=lambda source: rejection != "revoked",
                             _standalone_launch_scope=nullcontext)
    accepted = []
    started, release = asyncio.Event(), asyncio.Event()

    async def admit(event):
        started.set()
        await release.wait()
        accepted.append((event.text, event.source.to_dict(), event.metadata, event.allow_gateway_control))
        event._gateway_accepted = True

    monkeypatch.setattr(adapter, "handle_message", admit)
    menu = ReactionMenu.from_arguments("Choose a route", [
        {"emoji": "✅", "label": "First route", "payload": "/new is option text"},
        {"emoji": "❌", "label": "Second route", "payload": "Take the second route"},
    ], "route")
    delivery = MenuDelivery(runner, adapter, source, "lane", "conversation", None)
    await adapter.send_reaction_menu(menu, "lane", delivery.selected, {"chat_id": source.chat_id, "thread_id": "$thread", "requester_user_id": source.user_id})

    def reaction(event_id, **changes):
        raw = {"type": "m.reaction", "event_id": event_id, "room_id": source.chat_id,
               "sender": source.user_id, "origin_server_ts": 1,
               "content": {"m.relates_to": {"rel_type": "m.annotation", "event_id": "$menu", "key": "✅"}}}
        raw.update(changes)
        return ReactionEvent.deserialize(raw)

    bad = reaction("$bad")
    if rejection == "actor":
        bad.sender = "@bob:matrix.test"
    if rejection == "unauthorized":
        bad.sender = "@mallory:matrix.test"
    if rejection == "room":
        bad.room_id = "!other:matrix.test"
    if rejection == "target":
        bad.content.relates_to.event_id = "$other"
    if rejection == "key":
        bad.content.relates_to.key = "🟢"
    if rejection == "removed":
        bad = reaction("$bad", content={})
    if rejection == "expired":
        clock.now = 1000.0
    if rejection == "session":
        entry.session_id = "reset"
    if rejection == "approval":
        from plugins.platforms.matrix.approval_lifecycle import _MatrixApprovalPrompt
        prompt = _MatrixApprovalPrompt("lane", source.chat_id, "$menu", requester_user_id=source.user_id)
        adapter._approval_prompts_by_event["$menu"] = prompt
        monkeypatch.setattr("tools.approval.resolve_gateway_approval", lambda key, choice: 1)
        adapter._redact_bot_approval_reactions = AsyncMock()
        adapter.edit_message = AsyncMock(return_value=SendResult(success=True, message_id="$card-edit"))
    if rejection == "picker":
        from plugins.platforms.matrix.adapter import _MatrixPickerPrompt
        callback = AsyncMock()
        adapter._model_picker_prompts_by_event["$menu"] = _MatrixPickerPrompt(
            source.chat_id, "$menu", "lane", {"✅": "model"}, callback, requester_user_id=source.user_id)
    await adapter._on_reaction(bad)
    assert accepted == []
    if rejection == "approval":
        assert adapter._approval_prompts_by_event == {}
    if rejection == "picker":
        callback.assert_awaited_once_with(source.chat_id, "model")
    if rejection in {"expired", "session", "revoked"}:
        return

    good = reaction("$good")
    task = asyncio.create_task(adapter._on_reaction(good))
    await asyncio.wait_for(started.wait(), 2)
    await adapter._on_reaction(good)
    await adapter._on_reaction(reaction("$another"))
    release.set()
    await asyncio.wait_for(task, 2)
    assert accepted == [(
        '[menu-choice]\n{"prompt": "Choose a route", "context_id": "route", "emoji": "✅", "label": "First route", "payload": "/new is option text"}',
        source.to_dict(),
        {"gateway_session_key": "lane", "gateway_session_id": "conversation", "gateway_session_strict": True},
        False,
    )]
    assert adapter._choice_picker_prompts_by_event == {}


@pytest.mark.asyncio
async def test_menu_callback_reenters_profile_scope_and_bounds_pending_controls(tmp_path, monkeypatch, request):
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
    from gateway.run_turn_runner_menu import menu_callback
    from hermes_constants import get_hermes_home
    from plugins.platforms.matrix.adapter import _MatrixPickerPrompt
    from plugins.platforms.matrix.reaction_menu import MAX_PENDING_MENUS
    from tools.reaction_menu_model import ReactionMenu

    homes = {name: tmp_path / name for name in ("a", "b")}
    for home in homes.values():
        home.mkdir()
        (home / "config.yaml").write_text("terminal:\n  backend: local\n", encoding="utf-8")
    seen = []
    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.test"}))
    adapter._client = SimpleNamespace()
    adapter._user_id = "@bot:matrix.test"
    adapter._allowed_user_ids = set()
    adapter._send_reaction = AsyncMock(return_value="$seed")
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("plugins.platforms.matrix.adapter.time", SimpleNamespace(monotonic=lambda: clock.now))

    async def send(room, text, metadata: dict):
        seen.append(("send", get_hermes_home(), room, metadata["thread_id"]))
        return SendResult(success=True, message_id=f"$menu-{len(seen)}")

    async def admit(event):
        seen.append(("choice", get_hermes_home(), event.source.chat_id, event.source.thread_id))
        event._gateway_accepted = True

    monkeypatch.setattr(adapter, "send", send)
    monkeypatch.setattr(adapter, "handle_message", admit)
    args = {"prompt": "Choose", "options": [{"emoji": "✅", "label": "Route", "payload": "Go"}]}
    runner = SimpleNamespace(
        _profile_scope_key_for_source=lambda source: homes[source.profile],
        session_store=SimpleNamespace(lookup_by_session_key=lambda key: SimpleNamespace(session_id=key)),
        _is_user_authorized_for_source=lambda source: get_hermes_home() == homes[source.profile],
    )
    was_multiplexed = is_multiplex_active()
    set_multiplex_active(True)
    request.addfinalizer(lambda: set_multiplex_active(was_multiplexed))
    for profile in ("a", "b", "a"):
        source = SessionSource(platform=Platform.MATRIX, profile=profile, user_id="@alice:matrix.test",
                               chat_id=f"!{profile}:matrix.test", thread_id=f"$thread-{profile}")
        ctx = SimpleNamespace(source=source, enabled_toolsets=["reaction_menu"], _status_adapter=adapter,
                              session_key=profile, session_id=profile, _status_thread_metadata={"thread_id": source.thread_id},
                              _run_still_current=lambda: True, _loop_for_step=asyncio.get_running_loop())
        agent = SimpleNamespace(present_menu_callback=menu_callback(SimpleNamespace(_ctx=ctx, _runner=runner)))
        result = await asyncio.to_thread(INLINE_TOOL_EXECUTORS["present_menu"], agent, args, InlineToolContext(profile))
        assert '"status": "menu_presented"' in result
        message_id = next(iter(adapter._choice_picker_prompts_by_event))
        await adapter._handle_choice_picker_reaction(source.chat_id, message_id, "✅", "@alice:matrix.test")
    assert seen == [(kind, homes[profile], f"!{profile}:matrix.test", f"$thread-{profile}")
                    for profile in ("a", "b", "a") for kind in ("send", "choice")]

    registry = adapter._choice_picker_prompts_by_event
    for index in range(MAX_PENDING_MENUS):
        registry[f"$pending-{index}"] = _MatrixPickerPrompt(
            chat_id="!room", message_id=f"$pending-{index}", session_key=str(index), choices={},
            on_selected=AsyncMock(), is_menu=True, expires_at=101,
        )
    menu = ReactionMenu.from_arguments(**args)
    metadata = {"chat_id": "!room", "requester_user_id": "@alice:matrix.test", "thread_id": "$thread"}
    result = await adapter.send_reaction_menu(menu, "new-lane", AsyncMock(), metadata)
    assert result == SendResult(success=False, error="Too many pending Matrix menus")
    clock.now = 102
    await adapter.send_reaction_menu(menu, "new-lane", AsyncMock(), metadata)
    await adapter.send_reaction_menu(menu, "new-lane", AsyncMock(), metadata)
    assert [(prompt.session_key, prompt.is_menu) for prompt in registry.values()] == [("new-lane", True)]
