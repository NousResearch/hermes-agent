"""Matrix context stays refreshable through gateway prompt enrichment."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from gateway.session import SessionEntry, SessionSource, SessionStore
from plugins.platforms.matrix.reply_context import MatrixEventContext
from plugins.platforms.matrix.room_context import MatrixRoomState
from tests.gateway.test_matrix import _make_adapter
from tests.gateway.test_matrix_effective_event_state import (
    ROOM,
    SENDER,
    _edited,
    _original,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["room", "thread", "thread-fallback", "reply-only"])
@pytest.mark.parametrize(
    "boundary", ["references", "room-state", "images", "model-prepare", "proxy"]
)
@pytest.mark.parametrize(
    "change", ["edit", "redaction", "failed-recovery", "older-replacement"]
)
async def test_gateway_preparation_rechecks_context_after_enrichment(
    scope: str, boundary: str, change: str, monkeypatch, tmp_path
):
    started, release = asyncio.Event(), asyncio.Event()
    adapter = _make_adapter()
    adapter._room_backfill_limit = adapter._thread_backfill_limit = 1
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    cache = adapter._event_context_cache
    target = "$root" if scope.startswith("thread") else "$target"
    raw = _edited(_original(target, "initial draft"), "before enrichment")
    raw["unsigned"]["m.relations"]["m.replace"]["event_id"] = "$latest"
    cache.store(
        ROOM,
        target,
        MatrixEventContext(SENDER, "before enrichment", replacement_id="$latest"),
    )

    async def request(_method, path, **_kwargs):
        if "/event/" in path:
            if change == "failed-recovery":
                raise RuntimeError("recovery unavailable")
            return raw
        if "/context/" in path:
            return {"start": "boundary"}
        return {"chunk": [raw] if "/messages" in path else []}

    adapter._client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(side_effect=request))
    )
    source = SessionSource(
        Platform.MATRIX,
        ROOM,
        chat_type="group",
        user_id=SENDER,
        user_name="Alice",
        thread_id=target if scope.startswith("thread") else None,
    )
    content = {"msgtype": "m.text", "body": "question @file:notes"}
    relation = {"m.in_reply_to": {"event_id": target}}
    if scope in {"room", "thread"}:
        content["m.mentions"] = {"user_ids": [adapter._user_id]}
    if scope == "thread":
        relation.update(rel_type="m.thread", event_id=target, is_falling_back=False)
    content["m.relates_to"] = relation
    event = await adapter._build_inbound_event(
        ROOM,
        SENDER,
        "$current",
        content["body"],
        content,
        relation,
        ctx=(content["body"], False, "group", source.thread_id, "Alice", source),
    )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {Platform.MATRIX: adapter}
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    runner.session_store._entries["session"] = SessionEntry(
        "session", "id", now, now, origin=source
    )
    history = (
        []
        if scope == "thread-fallback"
        else [{"role": "user", "content": "previous turn before enrichment"}]
    )
    previous = [dict(message) for message in history]

    async def expand(_source: SessionSource, _key: str, text: str) -> str:
        if boundary == "references":
            started.set()
            await release.wait()
        return text.replace("@file:notes", "expanded user reference")

    async def room_state(_adapter, _room):
        if boundary == "room-state":
            started.set()
            await release.wait()
        return MatrixRoomState(None, None, None)

    async def enrich(
        _source: SessionSource, _key: str, text: str, _paths: list[str]
    ) -> str:
        started.set()
        await release.wait()
        return f"Image enrichment completed\n\n{text}"

    if boundary == "images":
        event.media_urls, event.media_types = ["/tmp/user-image.png"], ["image/png"]
        monkeypatch.setattr(runner, "_enrich_inbound_images", enrich)
    monkeypatch.setattr(runner, "_expand_inbound_context_references", expand)
    monkeypatch.setattr(type(adapter), "resolve_turn_room_state", room_state)

    async def process():
        message = await runner._prepare_profile_scoped_inbound_message_text(
            event=event,
            source=source,
            history=history,
            session_key="session",
        )
        assert message is not None
        if boundary == "proxy":
            from tests.gateway.test_proxy_mode import (
                _FakeSession,
                _FakeSSEResponse,
                _patch_aiohttp,
            )

            async def typing(_chat, **_kwargs):
                started.set()
                await release.wait()

            monkeypatch.setattr(adapter, "send_typing", typing)
            monkeypatch.setattr(runner, "_get_proxy_url", lambda: "http://proxy.test")
            monkeypatch.setattr(
                runner, "_run_still_current_fn", lambda *_args: lambda: True
            )
            monkeypatch.setattr(runner, "_proxy_stream_consumer", lambda *_args: None)
            session = _FakeSession(_FakeSSEResponse(sse_chunks=["data: [DONE]\n\n"]))
            kwargs = {}
            if getattr(event, "_prepared_inbound", None) is not None:
                kwargs["input_snapshot"] = event._prepared_inbound
            with _patch_aiohttp(session):
                await runner._run_agent_inner(
                    message,
                    "cached system prefix",
                    history,
                    source,
                    "id",
                    session_key="session",
                    **kwargs,
                )
            posted = session.captured_json
            assert posted is not None
            assert posted["messages"][:-1] == [
                {"role": "system", "content": "cached system prefix"},
                *previous,
            ]
            return posted["messages"][-1]["content"]
        if boundary != "model-prepare":
            return message
        started.set()
        await release.wait()
        if getattr(event, "_prepared_inbound", None) is not None:
            await event._prepared_inbound.snapshot.refresh()
        ctx = TurnContext(
            source=source,
            message=message,
            history=history,
            context_prompt="cached system prefix",
            session_key="session",
            session_id="id",
        )
        if getattr(event, "_prepared_inbound", None) is not None:
            ctx.input_snapshot = event._prepared_inbound
        turn = TurnRunner(runner, ctx)
        persist_message, timestamp = turn._prepare_turn_message(history)
        captured = {}

        def model_input(text, **kwargs):
            captured.update(text=text, history=kwargs["conversation_history"])
            return {"final_response": "ok"}

        turn._run_conversation_with_approval(
            SimpleNamespace(run_conversation=model_input),
            history,
            [],
            persist_message,
            timestamp,
        )
        assert captured["history"] == previous
        assert ctx.context_prompt == "cached system prefix"
        return captured["text"]

    pending = asyncio.create_task(process())
    try:
        await asyncio.wait_for(started.wait(), timeout=2.0)
        if change == "redaction":
            cache.redact(ROOM, target)
        elif change in {"failed-recovery", "older-replacement"}:
            cache.redact(ROOM, "$latest")
            raw = _edited(
                _original(target, "initial draft"), "validated older replacement"
            )
            raw["unsigned"]["m.relations"]["m.replace"]["event_id"] = "$earlier"
        else:
            cache.apply_edit(
                ROOM,
                SENDER,
                {
                    "m.relates_to": {"rel_type": "m.replace", "event_id": target},
                    "m.new_content": {"msgtype": "m.text", "body": "after enrichment"},
                },
                replacement_id="$next",
            )
    finally:
        release.set()
        result = await pending

    assert history == previous
    assert result is not None
    assert "expanded user reference" in result
    if boundary == "images":
        assert "Image enrichment completed" in result
    assert "before enrichment" not in result
    assert "initial draft" not in result
    if change == "redaction":
        assert "Replying to" not in result
        if scope != "reply-only":
            assert "[redacted]" in result
    elif change == "failed-recovery":
        assert "Replying to" not in result
        if scope != "reply-only":
            assert "[event content unavailable]" in result
    else:
        assert (
            "after enrichment" in result
            if change == "edit"
            else "validated older replacement" in result
        )
        assert "Replying to" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["text", "native"])
@pytest.mark.parametrize("change", ["unchanged", "redaction", "failed-recovery"])
async def test_quoted_images_are_rechecked_without_losing_authored_image_enrichment(
    tmp_path, mode: str, change: str, monkeypatch
):
    started, release = asyncio.Event(), asyncio.Event()
    authored_image, quoted_image = tmp_path / "authored.png", tmp_path / "quoted.png"
    authored_image.write_bytes(b"authored image")
    quoted_image.write_bytes(b"quoted image")
    adapter = _make_adapter()
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    cache = adapter._event_context_cache
    cache.store(
        ROOM,
        "$target",
        MatrixEventContext(
            SENDER,
            "quoted attachment",
            media_path=str(quoted_image),
            media_type="image/png",
            is_image=True,
            replacement_id="$latest",
        ),
    )
    adapter._client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(side_effect=RuntimeError("offline")))
    )
    source = SessionSource(Platform.MATRIX, ROOM, chat_type="dm", user_id=SENDER)
    event = await adapter._build_inbound_event(
        ROOM,
        SENDER,
        "$current",
        "question",
        {"body": "question"},
        {"m.in_reply_to": {"event_id": "$target"}},
        ctx=("question", True, "dm", None, "Alice", source),
        media_urls=[str(authored_image)],
        media_types=["image/png"],
    )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {Platform.MATRIX: adapter}
    state = runner._session_state("session")

    async def enrich(
        _source: SessionSource, _key: str, text: str, paths: list[str]
    ) -> str:
        if str(quoted_image) in paths:
            started.set()
            await release.wait()
        if mode == "native":
            state.persistent.native_image_paths = list(paths)
            return text
        descriptions = [
            "authored image description"
            if path == str(authored_image)
            else "quoted image description"
            for path in paths
        ]
        return "\n".join([*descriptions, text])

    monkeypatch.setattr(runner, "_enrich_inbound_images", enrich)
    pending = asyncio.create_task(
        runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
            session_key="session",
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=2.0)
        if change == "redaction":
            cache.redact(ROOM, "$target")
        elif change == "failed-recovery":
            cache.redact(ROOM, "$latest")
    finally:
        release.set()
        result = await pending
    assert result is not None
    if mode == "native":
        assert state.persistent.native_image_paths == [
            str(authored_image),
            *([str(quoted_image)] if change == "unchanged" else []),
        ]
    else:
        assert "authored image description" in result
        assert ("quoted image description" in result) == (change == "unchanged")
    assert ("quoted attachment" in result) == (change == "unchanged")
