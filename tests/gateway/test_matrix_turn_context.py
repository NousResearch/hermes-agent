"""Matrix context stays refreshable through gateway prompt enrichment."""

import asyncio
from dataclasses import replace
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import hermes_yaml as yaml
import pytest

from agent import secret_scope
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner, _profile_runtime_scope
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
        ctx=(content["body"], False, "group", source.thread_id, "Alice", True, source),
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

    async def room_identity(_room):
        if boundary == "room-state":
            started.set()
            await release.wait()
        return SimpleNamespace(room_state=MatrixRoomState(None, None, None))

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
    monkeypatch.setattr(adapter, "_resolve_room_identity", room_identity)

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
                proxy_result = await runner._run_agent_inner(
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
            assert proxy_result["messages"][0] == posted["messages"][-1]
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
    marker = {
        "redaction": "[redacted]",
        "failed-recovery": "[event content unavailable]",
    }.get(change)
    if marker is not None:
        assert f'[Replying to Alice: "{marker}"]' in result
    else:
        assert (
            "after enrichment" in result
            if change == "edit"
            else "validated older replacement" in result
        )
        assert "Replying to" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["text", "native"])
@pytest.mark.parametrize("change", ["unchanged", "redaction", "failed-recovery", "replacement", "unchanged-eviction", "redaction-eviction", "failed-recovery-eviction"])
@pytest.mark.parametrize("transform", ["direct", "rewrite", "pending", "parked", "photo", "text-batch", "queue-command", "shared-path"])
async def test_quoted_images_are_rechecked_without_losing_authored_image_enrichment(
    tmp_path, mode: str, change: str, transform: str, monkeypatch
):
    started, release = asyncio.Event(), asyncio.Event()
    authored_image, quoted_image = tmp_path / "authored.png", tmp_path / "quoted.png"
    authored_image.write_bytes(b"authored image")
    quoted_image.write_bytes(b"quoted image")
    if transform == "shared-path":
        authored_image = quoted_image
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
        ctx=("question", True, "dm", None, "Alice", False, source),
        media_urls=[] if transform in {"pending", "parked", "photo", "text-batch"} else [str(authored_image)],
        media_types=[] if transform in {"pending", "parked", "photo", "text-batch"} else ["image/png"],
    )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    runner.adapters = {Platform.MATRIX: adapter}
    if transform == "rewrite":
        monkeypatch.setattr(
            "hermes_cli.lifecycle.ainvoke_hook",
            AsyncMock(return_value=[{"action": "rewrite", "text": "rewritten question"}]),
        )
        event = await runner._hm_pre_gateway_dispatch_hook(event, source)
    elif transform in {"pending", "parked", "photo", "text-batch"}:
        from gateway.platforms.base_pending_merge import merge_pending_message_event
        from gateway.platforms.event import MessageEvent, MessageType

        incoming = event
        event = MessageEvent(
            "authored caption", source=source,
            message_type=MessageType.PHOTO if transform == "photo" else MessageType.TEXT,
            media_urls=[str(authored_image)], media_types=["image/png"],
        )
        if transform == "photo":
            incoming.message_type = MessageType.PHOTO
        if transform == "parked":
            incoming = replace(incoming, text=incoming.text)
        if transform == "text-batch":
            monkeypatch.setattr(adapter, "_drop_unresolved", lambda _event: False)
            monkeypatch.setattr(adapter, "_text_batch_key", lambda _event: "session")
            parked = asyncio.Event()
            monkeypatch.setattr(adapter, "_flush_text_batch", lambda _key: parked.wait())
            adapter._pending_text_batches["session"] = event
            adapter._enqueue_text_event(incoming)
            task = adapter._pending_text_batch_tasks["session"]
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        else:
            pending = {"session": event}
            merge_pending_message_event(pending, "session", incoming, merge_text=True)
    elif transform == "queue-command":
        event = replace(event, text="/queue question")
        queued = []
        monkeypatch.setattr(runner, "_enqueue_fifo", lambda _key, item, _adapter: queued.append(item))
        monkeypatch.setattr(runner, "_queue_depth", lambda *_args, **_kwargs: 1)
        await runner._busy_queue_command(event, "session", source)
        [event] = queued
    state = runner._session_state("session")

    async def enrich(
        _source: SessionSource, _key: str, text: str, paths: list[str]
    ) -> str:
        if str(quoted_image) in paths and (transform != "shared-path" or not text):
            started.set()
            await release.wait()
        if mode == "native":
            state.persistent.native_image_paths = list(paths)
            return text
        descriptions = [
            "authored image description"
            if path == str(authored_image) and (transform != "shared-path" or text)
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
        if change.startswith("redaction"):
            cache.redact(ROOM, "$target")
        elif change.startswith("failed-recovery"):
            cache.redact(ROOM, "$latest")
        elif change == "replacement":
            cache.apply_edit(
                ROOM, SENDER,
                {"m.relates_to": {"rel_type": "m.replace", "event_id": "$target"},
                 "m.new_content": {"msgtype": "m.text", "body": "new text parent"}},
                replacement_id="$next",
            )
        if change.endswith("-eviction"):
            import gc

            for index in range(cache.max_entries):
                cache.store(ROOM, f"$unrelated{index}", MatrixEventContext(SENDER, "unrelated"))
            gc.collect()
    finally:
        release.set()
        result = await pending
    assert result is not None
    if mode == "native":
        assert state.persistent.native_image_paths == list(dict.fromkeys([
            str(authored_image),
            *([str(quoted_image)] if change.startswith("unchanged") else []),
        ]))
    else:
        assert "authored image description" in result
        assert ("quoted image description" in result) == (change.startswith("unchanged"))
    assert ("quoted attachment" in result) == (change.startswith("unchanged"))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["text", "native"])
async def test_merged_quotes_refresh_each_parent_without_dropping_other_media(tmp_path, monkeypatch, mode):
    from gateway.platforms.base_pending_merge import merge_pending_message_event

    adapter = _make_adapter()
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    adapter._client = None
    source = SessionSource(Platform.MATRIX, ROOM, chat_type="dm", user_id=SENDER)
    paths = [tmp_path / "first.png", tmp_path / "second.png"]
    events = []
    for index, path in enumerate(paths):
        path.write_bytes(b"image")
        target = f"$image{index}"
        adapter._event_context_cache.store(ROOM, target, MatrixEventContext(
            SENDER, target, str(path), "image/png", is_image=True,
        ))
        events.append(await adapter._build_inbound_event(
            ROOM, SENDER, f"$reply{index}", "question", {"body": "question"},
            {"m.in_reply_to": {"event_id": target}},
            ctx=("question", True, "dm", None, "Alice", False, source),
        ))
    pending = {"session": events[0]}
    merge_pending_message_event(pending, "session", events[1])
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    runner.adapters = {Platform.MATRIX: adapter}
    state = runner._session_state("session")

    async def enrich(_source, _key, text, images):
        if mode == "native":
            state.persistent.native_image_paths = list(images)
        return "\n".join([f"description {Path(path).stem}" for path in images])

    monkeypatch.setattr(runner, "_enrich_inbound_images", enrich)
    await runner._prepare_inbound_message_text(
        event=events[0], source=source, history=[], session_key="session",
    )
    adapter._event_context_cache.redact(ROOM, "$image0")
    prepared = events[0]._prepared_inbound
    await prepared.snapshot.refresh()
    assert prepared.render(runner) == (
        '[Replying to Alice: "[redacted]"]\n\ndescription second\n\nquestion'
    )
    assert prepared.retained_image_paths([str(path) for path in paths]) == [str(paths[1])]


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["room", "thread"])
@pytest.mark.parametrize("change", ["unchanged", "replacement", "redaction"])
async def test_catch_up_preserves_only_current_quoted_pixels_at_model_input(tmp_path, monkeypatch, scope, change):
    import base64

    adapter = _make_adapter()
    adapter._room_backfill_limit = adapter._thread_backfill_limit = 1
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    image = tmp_path / "quoted.png"
    image.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aGNcAAAAASUVORK5CYII="
    ))
    raw = _original("$image", "quoted.png")
    raw["content"].update(msgtype="m.image", url="mxc://example.org/image", info={"mimetype": "image/png"})

    async def request(_method, path, **_kwargs):
        if "/event/" in path:
            return raw
        if "/context/" in path:
            return {"start": "boundary"}
        return {"chunk": [raw] if "/messages" in path else []}

    adapter._client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)))
    monkeypatch.setattr(adapter, "_cache_quoted_image", AsyncMock(return_value=(str(image), "image/png")))
    source = SessionSource(Platform.MATRIX, ROOM, chat_type="group", user_id=SENDER,
                           thread_id="$image" if scope == "thread" else None)
    content = {"body": "question", "msgtype": "m.text", "m.mentions": {"user_ids": [adapter._user_id]}}
    relation = {"m.in_reply_to": {"event_id": "$image"}}
    if scope == "thread":
        relation.update(rel_type="m.thread", event_id="$image", is_falling_back=False)
    content["m.relates_to"] = relation
    event = await adapter._build_inbound_event(
        ROOM, SENDER, "$current", "question", content, relation,
        ctx=("question", False, "group", source.thread_id, "Alice", True, source),
    )
    if change == "replacement":
        raw = _edited(raw, "replaced with text")
    elif change == "redaction":
        raw = {**raw, "content": {}, "unsigned": {"redacted_because": {"event_id": "$redaction"}}}
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    runner.adapters = {Platform.MATRIX: adapter}

    async def native(_source, key, text, paths):
        runner._session_state(key).persistent.native_image_paths = list(paths)
        return text

    monkeypatch.setattr(runner, "_enrich_inbound_images", native)
    message = await runner._prepare_inbound_message_text(event=event, source=source, history=[], session_key="session")
    ctx = TurnContext(source=source, message=message, history=[], context_prompt="cached system prefix",
                      session_key="session", session_id="id", input_snapshot=event._prepared_inbound)
    turn = TurnRunner(runner, ctx)
    captured = {}

    def model_input(text, **kwargs):
        captured.update(text=text, history=kwargs["conversation_history"])
        return {"final_response": "ok"}

    turn._run_conversation_with_approval(SimpleNamespace(run_conversation=model_input), [], [], None, None)
    parts = captured["text"]
    if change == "unchanged":
        assert isinstance(parts, list)
        images = [part for part in parts if part["type"] == "image_url"]
        assert len(images) == 1
        assert images[0]["image_url"]["url"].startswith("data:image/")
    else:
        assert isinstance(parts, str)
        assert ("replaced with text" in parts) == (change == "replacement")
        assert ("[redacted]" in parts) == (change == "redaction")
    assert captured["history"] == []
    assert ctx.context_prompt == "cached system prefix"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["sender", "missing-new-content", "valid", "redaction"])
@pytest.mark.parametrize("encrypted", [False, True])
async def test_typed_edit_keeps_quoted_media_until_authoritative_content_changes(
    tmp_path, monkeypatch, change: str, encrypted: bool,
):
    import copy
    from unittest.mock import patch
    from tests.gateway.test_matrix_effective_event_state import _edit_store

    mautrix_types = pytest.importorskip("mautrix.types")
    image = tmp_path / "quoted.png"
    image.write_bytes(b"image")
    original = _original("$target", "quoted.png")
    original["content"].update(msgtype="m.image", url="mxc://example.org/image")
    raw = _edited(original, "replacement text")
    replacement = raw["unsigned"]["m.relations"]["m.replace"]
    if change == "sender":
        replacement["sender"] = "@mallory:example.org"
    elif change == "missing-new-content":
        replacement["content"].pop("m.new_content")
    content = copy.deepcopy(replacement["content"])
    typed = mautrix_types.Event.deserialize({**copy.deepcopy(replacement), "origin_server_ts": 1})
    assert "m.new_content" in typed.content.serialize()
    typed_original = mautrix_types.Event.deserialize({**copy.deepcopy(original), "origin_server_ts": 1})
    if encrypted:
        typed["mautrix"] = {"was_encrypted": True}
        raw["type"] = "m.room.encrypted"
        raw["content"] = {"ciphertext": "original"}
        replacement["type"] = "m.room.encrypted"
        replacement["content"] = {"ciphertext": "replacement", "session_id": "sess",
                                  "m.relates_to": {"rel_type": "m.replace", "event_id": "$target"}}
    adapter = _make_adapter()
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    adapter._client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(return_value=copy.deepcopy(original))),
                                     crypto=SimpleNamespace(crypto_store=_edit_store(content)))
    loader = AsyncMock(return_value=(str(image), "image/png"))
    monkeypatch.setattr(adapter, "_cache_quoted_image", loader)
    source = SessionSource(Platform.MATRIX, ROOM, chat_type="dm", user_id=SENDER)
    event = await adapter._build_inbound_event(
        ROOM, SENDER, "$reply", "question", {"body": "question"}, {"m.in_reply_to": {"event_id": "$target"}},
        ctx=("question", True, "dm", None, "Alice", False, source),
    )
    snapshot = await adapter.fetch_inbound_context(event)
    adapter._client.api.request.return_value = raw
    if change == "redaction":
        await adapter._on_redaction(SimpleNamespace(room_id=ROOM, redacts="$target"))
    else:
        await adapter._on_room_message(typed)
    if change == "sender":
        assert snapshot.reply_image_paths() == [str(image)]
    with patch("plugins.platforms.matrix.effective_event._decrypt", new_callable=AsyncMock) as decrypt:
        decrypt.side_effect = [(typed_original, None), (typed, None)]
        await snapshot.refresh()
    current = snapshot.reply_event(event)
    assert (current.reply_to_text, snapshot.reply_image_paths(), loader.await_count) == (
        ("[image]", [str(image)], 1) if change in {"sender", "missing-new-content"} else
        ("replacement text", [], 1) if change == "valid" else ("[redacted]", [], 1)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["unchanged", "original", "replacement", "edit"])
@pytest.mark.parametrize("shared_path", [False, True])
async def test_native_conversion_revalidates_current_input_after_file_read(
    tmp_path, monkeypatch, change, shared_path
):
    import base64
    import threading

    image = tmp_path / "quoted.png"
    image.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        )
    )
    adapter = _make_adapter()
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    cache = adapter._event_context_cache
    cache.store(
        ROOM,
        "$target",
        MatrixEventContext(
            SENDER,
            "withdrawn quote",
            str(image),
            "image/png",
            is_image=True,
            replacement_id="$latest",
        ),
    )
    adapter._client = None
    source = SessionSource(Platform.MATRIX, ROOM, chat_type="dm", user_id=SENDER)
    event = await adapter._build_inbound_event(
        ROOM,
        SENDER,
        "$current",
        "question",
        {"body": "question"},
        {"m.in_reply_to": {"event_id": "$target"}},
        ctx=("question", True, "dm", None, "Alice", False, source),
        media_urls=[str(image)] if shared_path else [],
        media_types=["image/png"] if shared_path else [],
    )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    runner.adapters = {Platform.MATRIX: adapter}
    state = runner._session_state("session")

    async def enrich(_source, _key, text, paths):
        state.persistent.native_image_paths = list(paths)
        return text

    monkeypatch.setattr(runner, "_enrich_inbound_images", enrich)
    message = await runner._prepare_inbound_message_text(
        event=event, source=source, history=[{}], session_key="session"
    )
    history = [
        {"role": "user", "content": "previous quote stays"},
        {"role": "assistant", "content": "previous answer"},
    ]
    ctx = TurnContext(
        source=source,
        message=message,
        history=history,
        context_prompt="cached system prefix",
        session_key="session",
        session_id="id",
        input_snapshot=event._prepared_inbound,
    )
    runner._pending_model_notes = {"session": "Pending model note"}
    turn = TurnRunner(runner, ctx)
    persist, timestamp = turn._prepare_turn_message(history)
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original_read = Path.read_bytes

    def read(path):
        if path == image:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(2), "file conversion was not released"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    captured = {}

    def model_input(text, **kwargs):
        captured.update(
            text=text,
            history=kwargs["conversation_history"],
            persisted=kwargs.get("persist_user_message"),
        )
        return {"final_response": "ok"}

    pending = asyncio.create_task(
        asyncio.to_thread(
            turn._run_conversation_with_approval,
            SimpleNamespace(run_conversation=model_input),
            history,
            [],
            persist,
            timestamp,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        if change in {"original", "replacement"}:
            await adapter._on_redaction(
                SimpleNamespace(
                    room_id=ROOM,
                    redacts="$target" if change == "original" else "$latest",
                )
            )
        elif change == "edit":
            cache.apply_edit(
                ROOM,
                SENDER,
                {
                    "m.relates_to": {"rel_type": "m.replace", "event_id": "$target"},
                    "m.new_content": {"msgtype": "m.text", "body": "latest quote"},
                },
                replacement_id="$next",
            )
    finally:
        release.set()
        await pending
    current = captured["text"]
    parts = (
        current if isinstance(current, list) else [{"type": "text", "text": current}]
    )
    text = "\n".join(part["text"] for part in parts if part["type"] == "text")
    assert (
        "withdrawn quote" in text,
        "latest quote" in text,
        len([part for part in parts if part["type"] == "image_url"]),
    ) == (
        change == "unchanged",
        change == "edit",
        int(shared_path or change == "unchanged"),
    )
    assert "Pending model note" in text
    assert (
        "withdrawn quote" in captured["persisted"],
        "latest quote" in captured["persisted"],
    ) == (change == "unchanged", change == "edit")
    assert "Pending model note" not in captured["persisted"]
    assert (captured["history"], ctx.context_prompt) == (
        history,
        "cached system prefix",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "valid",
        "missing-new-content",
        "sender",
        "failed-fetch",
        "missing-keys",
        "redaction",
    ],
)
async def test_typed_edit_callback_resolves_current_native_input(
    tmp_path, monkeypatch, change
):
    import base64
    import copy
    import threading

    from hermes_constants import get_hermes_home
    from plugins.platforms.matrix.room_context import MatrixHistoryContext
    from plugins.platforms.matrix.turn_context import MatrixTurnContextUpdate

    mautrix_types = pytest.importorskip("mautrix.types")
    image = tmp_path / "quoted.png"
    pixels = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    )
    image.write_bytes(pixels)
    original = _original("$target", image.name)
    original["content"].update(msgtype="m.image", url="mxc://example.org/image")
    raw = _edited(original, "replacement text")
    replacement = raw["unsigned"]["m.relations"]["m.replace"]
    if change == "missing-new-content":
        replacement["content"].pop("m.new_content")
    if change == "sender":
        replacement["sender"] = "@mallory:example.org"
    typed = mautrix_types.Event.deserialize({
        **copy.deepcopy(replacement),
        "origin_server_ts": 100_000,
    })
    assert "m.new_content" in typed.content.serialize()
    adapter = _make_adapter()
    adapter._startup_ts = 100.0
    monkeypatch.setattr("plugins.platforms.matrix.adapter.time.time", lambda: 100.0)
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    loop = asyncio.get_running_loop()
    home = get_hermes_home()
    response = original

    async def request(_method, path, **_kwargs):
        assert asyncio.get_running_loop() is loop
        assert get_hermes_home() == home
        assert path.endswith("/event/%24target")
        if response is raw and change == "failed-fetch":
            raise RuntimeError("recovery unavailable")
        return copy.deepcopy(response)

    adapter._client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(side_effect=request)), crypto=None
    )
    monkeypatch.setattr(
        adapter,
        "_cache_quoted_image",
        AsyncMock(return_value=(str(image), "image/png")),
    )
    source = SessionSource(Platform.MATRIX, ROOM, chat_type="dm", user_id=SENDER)
    event = await adapter._build_inbound_event(
        ROOM,
        SENDER,
        "$current",
        "question",
        {"body": "question"},
        {"m.in_reply_to": {"event_id": "$target"}},
        ctx=("question", True, "dm", None, "Alice", False, source),
    )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    runner.adapters = {Platform.MATRIX: adapter}
    state = runner._session_state("session")

    async def enrich(_source, _key, text, paths):
        state.persistent.native_image_paths = list(paths)
        return text

    monkeypatch.setattr(runner, "_enrich_inbound_images", enrich)
    message = await runner._prepare_inbound_message_text(
        event=event, source=source, history=[{}], session_key="session"
    )
    snapshot = event._prepared_inbound.snapshot
    snapshot.use_turn_context(MatrixTurnContextUpdate(None, None, history=MatrixHistoryContext(
        adapter,
        ROOM,
        [snapshot.parent],
        "Recent room messages",
        "dm",
        {SENDER: "Alice"},
    )))
    history = [
        {"role": "user", "content": "previous quote stays"},
        {"role": "assistant", "content": "previous answer"},
    ]
    previous_history = copy.deepcopy(history)
    ctx = TurnContext(
        source=source,
        message=message,
        history=history,
        context_prompt="cached system prefix",
        session_key="session",
        session_id="id",
        input_snapshot=event._prepared_inbound,
    )
    turn = TurnRunner(runner, ctx)
    persist, timestamp = turn._prepare_turn_message(history)
    started, release = asyncio.Event(), threading.Event()
    original_read = Path.read_bytes

    def read(path):
        if path == image:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(2), "file conversion was not released"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    captured = {}

    def model_input(text, **kwargs):
        captured.update(
            text=text,
            persisted=kwargs.get("persist_user_message"),
            history=kwargs["conversation_history"],
        )
        return {"final_response": "ok"}

    pending = asyncio.create_task(
        asyncio.to_thread(
            turn._run_conversation_with_approval,
            SimpleNamespace(run_conversation=model_input),
            history,
            [],
            persist,
            timestamp,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        response = raw
        if change == "missing-keys":
            raw["type"] = "m.room.encrypted"
            raw["content"] = {"ciphertext": "unavailable"}
        await adapter._on_room_message(typed)
        if change == "redaction":
            await adapter._on_redaction(
                SimpleNamespace(room_id=ROOM, redacts="$target")
            )
    finally:
        release.set()
        await pending
    current = captured["text"]
    parts = (
        current if isinstance(current, list) else [{"type": "text", "text": current}]
    )
    text = "\n".join(part["text"] for part in parts if part["type"] == "text")
    unchanged = change in {"missing-new-content", "sender"}
    assert (
        snapshot.reply_event(event).reply_to_text,
        [part for part in parts if part["type"] == "image_url"],
        "replacement text" in text,
        "[image]" in text,
    ) == (
        "[image]" if unchanged
        else "replacement text" if change == "valid"
        else "[redacted]" if change == "redaction"
        else "[event content unavailable]",
        [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(pixels).decode('ascii')}",
                },
            }
        ]
        if unchanged
        else [],
        change == "valid",
        unchanged,
    )
    assert "[Recent room messages]" in text
    assert "question" in text
    assert (
        "replacement text" in captured["persisted"],
        "[image]" in captured["persisted"],
    ) == (change == "valid", unchanged)
    assert (captured["history"], ctx.context_prompt) == (
        previous_history,
        "cached system prefix",
    )


class _WorkerScanProbe:
    """A tracked event state that blocks a scan from a worker thread.

    While the worker waits, the event loop stores a newly observed event,
    which is what a sync callback does when a message arrives.
    """

    room_id_value = "!elsewhere:example.org"
    event_id = "$probe"

    def __init__(self, cache, loop):
        import threading

        self.cache, self.loop = cache, loop
        self.loop_thread = threading.get_ident()
        self.resumed = threading.Event()
        self.worker_reads = 0

    def _arrive(self):
        self.cache.retain(ROOM, "$arrived")
        self.resumed.set()

    @property
    def room_id(self):
        import threading

        if threading.get_ident() != self.loop_thread:
            self.worker_reads += 1
            self.loop.call_soon_threadsafe(self._arrive)
            assert self.resumed.wait(2), "the event loop did not store the arriving event"
        return self.room_id_value


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["turn-message", "native-image"])
async def test_worker_thread_input_preparation_reads_matrix_state_on_the_loop(
    tmp_path, monkeypatch, stage
):
    import base64

    image = tmp_path / "quoted.png"
    image.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        )
    )
    adapter = _make_adapter()
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    adapter._client = None
    cache = adapter._event_context_cache
    cache.store(
        ROOM,
        "$target",
        MatrixEventContext(
            SENDER,
            "quoted parent",
            str(image),
            "image/png",
            is_image=True,
            replacement_id="$latest",
        ),
    )
    source = SessionSource(Platform.MATRIX, ROOM, chat_type="dm", user_id=SENDER)
    event = await adapter._build_inbound_event(
        ROOM,
        SENDER,
        "$current",
        "question",
        {"body": "question"},
        {"m.in_reply_to": {"event_id": "$target"}},
        ctx=("question", True, "dm", None, "Alice", False, source),
    )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    runner.adapters = {Platform.MATRIX: adapter}
    state = runner._session_state("session")

    async def enrich(_source, _key, text, paths):
        state.persistent.native_image_paths = list(paths)
        return text

    monkeypatch.setattr(runner, "_enrich_inbound_images", enrich)
    message = await runner._prepare_inbound_message_text(
        event=event, source=source, history=[{}], session_key="session"
    )
    history = [{"role": "user", "content": "previous turn"}]
    ctx = TurnContext(
        source=source,
        message=message,
        history=history,
        context_prompt="cached system prefix",
        session_key="session",
        session_id="id",
        input_snapshot=event._prepared_inbound,
    )
    turn = TurnRunner(runner, ctx)
    captured = {}

    def model_input(text, **_kwargs):
        captured["text"] = text
        return {"final_response": "ok"}

    agent = SimpleNamespace(run_conversation=model_input)
    prepared = None if stage == "turn-message" else turn._prepare_turn_message(history)

    def work():
        if prepared is None:
            return turn._prepare_turn_message(history)
        return turn._run_conversation_with_approval(agent, history, [], *prepared)

    probe = _WorkerScanProbe(cache, asyncio.get_running_loop())
    cache._active_states.add(probe)

    await asyncio.wait_for(asyncio.to_thread(work), timeout=5)

    rendered = captured.get("text", ctx.message)
    text = (
        rendered
        if isinstance(rendered, str)
        else "\n".join(part["text"] for part in rendered if part["type"] == "text")
    )
    assert (probe.worker_reads, "quoted parent" in text, "question" in text) == (
        0,
        True,
        True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("parent", ["unreachable", "undecryptable"])
@pytest.mark.parametrize(
    ("body", "reply_line"),
    [
        (
            "> <@bob:example.org> the quoted parent\n\nwhat about this?",
            '[Replying to [unverified] Bob: "the quoted parent"]',
        ),
        ("what about this?", '[Replying to: "[event content unavailable]"]'),
    ],
)
async def test_reply_line_survives_a_parent_that_cannot_be_read(
    tmp_path, parent, body, reply_line
):
    adapter = _make_adapter()
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._get_display_name = AsyncMock(
        side_effect=lambda _room, user: "Bob" if user == "@bob:example.org" else "Alice"
    )
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    encrypted = {
        **_original("$parent", ""),
        "type": "m.room.encrypted",
        "content": {"ciphertext": "unavailable"},
    }
    adapter._client = SimpleNamespace(
        api=SimpleNamespace(
            request=AsyncMock(
                side_effect=RuntimeError("history not visible")
                if parent == "unreachable"
                else lambda *_args, **_kwargs: dict(encrypted)
            )
        ),
        crypto=None,
    )
    source = SessionSource(
        Platform.MATRIX, ROOM, chat_type="group", user_id=SENDER, user_name="Alice"
    )
    relation = {"m.in_reply_to": {"event_id": "$parent"}}
    content = {"msgtype": "m.text", "body": body, "m.relates_to": relation}
    event = await adapter._build_inbound_event(
        ROOM,
        SENDER,
        "$current",
        body,
        content,
        relation,
        ctx=(body, False, "group", None, "Alice", True, source),
    )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {Platform.MATRIX: adapter}
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    runner.session_store._entries["session"] = SessionEntry(
        "session", "id", now, now, origin=source
    )

    message = await runner._prepare_profile_scoped_inbound_message_text(
        event=event,
        source=source,
        history=[{"role": "user", "content": "earlier turn"}],
        session_key="session",
    )

    assert message is not None
    assert message.endswith(f"{reply_line}\n\nwhat about this?")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attachment",
    [
        {"file": {"url": "mxc://example.org/encrypted", "key": {}, "iv": "", "hashes": {}}},
        {},
    ],
    ids=["encrypted-download-fails", "no-url"],
)
async def test_media_without_a_cached_file_reaches_a_live_session_as_its_caption(
    tmp_path, monkeypatch, attachment
):
    adapter = _make_adapter()
    adapter._client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock()))
    source = SessionSource(Platform.MATRIX, ROOM, chat_type="group", user_id=SENDER)
    monkeypatch.setattr(
        adapter,
        "_resolve_message_context",
        AsyncMock(return_value=("what is this?", False, "group", None, "Alice", False, source)),
    )
    monkeypatch.setattr(
        adapter, "_download_and_cache_media", AsyncMock(side_effect=RuntimeError("decrypt failed"))
    )
    adapter.handle_message = AsyncMock()
    content = {
        "msgtype": "m.image",
        "body": "what is this?",
        "filename": "photo.png",
        "info": {"mimetype": "image/png"},
        **attachment,
    }
    await adapter._handle_media_message(ROOM, SENDER, "$photo", 1000.0, content, {}, "m.image")
    event = adapter.handle_message.await_args.args[0]
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    runner.adapters = {Platform.MATRIX: adapter}
    state = runner._session_state("session")

    message = await runner._prepare_inbound_message_text(
        event=event, source=source, history=[], session_key="session"
    )

    assert (event.media_urls, event.media_types, message, state.persistent.native_image_paths) == (
        [], [], "what is this?", [],
    )


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    launch = tmp_path / ".hermes"
    routed = launch / "profiles" / "b"
    routed.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    for home in (launch, routed):
        (home / "config.yaml").write_text(yaml.safe_dump({}), encoding="utf-8")
    secret_scope.set_multiplex_active(True)
    yield launch, routed
    secret_scope.set_multiplex_active(False)


@pytest.mark.asyncio
@pytest.mark.parametrize("move", ["moved", "move-fails"])
@pytest.mark.parametrize("mode", ["native", "text"])
async def test_quoted_image_is_the_file_in_the_cache_of_a_routed_profile(
    monkeypatch, tmp_path, two_homes, mode, move
):
    import base64

    launch, routed = two_homes
    adapter = _make_adapter()
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    parent = _original("$image", "photo.png")
    parent["content"].update(
        msgtype="m.image", url="mxc://example.org/image", info={"mimetype": "image/png"}
    )
    from tests.gateway.matrix_helpers import FakeMediaDownload

    adapter._client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(return_value=parent), session=SimpleNamespace()),
    )
    FakeMediaDownload(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aGNcAAAAASUVORK5CYII="
    )).install(adapter._client)
    source = SessionSource(Platform.MATRIX, ROOM, chat_type="dm", user_id=SENDER)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    runner.adapters = {Platform.MATRIX: adapter}
    monkeypatch.setattr(runner, "_decide_image_input_mode", lambda **_kwargs: mode)

    async def analyse(_text, paths):
        return "\n\n".join(f"<pixels of {path}>" for path in paths)

    monkeypatch.setattr(runner, "_enrich_message_with_vision", analyse)
    if move == "move-fails":
        monkeypatch.setattr(
            "gateway.run_inbound_media.shutil.copy2", lambda *_args: (_ for _ in ()).throw(OSError("busy"))
        )
    turns = []
    expected = []
    prior_files = []
    for turn, home in enumerate((launch, routed, routed, launch)):
        event = await adapter._build_inbound_event(
            ROOM, SENDER, f"$reply{turn}", "what is this?", {"body": "what is this?"},
            {"m.in_reply_to": {"event_id": "$image"}},
            ctx=("what is this?", True, "dm", None, "Alice", False, source),
        )
        session_key = f"session{turn}"
        runner._session_state(session_key)
        with _profile_runtime_scope(home):
            prepared = await runner._prepare_inbound_message_text(
                event=event, source=source, history=[], session_key=session_key
            )
        quoted = event.media_urls[0] if event.media_urls else None
        if quoted is not None:
            prior_files.append(Path(quoted))
        turns.append(
            {
                "retained_files": [path.is_file() for path in prior_files],
                "own": quoted is not None and Path(quoted).parent == home / "cache" / "images" and Path(quoted).is_file(),
                "pixels": runner._session_state(session_key).persistent.native_image_paths,
                "description": quoted is not None and f"<pixels of {quoted}>" in prepared,
            }
        )
        visible = move == "moved" or home == launch
        expected.append(
            {
                "retained_files": [True] * len(prior_files),
                "own": visible,
                "pixels": [quoted] if visible and mode == "native" else [],
                "description": visible and mode == "text",
            }
        )

    assert turns == expected
