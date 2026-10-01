"""Native Matrix actions and stickers use the normal message and context paths."""

from __future__ import annotations

import base64
import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import hermes_yaml as yaml
import pytest

from agent import secret_scope
from agent.context_references import preprocess_context_references_async
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import merge_pending_message_event
from gateway.platforms.event import MessageType
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionStore
from gateway.turn_context import TurnContext
from plugins.platforms.matrix.adapter import MatrixAdapter
from tests.gateway.matrix_helpers import FakeMediaDownload
from plugins.platforms.matrix.read_context import read_matrix_context
from plugins.platforms.matrix.reply_context import MatrixEventContext
from plugins.platforms.matrix.thread_context import history_entry


ROOM = "!room:example.org"
SENDER = "@alice:example.org"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _adapter(monkeypatch: pytest.MonkeyPatch) -> tuple[MatrixAdapter, AsyncMock]:
    monkeypatch.setattr("plugins.platforms.matrix.adapter.time.time", lambda: 1000.0)
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="test",
            extra={
                "homeserver": "https://example.org",
                "user_id": "@hermes:example.org",
                "require_mention": False,
                "auto_thread": False,
            },
        )
    )
    adapter._text_batch_delay_seconds = 0
    adapter._startup_ts = 999
    adapter._joined_rooms.add(ROOM)
    adapter._resolve_room_identity = AsyncMock(
        return_value=SimpleNamespace(
            chat_type="group",
            display_name="Room",
            room_topic=None,
            server_name="example.org",
            members_digest=None,
            room_state=None,
        )
    )
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter.set_authorization_check(lambda *_args, **_kwargs: True)
    received = AsyncMock()
    adapter.handle_message = received
    adapter._client = SimpleNamespace(
        api=SimpleNamespace(request=AsyncMock(return_value={"chunk": []}), session=SimpleNamespace()),
        crypto=None,
    )
    adapter._client._test_media_download = FakeMediaDownload(PNG).install(adapter._client)
    return adapter, received


def _event(kind: str) -> dict:
    content = (
        {"body": "/new waves"}
        if kind == "emote"
        else {
            "body": "Friendly fox.png",
            "url": "mxc://example.org/fox",
            "info": {"mimetype": "image/png", "size": len(PNG), "w": 1, "h": 1},
        }
    )
    if kind == "emote":
        content.update(
            msgtype="m.emote",
            format="org.matrix.custom.html",
            formatted_body="<i>/new waves</i>",
        )
    content["m.relates_to"] = {"rel_type": "m.thread", "event_id": "$root"}
    return {
        "type": "m.room.message" if kind == "emote" else "m.sticker",
        "room_id": ROOM,
        "sender": SENDER,
        "event_id": "$native",
        "origin_server_ts": 1000000,
        "content": content,
    }


def _typed(raw: dict):
    return pytest.importorskip("mautrix.types").Event.deserialize(deepcopy(raw))


def _body(kind: str, sender: str = SENDER) -> str:
    return (
        f"[emote by https://matrix.to/#/{sender}] /new waves"
        if kind == "emote"
        else "[sticker: Friendly fox.png]"
    )


def _runner(adapter: MatrixAdapter, tmp_path: Path) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    runner.adapters = {Platform.MATRIX: adapter}
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,scenario",
    [
        (kind, scenario)
        for kind in ("emote", "sticker")
        for scenario in (
            "accepted",
            "mention-denied",
            "room-denied",
            "sender-denied",
            "redacted-denied",
            "queued-redaction",
            "queued-edit",
        )
    ]
    + [
        ("sticker", scenario)
        for scenario in (
            "oversize",
            "download-failure",
            "missing-media-key",
            "conversion-redaction",
            "analysis",
            "analysis-redaction",
            "analysis-edit",
            "invalid-url-denied",
            "missing-url-denied",
            "download-oversize",
            "download-edit",
            "download-redaction",
        )
    ],
)
async def test_native_content_reaches_model_with_actor_description_and_pixels(
    monkeypatch, tmp_path, kind, scenario
):
    adapter, received = _adapter(monkeypatch)
    raw = _event(kind)
    if scenario == "mention-denied":
        adapter._require_mention = True
    if scenario == "room-denied":
        adapter._allowed_rooms = {"!other:example.org"}
    if scenario == "sender-denied":
        adapter.set_authorization_check(lambda *_args, **_kwargs: False)
    if scenario == "oversize":
        adapter._max_media_bytes = len(PNG) - 1
    if scenario == "download-failure":
        adapter._client._test_media_download.fail = True
    if scenario == "invalid-url-denied":
        raw["content"]["url"] = "https://example.org/fox.png"
    if scenario == "missing-url-denied":
        del raw["content"]["url"]
    if scenario == "redacted-denied":
        raw["content"] = {}
        raw["unsigned"] = {
            "age": 1,
            "redacted_because": {
                "type": "m.room.redaction",
                "room_id": ROOM,
                "sender": SENDER,
                "event_id": "$redaction",
                "origin_server_ts": 1000001,
                "redacts": "$native",
                "content": {},
            },
        }
    if scenario == "download-oversize":
        raw["content"]["info"]["size"] = 1
        adapter._max_media_bytes = len(PNG) - 1
    if scenario in {"download-edit", "download-redaction"}:

        async def changed_versions():
            if scenario == "download-redaction":
                await adapter._on_redaction(
                    SimpleNamespace(room_id=ROOM, redacts="$native")
                )
            else:
                adapter._event_context_cache.apply_edit(
                    ROOM,
                    SENDER,
                    {
                        "m.new_content": {
                            "msgtype": "m.sticker",
                            "body": "Changed sticker",
                            "url": "mxc://example.org/new",
                            "info": {},
                        },
                        "m.relates_to": {
                            "rel_type": "m.replace",
                            "event_id": "$native",
                        },
                    },
                    replacement_id="$edit",
                )
            return SimpleNamespace(supports=lambda _version: True)

        adapter._client.versions = AsyncMock(side_effect=changed_versions)
    if scenario == "missing-media-key":
        raw["content"]["file"] = {
            "url": raw["content"].pop("url"),
            "key": {
                "k": "",
                "kty": "oct",
                "alg": "A256CTR",
                "ext": True,
                "key_ops": ["encrypt", "decrypt"],
            },
            "iv": "",
            "hashes": {},
            "v": "v2",
        }
    await adapter._on_room_message(_typed(raw))
    if scenario.endswith("denied"):
        assert (
            received.await_count,
            len(adapter._client._test_media_download.requested),
        ) == (0, 0)
        return
    received.assert_awaited_once()
    call = received.await_args
    assert call is not None
    event = call.args[0]
    text = _body(kind)
    if scenario in {"oversize", "download-oversize"}:
        text += "\n[matrix sticker attachment too large]"
    if scenario in {"download-failure", "missing-media-key"}:
        text += "\n[matrix sticker image unavailable]"
    assert {
        "text": event.text,
        "type": event.message_type,
        "actor": event.user_id,
        "room": event.source.chat_id,
        "thread": event.source.thread_id,
        "id": event.message_id,
    } == {
        "text": text,
        "type": MessageType.TEXT
        if kind == "emote" or scenario in {"oversize", "download-oversize"}
        else MessageType.PHOTO,
        "actor": SENDER,
        "room": ROOM,
        "thread": "$root",
        "id": "$native",
    }
    runner = _runner(adapter, tmp_path)
    monkeypatch.setattr(runner, "_decide_image_input_mode", lambda **_kwargs: "native")
    if scenario.startswith("analysis"):
        monkeypatch.setattr(
            runner, "_decide_image_input_mode", lambda **_kwargs: "text"
        )

        async def analyze(_text, _paths):
            if scenario == "analysis-redaction":
                await adapter._on_redaction(
                    SimpleNamespace(room_id=ROOM, redacts="$native")
                )
            if scenario == "analysis-edit":
                adapter._event_context_cache.apply_edit(
                    ROOM,
                    SENDER,
                    {
                        "m.new_content": {
                            "msgtype": "m.sticker",
                            "body": "Changed sticker",
                            "url": "mxc://example.org/new",
                            "info": {},
                        },
                        "m.relates_to": {
                            "rel_type": "m.replace",
                            "event_id": "$native",
                        },
                    },
                    replacement_id="$edit",
                )
            return "Analysed sticker pixels"

        monkeypatch.setattr(runner, "_enrich_message_with_vision", analyze)
    if scenario == "queued-redaction":
        await adapter._on_redaction(SimpleNamespace(room_id=ROOM, redacts="$native"))
    if scenario == "queued-edit":
        replacement = (
            {"body": "Changed action"}
            if kind == "emote"
            else {
                "body": "Changed sticker",
                "url": "mxc://example.org/new",
                "info": {"mimetype": "image/png"},
            }
        )
        if kind == "emote":
            replacement["msgtype"] = "m.emote"
        edit = {
            **raw,
            "event_id": "$edit",
            "content": {
                "body": "* changed",
                "msgtype": "m.emote" if kind == "emote" else "m.sticker",
                "m.new_content": replacement,
                "m.relates_to": {"rel_type": "m.replace", "event_id": "$native"},
            },
        }

        async def edited(_method, path, **_kwargs):
            if "/event/" in path:
                return {**raw, "unsigned": {"m.relations": {"m.replace": edit}}}
            return {"chunk": []}

        adapter._client.api.request.side_effect = edited
        await adapter._on_room_message(_typed(edit))
    prepared = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[{}],
        session_key="session",
    )
    if scenario == "conversion-redaction":
        read_bytes = Path.read_bytes

        def withdraw(path):
            data = read_bytes(path)
            if str(path) in event.media_urls:
                adapter._event_context_cache.redact(ROOM, "$native")
            return data

        monkeypatch.setattr(Path, "read_bytes", withdraw)
    turn = TurnRunner(
        runner,
        TurnContext(
            session_key="session",
            input_snapshot=event._prepared_inbound,
            message=prepared,
        ),
    )
    content = turn._native_image_run_message()
    parts = (
        content if isinstance(content, list) else [{"type": "text", "text": content}]
    )
    model_text = "".join(part.get("text", "") for part in parts)
    if scenario in {
        "queued-redaction",
        "conversion-redaction",
        "analysis-redaction",
        "download-redaction",
    }:
        assert "[redacted]" in model_text and _body(kind) not in model_text
    elif scenario in {"queued-edit", "analysis-edit", "download-edit"}:
        assert (
            "Changed action" if kind == "emote" else "Changed sticker"
        ) in model_text
        assert _body(kind) not in model_text
    else:
        assert text in model_text
    assert ("Analysed sticker pixels" in model_text) is (scenario == "analysis")
    assert any(part["type"] == "image_url" for part in parts) is (
        kind == "sticker" and scenario == "accepted"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,scenario",
    [
        (kind, scenario)
        for kind in ("emote", "sticker")
        for scenario in ("visible", "withdrawn", "bounded", "missing-room-key")
    ]
    + [("sticker", "oversize")],
)
async def test_native_content_has_consistent_effective_reads_history_and_reply_pixels(
    monkeypatch, kind, scenario
):
    adapter, _received = _adapter(monkeypatch)
    raw = _event(kind)
    body = _body(kind)
    if scenario == "bounded":
        raw["content"]["body"] = "Untrusted description " * 1000
        body = (
            f"[emote by https://matrix.to/#/{SENDER}] " + raw["content"]["body"]
            if kind == "emote"
            else "[sticker: " + raw["content"]["body"].strip() + "]"
        ).strip()
    if scenario == "oversize":
        adapter._max_media_bytes = len(PNG) - 1
    if scenario == "missing-room-key":
        raw.update(
            type="m.room.encrypted",
            content={
                "algorithm": "m.megolm.v1.aes-sha2",
                "session_id": "missing",
                "ciphertext": "unavailable",
            },
        )

    async def request(_method, path, **_kwargs):
        return raw if "/event/" in path else {"chunk": []}

    adapter._client.api.request.side_effect = request
    read = await read_matrix_context(
        adapter, "event", ROOM, "$native", 1, requester=SENDER
    )
    if scenario == "missing-room-key":
        assert read == {
            "events": [],
            "errors": [{"event_id": "$native", "error": "missing decryption keys"}],
            "skipped": 0,
        }
        parsed = await history_entry(
            adapter._client, raw, adapter._event_context_cache, ROOM,
            before=adapter._event_context_cache.history_entry(ROOM, "$native"),
        )
        assert parsed is not None
        assert parsed[0] == MatrixEventContext(
            SENDER, "", state_error="missing decryption keys", event_id="$native",
        )
        return
    assert read == {
        "events": [
            {
                "event_id": "$native",
                "sender": SENDER,
                "body": body[:1200],
                "msgtype": "m.emote" if kind == "emote" else "m.sticker",
                "thread_id": "$root",
                "timestamp": 1000000,
                "sender_authorized": True,
            }
        ],
        "errors": [],
        "skipped": 0,
    }
    parsed = await history_entry(
        adapter._client, raw, adapter._event_context_cache, ROOM,
        before=adapter._event_context_cache.history_entry(ROOM, "$native"),
    )
    assert parsed is not None and parsed[0].text == body
    parent = await adapter._event_context_cache.resolve(
        adapter._client, ROOM, "$native", adapter._cache_quoted_image
    )
    assert parent is not None and parent.text == body
    assert bool(parent.media_path) is (kind == "sticker" and scenario != "oversize")
    reply = await adapter._build_inbound_event(
        ROOM,
        SENDER,
        "$question",
        "question",
        {"msgtype": "m.text", "body": "question"},
        {"m.in_reply_to": {"event_id": "$native"}},
    )
    assert reply is not None
    snapshot = await adapter.fetch_inbound_context(reply)
    if scenario == "withdrawn":
        await adapter._on_redaction(SimpleNamespace(room_id=ROOM, redacts="$native"))
    await snapshot.refresh()
    assert snapshot.reply_event(reply).reply_to_text == (
        "[redacted]" if scenario == "withdrawn" else body
    )
    assert bool(snapshot.reply_image_paths()) is (
        kind == "sticker" and scenario not in {"withdrawn", "oversize"}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,kinds",
    [
        ("queued", ("sticker", "sticker")),
        ("queued-duplicate", ("sticker", "sticker")),
        ("queued", ("text", "sticker")),
        ("queued", ("sticker", "emote")),
        ("queued-text", ("emote", "emote")),
        ("debounce", ("emote", "emote")),
        ("debounce-duplicate", ("emote", "emote")),
        ("debounce", ("text", "emote")),
        ("busy-debounce", ("emote", "emote")),
    ],
)
@pytest.mark.parametrize("withdrawn", [0, 1])
@pytest.mark.parametrize("mode", ["native", "text"])
async def test_coalesced_native_content_revalidates_each_authored_event(
    monkeypatch, tmp_path, method, kinds, withdrawn, mode
):
    adapter, received = _adapter(monkeypatch)
    duplicate = method.endswith("-duplicate")
    method = method.removesuffix("-duplicate")
    admitted = []
    for index, kind in enumerate(kinds):
        raw = _event(kind)
        raw["event_id"] = f"$native{index}"
        raw["content"]["body"] = f"Authored contribution {0 if duplicate else index}"
        if kind == "text":
            raw["type"] = "m.room.message"
            raw["content"]["msgtype"] = "m.text"
        incoming = _typed(raw)
        await adapter._on_room_message(incoming)
        admitted.append(received.await_args.args[0])
    expected_text = [event.text for event in admitted]
    expected_paths = [list(event.authored_media().media_urls) for event in admitted]
    if method in {"debounce", "busy-debounce"}:
        blocked = asyncio.Event()

        async def pause_flush(*_args):
            await blocked.wait()

        if method == "debounce":
            monkeypatch.setattr(adapter, "_flush_text_batch", pause_flush)
            for event in admitted:
                adapter._enqueue_text_event(event)
            event = next(iter(adapter._pending_text_batches.values()))
            tasks = tuple(adapter._pending_text_batch_tasks.values())
        else:
            monkeypatch.setattr(adapter, "_flush_text_debounce", pause_flush)
            for event in admitted:
                await adapter._queue_text_debounce("session", event)
            state = adapter._text_debounce_store()["session"]
            event = state.event
            tasks = (state.task,)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    else:
        pending = {}
        for event in admitted:
            merge_pending_message_event(
                pending, "session", event, merge_text=method == "queued-text"
            )
        event = pending["session"]
    all_paths = [path for paths in expected_paths for path in paths]
    runner = _runner(adapter, tmp_path)
    monkeypatch.setattr(runner, "_decide_image_input_mode", lambda **_kwargs: mode)

    async def withdraw():
        await adapter._on_redaction(
            SimpleNamespace(room_id=ROOM, redacts=f"$native{withdrawn}")
        )

    async def analyse(_text, paths):
        await withdraw()
        return "\n\n".join(f"<pixels of {path}>" for path in paths)

    if mode == "text" and all_paths:
        monkeypatch.setattr(runner, "_enrich_message_with_vision", analyse)
    else:
        await withdraw()
    prepared = await runner._prepare_inbound_message_text(
        event=event, source=event.source, history=[{}], session_key="session"
    )
    removed = kinds[withdrawn] != "text"
    retained = [
        path
        for index, paths in enumerate(expected_paths)
        if index != withdrawn or not removed
        for path in paths
    ]
    assert {
        "visible": [text in prepared for text in expected_text],
        "paths": event._prepared_inbound.retained_image_paths(all_paths),
        "descriptions": [f"<pixels of {path}>" in prepared for path in all_paths],
        "redacted": "[redacted]" in prepared,
    } == {
        "visible": [
            duplicate or index != withdrawn or not removed for index in range(2)
        ],
        "paths": retained,
        "descriptions": [mode == "text" and path in retained for path in all_paths],
        "redacted": removed,
    }


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
@pytest.mark.parametrize("mode", ["native", "text"])
async def test_withdrawn_sticker_leaves_the_input_of_a_routed_profile(
    monkeypatch, tmp_path, two_homes, mode
):
    launch, routed = two_homes
    adapter, received = _adapter(monkeypatch)
    runner = _runner(adapter, tmp_path)
    monkeypatch.setattr(runner, "_decide_image_input_mode", lambda **_kwargs: mode)

    async def analyse(_text, paths):
        return "\n\n".join(f"<pixels of {path}>" for path in paths)

    monkeypatch.setattr(runner, "_enrich_message_with_vision", analyse)
    turns = []
    expected = []
    for turn, home in enumerate((launch, routed, launch)):
        pending = {}
        for index in range(2):
            raw = _event("sticker")
            raw["event_id"] = f"$native{turn}.{index}"
            await adapter._on_room_message(_typed(raw))
            merge_pending_message_event(pending, "session", received.await_args.args[0])
        event = pending["session"]
        await adapter._on_redaction(
            SimpleNamespace(room_id=ROOM, redacts=f"$native{turn}.1")
        )
        session_key = f"session{turn}"
        runner._session_state(session_key)
        with _profile_runtime_scope(home):
            prepared = await runner._prepare_inbound_message_text(
                event=event, source=event.source, history=[{}], session_key=session_key
            )
        kept, withdrawn = event.media_urls
        turns.append(
            {
                "cache": [Path(path).parent == home / "cache" / "images" for path in (kept, withdrawn)],
                "pixels": runner._session_state(session_key).persistent.native_image_paths,
                "descriptions": [f"<pixels of {path}>" in prepared for path in (kept, withdrawn)],
            }
        )
        expected.append(
            {
                "cache": [True, True],
                "pixels": [kept] if mode == "native" else [],
                "descriptions": [mode == "text", False],
            }
        )

    assert turns == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["emote", "sticker"])
async def test_unchanged_effective_read_preserves_mention_stripped_native_input(
    monkeypatch, kind
):
    adapter, received = _adapter(monkeypatch)
    adapter._require_mention = True
    raw = _event(kind)
    raw["content"].update(
        body="@hermes:example.org Friendly fox",
        **{"m.mentions": {"user_ids": ["@hermes:example.org"]}},
    )
    incoming = _typed(raw)
    await adapter._on_room_message(incoming)
    event = received.await_args.args[0]
    original = {"text": event.text, "paths": list(event.authored_media().media_urls)}

    async def request(_method, path, **_kwargs):
        return raw if "/event/" in path else {"chunk": []}

    adapter._client.api.request.side_effect = request
    await read_matrix_context(adapter, "event", ROOM, "$native", 1, requester=SENDER)
    snapshot = await adapter.fetch_inbound_context(event)
    await snapshot.refresh()
    assert {
        "text": snapshot.authored_text(event.text),
        "paths": snapshot.media_event(event).media_urls,
    } == original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sender",
    [
        "@file:example.org",
        "@url:example.org",
        "@diff-bot:example.org",
        "@historical.@file:example.org",
    ],
)
async def test_emote_attribution_is_not_a_context_reference(
    monkeypatch, tmp_path, sender
):
    (tmp_path / "example.org").write_text("local file")
    adapter, received = _adapter(monkeypatch)
    raw = _event("emote")
    raw["sender"] = sender
    await adapter._on_room_message(_typed(raw))
    text = received.await_args.args[0].text
    result = await preprocess_context_references_async(
        text,
        cwd=tmp_path,
        context_length=100_000,
        url_fetcher=lambda _url: "fetched page",
        allowed_root=tmp_path,
    )
    assert (result.expanded, result.references, result.message) == (False, [], text)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["emote", "sticker"])
async def test_reply_to_unadmitted_native_event_quotes_only_its_own_text(
    monkeypatch, kind
):
    adapter, received = _adapter(monkeypatch)
    adapter._require_mention = True
    bob = "@bob:example.org"
    raw = _event(kind)
    raw["sender"] = bob
    raw["content"]["body"] = (
        "> <@carol:example.org> private grandparent\n\n" + raw["content"]["body"]
    )
    raw["content"]["m.relates_to"] = {"m.in_reply_to": {"event_id": "$grandparent"}}
    await adapter._on_room_message(_typed(raw))
    assert received.await_count == 0
    await adapter._on_room_message(_typed({
        "type": "m.room.message",
        "room_id": ROOM,
        "sender": SENDER,
        "event_id": "$question",
        "origin_server_ts": 1000000,
        "content": {
            "msgtype": "m.text",
            "body": "@hermes:example.org what was that?",
            "m.mentions": {"user_ids": ["@hermes:example.org"]},
            "m.relates_to": {"m.in_reply_to": {"event_id": "$native"}},
        },
    }))
    assert received.await_args.args[0].reply_to_text == _body(kind, bob)


@pytest.mark.asyncio
@pytest.mark.parametrize("withdrawn", [False, True])
async def test_withdrawal_replaces_the_message_not_an_identical_reply_quote(
    monkeypatch, tmp_path, withdrawn
):
    adapter, received = _adapter(monkeypatch)
    adapter._event_context_cache.store(
        ROOM, "$parent", MatrixEventContext("@bob:example.org", _body("sticker"))
    )
    raw = _event("sticker")
    raw["content"]["m.relates_to"] = {"m.in_reply_to": {"event_id": "$parent"}}
    await adapter._on_room_message(_typed(raw))
    event = received.await_args.args[0]
    if withdrawn:
        await adapter._on_redaction(SimpleNamespace(room_id=ROOM, redacts="$native"))
    runner = _runner(adapter, tmp_path)
    monkeypatch.setattr(runner, "_decide_image_input_mode", lambda **_kwargs: "native")

    prepared = await runner._prepare_inbound_message_text(
        event=event, source=event.source, history=[{}], session_key="session"
    )

    message = "[redacted]" if withdrawn else _body("sticker")
    assert prepared == f'[Replying to Alice: "{_body("sticker")}"]\n\n{message}'


@pytest.mark.asyncio
@pytest.mark.parametrize("encrypted", [False, True])
async def test_room_read_shows_stickers(monkeypatch, encrypted):
    adapter, _received = _adapter(monkeypatch)
    sticker = _event("sticker")
    wire = (
        {
            **sticker,
            "type": "m.room.encrypted",
            "content": {
                "algorithm": "m.megolm.v1.aes-sha2",
                "session_id": "session",
                "ciphertext": "sticker",
                "m.relates_to": sticker["content"]["m.relates_to"],
            },
        }
        if encrypted
        else sticker
    )

    async def decrypt(_client, _raw):
        return _typed(sticker)

    async def request(_method, path, **_kwargs):
        return {"chunk": [wire] if path.endswith("/messages") else []}

    monkeypatch.setattr(
        "plugins.platforms.matrix.effective_event.decrypt_history_event", decrypt
    )
    adapter._client.api.request.side_effect = request
    adapter._client.sync_store = SimpleNamespace(
        get_next_batch=AsyncMock(return_value="s1")
    )

    read = await read_matrix_context(adapter, "room", ROOM, None, 5, requester=SENDER)

    assert read == {
        "events": [
            {
                "event_id": "$native",
                "sender": SENDER,
                "body": _body("sticker"),
                "msgtype": "m.sticker",
                "thread_id": "$root",
                "timestamp": 1000000,
                "sender_authorized": True,
            }
        ],
        "errors": [],
        "skipped": 0,
    }
