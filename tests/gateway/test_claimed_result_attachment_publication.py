"""RED-first complete-response ownership for claimed-result attachments.

The rejected c110cd6c boundary durably owned only the visible text. These
regressions drive the real base/queued publication pipelines and require a
claimed result to remain non-terminal until every intended response part is
acknowledged.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    DeliveryOwnedReply,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.fixture(autouse=True)
def _isolated_ledger_and_media(tmp_path, monkeypatch):
    home = tmp_path / "isolated-hermes"
    home.mkdir()
    media_root = tmp_path / "media"
    media_root.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (media_root,),
    )
    yield media_root


class _OwnedMediaAdapter(BasePlatformAdapter):  # type: ignore[misc]
    """Concrete adapter that drives real extraction and publication code."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.calls: list[tuple[str, str]] = []
        self.part_metadata: list[str | None] = []
        self.outcomes: dict[str, list[object]] = {}
        self._get_human_delay = lambda: 0.0

    async def connect(self, *, is_reconnect: bool = False):  # pragma: no cover
        return True

    async def disconnect(self):  # pragma: no cover
        return None

    async def get_chat_info(self, chat_id):  # pragma: no cover
        return None

    def queue(self, kind: str, *outcomes: object) -> None:
        self.outcomes.setdefault(kind, []).extend(outcomes)

    async def _result(
        self, kind: str, private_value: str, metadata=None
    ) -> SendResult:
        self.calls.append((kind, private_value))
        self.part_metadata.append(
            None if metadata is None else metadata.get("_hermes_delivery_part_id")
        )
        queue = self.outcomes.get(kind) or []
        outcome = queue.pop(0) if queue else SendResult(
            success=True, message_id=f"{kind}-{len(self.calls)}"
        )
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, SendResult)
        return outcome

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return await self._result("text", content, metadata)

    async def send_image(self, chat_id, image_url, caption=None, **kwargs):
        return await self._result("image", image_url, kwargs.get("metadata"))

    async def send_image_file(self, chat_id, image_path, caption=None, **kwargs):
        return await self._result("image", image_path, kwargs.get("metadata"))

    async def send_voice(self, chat_id, audio_path, **kwargs):
        return await self._result("voice", audio_path, kwargs.get("metadata"))

    async def send_video(self, chat_id, video_path, **kwargs):
        return await self._result("video", video_path, kwargs.get("metadata"))

    async def send_document(self, chat_id, file_path, **kwargs):
        return await self._result("document", file_path, kwargs.get("metadata"))


def _source(chat_id: str = "claimed-media") -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        chat_id=chat_id,
        chat_type="channel",
        thread_id="claimed-thread",
    )


def _event(source: SessionSource) -> MessageEvent:
    return MessageEvent(
        text="continue",
        message_type=MessageType.TEXT,
        source=source,
        message_id="claimed-media-head",
        goal_continuation=True,
    )


def _record_owned(
    source: SessionSource,
    response: str,
    *,
    content: str,
    suffix: str,
) -> str:
    source_payload = source.to_dict()
    source_payload["is_bot"] = bool(source.is_bot)
    source_payload["role_authorized"] = False
    return dl.record_claimed_result(
        session_key=f"agent:main:slack:channel:{source.chat_id}",
        claim_id=f"claim-{suffix}",
        claim_event_id=f"event-{suffix}",
        platform=source.platform.value,
        chat_id=source.chat_id,
        thread_id=source.thread_id,
        content=content,
        raw_content=response,
        source_json=json.dumps(
            source_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        message_ref="claimed-media-head",
        adapter_profile=None,
    )


async def _run_owned(
    adapter: _OwnedMediaAdapter,
    source: SessionSource,
    response: str,
    obligation_id: str,
) -> None:
    session_key = f"agent:main:slack:channel:{source.chat_id}"
    adapter._message_handler = AsyncMock(
        return_value=DeliveryOwnedReply(response, obligation_id)
    )
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter._process_message_background(_event(source), session_key)


def _state(obligation_id: str) -> str:
    with dl._connect() as conn:
        return conn.execute(
            "SELECT state FROM delivery_obligations WHERE obligation_id=?",
            (obligation_id,),
        ).fetchone()[0]


def _media_file(root: Path, name: str) -> Path:
    path = root / name
    path.write_bytes(b"claimed-media")
    return path.resolve()


@pytest.mark.asyncio
async def test_text_success_then_image_failure_keeps_whole_response_failed(
    _isolated_ledger_and_media,
):
    image = _media_file(_isolated_ledger_and_media, "chart.png")
    response = f"owned text\nMEDIA:{image}"
    source = _source("image-failure")
    oid = _record_owned(source, response, content="owned text", suffix="image-failure")
    adapter = _OwnedMediaAdapter()
    adapter.queue("image", SendResult(success=False, error="upload rejected"))

    await _run_owned(adapter, source, response, oid)

    assert [kind for kind, _ in adapter.calls] == ["text", "image"]
    assert _state(oid) == "failed"
    assert [(row["kind"], row["state"]) for row in dl.get_claimed_result_parts(oid)] == [
        ("text", "delivered"),
        ("image", "failed"),
    ]


@pytest.mark.asyncio
async def test_text_success_then_image_crash_never_marks_whole_response_delivered(
    _isolated_ledger_and_media,
):
    image = _media_file(_isolated_ledger_and_media, "crash.png")
    response = f"owned text\nMEDIA:{image}"
    source = _source("image-crash")
    oid = _record_owned(source, response, content="owned text", suffix="image-crash")
    adapter = _OwnedMediaAdapter()
    adapter.queue("image", SystemExit(71))

    with pytest.raises(SystemExit, match="71"):
        await _run_owned(adapter, source, response, oid)

    assert _state(oid) == "attempting"
    assert [(row["kind"], row["state"]) for row in dl.get_claimed_result_parts(oid)] == [
        ("text", "delivered"),
        ("image", "attempting"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "prefix", "expected_kind"),
    [
        ("reply.ogg", "[[audio_as_voice]]\nMEDIA:", "voice"),
        ("reply.mp4", "MEDIA:", "video"),
        ("reply.pdf", "MEDIA:", "document"),
        ("reply.zip", "", "document"),
    ],
)
async def test_text_success_then_non_image_part_failure_is_not_terminal(
    _isolated_ledger_and_media,
    name,
    prefix,
    expected_kind,
):
    media = _media_file(_isolated_ledger_and_media, name)
    response = f"owned text\n{prefix}{media}"
    source = _source(f"failure-{name}")
    oid = _record_owned(source, response, content="owned text", suffix=name)
    adapter = _OwnedMediaAdapter()
    adapter.queue(expected_kind, SendResult(success=False, error="upload rejected"))

    await _run_owned(adapter, source, response, oid)

    assert [kind for kind, _ in adapter.calls] == ["text", expected_kind]
    assert _state(oid) == "failed"
    assert [row["state"] for row in dl.get_claimed_result_parts(oid)] == [
        "delivered",
        "failed",
    ]


@pytest.mark.asyncio
async def test_media_only_claimed_result_reaches_delivered_after_attachment_ack(
    _isolated_ledger_and_media,
):
    video = _media_file(_isolated_ledger_and_media, "only.mp4")
    response = f"MEDIA:{video}"
    source = _source("media-only-success")
    oid = _record_owned(source, response, content="", suffix="media-only-success")
    adapter = _OwnedMediaAdapter()

    await _run_owned(adapter, source, response, oid)

    assert adapter.calls == [("video", str(video))]
    assert _state(oid) == "delivered"
    assert [(row["kind"], row["state"]) for row in dl.get_claimed_result_parts(oid)] == [
        ("video", "delivered")
    ]


@pytest.mark.asyncio
async def test_media_only_claimed_failure_is_failed_not_false_success(
    _isolated_ledger_and_media,
):
    document = _media_file(_isolated_ledger_and_media, "only.pdf")
    response = f"MEDIA:{document}"
    source = _source("media-only-failure")
    oid = _record_owned(source, response, content="", suffix="media-only-failure")
    adapter = _OwnedMediaAdapter()
    adapter.queue("document", SendResult(success=False, error="upload rejected"))

    await _run_owned(adapter, source, response, oid)

    assert adapter.calls == [("document", str(document))]
    assert _state(oid) == "failed"
    assert dl.get_claimed_result_parts(oid)[0]["state"] == "failed"


@pytest.mark.asyncio
async def test_recovery_replays_only_incomplete_parts_after_acknowledged_parts(
    _isolated_ledger_and_media,
):
    first = _media_file(_isolated_ledger_and_media, "first.pdf")
    second = _media_file(_isolated_ledger_and_media, "second.pdf")
    response = f"owned text\nMEDIA:{first}\nMEDIA:{second}"
    source = _source("partial-recovery")
    oid = _record_owned(source, response, content="owned text", suffix="partial-recovery")
    first_adapter = _OwnedMediaAdapter()
    first_adapter.queue(
        "document",
        SendResult(success=True, message_id="first-ack"),
        SendResult(success=False, error="second rejected"),
    )

    await _run_owned(first_adapter, source, response, oid)

    assert _state(oid) == "failed"
    assert [row["state"] for row in dl.get_claimed_result_parts(oid)] == [
        "delivered",
        "delivered",
        "failed",
    ]
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1 "
            "WHERE obligation_id=?",
            (oid,),
        )
    claimed = dl.sweep_recoverable()
    assert len(claimed) == 1

    replay_event = GatewayRunner._claimed_result_replay_event(claimed[0])
    second_adapter = _OwnedMediaAdapter()
    second_adapter._message_handler = AsyncMock(
        side_effect=AssertionError("recovery must not invoke the agent")
    )
    session_key = f"agent:main:slack:channel:{source.chat_id}"
    second_adapter._active_sessions[session_key] = asyncio.Event()
    await second_adapter._process_message_background(replay_event, session_key)

    assert second_adapter.calls == [("document", str(second))]
    assert _state(oid) == "delivered"
    assert {row["state"] for row in dl.get_claimed_result_parts(oid)} == {"delivered"}


@pytest.mark.asyncio
async def test_queued_raw_media_uses_attachment_method_and_propagates_failure(
    _isolated_ledger_and_media,
):
    document = _media_file(_isolated_ledger_and_media, "queued.pdf")
    response = f"queued text\nMEDIA:{document}"
    source = _source("queued-raw-media")
    oid = _record_owned(source, response, content="queued text", suffix="queued-raw-media")
    adapter = _OwnedMediaAdapter()
    adapter.queue("document", SendResult(success=False, error="queued rejected"))
    runner = object.__new__(GatewayRunner)
    runner._session_key_for_source = lambda _source: (
        f"agent:main:slack:channel:{source.chat_id}"
    )
    runner._thread_metadata_for_source = lambda *_args: {}
    runner._reply_anchor_for_event = lambda *_args: None

    with pytest.raises(Exception, match="claimed continuation attachment delivery failed"):
        await runner._deliver_queued_first_response(
            response,
            source=source,
            adapter=adapter,
            metadata={},
            delivery_obligation_id=oid,
        )

    assert [kind for kind, _ in adapter.calls] == ["text", "document"]
    assert _state(oid) == "failed"


@pytest.mark.asyncio
async def test_queued_bare_local_file_is_manifested_without_path_disclosure(
    _isolated_ledger_and_media,
):
    document = _media_file(_isolated_ledger_and_media, "queued-bare.pdf")
    response = f"queued text\n{document}"
    source = _source("queued-bare-local")
    oid = _record_owned(source, response, content="queued text", suffix="queued-bare")
    adapter = _OwnedMediaAdapter()
    runner = object.__new__(GatewayRunner)
    runner._session_key_for_source = lambda _source: (
        f"agent:main:slack:channel:{source.chat_id}"
    )
    runner._thread_metadata_for_source = lambda *_args: {}
    runner._reply_anchor_for_event = lambda *_args: None

    await runner._deliver_queued_first_response(
        response,
        source=source,
        adapter=adapter,
        metadata={},
        delivery_obligation_id=oid,
    )

    assert adapter.calls == [("text", "queued text"), ("document", str(document))]
    assert [(row["kind"], row["state"]) for row in dl.get_claimed_result_parts(oid)] == [
        ("text", "delivered"),
        ("document", "delivered"),
    ]
    with dl._connect() as conn:
        durable_content = conn.execute(
            "SELECT content FROM delivery_obligations WHERE obligation_id=?",
            (oid,),
        ).fetchone()[0]
    assert durable_content == "queued text"
    assert str(document) not in durable_content
    assert str(document) not in adapter.calls[0][1]


@pytest.mark.asyncio
async def test_queued_attachment_snapshot_survives_file_disappearing_after_extraction(
    _isolated_ledger_and_media,
):
    document = _media_file(
        _isolated_ledger_and_media,
        "queued-disappearing.pdf",
    )
    response = f"queued text\n{document}"
    source = _source("queued-disappearing-local")
    oid = _record_owned(
        source,
        response,
        content="queued text",
        suffix="queued-disappearing",
    )
    adapter = _OwnedMediaAdapter()
    extraction_calls = {"media": 0, "images": 0, "local": 0}
    extract_media = adapter.extract_media
    extract_images = adapter.extract_images
    extract_local_files = adapter.extract_local_files

    def _extract_media_once(content, **kwargs):
        extraction_calls["media"] += 1
        return extract_media(content, **kwargs)

    def _extract_images_once(content):
        extraction_calls["images"] += 1
        return extract_images(content)

    def _extract_local_files_once(content, **kwargs):
        extraction_calls["local"] += 1
        paths, cleaned = extract_local_files(content, **kwargs)
        if extraction_calls["local"] == 1:
            document.unlink()
        return paths, cleaned

    adapter.extract_media = _extract_media_once
    adapter.extract_images = _extract_images_once
    adapter.extract_local_files = _extract_local_files_once
    runner = object.__new__(GatewayRunner)
    runner._session_key_for_source = lambda _source: (
        f"agent:main:slack:channel:{source.chat_id}"
    )
    runner._thread_metadata_for_source = lambda *_args: {}
    runner._reply_anchor_for_event = lambda *_args: None

    with pytest.raises(
        Exception,
        match="claimed continuation attachment delivery failed",
    ):
        await runner._deliver_queued_first_response(
            response,
            source=source,
            adapter=adapter,
            metadata={},
            delivery_obligation_id=oid,
        )

    assert extraction_calls == {"media": 1, "images": 1, "local": 1}
    assert adapter.calls == [("text", "queued text")]
    assert _state(oid) == "failed"
    assert [
        (row["kind"], row["state"])
        for row in dl.get_claimed_result_parts(oid)
    ] == [("text", "delivered"), ("document", "failed")]
    with dl._connect() as conn:
        durable_content = conn.execute(
            "SELECT content FROM delivery_obligations WHERE obligation_id=?",
            (oid,),
        ).fetchone()[0]
        durable_parts = repr(
            conn.execute(
                "SELECT part_id, part_ordinal, kind, state, last_error, remote_receipt "
                "FROM delivery_obligation_parts WHERE obligation_id=?",
                (oid,),
            ).fetchall()
        )
    assert durable_content == "queued text"
    assert str(document) not in durable_content
    assert str(document) not in durable_parts
    assert "claimed-media" not in durable_parts


@pytest.mark.asyncio
async def test_unavailable_queued_attachment_recovery_retries_the_same_durable_part(
    _isolated_ledger_and_media,
):
    document = _media_file(
        _isolated_ledger_and_media,
        "queued-recovery.pdf",
    )
    response = f"queued text\n{document}"
    source = _source("queued-unavailable-recovery")
    oid = _record_owned(
        source,
        response,
        content="queued text",
        suffix="queued-unavailable-recovery",
    )
    first_adapter = _OwnedMediaAdapter()
    extract_local_files = first_adapter.extract_local_files

    def _extract_then_remove(content, **kwargs):
        paths, cleaned = extract_local_files(content, **kwargs)
        document.unlink(missing_ok=True)
        return paths, cleaned

    first_adapter.extract_local_files = _extract_then_remove
    runner = object.__new__(GatewayRunner)
    runner._session_key_for_source = lambda _source: (
        f"agent:main:slack:channel:{source.chat_id}"
    )
    runner._thread_metadata_for_source = lambda *_args: {}
    runner._reply_anchor_for_event = lambda *_args: None

    with pytest.raises(
        Exception,
        match="claimed continuation attachment delivery failed",
    ):
        await runner._deliver_queued_first_response(
            response,
            source=source,
            adapter=first_adapter,
            metadata={},
            delivery_obligation_id=oid,
        )

    original_parts = dl.get_claimed_result_parts(oid)
    assert [(row["kind"], row["state"]) for row in original_parts] == [
        ("text", "delivered"),
        ("document", "failed"),
    ]

    def _claim_after_owner_exit():
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=999999999, "
                "owner_started_at=1 WHERE obligation_id=?",
                (oid,),
            )
        claimed = dl.sweep_recoverable()
        assert len(claimed) == 1
        return GatewayRunner._claimed_result_replay_event(claimed[0])

    missing_replay = _claim_after_owner_exit()
    missing_adapter = _OwnedMediaAdapter()
    missing_adapter._message_handler = AsyncMock(
        side_effect=AssertionError("recovery must not invoke the agent")
    )
    session_key = f"agent:main:slack:channel:{source.chat_id}"
    missing_adapter._active_sessions[session_key] = asyncio.Event()
    await missing_adapter._process_message_background(missing_replay, session_key)

    retry_parts = dl.get_claimed_result_parts(oid)
    assert [row["part_id"] for row in retry_parts] == [
        row["part_id"] for row in original_parts
    ]
    assert [(row["kind"], row["state"]) for row in retry_parts] == [
        ("text", "delivered"),
        ("document", "failed"),
    ]
    assert missing_adapter.calls == []
    with dl._connect() as conn:
        durable_content = conn.execute(
            "SELECT content FROM delivery_obligations WHERE obligation_id=?",
            (oid,),
        ).fetchone()[0]
    assert durable_content == "queued text"
    assert str(document) not in durable_content

    document.write_bytes(b"claimed-media")
    available_replay = _claim_after_owner_exit()
    available_adapter = _OwnedMediaAdapter()
    available_adapter._message_handler = AsyncMock(
        side_effect=AssertionError("recovery must not invoke the agent")
    )
    available_adapter._active_sessions[session_key] = asyncio.Event()
    await available_adapter._process_message_background(available_replay, session_key)

    assert available_adapter.calls == [("document", str(document))]
    assert _state(oid) == "delivered"
    assert [row["part_id"] for row in dl.get_claimed_result_parts(oid)] == [
        row["part_id"] for row in original_parts
    ]
    assert {row["state"] for row in dl.get_claimed_result_parts(oid)} == {"delivered"}


@pytest.mark.asyncio
async def test_queued_unsafe_attachment_is_manifested_failed_not_false_terminal(
    _isolated_ledger_and_media,
    monkeypatch,
):
    from gateway.platforms import base as base_module

    unsafe = _isolated_ledger_and_media.parent / "PRIVATE_BLOCKED_BYTES.pdf"
    unsafe.write_bytes(b"PRIVATE_BLOCKED_BYTES")
    unsafe = unsafe.resolve()
    original_denied = base_module._path_under_denied_prefix
    monkeypatch.setattr(
        base_module,
        "_path_under_denied_prefix",
        lambda path: Path(path) == unsafe or original_denied(path),
    )
    response = f"queued text\n{unsafe}"
    source = _source("queued-unsafe-local")
    oid = _record_owned(
        source,
        response,
        content="queued text",
        suffix="queued-unsafe",
    )
    adapter = _OwnedMediaAdapter()
    runner = object.__new__(GatewayRunner)
    runner._session_key_for_source = lambda _source: (
        f"agent:main:slack:channel:{source.chat_id}"
    )
    runner._thread_metadata_for_source = lambda *_args: {}
    runner._reply_anchor_for_event = lambda *_args: None

    with pytest.raises(
        Exception,
        match="claimed continuation attachment delivery failed",
    ):
        await runner._deliver_queued_first_response(
            response,
            source=source,
            adapter=adapter,
            metadata={},
            delivery_obligation_id=oid,
        )

    assert adapter.calls == [("text", "queued text")]
    assert _state(oid) == "failed"
    assert [
        (row["kind"], row["state"])
        for row in dl.get_claimed_result_parts(oid)
    ] == [("text", "delivered"), ("document", "failed")]
    with dl._connect() as conn:
        durable_content = conn.execute(
            "SELECT content FROM delivery_obligations WHERE obligation_id=?",
            (oid,),
        ).fetchone()[0]
        durable_parts = repr(
            conn.execute(
                "SELECT part_id, part_ordinal, kind, state, last_error, remote_receipt "
                "FROM delivery_obligation_parts WHERE obligation_id=?",
                (oid,),
            ).fetchall()
        )
    assert durable_content == "queued text"
    assert str(unsafe) not in durable_content
    assert str(unsafe) not in durable_parts
    assert "PRIVATE_BLOCKED_BYTES" not in durable_parts


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["Caddyfile", "payload.unsupported"])
async def test_unavailable_explicit_media_without_supported_extension_is_retryable(
    _isolated_ledger_and_media,
    name,
):
    intended = (_isolated_ledger_and_media / name).resolve()
    assert not intended.exists()
    response = f"queued text\nMEDIA:{intended}"
    source = _source(f"missing-explicit-{name}")
    oid = _record_owned(
        source,
        response,
        content="queued text",
        suffix=f"missing-explicit-{name}",
    )
    adapter = _OwnedMediaAdapter()
    runner = object.__new__(GatewayRunner)
    runner._session_key_for_source = lambda _source: (
        f"agent:main:slack:channel:{source.chat_id}"
    )

    with pytest.raises(
        Exception,
        match="claimed continuation attachment delivery failed",
    ):
        await runner._deliver_queued_first_response(
            response,
            source=source,
            adapter=adapter,
            metadata={},
            delivery_obligation_id=oid,
        )

    assert adapter.calls == [("text", "queued text")]
    assert _state(oid) == "failed"
    assert [
        (row["kind"], row["state"])
        for row in dl.get_claimed_result_parts(oid)
    ] == [("text", "delivered"), ("document", "failed")]
    with dl._connect() as conn:
        provider_facing = repr(
            conn.execute(
                "SELECT content, last_error FROM delivery_obligations "
                "WHERE obligation_id=?",
                (oid,),
            ).fetchone()
        )
        part_metadata = repr(
            conn.execute(
                "SELECT part_id, part_ordinal, kind, state, last_error, "
                "remote_receipt FROM delivery_obligation_parts "
                "WHERE obligation_id=?",
                (oid,),
            ).fetchall()
        )
    assert str(intended) not in provider_facing
    assert str(intended) not in part_metadata


@pytest.mark.asyncio
async def test_denied_extensionless_media_is_manifested_before_validation(
    _isolated_ledger_and_media,
    monkeypatch,
):
    from gateway.platforms import base as base_module

    denied = _media_file(
        _isolated_ledger_and_media.parent,
        "PRIVATE_DENIED_EXTENSIONLESS",
    )
    original_denied = base_module._path_under_denied_prefix
    monkeypatch.setattr(
        base_module,
        "_path_under_denied_prefix",
        lambda path: Path(path) == denied or original_denied(path),
    )
    validation_calls = []
    original_validation = base_module.validate_media_delivery_path

    def _track_delivery_validation(path, session_key=""):
        validation_calls.append((str(path), session_key))
        return original_validation(path, session_key=session_key)

    monkeypatch.setattr(
        base_module,
        "validate_media_delivery_path",
        _track_delivery_validation,
    )
    response = f"queued text\nMEDIA:{denied}"
    source = _source("denied-explicit-extensionless")
    oid = _record_owned(
        source,
        response,
        content="queued text",
        suffix="denied-explicit-extensionless",
    )
    adapter = _OwnedMediaAdapter()
    runner = object.__new__(GatewayRunner)
    runner._session_key_for_source = lambda _source: (
        f"agent:main:slack:channel:{source.chat_id}"
    )

    with pytest.raises(
        Exception,
        match="claimed continuation attachment delivery failed",
    ):
        await runner._deliver_queued_first_response(
            response,
            source=source,
            adapter=adapter,
            metadata={},
            delivery_obligation_id=oid,
        )

    assert adapter.calls == [("text", "queued text")]
    assert _state(oid) == "failed"
    assert [
        (row["kind"], row["state"])
        for row in dl.get_claimed_result_parts(oid)
    ] == [("text", "delivered"), ("document", "failed")]
    assert validation_calls == [
        (
            str(denied),
            f"agent:main:slack:channel:{source.chat_id}",
        )
    ]


@pytest.mark.asyncio
async def test_extensionless_media_disappearing_after_snapshot_remains_incomplete(
    _isolated_ledger_and_media,
):
    intended = _media_file(_isolated_ledger_and_media, "DISAPPEARING_EXTENSIONLESS")
    response = f"queued text\nMEDIA:{intended}"
    source = _source("disappearing-explicit-extensionless")
    oid = _record_owned(
        source,
        response,
        content="queued text",
        suffix="disappearing-explicit-extensionless",
    )
    adapter = _OwnedMediaAdapter()
    extract_media = adapter.extract_media

    def _extract_then_remove(content, **kwargs):
        media, cleaned = extract_media(content, **kwargs)
        intended.unlink()
        return media, cleaned

    adapter.extract_media = _extract_then_remove
    runner = object.__new__(GatewayRunner)
    runner._session_key_for_source = lambda _source: (
        f"agent:main:slack:channel:{source.chat_id}"
    )

    with pytest.raises(
        Exception,
        match="claimed continuation attachment delivery failed",
    ):
        await runner._deliver_queued_first_response(
            response,
            source=source,
            adapter=adapter,
            metadata={},
            delivery_obligation_id=oid,
        )

    assert adapter.calls == [("text", "queued text")]
    assert [
        (row["kind"], row["state"])
        for row in dl.get_claimed_result_parts(oid)
    ] == [("text", "delivered"), ("document", "failed")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "protected",
    [
        "Use `MEDIA:{path}` as an example.",
        "```text\nMEDIA:{path}\n```",
        "> MEDIA:{path}",
        '{{"result":"MEDIA:{path}"}}',
    ],
)
async def test_protected_unavailable_extensionless_media_remains_text(
    _isolated_ledger_and_media,
    protected,
):
    intended = (_isolated_ledger_and_media / "PROTECTED_EXTENSIONLESS").resolve()
    response = protected.format(path=intended)
    source = _source("protected-explicit-extensionless")
    oid = _record_owned(
        source,
        response,
        content=response,
        suffix="protected-explicit-extensionless",
    )
    adapter = _OwnedMediaAdapter()
    runner = object.__new__(GatewayRunner)
    runner._session_key_for_source = lambda _source: (
        f"agent:main:slack:channel:{source.chat_id}"
    )

    await runner._deliver_queued_first_response(
        response,
        source=source,
        adapter=adapter,
        metadata={},
        delivery_obligation_id=oid,
    )

    assert adapter.calls == [("text", response)]
    assert [(row["kind"], row["state"]) for row in dl.get_claimed_result_parts(oid)] == [
        ("text", "delivered")
    ]


def test_claimed_part_manifest_is_bounded_private_and_pruned(
    _isolated_ledger_and_media,
    monkeypatch,
):
    private = _media_file(_isolated_ledger_and_media, "PRIVATE_SENTINEL.pdf")
    source = _source("part-bounds")
    oid = _record_owned(source, f"MEDIA:{private}", content="", suffix="part-bounds")
    part_id = dl.compute_claimed_result_part_id(
        oid, 0, "document", str(private)
    )
    dl.prepare_claimed_result_delivery(
        oid,
        session_key=f"agent:main:slack:channel:{source.chat_id}",
        platform=source.platform.value,
        chat_id=source.chat_id,
        thread_id=source.thread_id,
        content="",
        adapter_profile=None,
    )
    dl.register_claimed_result_parts(oid, [(part_id, "document")])

    with dl._connect() as conn:
        persisted = repr(
            conn.execute(
                "SELECT part_id, part_ordinal, kind, state, last_error "
                "FROM delivery_obligation_parts WHERE obligation_id=?",
                (oid,),
            ).fetchall()
        )
    assert str(private) not in persisted
    assert "PRIVATE_SENTINEL" not in persisted

    with pytest.raises(dl.DeliveryObligationCapacityError):
        dl.register_claimed_result_parts(
            oid,
            [
                (
                    dl.compute_claimed_result_part_id(oid, index, "document", str(index)),
                    "document",
                )
                for index in range(dl.MAX_PARTS_PER_OBLIGATION + 1)
            ],
        )

    assert dl.mark_claimed_result_part_delivered(oid, part_id) is True
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET updated_at=0 WHERE obligation_id=?",
            (oid,),
        )
    dl._prune(now=dl._RETENTION_SECONDS + 1)
    with dl._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM delivery_obligation_parts WHERE obligation_id=?",
            (oid,),
        ).fetchone()[0] == 0


@pytest.mark.asyncio
async def test_stable_part_ids_reach_each_platform_send(
    _isolated_ledger_and_media,
):
    document = _media_file(_isolated_ledger_and_media, "part-id.pdf")
    response = f"answer\nMEDIA:{document}"
    source = _source("part-id-boundary")
    oid = _record_owned(source, response, content="answer", suffix="part-id")
    adapter = _OwnedMediaAdapter()

    await _run_owned(adapter, source, response, oid)

    parts = dl.get_claimed_result_parts(oid)
    assert adapter.part_metadata == [part["part_id"] for part in parts]
    assert len(set(adapter.part_metadata)) == 2
