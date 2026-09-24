"""Regression coverage for media emitted in interim commentary (#99420).

Interim assistant commentary is display-cleaned before it reaches the platform
(``GatewayStreamConsumer._clean_for_display`` strips deliverable ``MEDIA:`` tags),
so a file the model attaches mid-turn — "the report is ready, sending it now" with
a ``MEDIA:`` directive in commentary — was stripped from every display lane and
never delivered: only the final response was rescanned by the post-turn
attachment pass.

The fix captures raw interim payloads that contain ``MEDIA:`` in ``TurnRunner``
(``ctx.interim_media_responses``), carries them out on the turn result, and folds
them with the final response's directives into ONE post-turn delivery once the
turn succeeded — deduplicated by path, never on failed/interrupted turns.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_turn import _build_interim_media_delivery_payload
from gateway.session import SessionSource


def _event():
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123CHAN",
        chat_type="group",
        thread_id=None,
    )
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=source,
        message_id="171.000001",
    )


def _gate_self(deliver=None):
    """Minimal ``self`` for calling ``_hmwa_deliver_turn_response`` unbound.

    ``_delivery_adapter_for`` returns a real-extract adapter; the REAL voice-reply
    probe is bound (no voice mode configured → False) so the test isolates the
    media rail through production code paths, and the post-turn media delivery
    is an injected async mock.
    """
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
    )
    ns = SimpleNamespace(
        _delivery_adapter_for=lambda source: adapter,
        _voice_mode={},
        _voice_key_for_source=lambda source: "k",
        _deliver_media_from_response=deliver,
    )
    ns._should_send_voice_reply = GatewayRunner._should_send_voice_reply.__get__(ns)
    return ns


def _allowed_media_path(tmp_path, monkeypatch, name):
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"media")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (root,),
    )
    return media_file.resolve()


def test_interim_media_is_preserved_and_final_duplicate_is_deduplicated(tmp_path):
    interim_path = tmp_path / "interim.png"
    final_path = tmp_path / "final.png"
    interim_path.write_bytes(b"interim")
    final_path.write_bytes(b"final")

    payload, cleaned_final = _build_interim_media_delivery_payload(
        [f"first\nMEDIA:{interim_path}"],
        f"done\nMEDIA:{interim_path}\nMEDIA:{final_path}",
        BasePlatformAdapter,
    )

    assert payload is not None
    media, _cleaned_payload = BasePlatformAdapter.extract_media(payload)
    assert media == [
        (str(interim_path), False),
        (str(final_path), False),
    ]
    assert cleaned_final == "done"


def test_final_only_media_keeps_established_delivery_rail(tmp_path):
    final_path = tmp_path / "final.png"
    final_path.write_bytes(b"final")
    final_response = f"done\nMEDIA:{final_path}"

    payload, cleaned_final = _build_interim_media_delivery_payload(
        ["commentary without media"],
        final_response,
        BasePlatformAdapter,
    )

    assert payload is None
    assert cleaned_final == final_response


def test_voice_and_document_directives_survive_the_fold(tmp_path):
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio")

    payload, _cleaned = _build_interim_media_delivery_payload(
        ["[[audio_as_voice]]\nMEDIA:" + str(audio_path)],
        "here\nMEDIA:" + str(audio_path),
        BasePlatformAdapter,
    )

    assert payload is not None
    assert payload.splitlines()[0] == "[[audio_as_voice]]"
    media, _cleaned_payload = BasePlatformAdapter.extract_media(payload)
    assert media == [(str(audio_path), True)]


def test_invalid_interim_media_only_keeps_final_untouched():
    # A payload whose MEDIA tags extract to nothing (unknown-extension path that
    # does not exist) must not trigger the special rail at all.
    final_response = "done\nMEDIA:/tmp/definitely-final.png"
    payload, cleaned_final = _build_interim_media_delivery_payload(
        ["see MEDIA:/nonexistent/example.zzz for format"],
        final_response,
        BasePlatformAdapter,
    )

    assert payload is None
    assert cleaned_final == final_response


@pytest.mark.asyncio
async def test_successful_turn_delivers_interim_media_once_and_cleans_final(
    tmp_path, monkeypatch
):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.pdf")
    deliver = AsyncMock(return_value=SendResult(success=True, message_id="m"))
    returned = await GatewayRunner._hmwa_deliver_turn_response(
        _gate_self(deliver), _event(), None, None, "sess", 1,
        {"completed": True, "interim_media_responses": [
            f"Building the report now\nMEDIA:{media_file}"]},
        [], f"done\nMEDIA:{media_file}", None, False,
    )

    deliver.assert_awaited_once()
    payload = deliver.await_args.args[0]
    media, _cleaned = BasePlatformAdapter.extract_media(payload)
    assert media == [(str(media_file), False)]
    # The visible final must not reprint the directive the payload delivers.
    assert returned == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("result_extra", [
    {"failed": True},
    {"interrupted": True},
    {"completed": False},
], ids=["failed", "interrupted", "not_completed"])
async def test_unsuccessful_turns_never_publish_interim_media(
    tmp_path, monkeypatch, result_extra
):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.pdf")
    final = f"done\nMEDIA:{media_file}"
    deliver = AsyncMock(return_value=SendResult(success=True, message_id="m"))
    returned = await GatewayRunner._hmwa_deliver_turn_response(
        _gate_self(deliver), _event(), None, None, "sess", 1,
        {"completed": True, **result_extra,
         "interim_media_responses": [f"working\nMEDIA:{media_file}"]},
        [], final, None, False,
    )

    deliver.assert_not_awaited()
    # Failed/interrupted turns keep the response untouched (the failure text
    # itself is still delivered by the caller; interim media is not).
    assert returned == final


@pytest.mark.asyncio
async def test_no_interim_media_leaves_final_only_path_untouched():
    deliver = AsyncMock(return_value=SendResult(success=True, message_id="m"))
    final = "done\nMEDIA:/tmp/never-built.png"
    returned = await GatewayRunner._hmwa_deliver_turn_response(
        _gate_self(deliver), _event(), None, None, "sess", 1,
        {"completed": True, "interim_media_responses": []},
        [], final, None, False,
    )

    deliver.assert_not_awaited()
    assert returned == final
