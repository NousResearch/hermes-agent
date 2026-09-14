"""Post-stream ``MEDIA:`` delivery must honour the ``SendResult`` contract.

``GatewayRunner._deliver_media_from_response`` runs AFTER streaming already
delivered the visible text. It uploads every explicit ``MEDIA:`` file through
``adapter.send_document`` / ``send_voice`` / ``send_video`` /
``send_multiple_images``. Those calls return a ``SendResult`` and, by
contract, report most platform-side failures as ``SendResult(success=False)``
WITHOUT raising (Telegram: "Not connected", missing file, capped FloodWait;
Discord: message accepted but nothing attached, ...).

The post-stream loop only guarded against exceptions, so a
``success=False`` result was indistinguishable from a delivered file: the text
was already on screen, the tag was already stripped, and nobody told the user
the attachment never arrived (``TEXT_DELIVERED=YES`` /
``DOCUMENT_DELIVERED=NO`` with no failure path executed).

The non-streaming loop in ``gateway/platforms/base.py``
(``_deliver_media_attachments._send_one``) already checks ``result.success``
and calls ``_notify_media_delivery_failure`` (#66797). These tests pin the
same contract on the post-stream lane and on the queued lane that reuses it.

Scope note: #62361 (Telegram upload retries) and #91335 (notify from the
``except Exception`` branches) cover the EXCEPTION failure shape. This file
covers the SENDRESULT failure shape those leave open. No network, no tokens.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

SIMULATED_ERROR = "simulated document delivery failure"


def _failure() -> SendResult:
    return SendResult(success=False, error=SIMULATED_ERROR)


def _event(platform=Platform.TELEGRAM) -> MessageEvent:
    source = SessionSource(platform=platform, chat_id="67890", chat_type="dm", thread_id=None)
    return MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="msg-1")


def _fake_runner(thread_meta=None):
    runner = SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: thread_meta,
        _reply_anchor_for_event=lambda event: None,
    )
    # The queued lane calls ``self._deliver_media_from_response``; bind the real one.
    runner._deliver_media_from_response = (
        lambda *a, **k: GatewayRunner._deliver_media_from_response(runner, *a, **k))
    return runner


def _fake_adapter(*, document=None, voice=None, video=None, images=None, with_notify=True):
    ok = lambda mid: SendResult(success=True, message_id=mid)  # noqa: E731
    adapter = SimpleNamespace(
        name="fake",
        platform=Platform.TELEGRAM,
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        send=AsyncMock(return_value=ok("text")),
        send_document=AsyncMock(return_value=document if document is not None else ok("doc")),
        send_voice=AsyncMock(return_value=voice if voice is not None else ok("voice")),
        send_video=AsyncMock(return_value=video if video is not None else ok("video")),
        send_multiple_images=AsyncMock(return_value=images if images is not None else ok("imgs")),
    )
    if with_notify:
        adapter._notify_media_delivery_failure = AsyncMock(return_value=None)
    return adapter


def _allowed_media(tmp_path, monkeypatch, name: str):
    """A real file under an allowed media root. ``tmp_path`` stands in for ``/tmp``
    (``MEDIA_DELIVERY_SAFE_ROOTS`` is the gate the production path validator uses)."""
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    return media_file.resolve()


# ---------------------------------------------------------------------------
# The regression: success=False without an exception on the post-stream lane
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_stream_send_document_success_false_runs_failure_path(tmp_path, monkeypatch, caplog):
    """``send_document`` returns ``SendResult(success=False)`` (no exception): the
    failure path must run — the user gets the notice and the log says why.
    Before the fix this was silently treated as a delivered document."""
    pdf = _allowed_media(tmp_path, monkeypatch, "report.pdf")
    adapter = _fake_adapter(document=_failure())
    thread_meta = {"thread_id": "t-1"}

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await GatewayRunner._deliver_media_from_response(
            _fake_runner(thread_meta), f"Here is the report.\nMEDIA:{pdf}", _event(), adapter)

    # 1. the upload was attempted through the adapter contract
    adapter.send_document.assert_awaited_once()
    assert adapter.send_document.await_args.kwargs["file_path"] == str(pdf)
    assert adapter.send_document.await_args.kwargs["chat_id"] == "67890"
    # 2. + 4. success=False is NOT a confirmed delivery: Hermes' existing failure
    #    infrastructure (#66797) is invoked for exactly this file, on the same thread
    adapter._notify_media_delivery_failure.assert_awaited_once()
    args, kwargs = adapter._notify_media_delivery_failure.await_args
    assert args[0] == "67890"
    assert args[1] == str(pdf)
    assert kwargs.get("is_voice") is False
    assert kwargs.get("metadata") == thread_meta
    # 3. the failure is observable in the gateway log with the adapter's reason
    assert any(SIMULATED_ERROR in rec.getMessage() for rec in caplog.records), caplog.text


@pytest.mark.asyncio
async def test_post_stream_send_document_success_true_does_not_notify(tmp_path, monkeypatch):
    """A delivered document must not trigger the failure path (no false alarm)."""
    pdf = _allowed_media(tmp_path, monkeypatch, "report.pdf")
    adapter = _fake_adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"Here is the report.\nMEDIA:{pdf}", _event(), adapter)

    adapter.send_document.assert_awaited_once()
    adapter._notify_media_delivery_failure.assert_not_awaited()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_stream_real_base_adapter_sends_user_visible_notice(tmp_path, monkeypatch):
    """End-to-end through the real ``BasePlatformAdapter`` notifier: the user sees
    the "couldn't deliver" notice instead of nothing. This is the observable
    Hermes failure path, not a mock of it."""
    pdf = _allowed_media(tmp_path, monkeypatch, "report.pdf")

    class _DocFailsAdapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM)
            self.sent: list = []
            self.document_calls = 0

        async def connect(self, *, is_reconnect: bool = False) -> bool:
            return True

        async def disconnect(self) -> None:
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
            self.sent.append(content)
            return SendResult(success=True, message_id=f"m{len(self.sent)}")

        async def send_typing(self, chat_id: str, metadata=None) -> None:
            return None

        async def get_chat_info(self, chat_id: str):
            return {"id": chat_id}

        async def send_document(self, chat_id, file_path, caption=None, file_name=None,
                                reply_to=None, metadata=None, **kwargs) -> SendResult:
            self.document_calls += 1
            return _failure()

    adapter = _DocFailsAdapter()
    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"Here is the report.\nMEDIA:{pdf}", _event(), adapter)

    assert adapter.document_calls == 1
    assert len(adapter.sent) == 1, adapter.sent
    assert "Couldn't deliver the file attachment" in adapter.sent[0]
    assert "report.pdf" in adapter.sent[0]
    # the host path is never echoed into the chat
    assert str(pdf) not in adapter.sent[0]


@pytest.mark.asyncio
async def test_post_stream_legacy_none_result_is_tolerated(tmp_path, monkeypatch):
    """Legacy adapters may return ``None`` from media senders: neither a crash nor
    a failure notice (``None`` is "unknown", the same lenience ``record_delivery``
    and the cron sender apply)."""
    pdf = _allowed_media(tmp_path, monkeypatch, "report.pdf")
    adapter = _fake_adapter()
    adapter.send_document = AsyncMock(return_value=None)

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"MEDIA:{pdf}", _event(), adapter)

    adapter.send_document.assert_awaited_once()
    adapter._notify_media_delivery_failure.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_stream_adapter_without_notifier_logs_and_survives(tmp_path, monkeypatch, caplog):
    """An adapter that does not inherit the base notifier still gets the failure
    logged, and the loop does not blow up on a missing attribute."""
    pdf = _allowed_media(tmp_path, monkeypatch, "report.pdf")
    adapter = _fake_adapter(document=_failure(), with_notify=False)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await GatewayRunner._deliver_media_from_response(
            _fake_runner({}), f"MEDIA:{pdf}", _event(), adapter)

    adapter.send_document.assert_awaited_once()
    assert any(SIMULATED_ERROR in rec.getMessage() for rec in caplog.records), caplog.text


@pytest.mark.asyncio
async def test_post_stream_failure_on_one_file_does_not_stop_the_next(tmp_path, monkeypatch):
    """Per-file handling: the second attachment still goes out after the first fails."""
    pdf = _allowed_media(tmp_path, monkeypatch, "report.pdf")
    csv = pdf.parent / "data.csv"
    csv.write_bytes(b"a,b\n1,2\n")
    adapter = _fake_adapter()
    adapter.send_document = AsyncMock(side_effect=[_failure(), SendResult(success=True, message_id="doc2")])

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"MEDIA:{pdf}\nMEDIA:{csv.resolve()}", _event(), adapter)

    assert adapter.send_document.await_count == 2
    adapter._notify_media_delivery_failure.assert_awaited_once()
    assert adapter._notify_media_delivery_failure.await_args.args[1] == str(pdf)


@pytest.mark.parametrize(
    "name, sender, kwarg, is_voice",
    [
        ("clip.mp4", "send_video", "video_path", False),
        ("note.mp3", "send_voice", "audio_path", False),
    ],
)
@pytest.mark.asyncio
async def test_post_stream_video_and_voice_success_false_notify(tmp_path, monkeypatch, name, sender, kwarg, is_voice):
    """Same contract for the sibling senders on the post-stream lane."""
    media = _allowed_media(tmp_path, monkeypatch, name)
    adapter = _fake_adapter(**{sender.removeprefix("send_"): _failure()})

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"MEDIA:{media}", _event(), adapter)

    mock = getattr(adapter, sender)
    mock.assert_awaited_once()
    assert mock.await_args.kwargs[kwarg] == str(media)
    adapter._notify_media_delivery_failure.assert_awaited_once()
    assert adapter._notify_media_delivery_failure.await_args.args[1] == str(media)
    assert adapter._notify_media_delivery_failure.await_args.kwargs.get("is_voice") is is_voice


def _image_batch(tmp_path, monkeypatch, count: int):
    first = _allowed_media(tmp_path, monkeypatch, "chart01.png")
    paths = [first] + [first.parent / f"chart{i:02d}.png" for i in range(2, count + 1)]
    for p in paths[1:]:
        p.write_bytes(b"\x89PNG fake")
    return [p.resolve() for p in paths]


@pytest.mark.asyncio
async def test_post_stream_image_batch_success_false_notifies_exactly_once(tmp_path, monkeypatch, caplog):
    """``send_multiple_images`` is ONE logical operation: a failed batch of 10 images
    yields exactly one failure notice (not one per image), and the log names the batch."""
    pngs = _image_batch(tmp_path, monkeypatch, 10)
    adapter = _fake_adapter(images=_failure())

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await GatewayRunner._deliver_media_from_response(
            _fake_runner({}), "\n".join(f"MEDIA:{p}" for p in pngs), _event(), adapter)

    adapter.send_multiple_images.assert_awaited_once()
    assert len(adapter.send_multiple_images.await_args.kwargs["images"]) == 10
    adapter._notify_media_delivery_failure.assert_awaited_once()
    assert adapter._notify_media_delivery_failure.await_args.args[1] == str(pngs[0])
    assert any("batch of 10" in rec.getMessage() for rec in caplog.records), caplog.text
    assert any(SIMULATED_ERROR in rec.getMessage() for rec in caplog.records), caplog.text


@pytest.mark.parametrize("batch_result", [None, SendResult(success=True, message_id="imgs")],
                         ids=["legacy-none", "success-true"])
@pytest.mark.asyncio
async def test_post_stream_image_batch_none_or_success_does_not_notify(tmp_path, monkeypatch, batch_result):
    """A delivered batch, or a legacy adapter returning ``None``, never triggers the notice."""
    pngs = _image_batch(tmp_path, monkeypatch, 3)
    adapter = _fake_adapter()
    adapter.send_multiple_images = AsyncMock(return_value=batch_result)

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), "\n".join(f"MEDIA:{p}" for p in pngs), _event(), adapter)

    adapter.send_multiple_images.assert_awaited_once()
    adapter._notify_media_delivery_failure.assert_not_awaited()


# ---------------------------------------------------------------------------
# Failure shapes the SendResult handler must NOT own
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_stream_exception_takes_except_branch_not_sendresult_handler(tmp_path, monkeypatch, caplog):
    """A raised exception is the EXCEPTION shape: it stays in the pre-existing ``except``
    branch (warning log, loop continues with the next file) and never reaches the
    ``success=False`` handler. Whether the ``except`` branch itself should also notify is
    #91335's call, so this test pins only that the two branches are exclusive."""
    pdf = _allowed_media(tmp_path, monkeypatch, "report.pdf")
    csv = pdf.parent / "data.csv"
    csv.write_bytes(b"a,b\n1,2\n")
    adapter = _fake_adapter()
    adapter.send_document = AsyncMock(
        side_effect=[RuntimeError("simulated transport crash"), SendResult(success=True, message_id="doc2")])
    handler_spy = AsyncMock(return_value=None)
    monkeypatch.setattr("gateway.run_notifications._report_media_send_failure", handler_spy)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await GatewayRunner._deliver_media_from_response(
            _fake_runner({}), f"MEDIA:{pdf}\nMEDIA:{csv.resolve()}", _event(), adapter)

    # the exception did not escape and the next file was still attempted
    assert adapter.send_document.await_count == 2
    assert adapter.send_document.await_args_list[1].kwargs["file_path"] == str(csv.resolve())
    # the pre-existing warning is preserved ...
    assert any("Post-stream media delivery failed" in rec.getMessage()
               and "simulated transport crash" in rec.getMessage() for rec in caplog.records), caplog.text
    # ... and the SendResult(success=False) handler was never entered for it
    handler_spy.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_stream_result_without_success_attribute_is_unknown_not_failure(tmp_path, monkeypatch):
    """An object with no ``success`` attribute is "unknown" like ``None``: no notice."""
    pdf = _allowed_media(tmp_path, monkeypatch, "report.pdf")
    adapter = _fake_adapter()
    adapter.send_document = AsyncMock(return_value=object())

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"MEDIA:{pdf}", _event(), adapter)

    adapter.send_document.assert_awaited_once()
    adapter._notify_media_delivery_failure.assert_not_awaited()


# ---------------------------------------------------------------------------
# Queued lane: ``_deliver_queued_first_response`` reuses the post-stream loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queued_lane_media_success_false_runs_failure_path(tmp_path, monkeypatch):
    """A queued follow-up whose text already streamed delivers media through the
    same loop, so it inherits the same contract."""
    pdf = _allowed_media(tmp_path, monkeypatch, "report.pdf")
    adapter = _fake_adapter(document=_failure())
    runner = _fake_runner({})
    source = _event().source

    await GatewayRunner._deliver_queued_first_response(
        runner, f"Done.\nMEDIA:{pdf}", source, adapter,
        metadata={"thread_id": "t-9"}, event_message_id=None,
        text_already_delivered=True, deliver_media=True,
    )

    adapter.send_document.assert_awaited_once()
    adapter._notify_media_delivery_failure.assert_awaited_once()
    assert adapter._notify_media_delivery_failure.await_args.args[1] == str(pdf)
    assert adapter._notify_media_delivery_failure.await_args.kwargs.get("metadata") == {"thread_id": "t-9"}
