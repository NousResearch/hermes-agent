"""Every silent media-send failure path notifies the user (port of #91335).

The non-streaming dispatch loop strips ``MEDIA:`` tags from the visible reply
*before* the attachment is uploaded, so a dropped attachment is invisible:
neither the user nor the model gets any signal that the file never arrived, and
a model that fabricated or mistyped a path keeps repeating the same broken tag
with no feedback loop.

``_notify_media_delivery_failure()`` is the established notice for exactly that
case (see ``emit_media_warning``). Several handlers still only ``logger.warning``
(or ``suppress(Exception)``) the failure, which is a silent drop. This module
pins that those handlers notify too, on every delivery path that can drop media:

    - ``gateway/platforms/base.py``: ``send_multiple_images`` (per-image
      ``success=False`` and exception), the attachment queue in
      ``_deliver_media_attachments`` (exception), ``_send_image_batch``
      (batch exception, per image).
    - ``gateway/platforms/weixin.py``: the outbound media loop in ``send()``.
    - ``gateway/run_notifications.py``: post-stream image batch + post-stream
      media.
    - ``gateway/run_turn.py``: background-task image and media delivery.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

NOTICE_MARKER = "Couldn't deliver"


def _allowed_media_path(tmp_path, monkeypatch, name):
    """A media path that passes ``filter_media_delivery_paths`` (safe-root pinned)."""
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"media")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    return media_file.resolve()


def _event(chat_id="C123CHAN"):
    source = SessionSource(
        platform=Platform.SLACK, chat_id=chat_id, chat_type="group", thread_id=None,
    )
    return MessageEvent(
        text="hi", message_type=MessageType.TEXT, source=source, message_id="171.000001",
    )


class _StubAdapter(BasePlatformAdapter):
    """Minimal adapter whose media sends fail on demand and whose notices are recorded."""

    name = "stub"
    platform = Platform.SLACK

    def __init__(self, *, mode="return", error="no attachment"):
        self.mode = mode
        self.error = error
        self.notices = []
        self.sent_images = []
        self.sent_documents = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, **kwargs):
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {}

    async def emit_warning(self, chat_id, content, *, reply_to=None, metadata=None,
                           logical_platform=None):
        self.notices.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return SendResult(success=True)

    def _fail(self):
        if self.mode == "raise":
            raise RuntimeError(self.error)
        return SendResult(success=False, error=self.error)

    async def send_image(self, chat_id, image_url, caption=None, **kwargs):
        self.sent_images.append(image_url)
        return self._fail()

    async def send_animation(self, chat_id, animation_url, caption=None, **kwargs):
        return self._fail()

    async def send_image_file(self, chat_id, image_path, caption=None, **kwargs):
        return self._fail()

    async def send_document(self, chat_id, file_path, **kwargs):
        self.sent_documents.append(file_path)
        return self._fail()

    async def send_voice(self, chat_id, audio_path, caption=None, **kwargs):
        # defined here so the base class's voice-downgrade notice never fires: this
        # suite asserts on _notify_media_delivery_failure's notice only.
        self.sent_documents.append(audio_path)
        return self._fail()


def _notices(adapter):
    return [n for n in adapter.notices if NOTICE_MARKER in n["content"]]


# ---------------------------------------------------------------------------
# gateway/platforms/base.py — send_multiple_images
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_multiple_images_notifies_when_a_send_reports_failure():
    """A ``success=False`` per-image send is a silent drop: the tag is already stripped."""
    adapter = _StubAdapter(mode="return")

    result = await adapter.send_multiple_images("chat1", [("file:///tmp/report.png", "alt")])

    assert result.success is False
    notices = _notices(adapter)
    assert len(notices) == 1
    assert notices[0]["chat_id"] == "chat1"
    assert "report.png" in notices[0]["content"]


@pytest.mark.asyncio
async def test_send_multiple_images_notifies_when_a_send_raises():
    """An exception per image must not be swallowed either."""
    adapter = _StubAdapter(mode="raise")

    await adapter.send_multiple_images("chat1", [
        ("file:///tmp/one.png", "a"), ("file:///tmp/two.png", "b"),
    ])

    notices = _notices(adapter)
    assert len(notices) == 2
    assert {n["content"].split("(")[-1].rstrip(").") for n in notices} == {"one.png", "two.png"}


@pytest.mark.asyncio
async def test_send_multiple_images_still_reports_success_without_notice():
    """The happy path stays quiet — one notice per dropped attachment, none otherwise."""
    adapter = _StubAdapter(mode="return")
    adapter.send_image = AsyncMock(return_value=SendResult(success=True, message_id="m1"))

    result = await adapter.send_multiple_images("chat1", [("https://x.invalid/ok.png", "alt")])

    assert result.success is True
    assert _notices(adapter) == []


# ---------------------------------------------------------------------------
# gateway/platforms/base.py — _deliver_media_attachments / _send_image_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_media_queue_exception_notifies_with_voice_flag():
    """A raising ``send_voice`` in the attachment queue must notify (is_voice preserved)."""
    adapter = _StubAdapter(mode="raise")
    recorded = []

    await adapter._deliver_media_attachments(
        _event(), [("/tmp/note.ogg", True)], [], force_document_attachments=False,
        human_delay=0.0, metadata={"thread_id": "t1"}, record_delivery=recorded.append,
    )

    notices = _notices(adapter)
    assert len(notices) == 1
    assert notices[0]["chat_id"] == "C123CHAN"
    # audio routing is decided from the extension: the notice names the attachment kind
    # (the audio notice carries no file name, unlike the generic-file one)
    assert "audio attachment" in notices[0]["content"]
    assert recorded and recorded[0].success is False


@pytest.mark.asyncio
async def test_local_file_queue_exception_notifies():
    """Bare local files (no MEDIA tag) go through the same queue and must notify too."""
    adapter = _StubAdapter(mode="raise")

    await adapter._deliver_media_attachments(
        _event(), [], ["/tmp/invoice.pdf"], force_document_attachments=False,
        human_delay=0.0, metadata={"thread_id": "t1"}, record_delivery=lambda _r: None,
    )

    notices = _notices(adapter)
    assert len(notices) == 1
    assert "invoice.pdf" in notices[0]["content"]


@pytest.mark.asyncio
async def test_image_batch_exception_notifies_every_image():
    """A batch that raises delivers nothing — each image gets its own notice."""
    adapter = _StubAdapter(mode="return")
    adapter.send_multiple_images = AsyncMock(side_effect=RuntimeError("batch exploded"))

    await adapter._send_image_batch(
        _event(), [("file:///tmp/a.png", ""), ("file:///tmp/b.png", "")], {"thread_id": "t1"},
        0.0, lambda _r: None,
    )

    notices = _notices(adapter)
    assert len(notices) == 2
    assert {"a.png", "b.png"} == {n["content"].split("(")[-1].rstrip(").") for n in notices}


# ---------------------------------------------------------------------------
# gateway/platforms/weixin.py — outbound media loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_weixin_media_delivery_failure_notifies(tmp_path, monkeypatch):
    """Weixin's media loop logged the failure and moved on — the user never learned."""
    from gateway.platforms import weixin as wx

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    (tmp_path / "config.yaml").write_text("{}")
    media_file = _allowed_media_path(tmp_path, monkeypatch, "notes.txt")

    adapter = wx.WeixinAdapter(
        PlatformConfig(token="fixture-token", extra={"account_id": "fixture"}))
    adapter._send_session = object()
    adapter.emit_warning = AsyncMock(return_value=SendResult(success=True))
    monkeypatch.setattr(adapter, "send_document", AsyncMock(side_effect=RuntimeError("boom")))

    result = await adapter.send("recipient", f"MEDIA:{media_file}")

    assert result.success is True  # the text path is unaffected by a media failure
    adapter.emit_warning.assert_awaited()
    chat_id, content = adapter.emit_warning.await_args.args[:2]
    assert chat_id == "recipient"
    assert NOTICE_MARKER in content


# ---------------------------------------------------------------------------
# gateway/run_notifications.py — post-stream delivery
# ---------------------------------------------------------------------------


def _runner_stub(thread_meta):
    return SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: thread_meta,
        _reply_anchor_for_event=lambda event: None,
    )


def _post_stream_adapter(**overrides):
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        _notify_media_delivery_failure=AsyncMock(),
        send_voice=AsyncMock(return_value=SendResult(success=True)),
        send_document=AsyncMock(return_value=SendResult(success=True)),
        send_video=AsyncMock(return_value=SendResult(success=True)),
        send_multiple_images=AsyncMock(return_value=SendResult(success=True)),
    )
    adapter.__dict__.update(overrides)
    return adapter


@pytest.mark.asyncio
async def test_post_stream_image_batch_exception_notifies(tmp_path, monkeypatch):
    """Post-stream image batches swallowed the exception: nothing arrived, no notice."""
    media_file = _allowed_media_path(tmp_path, monkeypatch, "chart.png")
    adapter = _post_stream_adapter(
        send_multiple_images=AsyncMock(side_effect=RuntimeError("batch exploded")))

    await GatewayRunner._deliver_media_from_response(
        _runner_stub({}), f"Here is the chart.\nMEDIA:{media_file}", _event(), adapter)

    adapter._notify_media_delivery_failure.assert_awaited()
    args = adapter._notify_media_delivery_failure.await_args.args
    assert args[0] == "C123CHAN" and "chart.png" in str(args[1])


@pytest.mark.asyncio
async def test_post_stream_media_exception_notifies(tmp_path, monkeypatch):
    """Same for a non-image MEDIA attachment delivered post-stream."""
    media_file = _allowed_media_path(tmp_path, monkeypatch, "notes.txt")
    adapter = _post_stream_adapter(
        send_document=AsyncMock(side_effect=RuntimeError("upload rejected")))

    await GatewayRunner._deliver_media_from_response(
        _runner_stub({}), f"Attached.\nMEDIA:{media_file}", _event(), adapter)

    adapter._notify_media_delivery_failure.assert_awaited()
    args = adapter._notify_media_delivery_failure.await_args.args
    assert args[0] == "C123CHAN" and "notes.txt" in str(args[1])


# ---------------------------------------------------------------------------
# gateway/run_turn.py — background-task delivery
# ---------------------------------------------------------------------------


def _bare_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._background_tasks = set()
    store = MagicMock()
    store.get_model_override.return_value = None
    runner.session_store = store
    from gateway.hooks import HookRegistry
    runner.hooks = HookRegistry()
    return runner


@pytest.mark.asyncio
async def test_background_task_media_failures_notify(tmp_path, monkeypatch):
    """``suppress(Exception)`` on the background-task delivery paths hid failures entirely."""
    from unittest.mock import patch

    image_file = _allowed_media_path(tmp_path, monkeypatch, "p.png")
    document_file = _allowed_media_path(tmp_path, monkeypatch, "x.pdf")
    runner = _bare_runner()
    adapter = AsyncMock()
    adapter.send = AsyncMock()
    adapter.extract_media = MagicMock(return_value=([(str(document_file), False)], "done"))
    adapter.extract_images = MagicMock(return_value=([(f"file://{image_file}", "alt")], "done"))
    adapter.send_image = AsyncMock(side_effect=RuntimeError("image failed"))
    adapter.send_document = AsyncMock(side_effect=RuntimeError("document failed"))
    runner.adapters[Platform.TELEGRAM] = adapter

    source = SessionSource(
        platform=Platform.TELEGRAM, user_id="12345", chat_id="67890", user_name="testuser")

    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
         patch("gateway.run._load_gateway_config", return_value={}), \
         patch("run_agent.AIAgent") as MockAgent:
        instance = MagicMock()
        instance.shutdown_memory_provider = MagicMock()
        instance.close = MagicMock()
        instance.run_conversation.return_value = {"final_response": "done", "messages": []}
        MockAgent.return_value = instance

        await runner._run_background_task("say hello", source, "bg_test")

    notified = adapter._notify_media_delivery_failure.await_args_list
    assert notified, "background-task media failures must notify"
    paths = [str(call.args[1]) for call in notified]
    assert any("p.png" in p for p in paths) and any("x.pdf" in p for p in paths)
