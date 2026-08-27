"""Regression tests for #73771 — session-wide MEDIA dedup swallowing
explicit resend requests.

The history-dedup guard in the delivery pipeline used to drop ANY
``MEDIA:<path>`` whose path appeared earlier in the session transcript —
unconditionally, silently, for the whole session lifetime. A user asking
"send me that file again" got a reply reading "here it is" with no
attachment and nothing in the logs.

Fix (salvaged from PR #74158 by @webtecnica, widened to the streaming
sibling):

* Non-streaming (``BasePlatformAdapter._process_message_background``):
  explicit MEDIA tags are NO LONGER filtered against history. Stale
  auto-appended tags are already deduped upstream in
  ``_collect_auto_append_media_tags``.
* Streaming (``GatewayRunner._deliver_media_from_response``): same filter
  removed — that rescan is explicit-only by design (#20834), so anything
  it finds is a deliberate attachment request.
* Bare local file paths (auto-detected, not explicit) KEEP the history
  dedup on the non-streaming path, and the suppression is now logged.
"""

import asyncio
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.run import GatewayRunner, _collect_auto_append_media_tags, _collect_history_media_paths
from gateway.session import SessionSource, build_session_key


class _DummyAdapter(BasePlatformAdapter):
    """Minimal BasePlatformAdapter for non-streaming dispatch tests."""

    def __init__(self, platform: Platform = Platform.DISCORD):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), platform)
        self.sent: list[dict] = []
        self.documents: list[str] = []
        self.images_sent: list[str] = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="msg-1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def send_document(self, chat_id, file_path, caption=None, file_name=None, reply_to=None, metadata=None, **kwargs) -> SendResult:
        self.documents.append(str(file_path))
        return SendResult(success=True, message_id="doc-1")

    async def send_multiple_images(self, chat_id, images, metadata=None, human_delay=0.0):
        self.images_sent.extend(str(p) for p, _cap in images)

    async def send_image_file(self, chat_id, image_path, caption=None, reply_to=None, metadata=None, **kwargs) -> SendResult:
        self.images_sent.append(str(image_path))
        return SendResult(success=True, message_id="img-1")


class _StubStore:
    """Session store stub whose transcript claims ``paths`` were already
    delivered in prior turns (one assistant message per path, followed by a
    trailing assistant message representing the current turn — which
    _history_media_paths_for_session pops)."""

    def __init__(self, paths):
        self._transcript = [{"role": "user", "content": "make files"}]
        for p in paths:
            self._transcript.append({"role": "assistant", "content": f"Done! MEDIA:{p}"})
            self._transcript.append({"role": "user", "content": "send it again please"})
        # Current turn's already-persisted assistant reply (excluded by the
        # helper), so all earlier paths stay in the dedup set.
        self._transcript.append({"role": "assistant", "content": "resending now"})

    def peek_session_id(self, session_key):
        return "sess-1"

    def load_transcript(self, session_id):
        return list(self._transcript)


def _make_event(platform: Platform = Platform.DISCORD) -> MessageEvent:
    return MessageEvent(
        text="send me that file again",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=platform, chat_id="111", chat_type="dm"),
        message_id="m1",
    )


async def _hold_typing(_chat_id, interval=2.0, metadata=None, stop_event=None):
    if stop_event is not None:
        await stop_event.wait()
    else:
        await asyncio.Event().wait()


def _allowed_file(tmp_path, monkeypatch, name: str):
    root = tmp_path / "media-cache"
    f = root / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"payload")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    return f.resolve()


# ---------------------------------------------------------------------------
# Non-streaming path (base.py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_media_resend_is_delivered_despite_history(tmp_path, monkeypatch):
    """#73771 core repro: the same MEDIA path was delivered in a prior turn;
    the model re-emits it on an explicit user request — it MUST be sent."""
    pdf = _allowed_file(tmp_path, monkeypatch, "report.pdf")
    adapter = _DummyAdapter()
    adapter._keep_typing = _hold_typing
    adapter.set_session_store(_StubStore([str(pdf)]))

    async def handler(_event):
        return f"Here it is again.\nMEDIA:{pdf}"

    adapter.set_message_handler(handler)
    event = _make_event()
    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.documents == [str(pdf)], (
        f"explicit MEDIA resend was suppressed: docs={adapter.documents} "
        f"sent={adapter.sent}"
    )
    # And the visible text still went out.
    assert any("Here it is again." in s["content"] for s in adapter.sent)


@pytest.mark.asyncio
async def test_first_delivery_not_poisoned_by_current_turn_tool_output(tmp_path, monkeypatch):
    """A MEDIA path present in the CURRENT turn's tool result used to poison
    the history set and suppress even the first-ever delivery. With the tag
    filter removed this cannot happen regardless of what history contains."""
    docx = _allowed_file(tmp_path, monkeypatch, "letter.docx")
    adapter = _DummyAdapter()
    adapter._keep_typing = _hold_typing

    class _ToolEchoStore(_StubStore):
        def __init__(self):
            self._transcript = [
                {"role": "user", "content": "make a docx"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "t1", "function": {"name": "execute_code"}}]},
                {"role": "tool", "tool_call_id": "t1",
                 "content": f"wrote MEDIA:{docx}"},
                {"role": "assistant", "content": "current turn reply"},
            ]

    adapter.set_session_store(_ToolEchoStore())

    async def handler(_event):
        return f"DOCX:\nMEDIA:{docx}"

    adapter.set_message_handler(handler)
    event = _make_event()
    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.documents == [str(docx)], (
        f"first delivery suppressed by current-turn tool echo: {adapter.sent}"
    )


@pytest.mark.asyncio
async def test_bare_local_path_history_dedup_survives_and_logs(tmp_path, monkeypatch, caplog):
    """Bare-path auto-detect (non-explicit) KEEPS the history dedup — and the
    suppression is now observable in the logs instead of silent."""
    png = _allowed_file(tmp_path, monkeypatch, "chart.png")
    monkeypatch.setattr("gateway.platforms.base.LOCAL_DELIVERY_SAFE_ROOTS", (png.parent,), raising=False)
    adapter = _DummyAdapter()
    adapter._keep_typing = _hold_typing
    adapter.set_session_store(_StubStore([str(png)]))
    # Bypass the local-path safety filter so the test pins ONLY the history
    # dedup behavior, not the safe-root policy.
    monkeypatch.setattr(
        type(adapter), "filter_local_delivery_paths", staticmethod(lambda paths, session_key="": list(paths))
    )

    async def handler(_event):
        # Bare path, no MEDIA: directive — the auto-detect lane.
        return f"The chart is at {png} by the way."

    adapter.set_message_handler(handler)
    event = _make_event()
    with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
        await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.images_sent == [] and adapter.documents == [], (
        "bare-path history dedup regressed — stale path re-uploaded"
    )
    assert any("Suppressing" in r.getMessage() for r in caplog.records), (
        "suppression must be logged (#73771 observability)"
    )


@pytest.mark.asyncio
async def test_plain_text_response_does_not_load_transcript():
    """Ordinary text delivery must not touch SQLite-backed history at all."""
    adapter = _DummyAdapter()
    adapter._keep_typing = _hold_typing

    class _ExplodingStore:
        calls = 0

        def peek_session_id(self, _session_key):
            self.calls += 1
            raise AssertionError("plain text delivery loaded session history")

    store = _ExplodingStore()
    adapter.set_session_store(store)

    async def handler(_event):
        return "Plain response with no local attachment path."

    adapter.set_message_handler(handler)
    event = _make_event()
    await adapter._process_message_background(event, build_session_key(event.source))

    assert store.calls == 0
    assert any("Plain response" in item["content"] for item in adapter.sent)


@pytest.mark.asyncio
async def test_explicit_media_response_does_not_load_transcript(tmp_path, monkeypatch):
    """Explicit MEDIA delivery must not touch SQLite-backed history."""
    pdf = _allowed_file(tmp_path, monkeypatch, "explicit-no-history.pdf")
    adapter = _DummyAdapter()
    adapter._keep_typing = _hold_typing

    class _ExplodingStore:
        calls = 0

        def peek_session_id(self, _session_key):
            self.calls += 1
            raise AssertionError("explicit MEDIA delivery loaded session history")

    store = _ExplodingStore()
    adapter.set_session_store(store)

    async def handler(_event):
        return f"Here is the file.\nMEDIA:{pdf}"

    adapter.set_message_handler(handler)
    event = _make_event()
    await adapter._process_message_background(event, build_session_key(event.source))

    assert store.calls == 0
    assert adapter.documents == [str(pdf)]


@pytest.mark.asyncio
async def test_bare_path_history_lookup_does_not_block_event_loop(tmp_path, monkeypatch):
    """A slow transcript read must run outside the platform event loop."""
    pdf = _allowed_file(tmp_path, monkeypatch, "slow-history.pdf")
    monkeypatch.setattr("gateway.platforms.base.LOCAL_DELIVERY_SAFE_ROOTS", (pdf.parent,), raising=False)
    adapter = _DummyAdapter()
    adapter._keep_typing = _hold_typing
    monkeypatch.setattr(
        type(adapter), "filter_local_delivery_paths", staticmethod(lambda paths, session_key="": list(paths))
    )

    class _GatedStore:
        def __init__(self):
            self.release = threading.Event()

        def peek_session_id(self, _session_key):
            return "sess-slow"

        def load_transcript(self, _session_id):
            # Block until the test has proven the event loop stayed
            # responsive. An Event gate (instead of a fixed sleep) makes the
            # assertion deterministic on slow/loaded CI hosts.
            self.release.wait(timeout=5)
            return [{"role": "user", "content": "current"}]

    store = _GatedStore()
    adapter.set_session_store(store)

    async def handler(_event):
        return f"Generated file: {pdf}"

    adapter.set_message_handler(handler)
    event = _make_event()
    delivery = asyncio.create_task(
        adapter._process_message_background(event, build_session_key(event.source))
    )
    await asyncio.sleep(0.02)
    # The transcript read is still parked on the gate. If load_transcript ran
    # on the event-loop thread, the sleep above could never have resumed
    # while the read was in progress — reaching this point with the delivery
    # still pending proves the read happened off-loop.
    assert not delivery.done()
    assert not adapter.documents
    store.release.set()
    await delivery
    assert adapter.documents == [str(pdf)]


@pytest.mark.asyncio
async def test_bare_path_history_lookup_timeout_fails_open(tmp_path, monkeypatch):
    """A wedged transcript read must not hold response delivery indefinitely."""
    pdf = _allowed_file(tmp_path, monkeypatch, "timeout-history.pdf")
    monkeypatch.setattr("gateway.platforms.base.LOCAL_DELIVERY_SAFE_ROOTS", (pdf.parent,), raising=False)
    monkeypatch.setattr("gateway.platforms.base._HISTORY_MEDIA_LOOKUP_TIMEOUT_SECONDS", 0.02)
    adapter = _DummyAdapter()
    adapter._keep_typing = _hold_typing
    monkeypatch.setattr(
        type(adapter), "filter_local_delivery_paths", staticmethod(lambda paths, session_key="": list(paths))
    )

    class _WedgedStore:
        def peek_session_id(self, _session_key):
            return "sess-wedged"

        def load_transcript(self, _session_id):
            time.sleep(0.2)
            return []

    adapter.set_session_store(_WedgedStore())

    async def handler(_event):
        return f"Generated file: {pdf}"

    adapter.set_message_handler(handler)
    event = _make_event()
    started = time.monotonic()
    await adapter._process_message_background(event, build_session_key(event.source))

    # The lookup times out after 0.02s and fails open; the bound only guards
    # against delivery hanging on the wedged read indefinitely. 1.0s still
    # flaked on loaded CI runners (observed 1.55s on main run 33455779041),
    # so keep it >= 5s per the flake policy. Delivery of the document below
    # is the real fail-open assertion.
    assert time.monotonic() - started < 5.0
    assert adapter.documents == [str(pdf)]


@pytest.mark.asyncio
async def test_history_lookup_saturation_fails_open_without_new_worker(monkeypatch):
    """Wedged lookups are bounded and cannot consume unbounded worker threads."""
    monkeypatch.setattr("gateway.platforms.base._HISTORY_MEDIA_LOOKUP_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(
        "gateway.platforms.base._HISTORY_MEDIA_LOOKUP_ADMISSION",
        threading.BoundedSemaphore(2),
    )
    adapter = _DummyAdapter()
    release = threading.Event()
    two_started = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def blocked_lookup(_session_key):
        nonlocal calls
        with calls_lock:
            calls += 1
            if calls == 2:
                two_started.set()
        release.wait(timeout=10)
        return None

    monkeypatch.setattr(adapter, "_history_media_paths_for_session", blocked_lookup)
    first = asyncio.create_task(adapter._bounded_history_media_paths_for_session("one"))
    second = asyncio.create_task(adapter._bounded_history_media_paths_for_session("two"))
    deadline = time.monotonic() + 5
    while not two_started.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.005)
    assert two_started.is_set()

    began = time.monotonic()
    third = await adapter._bounded_history_media_paths_for_session("three")
    elapsed = time.monotonic() - began

    assert third is None
    # Saturation must fail open immediately (no waiting on the 5.0s lookup
    # timeout); 2.0s keeps the distinction while staying flake-free on
    # loaded CI runners.
    assert elapsed < 2.0
    assert calls == 2
    release.set()
    await asyncio.gather(first, second)


@pytest.mark.asyncio
async def test_history_lookup_worker_start_failure_fails_open_and_releases_slot(
    monkeypatch, caplog
):
    """If the worker thread cannot start, fail open without leaking a permit.

    Regression: ``Thread.start()`` raising (thread exhaustion) used to
    propagate into the delivery loop AND permanently leak the admission
    permit acquired just above it, because the worker's finally-release
    never runs when the worker never starts.
    """
    admission = threading.BoundedSemaphore(1)
    monkeypatch.setattr(
        "gateway.platforms.base._HISTORY_MEDIA_LOOKUP_ADMISSION", admission
    )
    adapter = _DummyAdapter()

    def _exhausted_start(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", _exhausted_start)

    with caplog.at_level(logging.WARNING):
        first = await adapter._bounded_history_media_paths_for_session("one")
        # With only one permit, a leak on the first call would force this
        # second call down the capacity-exhausted path instead of the
        # start-failure path.
        second = await adapter._bounded_history_media_paths_for_session("two")

    assert first is None
    assert second is None
    assert "capacity exhausted" not in caplog.text
    assert (
        caplog.text.count("Could not start media-delivery history lookup worker")
        == 2
    )
    # The permit was returned both times: it is still acquirable.
    assert admission.acquire(blocking=False)
    admission.release()


# ---------------------------------------------------------------------------
# Streaming sibling (run.py _deliver_media_from_response)
# ---------------------------------------------------------------------------


def _stream_event():
    return MessageEvent(
        text="send it again",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="group"),
        message_id="171.1",
    )


def _stream_adapter():
    return SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="v")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="d")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="i")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="vid")),
        send_multiple_images=AsyncMock(return_value=SendResult(success=True, message_id="ii")),
    )


@pytest.mark.asyncio
async def test_streamed_explicit_media_resend_is_delivered(tmp_path, monkeypatch):
    """The streaming rescan must deliver an explicit MEDIA tag even when the
    same path was already delivered in a prior turn (sibling of the base.py
    fix — post-stream delivery is explicit-only, so nothing it finds is an
    accidental echo of auto-appended history)."""
    img = _allowed_file(tmp_path, monkeypatch, "flyer.png")
    adapter = _stream_adapter()
    runner = SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None, event_metadata=None: {},
        _reply_anchor_for_event=lambda event: None,
    )

    await GatewayRunner._deliver_media_from_response(
        runner,
        f"Here's the flyer again.\nMEDIA:{img}",
        _stream_event(),
        adapter,
    )

    adapter.send_multiple_images.assert_awaited_once()
    sent_paths = [p for p, _cap in adapter.send_multiple_images.await_args.kwargs["images"]]
    assert str(img) in sent_paths[0]


def test_stream_rescan_accepts_no_history_dedup_input():
    """Contract pin for the run.py half of the fix: the explicit-only
    post-stream rescan must not accept a history-dedup set at all — with the
    old ``history_media_paths`` parameter present, the call site fed it the
    session transcript and explicit resends were silently filtered."""
    import inspect

    params = inspect.signature(GatewayRunner._deliver_media_from_response).parameters
    assert "history_media_paths" not in params, (
        "history dedup re-attached to the explicit-only post-stream rescan "
        "(#73771 regression)"
    )


# ---------------------------------------------------------------------------
# Invariant: the stale-echo protection that REMAINS
# ---------------------------------------------------------------------------


def test_auto_append_lane_still_dedups_stale_media():
    """Removing the delivery-side tag filter must not regress the upstream
    protection: auto-appended tags from history are still deduped via
    _collect_auto_append_media_tags + _collect_history_media_paths."""
    history = [
        {"role": "assistant",
         "tool_calls": [{"id": "c", "function": {"name": "image_generate"}}]},
        {"role": "tool", "tool_call_id": "c",
         "content": '{"success": true, "image": "/tmp/gen/dog.png"}'},
        {"role": "assistant", "content": "made it! MEDIA:/tmp/gen/dog.png"},
    ]
    paths = _collect_history_media_paths(history)
    assert "/tmp/gen/dog.png" in paths

    tags, _voice = _collect_auto_append_media_tags(
        history, history_offset=0, history_media_paths=paths
    )
    assert tags == [], f"stale auto-append tags re-emitted: {tags}"
