"""The agent reply path and kanban notifier must consult MEDIA_KINDS before
dispatching attachments, skipping (with a warning) any kind the adapter does
not natively deliver — never leaking a local file path as chat text."""

import asyncio
import logging
from types import SimpleNamespace

import pytest

import gateway.run as run_mod
from gateway.platforms.base import BasePlatformAdapter, MediaKind


class _RecordingAdapter:
    name = "fake"
    extract_media = staticmethod(BasePlatformAdapter.extract_media)
    extract_images = staticmethod(BasePlatformAdapter.extract_images)
    extract_local_files = staticmethod(BasePlatformAdapter.extract_local_files)

    def __init__(self, kinds):
        self.MEDIA_KINDS = kinds
        self.calls = []

    async def send_multiple_images(self, **kw):
        self.calls.append(("image", kw))

    async def send_voice(self, **kw):
        self.calls.append(("voice", kw))

    async def send_video(self, **kw):
        self.calls.append(("video", kw))

    async def send_document(self, **kw):
        self.calls.append(("document", kw))


class _StubRunner:
    _thread_metadata_for_source = lambda self, *a, **k: None
    _reply_anchor_for_event = lambda self, *a, **k: None
    _deliver_media_from_response = run_mod.GatewayRunner._deliver_media_from_response


@pytest.fixture(autouse=True)
def _accept_all_media_paths(monkeypatch):
    # Bypass the on-disk safety validator so tests exercise capability gating,
    # not path validation (covered elsewhere).
    monkeypatch.setattr("gateway.platforms.base.validate_media_delivery_path", lambda p: str(p))


def _deliver(adapter, response):
    event = SimpleNamespace(source=SimpleNamespace(platform="qqbot", chat_id="c", thread_id=None))
    asyncio.run(_StubRunner()._deliver_media_from_response(response, event, adapter))


def test_undeclared_kinds_are_skipped_without_leaking_path(caplog):
    adapter = _RecordingAdapter(frozenset())
    with caplog.at_level(logging.WARNING):
        _deliver(adapter, "here you go\nMEDIA:/tmp/report.pdf")
    assert adapter.calls == []
    assert any("/tmp/report.pdf" in r.message for r in caplog.records)


def test_declared_document_is_delivered():
    adapter = _RecordingAdapter(frozenset({MediaKind.DOCUMENT}))
    _deliver(adapter, "here\nMEDIA:/tmp/report.pdf")
    assert [c[0] for c in adapter.calls] == ["document"]


def test_image_skipped_when_image_not_declared():
    adapter = _RecordingAdapter(frozenset({MediaKind.DOCUMENT}))
    _deliver(adapter, "pic\nMEDIA:/tmp/shot.png")
    assert adapter.calls == []


def test_partial_capability_delivers_only_declared_kind():
    adapter = _RecordingAdapter(frozenset({MediaKind.IMAGE}))
    _deliver(adapter, "both\nMEDIA:/tmp/shot.png\nMEDIA:/tmp/report.pdf")
    assert [c[0] for c in adapter.calls] == ["image"]
