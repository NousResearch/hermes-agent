"""``[[as_document]]`` in cron output sends images as documents, as every other send path does.

``extract_media`` strips the directive, so the delivery has to read it first. Cron did not: the
live lane routed a ``.png`` to ``send_image_file`` and the standalone lane passed
``force_document=False``, so a requested original arrived as a recompressed photo.
"""
import asyncio
import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from hermes_agent.cron.scheduler import _deliver_result
from hermes_agent.cron.scheduler_delivery_media import _send_media_via_adapter
from hermes_agent.gateway.config import Platform, PlatformConfig
from hermes_agent.hermes_constants import get_hermes_home


class _Result:
    def __init__(self, message_id):
        self.success, self.message_id, self.raw_response, self.error = True, message_id, None, None


class _FakeTelegram:
    platform = Platform.TELEGRAM

    def __init__(self):
        self.calls = []

    async def send(self, chat_id, content, metadata=None, **_kw):
        self.calls.append(("send", None))
        return _Result(1)

    async def send_image_file(self, chat_id, image_path, metadata=None, **_kw):
        self.calls.append(("send_image_file", os.path.basename(image_path)))
        return _Result(2)

    async def send_document(self, chat_id, file_path, metadata=None, **_kw):
        self.calls.append(("send_document", os.path.basename(file_path)))
        return _Result(3)

    async def send_video(self, chat_id, video_path, metadata=None, **_kw):
        self.calls.append(("send_video", os.path.basename(video_path)))
        return _Result(4)


@pytest.fixture()
def loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def _media(name: str) -> str:
    folder = get_hermes_home() / "cache" / "images"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    return str(path)


def _gateway_config():
    cfg = MagicMock()
    cfg.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True, token="x")}
    cfg.get_home_channel = lambda _platform: None
    return cfg


_JOB = {"id": "j1", "name": "Chart", "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "-100123"}}


def test_live_lane_sends_an_as_document_image_as_a_document(loop):
    adapter = _FakeTelegram()
    content = f"Full-resolution chart.\n[[as_document]]\nMEDIA:{_media('chart.png')}"

    with patch("hermes_agent.gateway.config.load_gateway_config", return_value=_gateway_config()):
        error = _deliver_result(dict(_JOB), content, adapters={Platform.TELEGRAM: adapter}, loop=loop)

    assert error is None
    assert ("send_document", "chart.png") in adapter.calls
    assert not any(method == "send_image_file" for method, _ in adapter.calls)


def test_standalone_lane_passes_force_document(monkeypatch):
    seen = {}

    async def fake_sender(pconfig, chat_id, message, *, thread_id=None, media_files=None,
                          force_document=False, **_kw):
        seen.update(force_document=force_document,
                    media=[os.path.basename(p) for p, _ in media_files or []])
        return {"success": True, "platform": "telegram", "chat_id": chat_id, "message_id": 9}

    import hermes_agent.gateway.platform_registry as reg_mod
    import hermes_agent.hermes_cli.plugins as hp

    if reg_mod.platform_registry.get("telegram") is None:
        hp.discover_plugins()
    entry = reg_mod.platform_registry.get("telegram")
    if entry is None:
        pytest.skip("telegram platform entry not registered in this environment")
    monkeypatch.setattr(entry, "standalone_sender_fn", fake_sender)
    monkeypatch.setattr(hp, "discover_plugins", lambda *a, **k: None)

    content = f"Full-resolution chart.\n[[as_document]]\nMEDIA:{_media('chart.png')}"
    with patch("hermes_agent.gateway.config.load_gateway_config", return_value=_gateway_config()):
        error = _deliver_result(dict(_JOB), content, adapters=None, loop=None)

    assert error is None
    assert seen == {"force_document": True, "media": ["chart.png"]}


@pytest.mark.parametrize(
    ("name", "force_document", "expected"),
    [
        ("chart.png", False, "send_image_file"),
        ("chart.png", True, "send_document"),
        # As in the gateway, [[as_document]] changes only how images go; video keeps its own route.
        ("clip.mp4", True, "send_video"),
    ],
)
def test_live_media_routing(loop, name, force_document, expected):
    adapter = _FakeTelegram()

    errors = _send_media_via_adapter(
        adapter, "-100123", [(_media(name), False)], None, loop, {"id": "j1"},
        platform=Platform.TELEGRAM, force_document=force_document)

    assert errors == []
    assert adapter.calls == [(expected, name)]


def test_directive_in_the_job_name_does_not_force_a_document(loop):
    # cron.wrap_response (default on) puts the job name into the delivered text; only the job's
    # own output may carry the directive.
    adapter = _FakeTelegram()
    job = dict(_JOB, name="How to use [[as_document]]")
    content = f"Preview.\nMEDIA:{_media('chart.png')}"

    with patch("hermes_agent.gateway.config.load_gateway_config", return_value=_gateway_config()):
        error = _deliver_result(job, content, adapters={Platform.TELEGRAM: adapter}, loop=loop)

    assert error is None
    assert ("send_image_file", "chart.png") in adapter.calls
    assert not any(method == "send_document" for method, _ in adapter.calls)
