"""MEDIA audio delivered while the bot sits in a Discord voice channel plays IN the channel.

The delivery core hands the turn's media list to ``gateway.voice_media.route_media_attachments``
before attachments dispatch. The Discord adapter opts in (resolve_voice_playback_guild /
play_voice_media) with the same membership proof its auto-TTS already uses: the guild whose
bound text channel IS this turn's chat AND where a live VC connection exists. Audio routed to
the VC disappears from the attachment list; everything else (images, video, non-audio, URL
tags, adapters without the hook, gate off) keeps upstream behavior untouched.

Queue tests drive the per-guild worker on an explicit event loop, matching this repo's
sync-test style — the worker is fire-and-forget via ``asyncio.ensure_future`` and only drains
while its loop runs.
"""
from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from gateway import voice_media
from gateway.voice_media import route_media_attachments, stop_voice_media


class _FakeEvent:
    def __init__(self, chat_id: str = "700") -> None:
        self.source = SimpleNamespace(chat_id=chat_id, platform="discord")


class _OptInAdapter:
    """Duck-typed Discord adapter: bound guild 1 -> text channel 700, VC connected."""

    def __init__(self, *, in_vc: bool = True) -> None:
        self._voice_text_channels = {1: 700}
        self._in_vc = in_vc
        self.played: list[tuple[int, str]] = []
        self.stopped: list[int] = []

    def is_in_voice_channel(self, guild_id: int) -> bool:
        return self._in_vc

    def resolve_voice_playback_guild(self, event):
        chat_id = getattr(getattr(event, "source", None), "chat_id", None)
        for gid, text_ch in self._voice_text_channels.items():
            if str(text_ch) == str(chat_id) and self.is_in_voice_channel(gid):
                return gid
        return None

    async def play_voice_media(self, guild_id: int, audio_path: str) -> bool:
        self.played.append((guild_id, audio_path))
        return True

    def stop_voice_playback(self, guild_id: int) -> None:
        self.stopped.append(guild_id)


class _PlainAdapter:
    """Adapter without the opt-in hooks: routing must be a no-op."""


@pytest.fixture()
def loop():
    """The conftest autouse fixture already installs a current loop for sync tests;
    route_media_attachments calls ensure_future OUTSIDE run_until_complete, so the task binds
    to THAT loop — reuse it and drive it. (Creating a second loop here would bind the worker to
    a loop nothing ever runs.)"""
    lp = asyncio.get_event_loop()
    yield lp
    for task in list(voice_media._QUEUE_WORKERS.values()):
        task.cancel()
    voice_media._QUEUES.clear()
    voice_media._LAST_ADAPTER.clear()
    voice_media._QUEUE_WORKERS.clear()


@pytest.fixture()
def audio_file(tmp_path):
    path = tmp_path / "song.mp3"
    path.write_bytes(b"ID3fake")
    return str(path)


@pytest.fixture()
def gate_on(monkeypatch):
    monkeypatch.setattr(voice_media, "voice_media_playback_enabled", lambda platform="discord": True)


async def _wait_until(predicate, timeout=2.0):
    lp = asyncio.get_event_loop()
    deadline = lp.time() + timeout
    while not predicate():
        if lp.time() > deadline:
            return False
        await asyncio.sleep(0.02)
    return True


def test_gate_reads_config_per_turn(monkeypatch):
    import hermes_cli.config as hc

    state = {"value": True}
    monkeypatch.setattr(hc, "read_raw_config", lambda: {"discord": {"voice_media_playback": state["value"]}})
    assert voice_media.voice_media_playback_enabled() is True
    state["value"] = False
    assert voice_media.voice_media_playback_enabled() is False


def test_audio_plays_in_vc_and_leaves_attachments(loop, audio_file, gate_on):
    adapter = _OptInAdapter()
    media = [(audio_file, True), ("/tmp/report.pdf", False)]
    kept = route_media_attachments(adapter, _FakeEvent(), media)
    assert kept == [("/tmp/report.pdf", False)]  # non-audio stays an attachment

    async def _drain():
        assert await _wait_until(lambda: adapter.played), "queue worker never played"

    loop.run_until_complete(_drain())
    assert adapter.played == [(1, audio_file)]


def test_no_binding_or_no_vc_keeps_attachment(audio_file, gate_on):
    # different chat than the bound text channel -> no guild match
    kept = route_media_attachments(_OptInAdapter(), _FakeEvent(chat_id="999"), [(audio_file, True)])
    assert kept == [(audio_file, True)]
    # bound chat but not connected in a VC
    kept = route_media_attachments(_OptInAdapter(in_vc=False), _FakeEvent(), [(audio_file, True)])
    assert kept == [(audio_file, True)]


def test_gate_off_rewrites_voice_flag_keeps_attachment(audio_file, monkeypatch):
    """Gate off: audio keeps upstream attachment behavior but is forced to the file-upload
    routing (is_voice=False) so native Opus-only voice senders don't fail on mp3."""
    monkeypatch.setattr(voice_media, "voice_media_playback_enabled", lambda platform="discord": False)
    kept = route_media_attachments(_OptInAdapter(), _FakeEvent(), [(audio_file, True)])
    assert kept == [(audio_file, False)]


def test_adapter_without_hooks_untouched(audio_file):
    media = [(audio_file, True)]
    assert route_media_attachments(_PlainAdapter(), _FakeEvent(), media) == media


def test_url_and_missing_files_never_played(loop, gate_on):
    adapter = _OptInAdapter()
    media = [("https://example.com/song.mp3", True), ("/nonexistent/local.mp3", True)]
    kept = route_media_attachments(adapter, _FakeEvent(), media)
    assert kept == media
    loop.run_until_complete(asyncio.sleep(0.1))
    assert adapter.played == []


def test_queue_is_sequential_per_guild(loop, gate_on):
    """Two turns' worth of audio serializes onto one worker in FIFO order."""

    class _SlowAdapter(_OptInAdapter):
        async def play_voice_media(self, guild_id, audio_path):
            self.played.append((guild_id, audio_path))
            await asyncio.sleep(0.05)
            return True

    adapter = _SlowAdapter()
    paths = []
    for name in ("a", "b", "c"):
        p = f"/tmp/hermes_vm_test_{name}.mp3"
        open(p, "w").write("x")
        paths.append(p)
    route_media_attachments(adapter, _FakeEvent(), [(paths[0], True)])
    route_media_attachments(adapter, _FakeEvent(), [(paths[1], True), (paths[2], True)])

    async def _drain():
        assert await _wait_until(
            lambda: len(adapter.played) == 3 and not voice_media._QUEUE_WORKERS.get(1)
        ), f"only {len(adapter.played)} played"

    loop.run_until_complete(_drain())
    assert [p for _, p in adapter.played] == paths


def test_stop_cuts_clip_and_drops_queue(loop, gate_on):
    class _BlockingAdapter(_OptInAdapter):
        async def play_voice_media(self, guild_id, audio_path):
            self.played.append((guild_id, audio_path))
            await asyncio.sleep(30)  # blocked until stop cancels the worker
            return True

    adapter = _BlockingAdapter()
    p = "/tmp/hermes_vm_test_a.mp3"
    open(p, "w").write("x")
    route_media_attachments(adapter, _FakeEvent(), [(p, True)])
    route_media_attachments(adapter, _FakeEvent(), [("/tmp/hermes_vm_test_b.mp3", True),
                                                    ("/tmp/hermes_vm_test_c.mp3", True)])

    async def _stop_after_start():
        assert await _wait_until(lambda: adapter.played), "first clip never started"
        dropped = stop_voice_media(1)
        assert dropped == 2
        assert adapter.stopped == [1]
        worker = voice_media._QUEUE_WORKERS.get(1)
        if worker is not None:
            await asyncio.gather(worker, return_exceptions=True)
        assert len(adapter.played) == 1  # queued files never played

    loop.run_until_complete(_stop_after_start())
