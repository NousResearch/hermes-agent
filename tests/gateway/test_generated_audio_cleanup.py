"""Whole-file auto speech belongs to the turn, including a cancelled provider's late output."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import errno
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from tests.waiting import async_wait_until
from tools import tts_tool


class AudioAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.delivered = []
        self._auto_tts_default = True

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.delivered.append(("text", content))
        return SendResult(success=True, message_id="text")

    async def send_voice(self, chat_id, audio_path, caption=None, reply_to=None, metadata=None, **kwargs):
        self.delivered.append(("audio", Path(audio_path).read_bytes()))
        return SendResult(success=True, message_id="audio")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def stop_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def _keep_typing(self, *args, **kwargs):
        await asyncio.Event().wait()


def caller(lane):
    """Keep eligibility, synthesis, and delivery real; only the remote transport is inert."""
    adapter = AudioAdapter()
    event = MessageEvent(
        text="hello", message_id="trigger", message_type=MessageType.VOICE if lane == "base" else MessageType.TEXT,
        source=SessionSource(platform=adapter.platform, chat_id="chat", user_id="user", chat_type="dm"),
    )
    if lane == "base":
        async def handler(_event):
            return "The written answer."
        adapter.set_message_handler(handler)
        return adapter, lambda: adapter._process_message_background(event, build_session_key(event.source))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {runner._voice_key_for_source(event.source): "all"}
    return adapter, lambda: runner._hmwa_deliver_turn_response(
        event, event.source, SimpleNamespace(session_id="session"), build_session_key(event.source),
        1, {}, [], "The written answer.", "", False,
    )


@pytest.fixture
def audio_root(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "config.yaml").write_text(json.dumps({"tts": {"provider": "edge"}}))
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(tts_tool, "check_tts_requirements", lambda: True)
    return scratch / "hermes_voice"


@pytest.mark.parametrize("lane", ["base", "runner"])
def test_cancelled_caller_cleans_late_audio_after_loop_close(lane, audio_root, monkeypatch):
    """Cancelling delivery must not wait for synthesis or depend on a surviving event loop."""
    entered, release = threading.Event(), threading.Event()
    written = []

    async def provider(_text, output_path, _config):
        entered.set()
        assert release.wait(10), "test did not release provider"
        path = Path(output_path)
        path.write_bytes(b"late audio")
        written.append(path)
        return output_path

    monkeypatch.setattr(tts_tool, "_generate_edge_tts", provider)
    adapter, invoke = caller(lane)
    loop = asyncio.new_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)

    async def cancel():
        task = asyncio.create_task(invoke())
        try:
            assert await async_wait_until(entered.is_set, timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
            assert not release.is_set(), "cancellation waited for the provider"
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    try:
        loop.run_until_complete(cancel())
    finally:
        loop.close()
        release.set()
        executor.shutdown(wait=True)

    assert written, "the real TTS tool never reached the provider"
    assert written[0].name.startswith("tts_reply_") and written[0].suffix == ".mp3"
    assert not list(audio_root.rglob("*")), "cancelled turn leaked its late generated audio"
    assert adapter.delivered == []


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["base", "runner"])
@pytest.mark.parametrize("ending", ["success", "cancel", "raise"])
async def test_multipart_lives_until_delivery_exits(lane, ending, audio_root, monkeypatch):
    """All parts, including unplayed ones and unnamed sidecars, have one lifetime."""
    parts = []

    def tool(*, text, output_path):
        base = Path(output_path)
        parts.extend([base.with_suffix(".part01.mp3"), base.with_suffix(".part02.wav")])
        for path in parts:
            path.write_bytes(b"audio part")
        base.with_suffix(".unreported").write_bytes(b"provider sidecar")
        return json.dumps({"success": True, "file_paths": [str(p) for p in parts]})

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", tool)
    adapter, invoke = caller(lane)
    reached, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def send_voice(*, audio_path, **kwargs):
        assert all(path.exists() for path in parts), "deleted a part before multipart delivery ended"
        seen.append(Path(audio_path))
        reached.set()
        await release.wait()
        if ending == "raise":
            raise RuntimeError("transport refused audio")
        return SendResult(success=True, message_id="audio")

    monkeypatch.setattr(adapter, "send_voice", send_voice)
    task = asyncio.create_task(invoke())
    try:
        await asyncio.wait_for(reached.wait(), timeout=5)
        assert all(path.exists() for path in parts)
        if ending == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            await asyncio.wait_for(task, timeout=5)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert seen == (parts if ending == "success" else parts[:1])
    assert not list(audio_root.rglob("*"))
    if ending == "success" and lane == "base":
        assert adapter.delivered == [("text", "The written answer.")]


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["base", "runner"])
@pytest.mark.parametrize("result_kind", ["invalid", "failed", "raise", "missing"])
async def test_failed_synthesis_cleans_family_and_keeps_text(lane, result_kind, audio_root, monkeypatch):
    def tool(*, text, output_path):
        path = Path(output_path)
        path.write_bytes(b"partial")
        path.with_suffix(".wav").write_bytes(b"unreported conversion")
        if result_kind == "raise":
            raise RuntimeError("provider failed")
        if result_kind == "invalid":
            return "not JSON"
        return json.dumps({"success": result_kind == "missing", "file_path": str(path.with_suffix(".absent"))})

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", tool)
    adapter, invoke = caller(lane)
    text = await invoke()
    assert not list(audio_root.rglob("*"))
    if lane == "base":
        assert adapter.delivered == [("text", "The written answer.")]
    else:
        assert text == "The written answer."
        assert adapter.delivered == []


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["base", "runner"])
async def test_returned_path_does_not_grant_deletion_ownership(lane, audio_root, tmp_path, monkeypatch):
    """Keep the selected root/name, preexisting output and external provider-owned files."""
    from gateway.platforms import base
    from gateway import run_voice

    audio_root.mkdir()
    requested = audio_root / "tts_reply_selected.mp3"
    requested.write_bytes(b"preexisting")
    outside = tmp_path / "provider-owned.wav"
    outside.write_bytes(b"provider audio")
    roots = []

    def tool(*, text, output_path):
        path = Path(output_path)
        assert path.name == requested.name
        assert path.parent.parent == audio_root
        assert path != requested
        roots.append(path.parent)
        resolved, error = tts_tool._resolve_output_base(output_path, "edge", None, False)
        assert error is None and resolved == path
        path.write_bytes(b"owned requested output")
        path.with_suffix(".partial").write_bytes(b"owned sidecar")
        return json.dumps({"success": True, "file_path": str(outside)})

    monkeypatch.setattr(base, "build_auto_tts_output_path", lambda platform: str(requested))
    monkeypatch.setattr(run_voice, "build_auto_tts_output_path", lambda platform: str(requested))
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", tool)
    adapter, invoke = caller(lane)
    await invoke()
    assert roots and not roots[0].exists()
    assert requested.read_bytes() == b"preexisting"
    assert outside.read_bytes() == b"provider audio"
    assert adapter.delivered[0] == ("audio", b"provider audio")


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["base", "runner"])
async def test_real_multipart_synthesis_keeps_profile_scope(lane, audio_root, tmp_path, monkeypatch):
    """Use real config reads, path validation, chunking and packing under A -> B -> A scopes."""
    from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override
    from tools import tts_tool_delivery

    homes = [tmp_path / "profile-a", tmp_path / "profile-b"]
    for home in homes:
        home.mkdir()
        (home / "config.yaml").write_text(json.dumps({"tts": {
            "provider": "edge", "edge": {"voice": home.name, "max_text_length": 8},
        }}))
    before = {home: (home / "config.yaml").read_bytes() for home in homes}
    calls = []

    def provider(_provider, text, output_path, config, _instructions):
        home = get_hermes_home()
        assert config["edge"]["voice"] == home.name
        path = Path(output_path)
        assert path.parent.parent == audio_root
        assert path.suffix == ".mp3" and path.name.startswith("tts_reply_")
        path.write_bytes(text.encode())
        calls.append((home, text))

    monkeypatch.setattr(tts_tool, "_synthesize_builtin", provider)
    # No ffmpeg/network: exercise the real multipart fallback when combination is unavailable.
    monkeypatch.setattr(tts_tool_delivery, "_concat_audio_files", lambda *args, **kwargs: None)
    for home in [*homes, homes[0]]:
        token = set_hermes_home_override(str(home))
        try:
            adapter, invoke = caller(lane)
            start = len(calls)
            text = await invoke()
            current = calls[start:]
            assert len(current) > 1 and all(owner == home for owner, _ in current)
            assert "".join(part for _, part in current).replace(" ", "") == "Thewrittenanswer."
            assert len([item for item in adapter.delivered if item[0] == "audio"]) == len(current)
            if lane == "base":
                assert adapter.delivered[-1] == ("text", "The written answer.")
            else:
                assert text == "The written answer."
            assert not list(audio_root.rglob("*"))
        finally:
            reset_hermes_home_override(token)
    assert {home: (home / "config.yaml").read_bytes() for home in homes} == before


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["base", "runner"])
@pytest.mark.parametrize("provider_success", [True, False])
async def test_cleanup_failure_keeps_written_answer(lane, provider_success, audio_root, monkeypatch):
    """Best-effort file removal must not discard a completed answer."""
    generated, refused = [], []
    real_remove, real_unlink = os.remove, os.unlink

    def tool(*, text, output_path):
        path = Path(output_path)
        path.write_bytes(b"audio or partial output")
        generated.append(path)
        return json.dumps({"success": provider_success, "file_path": output_path})

    def refusal(real, path, *args, **kwargs):
        # rmtree unlinks a basename with dir_fd; legacy cleanup uses absolute paths.
        if Path(path).name in {p.name for p in generated}:
            refused.append(path)
            raise PermissionError(errno.EACCES, "audio file is locked", str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", tool)
    adapter, invoke = caller(lane)
    with monkeypatch.context() as filesystem:
        filesystem.setattr(os, "remove", lambda path, *args, **kwargs: refusal(real_remove, path, *args, **kwargs))
        filesystem.setattr(os, "unlink", lambda path, *args, **kwargs: refusal(real_unlink, path, *args, **kwargs))
        response = await invoke()
    assert generated and refused, "did not exercise actual cleanup I/O"
    if lane == "base":
        assert [item for item in adapter.delivered if item[0] == "text"] == [("text", "The written answer.")]
    else:
        assert response == "The written answer."
