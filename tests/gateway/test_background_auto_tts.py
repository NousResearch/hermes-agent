"""Discord attachment speech must never hold the conversation guard."""

import asyncio
from pathlib import Path
import json
import os
import shlex
import sys

import psutil

from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from plugins.platforms.discord.adapter import DiscordAdapter
from tests.waiting import async_wait_until


def adapter_and_event():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    adapter._should_auto_tts_for_chat = lambda chat_id: True
    adapter.send_typing = AsyncMock()
    adapter.stop_typing = AsyncMock()
    adapter._run_processing_hook = AsyncMock()
    adapter._record_delivery_obligation = AsyncMock(return_value=None)
    event = MessageEvent(
        text="hello", message_type=MessageType.VOICE, message_id="voice-1",
        source=SessionSource(platform=Platform.DISCORD, chat_id="123", thread_id="456", chat_type="group"),
    )
    return adapter, event


@pytest.mark.asyncio
async def test_blocked_synthesis_does_not_hold_text_or_next_turn(tmp_path):
    adapter, event = adapter_and_event()
    reply = "The complete spoken reply. " * 400
    adapter.set_message_handler(AsyncMock(return_value=reply))
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="text-1"))
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="audio-1"))
    started, release = asyncio.Event(), asyncio.Event()
    audio = tmp_path / "audio.mp3"

    async def synthesize(text, **kwargs):
        assert text == reply.strip()
        started.set()
        await release.wait()
        audio.write_bytes(b"audio")
        return [str(audio)], str(audio)

    adapter._synthesize_auto_tts = synthesize
    key = build_session_key(event.source)
    try:
        await adapter.handle_message(event)
        await asyncio.wait_for(started.wait(), 5)
        owner = adapter._session_tasks.get(key)
        if owner is not None:
            await asyncio.wait_for(asyncio.shield(owner), 2)
        assert adapter.send.await_args.kwargs["content"] == reply.strip()
        assert key not in adapter._active_sessions
        followup = MessageEvent(text="next", message_type=MessageType.TEXT,
                                message_id="next-1", source=event.source)
        await adapter.handle_message(followup)
        owner = adapter._session_tasks.get(key)
        if owner is not None:
            await asyncio.wait_for(asyncio.shield(owner), 2)
        assert adapter._message_handler.await_count == 2
        assert not adapter.send_voice.called
        release.set()
        await asyncio.gather(*list(adapter._background_tasks))
        adapter.send_voice.assert_awaited_once()
        assert adapter.send_voice.await_args.kwargs["reply_to"] == "text-1"
        assert adapter.send_voice.await_args.kwargs["metadata"]["thread_id"] == "456"
        assert not audio.exists()
    finally:
        release.set()
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_deadline_emits_one_notice_and_releases_job(monkeypatch):
    from gateway.platforms import base_auto_tts
    monkeypatch.setattr(base_auto_tts, "AUDIO_DEADLINE_SECONDS", 0.05, raising=False)
    adapter, event = adapter_and_event()
    adapter.set_message_handler(AsyncMock(return_value="reply"))
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="text-1"))
    adapter.send_voice = AsyncMock()
    started = asyncio.Event()

    async def blocked(text, **kwargs):
        started.set()
        await asyncio.Event().wait()

    adapter._synthesize_auto_tts = blocked
    try:
        await adapter.handle_message(event)
        await asyncio.wait_for(started.wait(), 5)
        await asyncio.wait_for(asyncio.gather(*list(adapter._background_tasks)), 2)
        assert adapter.send.await_count == 2
        notice = adapter.send.await_args
        assert "deadline" in str(notice).lower()
        assert notice.kwargs["reply_to"] == "text-1"
        adapter.send_voice.assert_not_called()
        assert not adapter._background_tasks
    finally:
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_pending_audio_is_bounded_and_visible_in_agents(tmp_path):
    from gateway.run import GatewayRunner
    adapter, event = adapter_and_event()
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._background_tasks = set()
    runner._session_key_for_source = lambda source: build_session_key(source)
    adapter.gateway_runner = runner
    adapter.set_message_handler(AsyncMock(return_value="reply"))
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="text-1"))
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True))
    started = []
    release = asyncio.Event()
    notice_started, notice_release = asyncio.Event(), asyncio.Event()

    async def send(*args, **kwargs):
        if "Audio skipped" in str(args):
            notice_started.set()
            await notice_release.wait()
        return SendResult(success=True, message_id="text-1")

    adapter.send.side_effect = send

    async def blocked(text, **kwargs):
        started.append(text)
        await release.wait()
        return [], None

    adapter._synthesize_auto_tts = blocked
    try:
        for i in range(3):
            source = SessionSource(platform=Platform.DISCORD, chat_id=str(i), chat_type="dm")
            turn = MessageEvent(text="voice", message_type=MessageType.VOICE,
                                message_id=str(i), source=source)
            await adapter.handle_message(turn)
            task = adapter._session_tasks.get(build_session_key(source))
            if task:
                await asyncio.wait_for(asyncio.shield(task), 2)
        assert len(started) == 2
        await asyncio.wait_for(notice_started.wait(), 2)
        followup = MessageEvent(text="next", message_type=MessageType.TEXT, source=source)
        await adapter.handle_message(followup)
        task = adapter._session_tasks.get(build_session_key(source))
        if task:
            await asyncio.wait_for(asyncio.shield(task), 2)
        assert adapter._message_handler.await_count == 4
        out = await runner._handle_agents_command(event)
        assert "Pending audio: 2" in out
        assert adapter.send.await_count == 5  # four answers and one overload notice
        notice_release.set()
        release.set()
        await asyncio.gather(*list(adapter._background_tasks))
        assert "Pending audio:" not in await runner._handle_agents_command(event)
    finally:
        notice_release.set()
        release.set()
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["synthesis", "upload", "oversize"])
async def test_failure_has_one_anchored_notice_and_cleans_files(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace
    adapter, event = adapter_and_event()
    adapter.set_message_handler(AsyncMock(return_value="reply"))
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="text-1"))
    channel = SimpleNamespace(send=AsyncMock(), id=456)
    adapter._resolve_channel = AsyncMock(return_value=channel)
    adapter._is_forum_parent = lambda channel: False
    monkeypatch.setattr(adapter, "_discord_upload_limit_bytes", lambda channel: 1)
    paths = []

    async def synthesize(text, **kwargs):
        paths.extend(Path(kwargs["output_dir"]) / name for name in ("first.mp3", "second.mp3"))
        for path in paths:
            path.write_bytes(b"audio")
        return ([] if failure == "synthesis" else [str(path) for path in paths]), str(paths[0])

    adapter._synthesize_auto_tts = synthesize
    if failure != "oversize":
        adapter.send_voice = AsyncMock(return_value=SendResult(success=False, error="private backend detail"))
    await adapter.handle_message(event)
    await asyncio.wait_for(asyncio.gather(*list(adapter._background_tasks)), 5)
    # The upload preflight must not emit a second, unthreaded failure notice.
    channel.send.assert_not_called()
    assert adapter.send.await_count == 2
    assert adapter.send.await_args.kwargs["reply_to"] == "text-1"
    assert "private backend detail" not in str(adapter.send.await_args)
    assert all(not path.exists() for path in paths)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["synthesis", "upload", "notice"])
async def test_failure_logs_traceback_without_provider_secrets(caplog, failure):
    from gateway.platforms import base_auto_tts

    adapter, event = adapter_and_event()
    secret = "private-provider-token-and-request-text"

    async def fail(*args, **kwargs):
        raise ValueError(secret)

    adapter.send = fail if failure == "notice" else AsyncMock()
    adapter.send_voice = fail
    adapter._synthesize_auto_tts = fail if failure == "synthesis" else AsyncMock(
        return_value=(["unused.mp3"], None))
    if failure == "notice":
        await base_auto_tts._notice(adapter, event, "text-1", {}, "notice")
    else:
        await base_auto_tts._deliver(adapter, event, "reply", "text-1", {})
    assert caplog.records
    assert all(record.exc_info and record.exc_info[2] for record in caplog.records)
    assert "ValueError" in caplog.text
    assert secret not in caplog.text


@pytest.mark.asyncio
async def test_live_voice_channel_keeps_audio_before_text(tmp_path):
    adapter, event = adapter_and_event()
    adapter._voice_text_channels = {1: event.source.chat_id}
    adapter.is_in_voice_channel = lambda guild_id: True
    adapter.set_message_handler(AsyncMock(return_value="reply"))
    audio = tmp_path / "voice.mp3"
    audio.write_bytes(b"audio")
    adapter._synthesize_auto_tts = AsyncMock(return_value=([str(audio)], str(audio)))
    order = []

    async def play(**kwargs):
        order.append("audio")
        return SendResult(success=True)

    async def send(**kwargs):
        order.append("text")
        return SendResult(success=True, message_id="text-1")

    adapter.play_tts = play
    adapter.send = send
    await adapter._process_message_background(event, build_session_key(event.source))
    assert order == ["audio", "text"]
    adapter._synthesize_auto_tts.assert_awaited_once_with("reply", event.source)
    assert not adapter._background_tasks
    assert not audio.exists()


@pytest.mark.platforms("any")
@pytest.mark.asyncio
@pytest.mark.parametrize("cancellations", [0, 1, 2])
async def test_failed_worker_spawn_preserves_delivery_cancellation(tmp_path, monkeypatch, cancellations):
    from gateway.platforms import base_auto_tts
    from hermes_constants import get_hermes_home

    adapter, event = adapter_and_event()
    adapter.send = AsyncMock(return_value=SendResult(success=True))
    adapter.send_voice = AsyncMock()
    entered, release = asyncio.Event(), asyncio.Event()
    spawn = asyncio.create_subprocess_exec

    async def failed_spawn(*args, **kwargs):
        entered.set()
        await release.wait()
        return await spawn(str(tmp_path / "missing-python-executable"), **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", failed_spawn)
    task = asyncio.create_task(base_auto_tts._deliver(adapter, event, "reply", "text-1", {}))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        for _ in range(cancellations):
            task.cancel()
            await asyncio.sleep(0)
        release.set()
        done, _ = await asyncio.wait({task}, timeout=5)
        assert task in done
        assert task.cancelled() == bool(cancellations)
        if not cancellations:
            task.result()
            adapter.send.assert_awaited_once()
            assert adapter.send.await_args is not None
            assert adapter.send.await_args.args[1] == base_auto_tts.FAILURE_NOTICE
        else:
            adapter.send.assert_not_awaited()
        adapter.send_voice.assert_not_awaited()
        assert not list((get_hermes_home() / "cache" / "auto_tts").iterdir())
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.platforms("any")
@pytest.mark.asyncio
@pytest.mark.parametrize("reap_failure", [False, True])
async def test_worker_cleanup_preserves_cancellation(tmp_path, monkeypatch, reap_failure):
    from gateway.platforms import base_auto_tts_worker as worker

    spawn = asyncio.create_subprocess_exec
    reap = worker._reap
    cleaning, release = asyncio.Event(), asyncio.Event()
    script = (
        "import json, pathlib, sys; request=json.load(sys.stdin); "
        "pathlib.Path(request['output_path']).with_name('result.json').write_text('{}')"
    )

    async def spawn_stub(*args, **kwargs):
        return await spawn(args[0], "-c", script, **kwargs)

    async def slow_reap(proc):
        cleaning.set()
        await release.wait()
        await reap(proc)
        assert proc.returncode is not None
        if reap_failure:
            raise OSError("Reaping failed after process exit")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_stub)
    monkeypatch.setattr(worker, "_reap", slow_reap)
    task = asyncio.create_task(worker.synthesize("reply", str(tmp_path / "audio.mp3")))
    try:
        await asyncio.wait_for(cleaning.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()  # runner and adapter may both cancel the same job
        release.set()
        done, _ = await asyncio.wait({task}, timeout=5)
        assert task in done
        assert task.cancelled()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.platforms("any")
@pytest.mark.asyncio
@pytest.mark.parametrize("result", [
    {"file_paths": ["first.mp3"]},
    {"success": True, "file_paths": ["first.mp3", "missing.mp3"]},
])
async def test_worker_rejects_incomplete_audio_result(tmp_path, monkeypatch, result):
    from gateway.platforms import base_auto_tts_worker as worker

    spawn = asyncio.create_subprocess_exec
    result = {**result, "file_paths": [str(tmp_path / name) for name in result["file_paths"]]}
    script = (
        "import pathlib; "
        f"pathlib.Path({str(tmp_path / 'first.mp3')!r}).write_bytes(b'audio'); "
        f"pathlib.Path({str(tmp_path / 'result.json')!r}).write_text({json.dumps(result)!r})"
    )

    async def spawn_stub(*args, **kwargs):
        return await spawn(args[0], "-c", script, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_stub)
    with pytest.raises(RuntimeError, match="Incomplete auto-TTS output"):
        await worker.synthesize("complete reply", str(tmp_path / "audio.mp3"))


def test_worker_accepts_utf8_bom_result_with_unicode_path(tmp_path):
    from gateway.platforms.base_auto_tts_worker import _read_result

    audio = tmp_path / "réponse.mp3"
    audio.write_bytes(b"audio")
    expected = {"success": True, "file_path": str(audio)}
    (tmp_path / "result.json").write_text(
        json.dumps(expected, ensure_ascii=False), encoding="utf-8-sig")
    assert _read_result(tmp_path) == expected


@pytest.mark.platforms("any")
@pytest.mark.asyncio
async def test_cancelled_worker_reaps_real_descendant(tmp_path, monkeypatch):
    from gateway.platforms import base_auto_tts_worker as worker

    spawn = asyncio.create_subprocess_exec
    ready = tmp_path / "ready.json"
    script = (
        "import json, pathlib, subprocess, sys, threading; "
        "request = json.load(sys.stdin); "
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import threading; threading.Event().wait()']); "
        f"pathlib.Path({str(ready)!r}).write_text(json.dumps(child.pid)); "
        "threading.Event().wait()"
    )

    async def spawn_provider(*args, **kwargs):
        return await spawn(args[0], "-c", script, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_provider)
    task = asyncio.create_task(worker.synthesize("reply", str(tmp_path / "audio.mp3")))
    pid = None
    try:
        await async_wait_until(ready.exists, timeout=10)
        pid = json.loads(ready.read_text())
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=10)
        assert task in done
        assert task.cancelled()
        await async_wait_until(
            lambda: not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE,
            timeout=10)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if pid and psutil.pid_exists(pid):
            psutil.Process(pid).kill()


@pytest.mark.asyncio
async def test_worker_reaped_when_process_inspection_is_denied(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from gateway.platforms import base_auto_tts_worker as worker

    proc = SimpleNamespace(pid=123, returncode=None, kill=Mock(), wait=AsyncMock())
    monkeypatch.setattr(psutil, "Process", Mock(side_effect=psutil.AccessDenied(123)))
    if os.name != "nt":
        monkeypatch.setattr(os, "killpg", Mock())
    await worker._reap(proc)
    proc.kill.assert_called_once()
    proc.wait.assert_awaited_once()


@pytest.mark.platforms("linux", "macos")
@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["deadline", "shutdown", "session", "slash", "interrupt"])
async def test_blocking_provider_is_killed_and_partial_files_removed(tmp_path, monkeypatch, stop):
    from gateway.platforms import base_auto_tts
    from hermes_constants import get_hermes_home
    monkeypatch.setattr(base_auto_tts, "AUDIO_DEADLINE_SECONDS", 2)
    home = get_hermes_home()
    ready, release = tmp_path / "ready.json", tmp_path / "release"
    script = tmp_path / "provider.py"
    script.write_text(
        "import json, os, pathlib, sys, time\n"
        "output, ready, release = map(pathlib.Path, sys.argv[1:])\n"
        "output.write_bytes(b'partial')\n"
        "ready.write_text(json.dumps({'pid': os.getpid(), 'output': str(output)}))\n"
        "while not release.exists(): time.sleep(0.02)\n"
    )
    command = " ".join(map(shlex.quote, [sys.executable, str(script)]))
    command += " {output_path} " + " ".join(map(shlex.quote, [str(ready), str(release)]))
    (home / "config.yaml").write_text(json.dumps({"tts": {
        "provider": "blocked-test", "providers": {"blocked-test": {
            "type": "command", "command": command, "timeout": 30, "format": "mp3"}}}}))
    adapter, event = adapter_and_event()
    adapter.set_message_handler(AsyncMock(return_value="Complete reply."))
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="text-1"))
    adapter.send_voice = AsyncMock()
    from types import SimpleNamespace
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._background_tasks = set()
    runner.adapters = {Platform.DISCORD: adapter}
    key = build_session_key(event.source)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_key=key, session_id="test"))
    adapter.gateway_runner = runner
    info = None
    try:
        await adapter.handle_message(event)
        await async_wait_until(ready.exists)
        info = json.loads(ready.read_text())
        async def slash_stop():
            from agent.i18n import t
            assert not adapter._active_sessions
            result = await runner._handle_stop_command(event)
            assert result == t("gateway.stop.stopped")

        async def wait_deadline():
            await asyncio.wait_for(asyncio.gather(*list(adapter._background_tasks)), 5)

        actions = {
            "shutdown": adapter.cancel_background_tasks,
            "session": lambda: adapter.cancel_session_processing(key),
            "interrupt": lambda: adapter.interrupt_session_activity(key, event.source.chat_id),
            "slash": slash_stop,
            "deadline": wait_deadline,
        }
        await actions[stop]()
        assert not psutil.pid_exists(info["pid"]) or psutil.Process(info["pid"]).status() == psutil.STATUS_ZOMBIE
        assert not Path(info["output"]).exists()
        assert adapter.send.await_count == (2 if stop == "deadline" else 1)
        adapter.send_voice.assert_not_called()
        assert not adapter._background_tasks
    finally:
        release.touch()
        await adapter.cancel_background_tasks()
        if info:
            await async_wait_until(lambda: not psutil.pid_exists(info["pid"]) or psutil.Process(info["pid"]).status() == psutil.STATUS_ZOMBIE)
