"""Google Meet capture behavior contracts."""

from __future__ import annotations
import io
import json
import subprocess
from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


def test_pcm_pump_receives_audio_appended_after_start(tmp_path, monkeypatch):
    """The pump used to read the empty speaker.pcm to EOF and exit before Realtime spoke (#80875)."""
    import subprocess
    import time

    from plugins.google_meet import meet_bot

    pcm, sink = tmp_path / "speaker.pcm", tmp_path / "device.bin"
    pcm.write_bytes(b"")
    real_popen = subprocess.Popen

    def cat_popen(cmd, **kw):  # `cat` stands in for paplay: same stdin / file-EOF semantics
        assert cmd[0] == "paplay" and cmd[-1] == "-"
        kw["stdout"] = open(sink, "wb")
        return real_popen(["cat"], **kw)

    monkeypatch.setattr(meet_bot.subprocess, "Popen", cat_popen)
    rt, stop = {}, {"stop": False}
    state = meet_bot._BotState(tmp_path, "abc-defg-hij", "https://meet.google.com/abc-defg-hij")
    meet_bot._start_pcm_pump(rt, {"platform": "linux", "write_target": "sink"}, pcm, state, stop)
    time.sleep(0.2)
    with open(pcm, "ab") as f:
        f.write(b"\x01\x02" * 2000)
    deadline = time.time() + 5
    while time.time() < deadline and sink.stat().st_size < 4000:
        time.sleep(0.05)
    assert rt["pcm_pump"].poll() is None
    assert sink.stat().st_size == 4000
    stop["stop"] = True
    meet_bot._teardown_realtime({**rt, "speaker_thread": None, "session": None, "bridge": None})
    assert not rt["pcm_tail_thread"].is_alive()
    assert state.mic_state is None and "micState" in state.status_path.read_text(encoding="utf-8")


def test_pcm_tail_loop_swallows_only_pipe_errors(tmp_path):
    """A closed pump pipe is expected and quiet; any other tail-thread bug must not be silenced."""
    from types import SimpleNamespace

    from plugins.google_meet.meet_bot import _BotState, _pcm_tail_loop

    pcm = tmp_path / "speaker.pcm"
    pcm.write_bytes(b"\x00" * 16)
    state = _BotState(tmp_path, "abc-defg-hij", "https://meet.google.com/abc-defg-hij")

    def proc(exc):
        def write(_chunk):
            raise exc

        return SimpleNamespace(
            poll=lambda: None,
            stdin=SimpleNamespace(write=write, flush=lambda: None, close=lambda: None),
        )

    _pcm_tail_loop(proc(BrokenPipeError()), pcm, {"stop": False}, state)
    assert state.realtime_audio_pump_status == "failed"
    with pytest.raises(RuntimeError):
        _pcm_tail_loop(proc(RuntimeError("bug")), pcm, {"stop": False}, state)


@pytest.mark.macos_only
def test_pcm_pump_darwin_uses_stream_input(tmp_path, monkeypatch):
    from plugins.google_meet import meet_bot

    captured = {}

    class Pump:
        pid = 123
        stdin = io.BytesIO()

        def poll(self):
            return None

    def launch(argv, **kwargs):
        captured.update(argv=argv, kwargs=kwargs)
        return Pump()

    monkeypatch.setattr(meet_bot.subprocess, "Popen", launch)
    monkeypatch.setattr(meet_bot, "_mac_audio_device_index", lambda _name: "7")
    state = meet_bot._BotState(
        tmp_path, "abc-defg-hij", "https://meet.google.com/abc-defg-hij"
    )
    pcm = tmp_path / "speaker.pcm"
    pcm.write_bytes(b"")
    rt = {}
    assert meet_bot._start_pcm_pump(
        rt,
        {"platform": "darwin", "write_target": "BlackHole 2ch"},
        pcm,
        state,
        {"stop": True},
    )
    rt["pcm_tail_thread"].join(timeout=5)
    args = captured["argv"]
    assert args[args.index("-i") + 1] in ("-", "pipe:0")
    assert args[args.index("-audio_device_index") + 1] == "7"
    assert captured["kwargs"]["stdin"] == subprocess.PIPE
    assert str(pcm) not in args


@pytest.mark.macos_only
def test_pcm_pump_darwin_fails_closed_without_audio_device(tmp_path, monkeypatch):
    from plugins.google_meet import meet_bot

    monkeypatch.setattr(meet_bot, "_mac_audio_device_index", lambda _name: None)
    state = meet_bot._BotState(
        tmp_path, "abc-defg-hij", "https://meet.google.com/abc-defg-hij"
    )
    with patch.object(meet_bot.subprocess, "Popen") as launch:
        assert not meet_bot._start_pcm_pump(
            {},
            {"platform": "darwin", "write_target": "BlackHole 2ch"},
            tmp_path / "speaker.pcm",
            state,
            {"stop": True},
        )
    launch.assert_not_called()
    status = json.loads(state.status_path.read_text())
    assert status["realtimeReady"] is False
    assert status["realtimeAudioPumpStatus"] != "ready"


def test_run_bot_tears_down_realtime_resources_on_navigation_failure(
    tmp_path, monkeypatch
):
    import sys
    import types

    from plugins.google_meet import audio_bridge
    from plugins.google_meet.meet_bot import run_bot

    closed = {"context": False, "browser": False, "bridge": False}

    class _FakeBridge:
        def setup(self):
            return {
                "platform": "linux",
                "device_name": "hermes_meet_src",
                "write_target": "hermes_meet_sink",
            }

        def teardown(self):
            closed["bridge"] = True

    class _FakePage:
        def goto(self, *_args, **_kwargs):
            raise RuntimeError("navigation failed")

        def evaluate(self, _script):
            return False

    class _FakeContext:
        def new_page(self):
            return _FakePage()

        def close(self):
            closed["context"] = True

    class _FakeBrowser:
        def new_context(self, **_kwargs):
            return _FakeContext()

        def close(self):
            closed["browser"] = True

    class _FakePlaywright:
        chromium = type(
            "_Chromium", (), {"launch": staticmethod(lambda **_kwargs: _FakeBrowser())}
        )()

    class _FakeSyncPlaywright:
        def __enter__(self):
            return _FakePlaywright()

        def __exit__(self, *_args):
            return False

    monkeypatch.setitem(
        sys.modules,
        "playwright.sync_api",
        types.SimpleNamespace(sync_playwright=lambda: _FakeSyncPlaywright()),
    )
    monkeypatch.setattr(audio_bridge, "AudioBridge", _FakeBridge)
    monkeypatch.setenv("HERMES_MEET_URL", "https://meet.google.com/abc-defg-hij")
    monkeypatch.setenv("HERMES_MEET_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("HERMES_MEET_MODE", "realtime")
    monkeypatch.setenv("HERMES_MEET_REALTIME_KEY", "sk-test")

    assert run_bot() == 4
    assert closed == {"context": True, "browser": True, "bridge": True}


def test_run_bot_tears_down_partial_realtime_bridge_on_setup_failure(tmp_path, monkeypatch):
    from plugins.google_meet import audio_bridge
    from plugins.google_meet.meet_bot import run_bot

    closed = {"bridge": False}

    class _FakeBridge:
        def setup(self):
            raise RuntimeError("virtual device missing")

        def teardown(self):
            closed["bridge"] = True

    monkeypatch.setattr(audio_bridge, "AudioBridge", _FakeBridge)
    monkeypatch.setenv("HERMES_MEET_URL", "https://meet.google.com/abc-defg-hij")
    monkeypatch.setenv("HERMES_MEET_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("HERMES_MEET_MODE", "realtime")
    monkeypatch.setenv("HERMES_MEET_REALTIME_KEY", "sk-test")

    assert run_bot() == 6
    assert closed["bridge"] is True


@pytest.mark.parametrize("failure", ["exit", "broken_pipe"])
def test_pcm_route_failure_revokes_speech_and_exits_meeting(tmp_path, failure):
    import os
    import sys
    from types import SimpleNamespace
    from plugins.google_meet import meet_bot, process_manager as pm

    state = meet_bot._BotState(
        tmp_path, "abc-defg-hij", "https://meet.google.com/abc-defg-hij"
    )
    pcm = tmp_path / "speaker.pcm"
    pcm.write_bytes(b"\x00" * 16)
    code = (
        "raise SystemExit(7)"
        if failure == "exit"
        else "import os,time; os.close(0); print('closed', flush=True); time.sleep(60)"
    )
    pump = subprocess.Popen(
        [sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    try:
        state.set(
            in_call=True,
            realtime=True,
            realtime_ready=True,
            realtime_audio_pump_status="ready",
            realtime_audio_pump_pid=pump.pid,
            local_microphone_on=True,
        )
        pm._write_active(
            {"pid": os.getpid(), "out_dir": str(tmp_path), "mode": "realtime"}
        )
        if failure == "exit":
            assert pump.wait(timeout=5) == 7
            # A consumer must reject stale ready status even before the bot notices.
            assert pm.enqueue_say("must not be queued")["ok"] is False
        else:
            assert pump.stdout.readline() == b"closed\n"
        meet_bot._pcm_tail_loop(pump, pcm, {"stop": False}, state)
        status = json.loads(state.status_path.read_text())
        assert status["realtimeReady"] is False
        assert status["realtimeAudioPumpStatus"] == "failed"
        assert pm.enqueue_say("must remain rejected")["ok"] is False
        assert not (tmp_path / "say_queue.jsonl").exists()
        meet_bot._drain_loop(
            None,
            SimpleNamespace(duration_s=None),
            state,
            {"pcm_pump": pump},
            {"stop": False},
        )
        assert state.exited and not state.in_call
        assert state.leave_reason == "realtime_audio_route_failed"
    finally:
        if pump.poll() is None:
            pump.terminate()
        pump.wait(timeout=5)
        if pump.stdin is not None:
            try:
                pump.stdin.close()
            except BrokenPipeError:
                pass
        pump.stdout.close()
