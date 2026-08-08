"""Tests for the google_meet plugin.

Covers the safety-gated pieces that don't require Playwright:

  * URL regex — only ``https://meet.google.com/`` URLs pass
  * Meeting-id extraction from Meet URLs
  * Status / transcript writes round-trip through the file-backed state
  * Tool handlers return well-formed JSON under all branches
  * Process manager refuses unsafe URLs and clears stale state cleanly
  * ``_on_session_end`` hook is defensive (no-ops when no bot active)

Does NOT spawn a real Chromium — we mock ``subprocess.Popen`` where needed.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


# ---------------------------------------------------------------------------
# URL safety gate
# ---------------------------------------------------------------------------

def test_is_safe_meet_url_accepts_standard_meet_codes():
    from plugins.google_meet.meet_bot import _is_safe_meet_url

    assert _is_safe_meet_url("https://meet.google.com/abc-defg-hij")
    assert _is_safe_meet_url("https://meet.google.com/abc-defg-hij?pli=1")
    assert _is_safe_meet_url("https://meet.google.com/new")
    assert _is_safe_meet_url("https://meet.google.com/lookup/ABC123")


def test_meeting_id_extraction():
    from plugins.google_meet.meet_bot import _meeting_id_from_url

    assert _meeting_id_from_url("https://meet.google.com/abc-defg-hij") == "abc-defg-hij"
    assert _meeting_id_from_url("https://meet.google.com/abc-defg-hij?pli=1") == "abc-defg-hij"
    # fallback for codes we can't parse (e.g. /new before redirect)
    fallback = _meeting_id_from_url("https://meet.google.com/new")
    assert fallback.startswith("meet-")


# ---------------------------------------------------------------------------
# _BotState — transcript + status file round-trip
# ---------------------------------------------------------------------------

def test_bot_state_dedupes_captions_and_flushes_status(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    state.record_caption("Alice", "Hey everyone")
    state.record_caption("Alice", "Hey everyone")  # dup — ignored
    state.record_caption("Bob", "Let's start")

    transcript = (out / "transcript.txt").read_text()
    assert "Alice: Hey everyone" in transcript
    assert "Bob: Let's start" in transcript
    # dedup — Alice line appears exactly once
    assert transcript.count("Alice: Hey everyone") == 1

    status = json.loads((out / "status.json").read_text())
    assert status["meetingId"] == "abc-defg-hij"
    assert status["transcriptLines"] == 2
    assert status["transcriptPath"].endswith("transcript.txt")


def test_parse_duration():
    from plugins.google_meet.meet_bot import _parse_duration

    assert _parse_duration("30m") == 30 * 60
    assert _parse_duration("2h") == 2 * 3600
    assert _parse_duration("90s") == 90
    assert _parse_duration("90") == 90
    assert _parse_duration("") is None
    assert _parse_duration("bogus") is None


# ---------------------------------------------------------------------------
# process_manager — refuses unsafe URLs, manages active pointer
# ---------------------------------------------------------------------------

def test_start_refuses_unsafe_url():
    from plugins.google_meet import process_manager as pm

    res = pm.start("https://evil.example.com/abc-defg-hij")
    assert res["ok"] is False
    assert "refusing" in res["error"]


def test_status_reports_no_active_meeting():
    from plugins.google_meet import process_manager as pm

    assert pm.status() == {"ok": False, "reason": "no active meeting"}
    assert pm.transcript() == {"ok": False, "reason": "no active meeting"}
    assert pm.stop() == {"ok": False, "reason": "no active meeting"}


def test_transcript_reads_last_n_lines(tmp_path):
    from plugins.google_meet import process_manager as pm

    meeting_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    meeting_dir.mkdir(parents=True)
    (meeting_dir / "transcript.txt").write_text(
        "[10:00:00] Alice: one\n"
        "[10:00:01] Bob: two\n"
        "[10:00:02] Alice: three\n"
    )
    pm._write_active({
        "pid": 0, "meeting_id": "abc-defg-hij",
        "out_dir": str(meeting_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
    })

    res = pm.transcript(last=2)
    assert res["ok"] is True
    assert res["total"] == 3
    assert len(res["lines"]) == 2
    assert res["lines"][-1].endswith("Alice: three")


def test_stop_signals_process_and_clears_pointer(tmp_path):
    from plugins.google_meet import process_manager as pm

    pm._write_active({
        "pid": 11111, "meeting_id": "x-y-z",
        "out_dir": str(tmp_path / "x-y-z"),
        "url": "https://meet.google.com/x-y-z",
        "started_at": 0,
    })

    alive_seq = iter([True, True, False])  # alive at first, gone after SIGTERM
    def _alive(pid):
        try:
            return next(alive_seq)
        except StopIteration:
            return False

    sent = []
    def _kill(pid, sig):
        sent.append((pid, sig))

    with patch.object(pm, "_pid_alive", side_effect=_alive), \
         patch.object(pm.os, "kill", side_effect=_kill), \
         patch.object(pm.time, "sleep", lambda _s: None):
        res = pm.stop()

    assert res["ok"] is True
    assert (11111, signal.SIGTERM) in sent
    # .active.json cleared
    assert pm._read_active() is None


# ---------------------------------------------------------------------------
# Tool handlers — JSON shape + safety gates
# ---------------------------------------------------------------------------

def test_meet_join_handler_missing_url_returns_error():
    from plugins.google_meet.tools import handle_meet_join

    out = json.loads(handle_meet_join({}))
    assert out["success"] is False
    assert "url is required" in out["error"]


# ---------------------------------------------------------------------------
# _on_session_end — defensive cleanup
# ---------------------------------------------------------------------------

def test_on_session_end_noop_when_nothing_active():
    from plugins.google_meet import _on_session_end
    # Should not raise and should not call stop().
    with patch("plugins.google_meet.pm.stop") as stop_mock:
        _on_session_end()
    stop_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Plugin register() — platform gating + tool registration
# ---------------------------------------------------------------------------

def test_register_refuses_on_windows():
    import plugins.google_meet as plugin

    calls = {"tools": [], "cli": [], "hooks": []}

    class _Ctx:
        def register_tool(self, **kw): calls["tools"].append(kw["name"])
        def register_cli_command(self, **kw): calls["cli"].append(kw["name"])
        def register_hook(self, name, fn): calls["hooks"].append(name)

    with patch.object(plugin.platform, "system", return_value="Windows"):
        plugin.register(_Ctx())

    assert calls == {"tools": [], "cli": [], "hooks": []}


# ---------------------------------------------------------------------------
# v2: process_manager.enqueue_say + realtime-mode passthrough
# ---------------------------------------------------------------------------

def test_enqueue_say_requires_text():
    from plugins.google_meet import process_manager as pm
    assert pm.enqueue_say("")["ok"] is False
    assert pm.enqueue_say("   ")["ok"] is False


# ---------------------------------------------------------------------------
# v3: NodeClient routing from tool handlers
# ---------------------------------------------------------------------------


def test_cli_register_includes_node_subcommand():
    """`hermes meet` argparse tree includes the node subtree."""
    import argparse
    from plugins.google_meet.cli import register_cli

    parser = argparse.ArgumentParser(prog="hermes meet")
    register_cli(parser)

    # Parse a known-good node invocation to prove the subtree is wired.
    ns = parser.parse_args(["node", "list"])
    assert ns.meet_command == "node"
    assert ns.node_cmd == "list"


# ---------------------------------------------------------------------------
# v2.1: new _BotState fields + status dict shape
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Admission detection + barge-in helper
# ---------------------------------------------------------------------------

def test_looks_like_human_speaker():
    from plugins.google_meet.meet_bot import _looks_like_human_speaker

    # Blank, "unknown", "you", and the bot's own name → not human (no barge-in)
    for s in ("", "   ", "Unknown", "unknown", "You", "you", "Hermes Agent", "hermes agent"):
        assert not _looks_like_human_speaker(s, "Hermes Agent"), f"{s!r} should NOT be human"
    # Real names → human (barge-in)
    for s in ("Alice", "Bob Lee", "@teknium"):
        assert _looks_like_human_speaker(s, "Hermes Agent"), f"{s!r} SHOULD be human"


def test_detect_admission_returns_false_on_error():
    from plugins.google_meet.meet_bot import _detect_admission

    class _FakePage:
        def evaluate(self, _js): raise RuntimeError("boom")

    assert _detect_admission(_FakePage()) is False


# ---------------------------------------------------------------------------
# Post-admission microphone unmute (_ensure_mic_on)
# ---------------------------------------------------------------------------

class _Locator:
    """Fake Playwright locator whose ``.first`` is one of the fake buttons."""

    def __init__(self, btn):
        self._btn = btn

    @property
    def first(self):
        return self._btn


class _PresentBtn:
    def __init__(self):
        self.clicks = 0

    def count(self):
        return 1

    def is_visible(self):
        return True

    def click(self, timeout=0):
        self.clicks += 1


class _AbsentBtn:
    def __init__(self):
        self.clicks = 0  # never incremented; keeps the fake's interface uniform

    def count(self):
        return 0

    def is_visible(self):
        return False


class _MicPage:
    """Fake page: ``muted=True`` means the "Turn on microphone" toggle exists."""

    def __init__(self, muted):
        self._muted = muted
        self.muted_btn = _PresentBtn() if muted else _AbsentBtn()
        self.unmuted_btn = _AbsentBtn() if muted else _PresentBtn()

    def locator(self, sel):
        if "Turn on microphone" in sel:
            return _Locator(self.muted_btn)
        return _Locator(self.unmuted_btn)


def test_ensure_mic_on_clicks_toggle_when_muted():
    from plugins.google_meet.meet_bot import _ensure_mic_on

    page = _MicPage(muted=True)
    assert _ensure_mic_on(page) == "unmuted_clicked"
    assert page.muted_btn.clicks == 1


def test_ensure_mic_on_leaves_already_unmuted_alone():
    from plugins.google_meet.meet_bot import _ensure_mic_on

    page = _MicPage(muted=False)
    assert _ensure_mic_on(page) == "unmuted"
    assert page.unmuted_btn.clicks == 0


def test_ensure_mic_on_unknown_when_no_toggle_found():
    from plugins.google_meet.meet_bot import _ensure_mic_on

    class _Page:
        def locator(self, _sel):
            return _Locator(_AbsentBtn())

    assert _ensure_mic_on(_Page()) == "unknown"


def test_ensure_mic_on_survives_locator_errors():
    from plugins.google_meet.meet_bot import _ensure_mic_on

    class _Page:
        def locator(self, _sel):
            raise RuntimeError("boom")

    assert _ensure_mic_on(_Page()) == "unknown"


# ---------------------------------------------------------------------------
# Streaming PCM pump tail loop (_pcm_tail_loop)
# ---------------------------------------------------------------------------

class _FakeStdin:
    def __init__(self):
        self.received = bytearray()
        self.closed = False

    def write(self, b):
        self.received.extend(b)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class _FakePump:
    def __init__(self, stdin, poll_result=None):
        self.stdin = stdin
        self._poll_result = poll_result

    def poll(self):
        return self._poll_result


def _append_pcm(path: Path, data: bytes) -> None:
    # Mirrors RealtimeSession.speak(): open 'ab', write, close.
    with open(path, "ab") as f:
        f.write(data)


def test_pcm_tail_loop_survives_empty_file_and_streams_appended_bytes(tmp_path):
    from plugins.google_meet.meet_bot import _pcm_tail_loop

    pcm = tmp_path / "speaker.pcm"
    pcm.write_bytes(b"")  # production truncates the sink before pump start
    stdin = _FakeStdin()
    stop = {"stop": False}

    t = threading.Thread(target=_pcm_tail_loop, args=(_FakePump(stdin), pcm, stop, 0.01))
    t.start()
    try:
        # The pump must NOT exit just because the file is empty at start.
        time.sleep(0.1)
        assert t.is_alive(), "tail loop exited on an empty sink file"

        # Bytes appended after startup must reach the pump's stdin.
        chunk = b"\x00\x01" * 10
        _append_pcm(pcm, chunk)
        deadline = time.time() + 2.0
        while len(stdin.received) < len(chunk) and time.time() < deadline:
            time.sleep(0.01)
        assert bytes(stdin.received) == chunk

        # Later appends keep flowing.
        chunk2 = b"\xff" * 25
        _append_pcm(pcm, chunk2)
        deadline = time.time() + 2.0
        while len(stdin.received) < len(chunk) + len(chunk2) and time.time() < deadline:
            time.sleep(0.01)
        assert bytes(stdin.received) == chunk + chunk2
    finally:
        stop["stop"] = True
        t.join(timeout=1.0)
    assert stdin.closed


def test_pcm_tail_loop_stops_when_told(tmp_path):
    from plugins.google_meet.meet_bot import _pcm_tail_loop

    pcm = tmp_path / "speaker.pcm"
    pcm.write_bytes(b"")
    stdin = _FakeStdin()
    stop = {"stop": False}

    t = threading.Thread(target=_pcm_tail_loop, args=(_FakePump(stdin), pcm, stop, 0.01))
    t.start()
    time.sleep(0.05)
    stop["stop"] = True
    t.join(timeout=1.0)
    assert not t.is_alive()
    assert stdin.closed


def test_pcm_tail_loop_exits_when_pump_dies(tmp_path):
    from plugins.google_meet.meet_bot import _pcm_tail_loop

    class _DeadStdin:
        def write(self, _b):
            raise BrokenPipeError("pump gone")

        def flush(self):
            pass

        def close(self):
            pass

    pcm = tmp_path / "speaker.pcm"
    pcm.write_bytes(b"")
    stop = {"stop": False}

    t = threading.Thread(
        target=_pcm_tail_loop,
        args=(_FakePump(_DeadStdin()), pcm, stop, 0.01),
    )
    t.start()
    _append_pcm(pcm, b"\x00" * 100)  # triggers a write that fails
    t.join(timeout=1.0)
    assert not t.is_alive()


def test_pcm_tail_loop_exits_when_pump_process_gone_while_idle(tmp_path):
    from plugins.google_meet.meet_bot import _pcm_tail_loop

    pcm = tmp_path / "speaker.pcm"
    pcm.write_bytes(b"")
    stdin = _FakeStdin()
    stop = {"stop": False}

    # poll() != None means the pump process has exited; the loop must
    # notice even while the sink file is idle (no pending writes).
    t = threading.Thread(
        target=_pcm_tail_loop,
        args=(_FakePump(stdin, poll_result=1), pcm, stop, 0.01),
    )
    t.start()
    t.join(timeout=1.0)
    assert not t.is_alive()
    assert len(stdin.received) == 0


def test_pcm_tail_loop_waits_for_late_sink_file(tmp_path):
    from plugins.google_meet.meet_bot import _pcm_tail_loop

    pcm = tmp_path / "speaker.pcm"  # deliberately NOT created yet
    stdin = _FakeStdin()
    stop = {"stop": False}

    t = threading.Thread(target=_pcm_tail_loop, args=(_FakePump(stdin), pcm, stop, 0.01))
    t.start()
    try:
        time.sleep(0.05)
        assert t.is_alive(), "tail loop gave up before the sink file appeared"
        # The file appears empty (as in production); let the loop open it
        # at EOF, then grow it and confirm appended bytes still flow.
        pcm.write_bytes(b"")
        time.sleep(0.05)
        _append_pcm(pcm, b"\x00" * 10)
        deadline = time.time() + 2.0
        while len(stdin.received) < 10 and time.time() < deadline:
            time.sleep(0.01)
        assert bytes(stdin.received) == b"\x00" * 10
    finally:
        stop["stop"] = True
        t.join(timeout=1.0)


def test_realtime_speaker_reports_missing_pump_binary(monkeypatch, tmp_path):
    import plugins.google_meet.meet_bot as mb
    from plugins.google_meet.meet_bot import _start_realtime_speaker

    class _FakeSession:
        def __init__(self, **kwargs):
            self.audio_bytes_out = 0

        def connect(self):
            pass

    class _FakeSpeaker:
        def __init__(self, **kwargs):
            pass

        def run_until_stopped(self, stop_fn):
            return None

    monkeypatch.setattr(
        "plugins.google_meet.realtime.openai_client.RealtimeSession", _FakeSession
    )
    monkeypatch.setattr(
        "plugins.google_meet.realtime.openai_client.RealtimeSpeaker", _FakeSpeaker
    )

    def _missing_pump(*args, **kwargs):
        raise FileNotFoundError("paplay")

    monkeypatch.setattr("subprocess.Popen", _missing_pump)

    out = tmp_path / "meet"
    state = mb._BotState(out_dir=out, meeting_id="abc", url="https://meet.google.com/abc-defg-hij")
    rt = {
        "session": None,
        "speaker_thread": None,
        "speaker_stop": None,
        "pcm_pump": None,
        "pcm_tail_thread": None,
    }
    _start_realtime_speaker(
        rt=rt,
        out_dir=out,
        bridge_info={"platform": "linux", "write_target": "hermes_meet_sink"},
        api_key="sk-test",
        model="gpt-realtime",
        voice="alloy",
        instructions="",
        stop_flag={"stop": False},
        state=state,
    )
    assert "paplay not found" in (state.error or "")
    assert rt["pcm_pump"] is None
    assert rt["pcm_tail_thread"] is None


def test_realtime_speaker_builds_macos_ffmpeg_pump_argv(monkeypatch, tmp_path):
    import plugins.google_meet.meet_bot as mb
    from plugins.google_meet.meet_bot import _start_realtime_speaker

    class _FakeSession:
        def __init__(self, **kwargs):
            self.audio_bytes_out = 0

        def connect(self):
            pass

    class _FakeSpeaker:
        def __init__(self, **kwargs):
            pass

        def run_until_stopped(self, stop_fn):
            return None

    monkeypatch.setattr(
        "plugins.google_meet.realtime.openai_client.RealtimeSession", _FakeSession
    )
    monkeypatch.setattr(
        "plugins.google_meet.realtime.openai_client.RealtimeSpeaker", _FakeSpeaker
    )
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None
    )
    monkeypatch.setattr(mb, "_mac_audio_device_index", lambda device_name: "3")

    captured = {}

    def _fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _FakePump(_FakeStdin())

    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    out = tmp_path / "meet"
    state = mb._BotState(out_dir=out, meeting_id="abc", url="https://meet.google.com/abc-defg-hij")
    stop = {"stop": False}
    rt = {
        "session": None,
        "speaker_thread": None,
        "speaker_stop": None,
        "pcm_pump": None,
        "pcm_tail_thread": None,
    }
    _start_realtime_speaker(
        rt=rt,
        out_dir=out,
        bridge_info={"platform": "darwin", "write_target": "BlackHole 2ch"},
        api_key="sk-test",
        model="gpt-realtime",
        voice="alloy",
        instructions="",
        stop_flag=stop,
        state=state,
    )
    argv = captured["argv"]
    assert argv[0] == "ffmpeg"
    assert argv[argv.index("-i") + 1] == "-"  # raw PCM from stdin
    # The output-side "-f" (after the "-i -" input) selects audiotoolbox.
    assert argv[argv.index("-f", argv.index("-i")) + 1] == "audiotoolbox"
    # ffmpeg rejects the command without a positional output arg.
    assert argv[-1] == "-"
    assert captured["kwargs"]["stdin"] is not None  # PIPE
    assert rt["pcm_pump"] is not None
    assert rt["pcm_tail_thread"] is not None
    stop["stop"] = True
    rt["pcm_tail_thread"].join(timeout=1.0)


# ---------------------------------------------------------------------------
# Realtime session counters + cancel_response (barge-in)
# ---------------------------------------------------------------------------

def test_realtime_session_cancel_response_when_disconnected():
    from plugins.google_meet.realtime.openai_client import RealtimeSession

    sess = RealtimeSession(api_key="sk-test", audio_sink_path=None)
    # No _ws yet — cancel should no-op and return False.
    assert sess.cancel_response() is False


# ---------------------------------------------------------------------------
# hermes meet install CLI
# ---------------------------------------------------------------------------


def test_cmd_install_refuses_windows(capsys):
    from plugins.google_meet.cli import _cmd_install

    with patch("plugins.google_meet.cli.platform" if False else "platform.system",
               return_value="Windows"):
        rc = _cmd_install(realtime=False, assume_yes=True)
    assert rc == 1
    out = capsys.readouterr().out
    assert "Windows" in out


