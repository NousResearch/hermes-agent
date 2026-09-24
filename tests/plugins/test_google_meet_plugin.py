"""Google Meet plugin behavior contracts."""

from __future__ import annotations
import asyncio
import json
import os
import signal
import subprocess
import sys
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


def test_parse_duration():
    from plugins.google_meet.meet_bot import _parse_duration

    assert _parse_duration("30m") == 30 * 60
    assert _parse_duration("2h") == 2 * 3600
    assert _parse_duration("90s") == 90
    assert _parse_duration("90") == 90
    assert _parse_duration("") is None
    assert _parse_duration("bogus") is None


def test_start_refuses_unsafe_url():
    from plugins.google_meet import process_manager as pm

    res = pm.start("https://evil.example.com/abc-defg-hij")
    assert res["ok"] is False
    assert "refusing" in res["error"]


def test_status_reports_no_active_meeting():
    from plugins.google_meet import process_manager as pm

    assert pm.status()["ok"] is False
    assert pm.transcript()["ok"] is False
    assert pm.stop()["ok"] is False


def test_start_spawns_subprocess_and_writes_active_pointer(tmp_path):
    """Verify start() wires env vars correctly and records the pid."""
    from plugins.google_meet import process_manager as pm

    class _FakeProc:
        def __init__(self, pid):
            self.pid = pid

    captured_env = {}
    captured_argv = []

    def _fake_popen(argv, **kwargs):
        captured_argv.extend(argv)
        captured_env.update(kwargs.get("env") or {})
        return _FakeProc(99999)

    with patch.object(pm.subprocess, "Popen", side_effect=_fake_popen):
        # Also prevent pid liveness probe from stomping on our real pids
        with patch.object(pm, "_pid_alive", return_value=False):
            res = pm.start(
                "https://meet.google.com/abc-defg-hij",
                guest_name="Test Bot",
                duration="15m",
            )

    assert res["ok"] is True
    assert res["meeting_id"] == "abc-defg-hij"
    assert res["pid"] == 99999
    assert captured_env["HERMES_MEET_URL"] == "https://meet.google.com/abc-defg-hij"
    assert captured_env["HERMES_MEET_GUEST_NAME"] == "Test Bot"
    assert captured_env["HERMES_MEET_DURATION"] == "15m"
    # python -m plugins.google_meet.meet_bot
    assert any("plugins.google_meet.meet_bot" in a for a in captured_argv)

    # .active.json points at the bot
    active = pm._read_active()
    assert active is not None
    assert active["pid"] == 99999
    assert active["meeting_id"] == "abc-defg-hij"
    assert active["duration"] == "15m"


def _capture_start_env_for_profile(profile_home, config_text):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli import config as config_mod
    from plugins.google_meet import process_manager as pm

    profile_home.mkdir(parents=True, exist_ok=True)
    (profile_home / "config.yaml").write_text(config_text, encoding="utf-8")
    config_mod._LOAD_CONFIG_CACHE.clear()

    captured = {}

    class _FakeProc:
        pid = 99996

    def _fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["env"] = dict(kwargs["env"])
        return _FakeProc()

    token = set_hermes_home_override(profile_home)
    try:
        with patch.object(pm.subprocess, "Popen", side_effect=_fake_popen), \
             patch.object(pm, "_pid_alive", return_value=False):
            result = pm.start("https://meet.google.com/abc-defg-hij")
    finally:
        reset_hermes_home_override(token)

    assert result["ok"] is True
    return captured


def test_start_bridges_profile_meet_config_without_ambient_leak(tmp_path, monkeypatch):
    ambient = {
        "HERMES_MEET_DEBUG_STATUS": "ambient-debug",
        "HERMES_MEET_PROXY_SERVER": "http://ambient.invalid:8080",
        "HERMES_MEET_PROXY_BYPASS": "ambient-bypass",
        "HERMES_MEET_REALTIME_READY_TIMEOUT": "999",
        "HERMES_MEET_STALL_AFTER": "999",
        "HERMES_MEET_XVFB": "force",
    }
    for key, value in ambient.items():
        monkeypatch.setenv(key, value)

    profile_a = _capture_start_env_for_profile(
        tmp_path / "profile-a",
        """
google_meet:
  debug_status: true
  xvfb: disabled
  proxy:
    server: http://profile-a.example:8080
    bypass: ""
  realtime_ready_timeout: 23
  stall_after: 47
""",
    )
    profile_b = _capture_start_env_for_profile(
        tmp_path / "profile-b",
        "google_meet: {}\n",
    )

    assert profile_a["env"]["HERMES_MEET_DEBUG_STATUS"] == "1"
    assert profile_a["env"]["HERMES_MEET_PROXY_SERVER"] == "http://profile-a.example:8080"
    assert profile_a["env"]["HERMES_MEET_PROXY_BYPASS"] == ""
    assert profile_a["env"]["HERMES_MEET_REALTIME_READY_TIMEOUT"] == "23"
    assert profile_a["env"]["HERMES_MEET_STALL_AFTER"] == "47"
    assert "HERMES_MEET_XVFB" not in profile_a["env"]

    assert "HERMES_MEET_DEBUG_STATUS" not in profile_b["env"]
    assert "HERMES_MEET_PROXY_SERVER" not in profile_b["env"]
    assert "HERMES_MEET_PROXY_BYPASS" not in profile_b["env"]
    assert profile_b["env"]["HERMES_MEET_REALTIME_READY_TIMEOUT"] == "15"
    assert profile_b["env"]["HERMES_MEET_STALL_AFTER"] == "90"
    assert "HERMES_MEET_XVFB" not in profile_b["env"]


@pytest.mark.parametrize(
    ("profile_name", "bypass_yaml", "expected"),
    [
        ("default", "null", None),
        ("disabled", '\"\"', ""),
        ("custom", '\"custom.internal\"', "custom.internal"),
    ],
)
def test_start_preserves_profile_proxy_bypass_tristate(
    tmp_path, monkeypatch, profile_name, bypass_yaml, expected
):
    monkeypatch.setenv("HERMES_MEET_PROXY_SERVER", "http://ambient.invalid:8080")
    monkeypatch.setenv("HERMES_MEET_PROXY_BYPASS", "ambient-bypass")
    captured = _capture_start_env_for_profile(
        tmp_path / profile_name,
        (
            "google_meet:\n"
            "  proxy:\n"
            "    server: http://proxy.example:8080\n"
            f"    bypass: {bypass_yaml}\n"
        ),
    )

    assert captured["env"]["HERMES_MEET_PROXY_SERVER"] == "http://proxy.example:8080"
    if expected is None:
        assert "HERMES_MEET_PROXY_BYPASS" not in captured["env"]
    else:
        assert captured["env"]["HERMES_MEET_PROXY_BYPASS"] == expected


@pytest.mark.linux_only
def test_start_headed_uses_xvfb_when_display_is_missing(monkeypatch):
    """A service-mode headed launch must be wrapped with xvfb-run."""
    from plugins.google_meet import process_manager as pm

    class _FakeProc:
        pid = 99998

    captured_env = {}
    captured_argv = []

    def _fake_popen(argv, **kwargs):
        captured_argv.extend(argv)
        captured_env.update(kwargs.get("env") or {})
        return _FakeProc()

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("HERMES_MEET_XVFB", "disabled")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(pm.shutil, "which", lambda name: f"/usr/bin/{name}")

    with patch.object(pm.subprocess, "Popen", side_effect=_fake_popen), \
         patch.object(pm, "_pid_alive", return_value=False):
        res = pm.start("https://meet.google.com/abc-defg-hij", headed=True)

    assert res["ok"] is True
    assert Path(captured_argv[0]).name == "xvfb-run"
    assert captured_argv[1] == "-a"
    assert any("plugins.google_meet.meet_bot" in arg for arg in captured_argv)
    assert captured_env["HERMES_MEET_HEADED"] == "1"
    assert res["headed"] is True
    assert res["xvfb"] is True

    active = pm._read_active()
    assert active is not None
    assert active["headed"] is True
    assert active["xvfb"] is True


@pytest.mark.linux_only
def test_start_headed_rejects_without_display_or_xvfb(monkeypatch):
    """Without DISPLAY or xvfb-run, fail before spawning Chromium."""
    from plugins.google_meet import process_manager as pm

    popen_called = False

    def _fake_popen(_argv, **_kwargs):
        nonlocal popen_called
        popen_called = True
        class _FakeProc:
            pid = 99997
        return _FakeProc()

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("HERMES_MEET_XVFB", "force")
    monkeypatch.setenv("PATH", "/definitely-no-xvfb-here")

    with patch.object(pm.subprocess, "Popen", side_effect=_fake_popen), \
         patch.object(pm, "_pid_alive", return_value=False):
        res = pm.start("https://meet.google.com/abc-defg-hij", headed=True)

    assert res["ok"] is False
    assert "headed" in res["error"].lower()
    assert "xvfb-run" in res["error"]
    assert popen_called is False
    assert pm._read_active() is None


@pytest.mark.macos_only
def test_start_headed_uses_native_browser_on_darwin(monkeypatch):
    from plugins.google_meet import process_manager as pm

    class _FakeProc:
        pid = 99995

    captured_argv = []

    def _fake_popen(argv, **_kwargs):
        captured_argv.extend(argv)
        return _FakeProc()

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("HERMES_MEET_XVFB", "force")

    with patch.object(pm.shutil, "which", return_value=None) as which, \
         patch.object(pm.subprocess, "Popen", side_effect=_fake_popen), \
         patch.object(pm, "_pid_alive", return_value=False):
        result = pm.start(
            "https://meet.google.com/abc-defg-hij",
            headed=True,
        )

    assert result["ok"] is True
    assert result["xvfb"] is False
    assert captured_argv[:3] == [
        sys.executable,
        "-m",
        "plugins.google_meet.meet_bot",
    ]
    which.assert_not_called()


def test_status_clears_stale_active_pointer_when_bot_exited(tmp_path):
    """A dead bot with a final status file is no longer an active meeting."""
    from plugins.google_meet import process_manager as pm

    out_dir = tmp_path / "abc-defg-hij"
    out_dir.mkdir()
    (out_dir / "status.json").write_text(json.dumps({
        "meetingId": "abc-defg-hij",
        "exited": True,
        "leaveReason": "meet_landing",
        "error": "meet returned to landing before captions",
    }))
    pm._write_active({
        "pid": 11111,
        "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
    })

    with patch.object(pm, "_pid_alive", return_value=False):
        res = pm.status()

    assert res["ok"] is False
    assert res["reason"] == "no active meeting"
    assert res["lastStatus"]["leaveReason"] == "meet_landing"
    assert pm._read_active() is None


def test_transcript_does_not_read_last_meeting_by_default_after_status_clears_dead_active_pointer(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = tmp_path / "abc-defg-hij"
    out_dir.mkdir()
    (out_dir / "status.json").write_text(json.dumps({
        "meetingId": "abc-defg-hij",
        "exited": True,
        "leaveReason": "duration_expired",
    }))
    (out_dir / "transcript.txt").write_text(
        "[10:00:00] Alex Rivera: one\n"
        "[10:00:01] Morgan Lee: two\n",
        encoding="utf-8",
    )
    pm._write_active({
        "pid": 11111,
        "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
    })

    with patch.object(pm, "_pid_alive", return_value=False):
        status = pm.status()
    transcript = pm.transcript()

    assert status["ok"] is False
    assert transcript["ok"] is False
    assert "no active meeting" in transcript["reason"]


def test_transcript_clears_dead_active_pointer_without_prior_status_call(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = tmp_path / "abc-defg-hij"
    out_dir.mkdir()
    (out_dir / "status.json").write_text(json.dumps({
        "meetingId": "abc-defg-hij",
        "exited": True,
        "leaveReason": "duration_expired",
    }))
    (out_dir / "transcript.txt").write_text(
        "[10:00:00] Alex Rivera: one\n",
        encoding="utf-8",
    )
    pm._write_active({
        "pid": 11111,
        "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
        "session_id": "session-a",
    })

    with patch.object(pm, "_pid_alive", return_value=False):
        default_read = pm.transcript()
        finished_read = pm.transcript(include_finished=True, session_id="session-a")

    assert default_read["ok"] is False
    assert "no active meeting" in default_read["reason"]
    assert finished_read["ok"] is True
    assert finished_read["active"] is False
    assert finished_read["fromLast"] is True
    assert finished_read["stale"] is True
    assert pm._read_active() is None


def test_transcript_can_explicitly_read_finished_meeting_after_status_clears_dead_active_pointer(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = tmp_path / "abc-defg-hij"
    out_dir.mkdir()
    (out_dir / "status.json").write_text(json.dumps({
        "meetingId": "abc-defg-hij",
        "exited": True,
        "leaveReason": "duration_expired",
    }))
    (out_dir / "transcript.txt").write_text(
        "[10:00:00] Alex Rivera: one\n"
        "[10:00:01] Morgan Lee: two\n",
        encoding="utf-8",
    )
    pm._write_active({
        "pid": 11111,
        "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
        "session_id": "session-a",
    })

    with patch.object(pm, "_pid_alive", return_value=False):
        status = pm.status()
    transcript = pm.transcript(include_finished=True, session_id="session-a")

    assert status["ok"] is False
    assert transcript["ok"] is True
    assert transcript["active"] is False
    assert transcript["fromLast"] is True
    assert transcript["stale"] is True
    assert transcript["sessionId"] == "session-a"
    assert transcript["leaveReason"] == "duration_expired"
    assert transcript["total"] == 2
    assert transcript["lines"][-1].endswith("Morgan Lee: two")


def test_meet_transcript_finished_meeting_requires_matching_session_id(tmp_path):
    import subprocess
    import sys

    from plugins.google_meet import process_manager as pm
    from plugins.google_meet.tools import handle_meet_transcript

    out_dir = tmp_path / "abc-defg-hij"
    out_dir.mkdir()
    (out_dir / "status.json").write_text(
        json.dumps(
            {
                "meetingId": "abc-defg-hij",
                "exited": True,
                "leaveReason": "duration_expired",
            }
        )
    )
    line = "[10:00:00] Alex Rivera: private meeting notes"
    (out_dir / "transcript.txt").write_text(line + "\n", encoding="utf-8")
    with subprocess.Popen([sys.executable, "-c", "pass"]) as process:
        process.wait(timeout=10)
    pm._write_active(
        {
            "pid": process.pid,
            "meeting_id": "abc-defg-hij",
            "out_dir": str(out_dir),
            "url": "https://meet.google.com/abc-defg-hij",
            "started_at": 0,
            "session_id": "session-a",
        }
    )
    pm.status()

    for args, session_id in (
        ({}, "session-a"),
        ({"include_finished": True}, None),
        ({"include_finished": True}, "session-b"),
    ):
        denied = json.loads(handle_meet_transcript(args, session_id=session_id))
        assert denied["success"] is False
        assert "lines" not in denied
    allowed = json.loads(
        handle_meet_transcript({"include_finished": True}, session_id="session-a")
    )
    assert allowed["success"] is True
    assert allowed["lines"] == [line]
    assert allowed["sessionId"] == "session-a"
    assert allowed["active"] is False
    assert allowed["fromLast"] is True


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
        "pid": 12345, "meeting_id": "abc-defg-hij",
        "out_dir": str(meeting_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
    })

    with patch.object(pm, "_pid_alive", return_value=True):
        res = pm.transcript(last=2)
    assert res["ok"] is True
    assert res["total"] == 3
    assert len(res["lines"]) == 2
    assert res["lines"][-1].endswith("Alice: three")


def test_stop_signals_process_and_clears_pointer(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = tmp_path / "x-y-z"
    out_dir.mkdir()
    (out_dir / "status.json").write_text(json.dumps({
        "meetingId": "x-y-z",
        "exited": False,
        "leaveReason": None,
    }))
    pm._write_active({
        "pid": 11111, "meeting_id": "x-y-z",
        "out_dir": str(out_dir),
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
    status = json.loads((out_dir / "status.json").read_text())
    assert status["exited"] is True
    assert status["leaveReason"] == "requested"
    # .active.json cleared
    assert pm._read_active() is None


def test_meet_join_handler_missing_url_returns_error():
    from plugins.google_meet.tools import handle_meet_join

    out = json.loads(handle_meet_join({}))
    assert out["success"] is False
    assert out["error"]


def test_meet_join_handler_respects_safety_gate():
    from plugins.google_meet.tools import handle_meet_join

    with patch("plugins.google_meet.tools.check_meet_requirements", return_value=True):
        out = json.loads(handle_meet_join({"url": "https://evil.example.com/foo"}))
    assert out["success"] is False
    assert "refusing" in out["error"]


def test_meet_join_handler_returns_error_when_playwright_missing():
    from plugins.google_meet.tools import handle_meet_join

    with patch("plugins.google_meet.tools.check_meet_requirements", return_value=False):
        out = json.loads(handle_meet_join({"url": "https://meet.google.com/abc-defg-hij"}))
    assert out["success"] is False
    assert "prerequisites missing" in out["error"]


def test_meet_say_requires_text():
    from plugins.google_meet.tools import handle_meet_say

    out = json.loads(handle_meet_say({}))
    assert out["success"] is False
    assert "text is required" in out["error"]


def test_meet_say_no_active_meeting():
    from plugins.google_meet.tools import handle_meet_say

    out = json.loads(handle_meet_say({"text": "hello everyone"}))
    assert out["success"] is False
    # Falls through to pm.enqueue_say which reports no active meeting.
    assert "no active meeting" in out.get("reason", "")


def test_meet_status_and_transcript_no_active():
    from plugins.google_meet.tools import handle_meet_status, handle_meet_transcript

    assert json.loads(handle_meet_status({}))["success"] is False
    assert json.loads(handle_meet_transcript({}))["success"] is False


def test_meet_leave_no_active():
    from plugins.google_meet.tools import handle_meet_leave

    out = json.loads(handle_meet_leave({}))
    assert out["success"] is False


@pytest.mark.parametrize(
    "updates, ending_session, should_stop",
    [
        ({}, "owner", True),
        ({}, "other", False),
        ({}, "", False),
        ({"sessionId": None}, "owner", False),
        ({"persistAfterSession": True}, "owner", False),
        ({"alive": False}, "owner", False),
        ({"ok": False}, "owner", False),
    ],
)
def test_finalize_policy_requires_live_attached_session_ownership(
    updates, ending_session, should_stop
):
    from plugins.google_meet import _should_stop_owned_bot

    status = {
        "ok": True,
        "alive": True,
        "sessionId": "owner",
        "duration": "20m",
        "persistAfterSession": False,
    }
    status.update(updates)
    assert _should_stop_owned_bot(status, ending_session) is should_stop


@pytest.mark.linux_only
def test_registered_cleanup_preserves_turns_and_stops_only_owning_session(
    tmp_path, monkeypatch
):
    import plugins.google_meet as plugin
    from hermes_cli.plugins import PluginManager
    from plugins.google_meet import process_manager as pm

    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text("plugins:\n  enabled: [google_meet]\n")
    user_plugins = home / "plugins"
    user_plugins.mkdir()
    (user_plugins / "google_meet").symlink_to(
        Path(plugin.__file__).parent, target_is_directory=True
    )
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "no-bundled-plugins"))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    manager = PluginManager()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    out_dir = home / "workspace" / "meetings" / "abc-defg-hij"
    out_dir.mkdir(parents=True)
    pm._write_active(
        {
            "pid": process.pid,
            "meeting_id": "abc-defg-hij",
            "out_dir": str(out_dir),
            "session_id": "owner",
            "persist_after_session": False,
            "mode": "transcribe",
        }
    )
    try:
        manager.discover_and_load()
        manager.invoke_hook("on_session_end", session_id="owner")
        assert process.poll() is None
        manager.invoke_hook("on_session_finalize", session_id="different-session")
        assert process.poll() is None
        manager.invoke_hook("on_session_finalize", session_id="owner")
        process.wait(timeout=5)
        assert not pm.status()["ok"]
        status = json.loads((out_dir / "status.json").read_text())
        assert status["leaveReason"] == "session ended"
    finally:
        manager.unload()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_enqueue_say_requires_text():
    from plugins.google_meet import process_manager as pm
    assert pm.enqueue_say("")["ok"] is False
    assert pm.enqueue_say("   ")["ok"] is False


def test_enqueue_say_no_active_meeting():
    from plugins.google_meet import process_manager as pm
    res = pm.enqueue_say("hi team")
    assert res["ok"] is False
    assert "no active meeting" in res["reason"]


def test_enqueue_say_rejects_transcribe_mode(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    out_dir.mkdir(parents=True)
    pm._write_active({
        "pid": 0, "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir), "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0, "mode": "transcribe",
    })
    res = pm.enqueue_say("hi team")
    assert res["ok"] is False
    assert "transcribe mode" in res["reason"]


def test_enqueue_say_rejects_dead_realtime_bot(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = (
        Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    )
    out_dir.mkdir(parents=True)
    (out_dir / "status.json").write_text(
        json.dumps(
            {
                "inCall": True,
                "realtime": True,
                "realtimeReady": True,
                "realtimeAudioPumpStatus": "ready",
                "realtimeAudioPumpPid": os.getpid(),
                "localMicrophoneOn": True,
            }
        )
    )
    pm._write_active(
        {
            "pid": 12345,
            "meeting_id": "abc-defg-hij",
            "out_dir": str(out_dir),
            "url": "https://meet.google.com/abc-defg-hij",
            "started_at": 0,
            "mode": "realtime",
        }
    )

    with patch.object(pm, "_pid_alive", return_value=False):
        res = pm.enqueue_say("hello everyone")

    assert res["ok"] is False
    assert "no active meeting" in res["reason"]
    assert not (out_dir / "say_queue.jsonl").exists()


def test_enqueue_say_rejects_realtime_before_bot_is_ready(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    out_dir.mkdir(parents=True)
    (out_dir / "status.json").write_text(json.dumps({
        "inCall": True,
        "realtime": True,
        "realtimeReady": False,
    }))
    pm._write_active({
        "pid": 12345, "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir), "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0, "mode": "realtime",
    })

    with patch.object(pm, "_pid_alive", return_value=True):
        res = pm.enqueue_say("hello everyone")

    assert res["ok"] is False
    assert "realtime is not ready" in res["reason"]
    assert not (out_dir / "say_queue.jsonl").exists()


def test_enqueue_say_rejects_realtime_when_audio_pump_is_not_ready(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    out_dir.mkdir(parents=True)
    (out_dir / "status.json").write_text(json.dumps({
        "inCall": True,
        "realtime": True,
        "realtimeReady": True,
        "realtimeAudioPumpStatus": "exited",
        "realtimeAudioPumpReturnCode": 1,
        "error": None,
        "exited": False,
    }))
    pm._write_active({
        "pid": 12345, "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir), "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0, "mode": "realtime",
    })

    with patch.object(pm, "_pid_alive", return_value=True):
        res = pm.enqueue_say("hello everyone")

    assert res["ok"] is False
    assert "audio pump is not ready" in res["reason"]
    assert not (out_dir / "say_queue.jsonl").exists()


def test_enqueue_say_rejects_realtime_when_meet_microphone_is_off(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = (
        Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    )
    out_dir.mkdir(parents=True)
    (out_dir / "status.json").write_text(
        json.dumps(
            {
                "inCall": True,
                "realtime": True,
                "realtimeReady": True,
                "realtimeAudioPumpStatus": "ready",
                "realtimeAudioPumpPid": os.getpid(),
                "localMicrophoneOn": False,
                "error": None,
                "exited": False,
            }
        )
    )
    pm._write_active(
        {
            "pid": 12345,
            "meeting_id": "abc-defg-hij",
            "out_dir": str(out_dir),
            "url": "https://meet.google.com/abc-defg-hij",
            "started_at": 0,
            "mode": "realtime",
        }
    )

    with patch.object(pm, "_pid_alive", return_value=True):
        res = pm.enqueue_say("hello everyone")

    assert res["ok"] is False
    assert "microphone is not enabled" in res["reason"]
    assert not (out_dir / "say_queue.jsonl").exists()


def test_enqueue_say_writes_jsonl_in_realtime_mode():
    from plugins.google_meet import process_manager as pm

    out_dir = (
        Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    )
    out_dir.mkdir(parents=True)
    (out_dir / "status.json").write_text(
        json.dumps(
            {
                "inCall": True,
                "realtime": True,
                "realtimeReady": True,
                "realtimeAudioPumpStatus": "ready",
                "realtimeAudioPumpPid": os.getpid(),
                "localMicrophoneOn": True,
                "error": None,
                "exited": False,
            }
        )
    )
    pm._write_active(
        {
            "pid": 12345,
            "meeting_id": "abc-defg-hij",
            "out_dir": str(out_dir),
            "url": "https://meet.google.com/abc-defg-hij",
            "started_at": 0,
            "mode": "realtime",
        }
    )
    with patch.object(pm, "_pid_alive", return_value=True):
        res = pm.enqueue_say("hello everyone")
    assert res["ok"] is True
    assert "enqueued_id" in res

    queue = out_dir / "say_queue.jsonl"
    assert queue.is_file()
    lines = [json.loads(ln) for ln in queue.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0]["text"] == "hello everyone"


def test_realtime_queue_preserves_messages_enqueued_while_speaking():
    from plugins.google_meet import process_manager as pm
    from plugins.google_meet.realtime.openai_client import RealtimeSpeaker

    out_dir = (
        Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    )
    out_dir.mkdir(parents=True)
    queue_path = out_dir / "say_queue.jsonl"
    queue_path.write_text(json.dumps({"id": "first", "text": "first"}) + "\n")
    (out_dir / "status.json").write_text(
        json.dumps(
            {
                "inCall": True,
                "realtime": True,
                "realtimeReady": True,
                "realtimeAudioPumpStatus": "ready",
                "realtimeAudioPumpPid": os.getpid(),
                "localMicrophoneOn": True,
            }
        )
    )
    pm._write_active(
        {
            "pid": os.getpid(),
            "meeting_id": "abc-defg-hij",
            "out_dir": str(out_dir),
            "url": "https://meet.google.com/abc-defg-hij",
            "started_at": 0,
            "mode": "realtime",
        }
    )
    enqueued = []

    class Session:
        def speak(self, text):
            assert text == "first"
            # The producer appends after the consumer took its queue snapshot,
            # before the completed first message is removed.
            enqueued.append(pm.enqueue_say("second"))
            return {"ok": True, "bytes_written": 0, "duration_ms": 0.0}

    speaker = RealtimeSpeaker(session=Session(), queue_path=queue_path)
    speaker.run_until_stopped(lambda: bool(enqueued), poll_interval=0.01)
    assert enqueued[0]["ok"] is True
    remaining = [json.loads(line) for line in queue_path.read_text().splitlines()]
    assert remaining == [{"id": enqueued[0]["enqueued_id"], "text": "second"}]


def test_start_passes_mode_into_active_record():
    from plugins.google_meet import process_manager as pm

    class _FakeProc:
        def __init__(self, pid): self.pid = pid

    with patch.object(pm.subprocess, "Popen", return_value=_FakeProc(12345)), \
         patch.object(pm, "_pid_alive", return_value=False):
        res = pm.start(
            "https://meet.google.com/abc-defg-hij",
            mode="realtime",
        )
    assert res["ok"] is True
    assert res["mode"] == "realtime"
    assert pm._read_active()["mode"] == "realtime"


def test_start_realtime_env_vars_threaded_through():
    from plugins.google_meet import process_manager as pm

    class _FakeProc:
        def __init__(self, pid): self.pid = pid

    captured_env = {}
    def _fake_popen(argv, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return _FakeProc(11111)

    with patch.object(pm.subprocess, "Popen", side_effect=_fake_popen), \
         patch.object(pm, "_pid_alive", return_value=False):
        pm.start(
            "https://meet.google.com/abc-defg-hij",
            mode="realtime",
            realtime_model="gpt-realtime",
            realtime_voice="alloy",
            realtime_instructions="Be brief.",
            realtime_api_key="sk-test",
        )
    assert captured_env["HERMES_MEET_MODE"] == "realtime"
    assert captured_env["HERMES_MEET_REALTIME_MODEL"] == "gpt-realtime"
    assert captured_env["HERMES_MEET_REALTIME_VOICE"] == "alloy"
    assert captured_env["HERMES_MEET_REALTIME_INSTRUCTIONS"] == "Be brief."
    assert captured_env["HERMES_MEET_REALTIME_KEY"] == "sk-test"


def test_meet_join_rejects_bad_mode():
    from plugins.google_meet.tools import handle_meet_join

    out = json.loads(handle_meet_join({
        "url": "https://meet.google.com/abc-defg-hij",
        "mode": "bogus",
    }))
    assert out["success"] is False
    assert "mode must be" in out["error"]


def test_meet_join_unknown_node_returns_clear_error():
    from plugins.google_meet.tools import handle_meet_join

    out = json.loads(handle_meet_join({
        "url": "https://meet.google.com/abc-defg-hij",
        "node": "my-mac",
    }))
    assert out["success"] is False
    assert "no registered meet node" in out["error"]


def test_meet_join_rejects_auth_state_for_remote_node():
    from plugins.google_meet.tools import handle_meet_join
    from plugins.google_meet.node.registry import NodeRegistry

    reg = NodeRegistry()
    reg.add("my-mac", "ws://1.2.3.4:18789", "tok")

    out = json.loads(handle_meet_join({
        "url": "https://meet.google.com/abc-defg-hij",
        "node": "my-mac",
        "use_auth_state": True,
    }))

    assert out["success"] is False
    assert "use_auth_state is local-only" in out["error"]


def test_node_server_say_rejects_without_active_meeting(tmp_path):
    from plugins.google_meet.node import protocol as proto
    from plugins.google_meet.node.server import NodeServer

    server = NodeServer(token_path=tmp_path / "node_token.json")
    server._token = "tok"

    response = asyncio.run(server._handle_request(
        proto.make_request("say", "tok", {"text": "hello"})
    ))

    assert response["type"] == "response"
    assert response["payload"]["ok"] is False
    assert "no active meeting" in response["payload"]["reason"]


def test_node_server_say_rejects_transcribe_mode(tmp_path):
    from plugins.google_meet import process_manager as pm
    from plugins.google_meet.node import protocol as proto
    from plugins.google_meet.node.server import NodeServer

    out_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    out_dir.mkdir(parents=True)
    pm._write_active({
        "pid": 0,
        "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
        "mode": "transcribe",
    })
    server = NodeServer(token_path=tmp_path / "node_token.json")
    server._token = "tok"

    response = asyncio.run(server._handle_request(
        proto.make_request("say", "tok", {"text": "hello"})
    ))

    assert response["type"] == "response"
    assert response["payload"]["ok"] is False
    assert "transcribe mode" in response["payload"]["reason"]


def test_node_server_say_uses_realtime_queue(tmp_path):
    from plugins.google_meet import process_manager as pm
    from plugins.google_meet.node import protocol as proto
    from plugins.google_meet.node.server import NodeServer

    out_dir = (
        Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    )
    out_dir.mkdir(parents=True)
    (out_dir / "status.json").write_text(
        json.dumps(
            {
                "inCall": True,
                "realtime": True,
                "realtimeReady": True,
                "realtimeAudioPumpStatus": "ready",
                "realtimeAudioPumpPid": os.getpid(),
                "localMicrophoneOn": True,
            }
        )
    )
    pm._write_active(
        {
            "pid": 12345,
            "meeting_id": "abc-defg-hij",
            "out_dir": str(out_dir),
            "url": "https://meet.google.com/abc-defg-hij",
            "started_at": 0,
            "mode": "realtime",
        }
    )
    server = NodeServer(token_path=tmp_path / "node_token.json")
    server._token = "tok"

    with patch.object(pm, "_pid_alive", return_value=True):
        response = asyncio.run(
            server._handle_request(proto.make_request("say", "tok", {"text": "hello"}))
        )

    assert response["type"] == "response"
    assert response["payload"]["ok"] is True
    assert response["payload"]["enqueued_id"]
    queued = [
        json.loads(line)
        for line in (out_dir / "say_queue.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert queued == [{"id": response["payload"]["enqueued_id"], "text": "hello"}]


def test_meet_join_auto_node_ambiguous_returns_error():
    from plugins.google_meet.tools import handle_meet_join
    from plugins.google_meet.node.registry import NodeRegistry

    reg = NodeRegistry()
    reg.add("a", "ws://1.2.3.4:18789", "tok")
    reg.add("b", "ws://5.6.7.8:18789", "tok")

    out = json.loads(handle_meet_join({
        "url": "https://meet.google.com/abc-defg-hij",
        "node": "auto",
    }))
    assert out["success"] is False
    assert "no registered meet node" in out["error"]


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


def test_cli_join_rejects_auth_state_for_remote_node(capsys):
    from plugins.google_meet.cli import _cmd_join
    from plugins.google_meet.node.registry import NodeRegistry

    reg = NodeRegistry()
    reg.add("my-mac", "ws://1.2.3.4:18789", "tok")

    rc = _cmd_join(
        "https://meet.google.com/abc-defg-hij",
        guest_name="Hermes Agent",
        duration=None,
        headed=False,
        mode="transcribe",
        node="my-mac",
        use_auth_state=True,
        persist_after_session=False,
    )

    assert rc == 1
    assert "use_auth_state is local-only" in capsys.readouterr().out


def test_cli_say_subcommand_exists():
    import argparse
    from plugins.google_meet.cli import register_cli

    parser = argparse.ArgumentParser(prog="hermes meet")
    register_cli(parser)

    ns = parser.parse_args(["say", "hello team", "--node", "my-mac"])
    assert ns.text == "hello team"
    assert ns.node == "my-mac"


def test_looks_like_human_speaker():
    from plugins.google_meet.meet_bot import _looks_like_human_speaker

    # Blank, "unknown", "you", and the bot's own name → not human (no barge-in)
    for s in ("", "   ", "Unknown", "unknown", "You", "you", "Hermes Agent", "hermes agent"):
        assert not _looks_like_human_speaker(s, "Hermes Agent"), f"{s!r} should NOT be human"
    # Real names → human (barge-in)
    for s in ("Alice", "Bob Lee", "@teknium"):
        assert _looks_like_human_speaker(s, "Hermes Agent"), f"{s!r} SHOULD be human"


def test_realtime_session_cancel_response_when_disconnected():
    from plugins.google_meet.realtime.openai_client import RealtimeSession

    sess = RealtimeSession(api_key="sk-test", audio_sink_path=None)
    # No _ws yet — cancel should no-op and return False.
    assert sess.cancel_response() is False


def test_realtime_session_cancel_response_sends_cancel_frame():
    from plugins.google_meet.realtime.openai_client import RealtimeSession

    sess = RealtimeSession(api_key="sk-test", audio_sink_path=None)
    sent = []

    class _FakeWs:
        def send(self, msg): sent.append(msg)

    sess._ws = _FakeWs()
    assert sess.cancel_response() is True
    assert len(sent) == 1
    import json as _j
    envelope = _j.loads(sent[0])
    assert envelope == {"type": "response.cancel"}


def test_realtime_session_counters_initialized():
    from plugins.google_meet.realtime.openai_client import RealtimeSession

    sess = RealtimeSession(api_key="sk-test", audio_sink_path=None)
    assert sess.audio_bytes_out == 0
    assert sess.last_audio_out_at is None


def test_cli_install_subcommand_is_registered():
    import argparse
    from plugins.google_meet.cli import register_cli

    parser = argparse.ArgumentParser(prog="hermes meet")
    register_cli(parser)

    ns = parser.parse_args(["install"])
    assert ns.meet_command == "install"
    assert ns.realtime is False
    assert ns.yes is False


def test_cli_install_flags_parse():
    import argparse
    from plugins.google_meet.cli import register_cli

    parser = argparse.ArgumentParser(prog="hermes meet")
    register_cli(parser)

    ns = parser.parse_args(["install", "--realtime", "--yes"])
    assert ns.realtime is True
    assert ns.yes is True


@pytest.mark.linux_only
def test_cmd_install_runs_pip_and_playwright(capsys):
    """Dependency installation selects the supported commands."""
    from plugins.google_meet.cli import _cmd_install

    calls = []

    class _FakeRes:
        def __init__(self, rc=0):
            self.returncode = rc

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        return _FakeRes(0)

    with (
        patch("subprocess.run", side_effect=_fake_run),
        patch("shutil.which", return_value="/usr/bin/paplay"),
    ):
        rc = _cmd_install(realtime=False, assume_yes=True)
    assert rc == 0
    # First invocation: dependency install via the uv→pip ladder
    # (shutil.which is mocked truthy, so the uv tier is taken: `<uv> pip install ...`)
    pip_cmds = [
        c for c in calls if "install" in c and "playwright" in c and "websockets" in c
    ]
    assert pip_cmds, f"no dependency install run: {calls}"
    # Second: playwright install chromium
    pw_cmds = [
        c for c in calls if len(c) > 2 and c[1:4] == ["-m", "playwright", "install"]
    ]
    assert pw_cmds, f"no playwright install run: {calls}"
    assert "chromium" in pw_cmds[0]


@pytest.mark.linux_only
def test_cmd_install_realtime_skips_when_deps_present():
    """When paplay + pactl are already on PATH, no sudo call happens."""
    from plugins.google_meet.cli import _cmd_install

    calls = []

    class _FakeRes:
        def __init__(self, rc=0):
            self.returncode = rc

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        return _FakeRes(0)

    with (
        patch("subprocess.run", side_effect=_fake_run),
        patch("shutil.which", return_value="/usr/bin/paplay"),
    ):
        rc = _cmd_install(realtime=True, assume_yes=True)
    assert rc == 0
    # No sudo apt-get call — paplay was already on PATH.
    sudo_calls = [c for c in calls if c and c[0] == "sudo"]
    assert sudo_calls == [], f"unexpected sudo invocation: {sudo_calls}"


@pytest.mark.linux_only
@pytest.mark.parametrize("frontdoor", ["tool", "cli", "remote"])
@pytest.mark.parametrize(
    "scenario", ["guest", "authenticated", "detached_duration", "realtime"]
)
def test_linux_join_frontdoors_preserve_auth_and_session_contracts(
    tmp_path, monkeypatch, frontdoor, scenario
):
    _assert_join_frontdoor_contract(tmp_path, monkeypatch, frontdoor, scenario)


@pytest.mark.macos_only
@pytest.mark.parametrize("frontdoor", ["tool", "cli", "remote"])
@pytest.mark.parametrize(
    "scenario", ["guest", "authenticated", "detached_duration", "realtime"]
)
def test_macos_join_frontdoors_preserve_auth_and_session_contracts(
    tmp_path, monkeypatch, frontdoor, scenario
):
    _assert_join_frontdoor_contract(tmp_path, monkeypatch, frontdoor, scenario)


def _assert_join_frontdoor_contract(tmp_path, monkeypatch, frontdoor, scenario):
    import argparse

    pytest.importorskip("playwright")
    from plugins.google_meet import cli, process_manager as pm
    from plugins.google_meet.tools import handle_meet_join

    auth_path = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "auth.json"
    auth_path.parent.mkdir(parents=True)
    auth_path.write_text(json.dumps({"cookies": [], "origins": []}))
    real_popen = subprocess.Popen
    children = []
    # Replace only the external browser process with a local consumer of the
    # exact production config bridge; do not mock either launch entry point.
    child_code = """
import json
from plugins.google_meet.meet_bot import _config_from_env
cfg = _config_from_env()
(cfg.out_dir / 'launch-proof.json').write_text(json.dumps({
    'auth_state': cfg.auth_state, 'duration_s': cfg.duration_s,
    'realtime': cfg.realtime, 'guest_name': cfg.guest_name,
}))
"""

    def local_consumer(_argv, **kwargs):
        process = real_popen([sys.executable, "-c", child_code], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(pm.subprocess, "Popen", local_consumer)
    options: dict[str, str | bool] = {
        "url": "https://meet.google.com/abc-defg-hij",
        "guest_name": "Test participant",
    }
    if scenario == "authenticated":
        options["use_auth_state"] = True
    elif scenario == "detached_duration":
        options.update(duration="15m", persist_after_session=True)
    elif scenario == "realtime":
        options["mode"] = "realtime"
    server = None
    server_thread = None
    if frontdoor == "remote":
        from plugins.google_meet.node import protocol
        from plugins.google_meet.node.registry import NodeRegistry
        from plugins.google_meet.node.server import NodeServer
        from websockets.sync.server import serve

        node = NodeServer(token_path=tmp_path / "node-token.json")
        token = node.ensure_token()

        def handler(connection):
            request = protocol.decode(connection.recv())
            response = asyncio.run(node._handle_request(request))
            connection.send(protocol.encode(response))

        server = serve(handler, "127.0.0.1", 0)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        port = server.socket.getsockname()[1]
        NodeRegistry().add("test-node", f"ws://127.0.0.1:{port}", token)
        options["node"] = "auto"
    try:
        if frontdoor != "cli":
            result = json.loads(handle_meet_join(options, session_id="owner"))
            if frontdoor == "remote" and scenario == "authenticated":
                assert result["success"] is False
                assert "local-only" in result["error"]
                assert not children
                return
            assert result["success"] is True, result
        else:
            parser = argparse.ArgumentParser()
            cli.register_cli(parser)
            argv = [
                "join",
                str(options["url"]),
                "--guest-name",
                str(options["guest_name"]),
            ]
            for key, value in options.items():
                if key in {"url", "guest_name"}:
                    continue
                argv.append("--" + key.replace("_", "-"))
                if value is not True:
                    argv.append(str(value))
            args = parser.parse_args(argv)
            assert args.func(args) == 0
        record = pm._read_active()
        assert children[0].wait(timeout=10) == 0
        observed = json.loads(
            (Path(record["out_dir"]) / "launch-proof.json").read_text()
        )
        if scenario == "authenticated":
            assert observed["auth_state"] == str(auth_path)
        else:
            assert not observed["auth_state"]
        assert observed["duration_s"] == (
            900.0 if scenario == "detached_duration" else None
        )
        assert observed["realtime"] is (scenario == "realtime")
        assert observed["guest_name"] == "Test participant"
        assert record["persist_after_session"] is (scenario == "detached_duration")
        assert record["session_id"] == (None if frontdoor == "cli" else "owner")
    finally:
        if server is not None:
            server.shutdown()
        if server_thread is not None:
            server_thread.join(timeout=5)
        for process in children:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
