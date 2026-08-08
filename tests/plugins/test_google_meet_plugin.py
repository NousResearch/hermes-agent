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
import sys
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

    transcript = (out / "transcript.txt").read_text(encoding="utf-8")
    assert "Alice: Hey everyone" in transcript
    assert "Bob: Let's start" in transcript
    # dedup — Alice line appears exactly once
    assert transcript.count("Alice: Hey everyone") == 1

    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
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


# ---------------------------------------------------------------------------
# Sidecar env hygiene: meet_bot Chromium env + meet_bot spawn env must not
# carry gateway credentials (BWS vault token, provider keys, *_PASSWORD)
# into the child, while the plugin's own keys survive.
# ---------------------------------------------------------------------------


def test_build_chrome_env_scrubs_credentials_keeps_bot_vars(monkeypatch):
    from plugins.google_meet.meet_bot import _build_chrome_env

    monkeypatch.setenv("GATEWAY_RELAY_SECRET", "relay-secret")
    monkeypatch.setenv("EMAIL_PASSWORD", "mail-pass")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setenv("HERMES_MEET_LOBBY_TIMEOUT", "42")

    env = _build_chrome_env(
        {"enabled": True, "bridge_info": {"platform": "linux", "device_name": "virtsrc"}}
    )

    # Gateway credentials scrubbed from the Chromium child env.
    assert "GATEWAY_RELAY_SECRET" not in env
    assert "EMAIL_PASSWORD" not in env
    assert "OPENAI_API_KEY" not in env
    # Non-secret meeting/display vars survive; the bot's own PULSE_SOURCE
    # override is applied on top of the sanitized base.
    assert env["DISPLAY"] == ":99"
    assert env["HERMES_MEET_LOBBY_TIMEOUT"] == "42"
    assert env["PULSE_SOURCE"] == "virtsrc"


def test_apply_chrome_env_replaces_process_env_cleanly(monkeypatch):
    from plugins.google_meet.meet_bot import _apply_chrome_env

    monkeypatch.setenv("GATEWAY_RELAY_SECRET", "relay-secret")
    monkeypatch.setenv("EMAIL_PASSWORD", "mail-pass")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DISPLAY", ":99")

    snapshot = os.environ.copy()
    try:
        _apply_chrome_env(
            {"PULSE_SOURCE": "virtsrc", "DISPLAY": ":99", "PATH": snapshot["PATH"]}
        )
        # The merge replaces os.environ wholesale — scrubbed credentials
        # cannot linger in the env Playwright's Chromium inherits.
        assert "GATEWAY_RELAY_SECRET" not in os.environ
        assert "EMAIL_PASSWORD" not in os.environ
        assert "OPENAI_API_KEY" not in os.environ
        assert os.environ["PULSE_SOURCE"] == "virtsrc"
        assert os.environ["DISPLAY"] == ":99"
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


def test_start_spawns_meet_bot_with_scrubbed_env(tmp_path, monkeypatch):
    from plugins.google_meet import process_manager as pm

    monkeypatch.setenv("GATEWAY_RELAY_SECRET", "relay-secret")
    monkeypatch.setenv("EMAIL_PASSWORD", "mail-pass")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(pm, "_read_active", lambda: None)

    captured = {}

    class _Proc:
        pid = 4242

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return _Proc()

    monkeypatch.setattr(pm.subprocess, "Popen", _fake_popen)

    res = pm.start(
        "https://meet.google.com/abc-defg-hij",
        out_dir=tmp_path / "abc-defg-hij",
        realtime_api_key="rt-key",
    )
    assert res["ok"] is True

    env = captured["env"]
    # Gateway credentials must not reach the meet_bot child…
    assert "GATEWAY_RELAY_SECRET" not in env
    assert "EMAIL_PASSWORD" not in env
    assert "OPENAI_API_KEY" not in env
    # …while the meeting config (incl. the explicitly-passed realtime key)
    # still travels.
    assert env["HERMES_MEET_URL"] == "https://meet.google.com/abc-defg-hij"
    assert env["HERMES_MEET_REALTIME_KEY"] == "rt-key"
    assert captured["cmd"] == [sys.executable, "-m", "plugins.google_meet.meet_bot"]


