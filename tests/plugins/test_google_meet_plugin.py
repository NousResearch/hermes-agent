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


def test_is_safe_meet_url_rejects_non_meet_urls():
    from plugins.google_meet.meet_bot import _is_safe_meet_url

    # wrong host
    assert not _is_safe_meet_url("https://evil.example.com/abc-defg-hij")
    # wrong scheme
    assert not _is_safe_meet_url("http://meet.google.com/abc-defg-hij")
    # malformed code
    assert not _is_safe_meet_url("https://meet.google.com/not-a-meet-code")
    # subdomain hijack attempts
    assert not _is_safe_meet_url("https://meet.google.com.evil.com/abc-defg-hij")
    assert not _is_safe_meet_url("https://notmeet.google.com/abc-defg-hij")
    # empty / wrong type
    assert not _is_safe_meet_url("")
    assert not _is_safe_meet_url(None)  # type: ignore[arg-type]
    assert not _is_safe_meet_url(123)  # type: ignore[arg-type]


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


def test_bot_state_ignores_blank_text(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    state = _BotState(out_dir=tmp_path / "s", meeting_id="x-y-z",
                      url="https://meet.google.com/x-y-z")
    state.record_caption("Alice", "")
    state.record_caption("Alice", "   ")
    state.record_caption("", "text but no speaker")

    status = json.loads((tmp_path / "s" / "status.json").read_text())
    assert status["transcriptLines"] == 1
    # blank-speaker falls back to "Unknown"
    assert "Unknown: text but no speaker" in (tmp_path / "s" / "transcript.txt").read_text()


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


# ---------------------------------------------------------------------------
# _on_session_end — defensive cleanup
# ---------------------------------------------------------------------------

def test_on_session_end_noop_when_nothing_active():
    from plugins.google_meet import _on_session_end
    # Should not raise and should not call stop().
    with patch("plugins.google_meet.pm.stop") as stop_mock:
        _on_session_end()
    stop_mock.assert_not_called()


def test_on_session_end_stops_live_bot():
    from plugins.google_meet import _on_session_end
    from plugins.google_meet import pm

    with patch.object(pm, "status", return_value={"ok": True, "alive": True}), \
         patch.object(pm, "stop") as stop_mock:
        _on_session_end()
    stop_mock.assert_called_once()


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


def test_register_wires_tools_cli_and_hook_on_linux():
    import plugins.google_meet as plugin

    calls = {"tools": [], "cli": [], "hooks": []}

    class _Ctx:
        def register_tool(self, **kw): calls["tools"].append(kw["name"])
        def register_cli_command(self, **kw): calls["cli"].append(kw["name"])
        def register_hook(self, name, fn): calls["hooks"].append(name)

    with patch.object(plugin.platform, "system", return_value="Linux"):
        plugin.register(_Ctx())

    assert set(calls["tools"]) == {
        "meet_join", "meet_status", "meet_transcript", "meet_leave", "meet_say",
    }
    assert calls["cli"] == ["meet"]
    assert calls["hooks"] == ["on_session_end"]


# ---------------------------------------------------------------------------
# v2: process_manager.enqueue_say + realtime-mode passthrough
# ---------------------------------------------------------------------------

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


def test_enqueue_say_writes_jsonl_in_realtime_mode():
    from plugins.google_meet import process_manager as pm

    out_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    out_dir.mkdir(parents=True)
    pm._write_active({
        "pid": 0, "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir), "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0, "mode": "realtime",
    })
    res = pm.enqueue_say("hello everyone")
    assert res["ok"] is True
    assert "enqueued_id" in res

    queue = out_dir / "say_queue.jsonl"
    assert queue.is_file()
    lines = [json.loads(ln) for ln in queue.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0]["text"] == "hello everyone"


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


def test_meet_join_accepts_realtime_mode():
    from plugins.google_meet.tools import handle_meet_join

    with patch("plugins.google_meet.tools.check_meet_requirements", return_value=True), \
         patch("plugins.google_meet.tools.pm.start", return_value={"ok": True, "meeting_id": "x-y-z"}) as start_mock:
        out = json.loads(handle_meet_join({
            "url": "https://meet.google.com/abc-defg-hij",
            "mode": "realtime",
        }))
    assert out["success"] is True
    assert start_mock.call_args.kwargs["mode"] == "realtime"


def test_meet_join_rejects_bad_mode():
    from plugins.google_meet.tools import handle_meet_join

    out = json.loads(handle_meet_join({
        "url": "https://meet.google.com/abc-defg-hij",
        "mode": "bogus",
    }))
    assert out["success"] is False
    assert "mode must be" in out["error"]


# ---------------------------------------------------------------------------
# v3: NodeClient routing from tool handlers
# ---------------------------------------------------------------------------

def test_meet_join_unknown_node_returns_clear_error():
    from plugins.google_meet.tools import handle_meet_join

    out = json.loads(handle_meet_join({
        "url": "https://meet.google.com/abc-defg-hij",
        "node": "my-mac",
    }))
    assert out["success"] is False
    assert "no registered meet node" in out["error"]


def test_meet_join_routes_to_registered_node():
    from plugins.google_meet.tools import handle_meet_join
    from plugins.google_meet.node.registry import NodeRegistry

    reg = NodeRegistry()
    reg.add("my-mac", "ws://1.2.3.4:18789", "tok")

    with patch("plugins.google_meet.node.client.NodeClient.start_bot",
               return_value={"ok": True, "meeting_id": "a-b-c"}) as call_mock:
        out = json.loads(handle_meet_join({
            "url": "https://meet.google.com/abc-defg-hij",
            "node": "my-mac",
            "mode": "realtime",
        }))
    assert out["success"] is True
    assert out["node"] == "my-mac"
    assert call_mock.call_args.kwargs["mode"] == "realtime"


def test_meet_say_routes_to_node():
    from plugins.google_meet.tools import handle_meet_say
    from plugins.google_meet.node.registry import NodeRegistry

    reg = NodeRegistry()
    reg.add("my-mac", "ws://1.2.3.4:18789", "tok")

    with patch("plugins.google_meet.node.client.NodeClient.say",
               return_value={"ok": True, "enqueued_id": "abc"}) as call_mock:
        out = json.loads(handle_meet_say({"text": "hello", "node": "my-mac"}))
    assert out["success"] is True
    assert out["node"] == "my-mac"
    call_mock.assert_called_once_with("hello")


def test_meet_join_auto_node_selects_sole_registered():
    from plugins.google_meet.tools import handle_meet_join
    from plugins.google_meet.node.registry import NodeRegistry

    reg = NodeRegistry()
    reg.add("only-one", "ws://1.2.3.4:18789", "tok")

    with patch("plugins.google_meet.node.client.NodeClient.start_bot",
               return_value={"ok": True}) as call_mock:
        out = json.loads(handle_meet_join({
            "url": "https://meet.google.com/abc-defg-hij",
            "node": "auto",
        }))
    assert out["success"] is True
    assert out["node"] == "only-one"
    assert call_mock.called


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


def test_cli_join_accepts_mode_and_node_flags():
    import argparse
    from plugins.google_meet.cli import register_cli

    parser = argparse.ArgumentParser(prog="hermes meet")
    register_cli(parser)

    ns = parser.parse_args([
        "join", "https://meet.google.com/abc-defg-hij",
        "--mode", "realtime", "--node", "my-mac",
    ])
    assert ns.mode == "realtime"
    assert ns.node == "my-mac"


def test_cli_say_subcommand_exists():
    import argparse
    from plugins.google_meet.cli import register_cli

    parser = argparse.ArgumentParser(prog="hermes meet")
    register_cli(parser)

    ns = parser.parse_args(["say", "hello team", "--node", "my-mac"])
    assert ns.text == "hello team"
    assert ns.node == "my-mac"
