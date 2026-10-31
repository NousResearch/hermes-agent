"""Tests for the ``/watch`` slash command — the GUI-facing surface.

A slash command returns one string and finishes, which is a bad fit for a
long-running loop. What these pin is that mismatch being handled: live watching
detaches, a failed start is reported instead of being claimed as success, and
nothing is left running when a session ends.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(module_name: str):
    path = REPO_ROOT / "plugins" / "watch" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"_watch_slash_{module_name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def slash(monkeypatch, tmp_path):
    """A fresh slash module with capture and recording stubbed out.

    The start grace is dropped to near zero: it exists to catch a synchronous
    failure, and paying 2.5s per case for that would make the suite slow enough
    that nobody runs it.
    """
    module = _load("slash")
    monkeypatch.setattr(module.rec, "watch_dir", lambda: tmp_path)
    monkeypatch.setattr(module.rec, "status", lambda: {"recording": False})
    monkeypatch.setattr(module, "START_GRACE_SECONDS", 0.15)
    return module


# ══ Help and dispatch ═════════════════════════════════════════════════════

@pytest.mark.parametrize("args", ["", "help", "--help", "  "])
def test_bare_invocation_explains_itself(slash, args):
    out = slash.handle(args)
    assert "/watch live" in out
    assert "/watch stop" in out


def test_unknown_subcommand_shows_help_rather_than_failing(slash):
    out = slash.handle("frobnicate")
    assert "Unknown subcommand" in out
    assert "/watch live" in out


def test_help_states_that_quiet_is_intended(slash):
    """The first thing a user misreads: silence looks like breakage."""
    assert "quiet by design" in slash.handle("").lower()


# ══ Live: detaching, and failing loudly ═══════════════════════════════════

def test_a_failed_start_is_reported_not_claimed_as_watching(slash, monkeypatch):
    """Saying "watching" while nothing is captured is the recurring failure.

    A blocked platform (Wayland without XWayland) or a missing ffmpeg is known
    immediately, so the command waits briefly and surfaces it rather than
    detaching into a loop that can never produce a frame.
    """
    monkeypatch.setattr(
        slash.livemod, "run_live",
        lambda **kw: {"success": False, "error": "Wayland session without XWayland"},
    )
    out = slash.handle("live my rotation")
    assert "could not start" in out.lower()
    assert "wayland" in out.lower()


def test_a_successful_start_returns_immediately(slash, monkeypatch):
    """The loop runs for as long as the user wants; the command cannot block."""
    started = threading.Event()

    def fake_run_live(**kwargs):
        started.set()
        kwargs["stop"].wait(timeout=30)
        return {"success": True, "seconds": 1.0, "ticks": 1, "model_calls": 0, "spoke": 0}

    monkeypatch.setattr(slash.livemod, "run_live", fake_run_live)

    began = time.monotonic()
    out = slash.handle("live how I use my synths")
    elapsed = time.monotonic() - began

    assert "how I use my synths" in out
    assert elapsed < 10.0
    assert started.is_set()
    slash.shutdown()


def test_the_brief_is_passed_through(slash, monkeypatch):
    seen = {}

    def fake_run_live(**kwargs):
        seen.update(kwargs)
        kwargs["stop"].wait(timeout=30)
        return {"success": True}

    monkeypatch.setattr(slash.livemod, "run_live", fake_run_live)
    slash.handle("live my cooldown drift")
    assert seen["brief"] == "my cooldown drift"
    slash.shutdown()


def test_a_bare_live_still_gets_a_usable_brief(slash, monkeypatch):
    seen = {}

    def fake_run_live(**kwargs):
        seen.update(kwargs)
        kwargs["stop"].wait(timeout=30)
        return {"success": True}

    monkeypatch.setattr(slash.livemod, "run_live", fake_run_live)
    slash.handle("live")
    assert seen["brief"].strip()
    slash.shutdown()


def test_starting_twice_is_refused(slash, monkeypatch):
    """One screen, one loop. A second would double the bill for one view."""
    monkeypatch.setattr(
        slash.livemod, "run_live",
        lambda **kw: (kw["stop"].wait(timeout=30), {"success": True})[1],
    )
    slash.handle("live first")
    second = slash.handle("live second")
    assert "already watching" in second.lower()
    slash.shutdown()


# ══ Stop and status ═══════════════════════════════════════════════════════

def test_stop_reports_what_was_said(slash, monkeypatch):
    def fake_run_live(**kwargs):
        kwargs["on_speak"](
            type("D", (), {"at": 65.0, "text": "You clipped that cooldown.", "deferred": False})()
        )
        kwargs["stop"].wait(timeout=30)
        return {
            "success": True, "seconds": 70.0, "ticks": 70,
            "model_calls": 4, "spoke": 1, "log_path": "/tmp/x.jsonl",
        }

    monkeypatch.setattr(slash.livemod, "run_live", fake_run_live)
    slash.handle("live timing")
    time.sleep(0.1)
    out = slash.handle("stop")
    assert "1:05" in out
    assert "clipped that cooldown" in out
    assert "replay" in out.lower()


def test_stop_when_nothing_is_running_says_so(slash):
    assert "not watching" in slash.handle("stop").lower()


def test_stop_finalizes_a_recording_when_that_is_what_is_running(slash, monkeypatch):
    """One verb for both modes: the user does not track which one they started."""
    monkeypatch.setattr(slash.rec, "status", lambda: {"recording": True})
    monkeypatch.setattr(
        slash.rec, "stop",
        lambda: {
            "success": True, "video_path": "/takes/a.mp4",
            "duration_seconds": 42.0, "size_bytes": 5 * 1048576,
        },
    )
    out = slash.handle("stop")
    assert "/takes/a.mp4" in out
    assert "42s" in out


def test_status_shows_recent_comments(slash, monkeypatch):
    def fake_run_live(**kwargs):
        for i in range(3):
            kwargs["on_speak"](
                type("D", (), {"at": float(i * 30), "text": f"Note {i}.", "deferred": False})()
            )
        kwargs["stop"].wait(timeout=30)
        return {"success": True}

    monkeypatch.setattr(slash.livemod, "run_live", fake_run_live)
    slash.handle("live x")
    time.sleep(0.1)
    out = slash.handle("status")
    assert "watching live" in out.lower()
    assert "Note 2." in out
    slash.shutdown()


def test_status_when_idle(slash):
    assert "not watching" in slash.handle("status").lower()


# ══ Recording and takes ═══════════════════════════════════════════════════

def test_record_surfaces_the_audio_warning(slash, monkeypatch):
    """A silent take is worth knowing about before performing, not after."""
    monkeypatch.setattr(
        slash.rec, "start",
        lambda **kw: {
            "success": True, "video_path": "/takes/b.mp4", "audio": False,
            "notes": ["No audio source set — recording video only."],
        },
    )
    out = slash.handle("record solo")
    assert "/takes/b.mp4" in out
    assert "no audio" in out.lower()


def test_record_failure_is_reported(slash, monkeypatch):
    monkeypatch.setattr(
        slash.rec, "start", lambda **kw: {"success": False, "error": "ffmpeg not found"}
    )
    assert "ffmpeg not found" in slash.handle("record")


def test_takes_lists_recordings(slash, monkeypatch):
    monkeypatch.setattr(
        slash.rec, "takes",
        lambda limit=10: [
            {"path": "/takes/a.mp4", "size_bytes": 3 * 1048576, "has_timeline": True}
        ],
    )
    out = slash.handle("takes")
    assert "/takes/a.mp4" in out
    assert "+timeline" in out


def test_takes_when_empty_suggests_recording(slash, monkeypatch):
    monkeypatch.setattr(slash.rec, "takes", lambda limit=10: [])
    assert "/watch record" in slash.handle("takes")


# ══ Replay ════════════════════════════════════════════════════════════════

def test_replay_sweeps_thresholds_from_the_last_session(slash, tmp_path):
    import json

    directory = tmp_path / "live"
    directory.mkdir()
    log = directory / "session.jsonl"
    with log.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"brief": "my timing", "ticks": 3, "policy": {}}) + "\n")
        for i in range(3):
            handle.write(
                json.dumps(
                    {"at": float(i * 20), "reason": "spoke", "spoke": True,
                     "called_model": True, "text": f"Note {i}.",
                     "signals": {"screen": 0.9}}
                ) + "\n"
            )

    out = slash.handle("replay")
    assert "my timing" in out
    assert "threshold" in out
    assert "speaks less" in out


def test_replay_without_a_session_says_how_to_make_one(slash):
    assert "/watch live" in slash.handle("replay")


# ══ Teardown ══════════════════════════════════════════════════════════════

def test_shutdown_stops_a_running_loop(slash, monkeypatch):
    """An orphaned loop keeps grabbing frames and calling a model for nobody."""
    stopped = threading.Event()

    def fake_run_live(**kwargs):
        kwargs["stop"].wait(timeout=30)
        stopped.set()
        return {"success": True}

    monkeypatch.setattr(slash.livemod, "run_live", fake_run_live)
    slash.handle("live x")
    slash.shutdown()
    assert stopped.wait(timeout=5)


def test_shutdown_is_safe_when_nothing_is_running(slash):
    slash.shutdown()
    slash.shutdown()
