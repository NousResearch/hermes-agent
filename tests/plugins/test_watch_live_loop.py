"""Tests for the live loop — the driver, its argv, and offline replay.

The loop itself is exercised with an injected ``ask`` and a fake grab, so these
run with no network and no screen. What they pin is the plumbing that broke when
it met real ffmpeg: per-output frame limits, device framerate, and the
distinction between "nothing happened" and "capture is broken".
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "watch_frames"


def _load(module_name: str):
    path = REPO_ROOT / "plugins" / "watch" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"_watch_{module_name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fr():
    return _load("frames")


@pytest.fixture
def live(monkeypatch, tmp_path):
    """The live module with its output directory redirected to tmp."""
    module = _load("live")
    monkeypatch.setattr(module, "watch_dir", lambda: tmp_path)
    return module


@pytest.fixture(scope="module")
def dec():
    return _load("decider")


def _thumb(index: int) -> bytes:
    return (FIXTURES / f"gui{index:02d}.gray").read_bytes()


# ══ grab argv: the bugs real ffmpeg found ═════════════════════════════════

def test_each_output_gets_its_own_frame_limit(fr):
    """``-frames:v`` is PER OUTPUT, and one leading copy bounds only the first.

    Measured: with a single leading ``-frames:v 1`` the thumbnail output ran
    until killed, so every grab hit its timeout and a 13-second live run
    reported zero ticks — indistinguishable from a quiet screen.
    """
    args = fr.grab_args(["-f", "x11grab", "-i", ":0"], jpeg_path="f.jpg", thumb_path="f.gray")
    assert args.count("-frames:v") == 2

    # And each one must precede an output, not trail it.
    jpeg_at = args.index("f.jpg")
    thumb_at = args.index("f.gray")
    limits = [i for i, a in enumerate(args) if a == "-frames:v"]
    assert limits[0] < jpeg_at
    assert limits[1] < thumb_at
    assert limits[1] > jpeg_at


def test_a_one_shot_grab_does_not_use_the_loop_cadence(fr):
    """Device framerate and loop cadence are different things.

    Grabbing at 1 fps makes the capture device wait a full second before it can
    hand over a frame, so every tick paid a second of latency for nothing. The
    loop enforces its own interval.
    """
    assert fr.GRAB_FRAMERATE > 1.0


def test_grab_argv_is_a_single_ffmpeg_invocation(fr):
    """Two passes would hash a different instant than the one sent."""
    args = fr.grab_args(["-i", "x"], jpeg_path="a.jpg", thumb_path="b.gray")
    assert args.count("ffmpeg") == 1
    assert args.count("-i") == 1


# ══ Signals per tick ══════════════════════════════════════════════════════

def test_the_screen_is_always_a_signal(live, fr):
    session = live.LiveSession(brief="x")
    session.ring.add(fr.Frame(at=0.0, jpeg=b"j", thumb=_thumb(1)))
    names = [s.name for s in live.tick_signals(session, now=0.0)]
    assert names == ["screen"]


def test_enrichment_tracks_appear_only_when_fed(live, fr):
    """Frames are the mechanism; keys and notes are optional resolution."""
    inputs = _load("inputs")
    session = live.LiveSession(brief="x")
    session.ring.add(fr.Frame(at=0.0, jpeg=b"j", thumb=_thumb(1)))
    session.key_events.append(inputs.InputEvent(at=0.0, symbol="1", app="game"))

    names = [s.name for s in live.tick_signals(session, now=0.0)]
    assert "screen" in names
    assert "keys" in names
    assert "notes" not in names


def test_tracks_are_trimmed_so_they_cannot_grow_forever(live):
    """The unbounded-context bug arriving through a side door."""
    inputs = _load("inputs")
    session = live.LiveSession(brief="x")
    session.key_events.extend(
        inputs.InputEvent(at=float(i), symbol="1", app="g") for i in range(500)
    )
    live.trim_tracks(session, now=500.0, keep=120.0)
    assert len(session.key_events) < 200
    assert all(e.at >= 380.0 for e in session.key_events)


# ══ The tick ══════════════════════════════════════════════════════════════

def test_a_failed_grab_is_a_missing_tick_not_a_dead_session(live, monkeypatch):
    """Capture fails transiently (display sleep, fullscreen switch)."""
    monkeypatch.setattr(live, "grab_once", lambda *a, **k: None)
    session = live.LiveSession(brief="x")
    assert live.run_tick(session, ["-i", "x"], ask=lambda s, u: "hi", now=1.0) is None
    assert session.ticks == 0


def test_a_successful_tick_advances_the_ring_and_logs(live, fr, monkeypatch):
    monkeypatch.setattr(
        live, "grab_once",
        lambda *a, **k: fr.Frame(at=k.get("at", 0.0), jpeg=b"j", thumb=_thumb(5)),
    )
    session = live.LiveSession(brief="x")
    decision = live.run_tick(session, ["-i", "x"], ask=lambda s, u: "NO REPLY", now=1.0)
    assert decision is not None
    assert session.ticks == 1
    assert len(session.ring.frames) == 1
    assert session.state.log


def test_on_speak_fires_only_when_it_speaks(live, fr, monkeypatch):
    monkeypatch.setattr(
        live, "grab_once",
        lambda *a, **k: fr.Frame(at=k.get("at", 0.0), jpeg=b"j", thumb=_thumb(9)),
    )
    heard = []
    session = live.LiveSession(brief="x")
    live.run_tick(
        session, ["-i", "x"], ask=lambda s, u: "A specific note.",
        now=1.0, on_speak=heard.append,
    )
    assert len(heard) == 1
    assert heard[0].text == "A specific note."


# ══ The loop ══════════════════════════════════════════════════════════════

def test_persistent_capture_failure_reports_why(live, monkeypatch):
    """The trap this exists for: zero ticks looks exactly like a quiet screen.

    A 13-second run that grabbed nothing returned success with ticks=0 and no
    hint that ffmpeg had failed every single time.
    """
    monkeypatch.setattr(live.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(live, "grab_once", lambda *a, **k: None)
    monkeypatch.setenv("DISPLAY", ":0")

    result = live.run_live(
        brief="x", duration=30.0, interval=0.0, platform="linux",
        ask=lambda s, u: "hi",
    )
    assert result["success"] is False
    assert "capture failed" in result["error"].lower()


def test_a_blocked_platform_is_reported_before_anything_starts(live, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    result = live.run_live(brief="x", duration=1.0, platform="linux", ask=lambda s, u: "hi")
    assert result["success"] is False
    assert "wayland" in result["error"].lower()


def test_missing_ffmpeg_is_reported(live, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(live.shutil, "which", lambda _name: None)
    result = live.run_live(brief="x", duration=1.0, platform="linux", ask=lambda s, u: "hi")
    assert result["success"] is False
    assert "ffmpeg" in result["error"].lower()


def test_the_loop_stops_when_asked(live, fr, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(live.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        live, "grab_once",
        lambda *a, **k: fr.Frame(at=k.get("at", 0.0), jpeg=b"j", thumb=_thumb(1)),
    )
    stop = threading.Event()
    ticks = []

    def ask(system, user):
        ticks.append(1)
        if len(ticks) >= 2:
            stop.set()
        return "NO REPLY"

    result = live.run_live(
        brief="x", interval=0.0, platform="linux", ask=ask, stop=stop,
        policy=_load("decider").Policy(call_cooldown=0.0, min_salience=0.0),
    )
    assert result["success"] is True
    assert result["ticks"] >= 2


def test_the_loop_writes_a_decision_log(live, fr, monkeypatch, tmp_path):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(live.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        live, "grab_once",
        lambda *a, **k: fr.Frame(at=k.get("at", 0.0), jpeg=b"j", thumb=_thumb(6)),
    )
    result = live.run_live(
        brief="my timing", duration=0.2, interval=0.0, platform="linux",
        ask=lambda s, u: "NO REPLY",
    )
    assert result["log_path"]
    header, rows = live.read_log(Path(result["log_path"]))
    assert header["brief"] == "my timing"
    assert rows


# ══ Replay ════════════════════════════════════════════════════════════════

def _write_log(path: Path, rows: list[dict], brief: str = "test") -> Path:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"brief": brief, "ticks": len(rows), "policy": {}}) + "\n")
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def test_replay_re_evaluates_the_free_gates(live, dec, tmp_path):
    """Tuning must cost nothing after the first recording."""
    rows = [
        {"at": float(i), "reason": "quiet", "spoke": False, "called_model": False,
         "text": "", "signals": {"screen": 0.1 if i % 10 else 0.9}}
        for i in range(100)
    ]
    path = _write_log(tmp_path / "log.jsonl", rows)

    loose = live.replay(path, dec.Policy(min_salience=0.05, call_cooldown=0.0))
    strict = live.replay(path, dec.Policy(min_salience=0.5, call_cooldown=0.0))
    assert loose["model_calls"] > strict["model_calls"]


def test_replay_reports_the_brief_it_was_recorded_with(live, tmp_path):
    path = _write_log(tmp_path / "log.jsonl", [], brief="my rotation")
    assert live.replay(path)["brief"] == "my rotation"


def test_replay_honours_the_call_cooldown(live, dec, tmp_path):
    rows = [
        {"at": float(i), "reason": "spoke", "spoke": True, "called_model": True,
         "text": f"Note {i}.", "signals": {"screen": 0.9}}
        for i in range(60)
    ]
    path = _write_log(tmp_path / "log.jsonl", rows)
    cheap = live.replay(path, dec.Policy(call_cooldown=20.0, refractory=0.0))
    assert cheap["model_calls"] < 10


def test_replay_suppresses_repeats_the_same_way_the_loop_does(live, dec, tmp_path):
    rows = [
        {"at": float(i * 30), "reason": "spoke", "spoke": True, "called_model": True,
         "text": "Watch your timing on the transition.", "signals": {"screen": 0.9}}
        for i in range(5)
    ]
    path = _write_log(tmp_path / "log.jsonl", rows)
    stats = live.replay(path, dec.Policy(refractory=0.0, call_cooldown=0.0))
    assert stats["spoke"] == 1, "a repeated line must not count five times"


def test_replay_on_an_empty_log_does_not_divide_by_zero(live, tmp_path):
    path = _write_log(tmp_path / "log.jsonl", [])
    assert live.replay(path)["call_rate"] == 0.0


def test_a_corrupt_line_does_not_lose_the_log(live, tmp_path):
    """A loop killed mid-write leaves a truncated final row."""
    path = tmp_path / "log.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"brief": "x", "ticks": 2, "policy": {}}) + "\n")
        handle.write(json.dumps({"at": 0.0, "signals": {"screen": 0.9}}) + "\n")
        handle.write('{"at": 1.0, "signals": {"scre\n')
    _header, rows = live.read_log(path)
    assert len(rows) == 1
