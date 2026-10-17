"""Tests for the watch plugin's pure layers.

Deliberately covers the arithmetic and the ladders rather than ffmpeg: the
capture/prepare argv builders are pure functions over an explicit platform, so
Windows and macOS argv are asserted from the Linux lane without faking the host
(no ``sys.platform`` patching — see the root AGENTS.md rule).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(module_name: str):
    """Import a watch module by path.

    The plugin lives under ``plugins/watch/`` and its package ``__init__``
    imports the CLI (and through it the recorder), so loading a leaf module by
    path keeps these tests free of that tree.
    """
    path = REPO_ROOT / "plugins" / "watch" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"_watch_{module_name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prep():
    return _load("prepare")


@pytest.fixture(scope="module")
def cap():
    return _load("capture")


@pytest.fixture(scope="module")
def tl():
    return _load("timeline")


# ── Cost model ────────────────────────────────────────────────────────────

def test_file_cap_stays_under_the_base64_payload_limit(prep):
    """The file ceiling must leave room for base64 inflation + envelope.

    An invariant, not a snapshot: whatever the numbers are, a file at the cap
    must still fit the payload limit after 4/3 expansion.
    """
    assert prep.MAX_FILE_BYTES * prep.BASE64_OVERHEAD < prep.MAX_BASE64_BYTES


def test_speed_trades_tokens_against_effective_frame_rate(prep):
    """Retiming is the one dial that moves BOTH cost and temporal detail.

    Faster = fewer billed seconds = fewer tokens, but the provider's 1 fps
    sampling then covers more real time per frame. Slower is the reverse. This
    relationship is the plugin's central claim, so it is pinned directly.
    """
    base = prep.PrepareSpec(speed=1.0)
    fast = prep.PrepareSpec(speed=2.0)
    slow = prep.PrepareSpec(speed=0.5)

    duration = 600.0
    tokens_base = prep.estimate_tokens(duration, base)
    tokens_fast = prep.estimate_tokens(duration, fast)
    tokens_slow = prep.estimate_tokens(duration, slow)

    assert tokens_fast == pytest.approx(tokens_base / 2, rel=0.01)
    assert tokens_slow == pytest.approx(tokens_base * 2, rel=0.01)

    assert prep.effective_fps_for(fast) == pytest.approx(0.5)
    assert prep.effective_fps_for(slow) == pytest.approx(2.0)


def test_resolution_does_not_change_token_cost(prep):
    """Shrinking the picture makes the FILE smaller, not the bill.

    The counter-intuitive half of the two-budget model: providers bill per
    second of timeline at their own sample rate, so width is a bytes lever
    only. A regression here would have us "optimizing" cost by scaling down.
    """
    duration = 300.0
    wide = prep.PrepareSpec(width=1280)
    narrow = prep.PrepareSpec(width=320)

    assert prep.estimate_tokens(duration, wide) == prep.estimate_tokens(duration, narrow)


def test_low_media_resolution_cuts_frame_tokens_but_not_audio(prep):
    """media_resolution only discounts the picture; audio is billed flat."""
    duration = 100.0
    default_spec = prep.PrepareSpec(audio=False)
    low_spec = prep.PrepareSpec(audio=False, media_resolution="low")

    ratio = prep.estimate_tokens(duration, default_spec) / prep.estimate_tokens(duration, low_spec)
    assert ratio == pytest.approx(
        prep.TOKENS_PER_FRAME_DEFAULT / prep.TOKENS_PER_FRAME_LOW, rel=0.01
    )

    with_audio = prep.estimate_tokens(duration, prep.PrepareSpec(media_resolution="low"))
    assert with_audio > prep.estimate_tokens(duration, low_spec)


def test_audio_is_cheaper_per_second_than_video(prep):
    """Why a music take should be audio-primary: audio is ~8x cheaper."""
    assert prep.TOKENS_PER_AUDIO_SECOND < prep.TOKENS_PER_FRAME_DEFAULT


def test_dropping_audio_reduces_the_estimate(prep):
    duration = 120.0
    assert prep.estimate_tokens(duration, prep.PrepareSpec(audio=False)) < prep.estimate_tokens(
        duration, prep.PrepareSpec(audio=True)
    )


def test_zero_duration_costs_nothing(prep):
    assert prep.estimate_tokens(0.0, prep.PrepareSpec()) == 0


# ── Filter chain ──────────────────────────────────────────────────────────

def test_decimate_and_fps_are_mutually_exclusive(prep):
    """An fps filter resamples the timeline and undoes mpdecimate.

    Emitting both would silently reintroduce the duplicate frames the decimate
    pass just removed, making the flag a no-op that still looks applied.
    """
    both = prep.PrepareSpec(fps=1.0, decimate=True)
    stages = prep.video_filters(both)
    assert any(s.startswith("fps=") for s in stages)
    assert "mpdecimate" not in stages

    decimate_only = prep.PrepareSpec(fps=None, decimate=True)
    assert "mpdecimate" in prep.video_filters(decimate_only)


def test_scale_runs_after_timeline_thinning(prep):
    """Order matters for cost: scale the frames that survived, not all of them."""
    stages = prep.video_filters(prep.PrepareSpec(width=640, fps=1.0))
    fps_at = next(i for i, s in enumerate(stages) if s.startswith("fps="))
    scale_at = next(i for i, s in enumerate(stages) if s.startswith("scale="))
    assert fps_at < scale_at


def test_retiming_precedes_frame_rate_resampling(prep):
    """setpts MUST come before fps, or the clip's duration is a lie.

    Measured against real ffmpeg: with ``fps=1`` first, a 20 s source at 2x
    produced a 12 s / 12-frame clip instead of 10 s / 10 frames — ``fps`` fixes
    the output cadence and the later ``setpts`` drops frames into slots already
    committed. Both ``render_timeline`` (which divides stamps by ``speed``) and
    ``estimate_tokens`` (which bills ``duration / speed``) assume
    ``clip == source / speed``, so the wrong order silently misdates every event
    the model reports on. This is the cheapest possible guard on that.
    """
    stages = prep.video_filters(prep.PrepareSpec(width=640, fps=1.0, speed=2.0))
    setpts_at = next(i for i, s in enumerate(stages) if s.startswith("setpts="))
    fps_at = next(i for i, s in enumerate(stages) if s.startswith("fps="))
    assert setpts_at < fps_at


def test_speed_retimes_both_tracks_together(prep):
    """Video setpts without matching audio atempo desyncs the take."""
    spec = prep.PrepareSpec(speed=2.0, audio=True)
    assert any(s.startswith("setpts=") for s in prep.video_filters(spec))
    assert prep.audio_filters(spec), "audio must be retimed alongside video"


def test_large_speedups_split_into_multiple_atempo_stages(prep):
    """atempo artefacts badly past 2x, so a big factor is staged."""
    stages = prep.audio_filters(prep.PrepareSpec(speed=8.0))
    assert len(stages) > 1
    product = 1.0
    for stage in stages:
        product *= float(stage.split("=")[1])
    assert product == pytest.approx(8.0, rel=0.01)


def test_no_speed_change_emits_no_retiming_filters(prep):
    spec = prep.PrepareSpec(speed=1.0)
    assert not prep.audio_filters(spec)
    assert not any(s.startswith("setpts=") for s in prep.video_filters(spec))


def test_prepare_args_drops_audio_flags_when_audio_is_off(prep):
    """Codec flags for a stream that does not exist make ffmpeg fail."""
    args = prep.prepare_args("in.mp4", "out.mp4", prep.PrepareSpec(audio=False))
    assert "-an" in args
    assert "-c:a" not in args


def test_prepare_args_overwrites_so_the_ladder_can_retry(prep):
    """Each rung rewrites the same destination; a prompt would hang the ladder."""
    args = prep.prepare_args("in.mp4", "out.mp4", prep.PrepareSpec())
    assert "-y" in args


@pytest.mark.parametrize(
    "spec_kwargs",
    [
        {"width": 10},
        {"fps": 0},
        {"fps": 120},
        {"crf": 99},
        {"speed": 100.0},
        {"speed": 0.01},
        {"media_resolution": "ultra"},
    ],
)
def test_invalid_specs_are_rejected(prep, spec_kwargs):
    with pytest.raises(ValueError):
        prep.prepare_args("in.mp4", "out.mp4", prep.PrepareSpec(**spec_kwargs))


# ── Fit ladder ────────────────────────────────────────────────────────────

def test_ladder_descends_monotonically(prep):
    """Later rungs must never be cheaper-looking than earlier ones.

    Quality is spent before information: CRF/width degrade first so the model
    still sees every moment, just less sharply.
    """
    specs = prep.ladder_specs(prep.PrepareSpec())
    crfs = [s.crf for s in specs]
    assert crfs == sorted(crfs)
    widths = [s.width for s in specs if s.width is not None]
    assert widths == sorted(widths, reverse=True)


def test_ladder_never_upgrades_an_explicit_request(prep):
    """A user who asked for 320px must not be widened back to 480 mid-descent.

    The rung's width is an upper bound, not an assignment — otherwise the
    ladder inflates the very file it is trying to shrink.
    """
    base = prep.PrepareSpec(width=320, crf=40)
    for spec in prep.ladder_specs(base):
        assert spec.width is not None and spec.width <= 320
        assert spec.crf >= 40


def test_ladder_preserves_unrelated_caller_choices(prep):
    """Audio-off and retiming are the user's decisions, not the ladder's."""
    base = prep.PrepareSpec(audio=False, speed=2.0, media_resolution="low")
    for spec in prep.ladder_specs(base):
        assert spec.audio is False
        assert spec.speed == 2.0
        assert spec.media_resolution == "low"


def test_ladder_has_no_duplicate_rungs(prep):
    """A clamped rung that collapses onto its predecessor must not re-encode."""
    specs = prep.ladder_specs(prep.PrepareSpec(width=320, crf=40, fps=0.5))
    assert len(specs) == len(set(specs))


def test_fits_rejects_empty_and_oversized(prep):
    assert prep.fits(1024)
    assert prep.fits(prep.MAX_FILE_BYTES)
    assert not prep.fits(prep.MAX_FILE_BYTES + 1)
    assert not prep.fits(0)


def test_speedup_suggestion_would_actually_fit(prep):
    """The advice has to be true: applying it must bring the file under cap."""
    oversize = int(prep.MAX_FILE_BYTES * 2.5)
    factor = prep.suggest_speedup(oversize, duration_s=600.0)
    assert factor is not None
    assert oversize / factor <= prep.MAX_FILE_BYTES


def test_no_speedup_suggested_for_a_file_that_already_fits(prep):
    assert prep.suggest_speedup(1024, duration_s=60.0) is None


def test_hopeless_oversize_gets_no_suggestion(prep):
    """Past the atempo ceiling, honest refusal beats unusable advice."""
    assert prep.suggest_speedup(int(prep.MAX_FILE_BYTES * 50), duration_s=7200.0) is None


# ── Capture plans (all hosts, from any host) ──────────────────────────────

def test_windows_plan_uses_gdigrab_and_named_audio_device(cap):
    plan = cap.capture_plan(cap.WINDOWS, fps=30, audio_device="Stereo Mix (Realtek)")
    assert "gdigrab" in plan.args
    assert "audio=Stereo Mix (Realtek)" in plan.args
    assert plan.audio is True
    assert plan.blocked is None


def test_windows_without_audio_device_records_video_and_says_how_to_fix_it(cap):
    plan = cap.capture_plan(cap.WINDOWS)
    assert plan.audio is False
    assert plan.blocked is None
    assert any("list_devices" in note for note in plan.notes)


def test_macos_combines_screen_and_audio_into_one_input(cap):
    """avfoundation takes ``<screen>:<audio>`` as a single -i, not two inputs."""
    plan = cap.capture_plan(cap.MACOS, screen_index=3, audio_index=1)
    assert "avfoundation" in plan.args
    assert "3:1" in plan.args
    assert plan.args.count("-i") == 1
    assert plan.audio is True


def test_macos_without_audio_still_pins_the_screen_index(cap):
    plan = cap.capture_plan(cap.MACOS, screen_index=2)
    assert "2:none" in plan.args
    assert plan.audio is False
    assert any("loopback" in note for note in plan.notes)


def test_macos_rejects_a_named_audio_device_with_a_usable_message(cap):
    """Names are a category error on avfoundation — say so, don't ignore it."""
    plan = cap.capture_plan(cap.MACOS, audio_device="BlackHole 2ch")
    assert plan.blocked is not None
    assert "INDEX" in plan.blocked


def test_macos_accepts_a_numeric_audio_device_string(cap):
    plan = cap.capture_plan(cap.MACOS, screen_index=1, audio_device="2")
    assert plan.blocked is None
    assert "1:2" in plan.args


def test_linux_x11_capture_uses_display_and_pulse(cap):
    plan = cap.capture_plan(cap.LINUX, display=":0", audio_device="mic.monitor")
    assert "x11grab" in plan.args
    assert ":0" in plan.args
    assert "pulse" in plan.args
    assert plan.audio is True


def test_wayland_without_xwayland_is_refused_not_silently_black(cap):
    """x11grab on pure Wayland yields a black video — worse than refusing."""
    plan = cap.capture_plan(cap.LINUX, wayland_display="wayland-0", display=None)
    assert plan.blocked is not None
    assert "wayland" in plan.blocked.lower()
    assert not plan.args


def test_wayland_with_xwayland_captures_through_x11(cap):
    """Both vars set means XWayland is present and x11grab works."""
    plan = cap.capture_plan(cap.LINUX, wayland_display="wayland-0", display=":0")
    assert plan.blocked is None
    assert "x11grab" in plan.args


def test_headless_linux_is_refused(cap):
    plan = cap.capture_plan(cap.LINUX, display=None, wayland_display=None)
    assert plan.blocked is not None


def test_unknown_platform_is_refused_rather_than_guessed(cap):
    plan = cap.capture_plan("sunos5")
    assert plan.blocked is not None


def test_region_capture_restricts_the_grab_on_both_desktop_backends(cap):
    win = cap.capture_plan(cap.WINDOWS, region=(10, 20, 640, 480))
    assert "640x480" in win.args
    assert "-offset_x" in win.args

    linux = cap.capture_plan(cap.LINUX, display=":0", region=(10, 20, 640, 480))
    assert "640x480" in linux.args
    assert ":0+10,20" in linux.args


def test_encoder_args_omit_audio_codec_when_there_is_no_audio_stream(cap):
    """Audio flags with no audio input is an ffmpeg error, not a warning."""
    silent = cap.encoder_args(audio=False)
    assert "-an" in silent
    assert "-c:a" not in silent
    assert "-c:a" in cap.encoder_args(audio=True)


# ── Window timeline ───────────────────────────────────────────────────────

def _samples(tl, rows):
    return [tl.WindowSample(at=at, app=app, title=title) for at, app, title in rows]


def test_repeated_samples_of_one_app_collapse_into_one_segment(tl):
    """A 1.5s poll over 40 minutes of one app is one row, not 1,600."""
    rows = [(float(i) * 1.5, "Ableton Live", "Set 1") for i in range(400)]
    segments = tl.segments_from_samples(_samples(tl, rows), total_duration=600.0)
    assert len(segments) == 1
    assert segments[0].app == "Ableton Live"
    assert segments[0].end == 600.0


def test_title_churn_within_one_app_does_not_split_a_segment(tl):
    """A DAW rewriting its title on every save must not fragment the take."""
    rows = [(0.0, "Ableton Live", "Set 1"), (5.0, "Ableton Live", "Set 1*"), (10.0, "Ableton Live", "Set 2")]
    segments = tl.segments_from_samples(_samples(tl, rows), total_duration=60.0)
    assert len(segments) == 1


def test_brief_flicks_are_absorbed_by_the_previous_segment(tl):
    """Alt-tabbing through a switcher is not a change of activity.

    The absorbed time stays accounted for — the previous segment extends over
    it rather than the timeline developing a hole.
    """
    rows = [(0.0, "Ableton Live", ""), (30.0, "Switcher", ""), (30.5, "Chrome", "")]
    segments = tl.segments_from_samples(_samples(tl, rows), total_duration=90.0)
    apps = [s.app for s in segments]
    assert "Switcher" not in apps
    assert apps == ["Ableton Live", "Chrome"]
    assert segments[0].end == segments[1].start


def test_segments_tile_the_recording_without_gaps(tl):
    rows = [(0.0, "A", ""), (20.0, "B", ""), (50.0, "C", "")]
    segments = tl.segments_from_samples(_samples(tl, rows), total_duration=80.0)
    assert segments[0].start == 0.0
    assert segments[-1].end == 80.0
    for earlier, later in zip(segments, segments[1:]):
        assert earlier.end == later.start


def test_empty_sample_stream_yields_no_timeline(tl):
    assert tl.segments_from_samples([], total_duration=60.0) == []
    assert tl.render_timeline([]) == ""


def test_rendered_stamps_are_divided_by_the_clip_speed(tl):
    """A 2x clip puts a 4:00 real-time event at 2:00 of footage.

    Getting this backwards makes every timestamp the model quotes wrong, which
    is invisible in review — hence a direct assertion.
    """
    segments = [tl.TimelineSegment(start=240.0, end=300.0, app="Ableton Live", title="")]
    assert "02:00" in tl.render_timeline(segments, speed=2.0)
    assert "04:00" in tl.render_timeline(segments, speed=1.0)


def test_capture_shortfall_compresses_the_stamps_onto_the_file(tl):
    """Window samples are wall-clock; the file may be shorter than real time.

    Measured live: a 21 s x11grab session produced 9 s of video. A sample
    stamped at 20 s of wall clock therefore belongs near the END of that 9 s
    file, not 11 s past it — without the scale the whole track points past the
    end of the footage.
    """
    segments = [tl.TimelineSegment(start=20.0, end=21.0, app="Chrome", title="")]
    scaled = tl.render_timeline(segments, capture_scale=9.0 / 21.0)
    assert "00:09" in scaled
    assert "00:20" in tl.render_timeline(segments, capture_scale=1.0)


def test_capture_scale_and_speed_compose(tl):
    """Both retimings apply — capture shortfall first, then prepare's speed."""
    segments = [tl.TimelineSegment(start=100.0, end=110.0, app="WoW", title="")]
    # Half the frames captured, then played back 2x: 100s -> 50s -> 25s.
    assert "00:25" in tl.render_timeline(segments, capture_scale=0.5, speed=2.0)


def test_timeline_for_threads_capture_scale_through(tl):
    rows = [{"at": 0.0, "app": "A", "title": ""}, {"at": 40.0, "app": "B", "title": ""}]
    block, _ = tl.timeline_for(rows, total_duration=60.0, capture_scale=0.5)
    assert "00:20" in block


def test_long_recordings_render_hours(tl):
    segments = [tl.TimelineSegment(start=3725.0, end=3800.0, app="WoW", title="")]
    assert "1:02:05" in tl.render_timeline(segments)


def test_titles_can_be_dropped_while_keeping_app_names(tl):
    """Titles carry document names and URLs; app names are the useful part."""
    segments = [tl.TimelineSegment(start=0.0, end=10.0, app="Chrome", title="bank statement.pdf")]
    private = tl.render_timeline(segments, include_titles=False)
    assert "Chrome" in private
    assert "bank statement" not in private
    assert "bank statement" in tl.render_timeline(segments, include_titles=True)


def test_long_titles_are_truncated_so_one_row_cannot_dominate(tl):
    segments = [tl.TimelineSegment(start=0.0, end=5.0, app="Code", title="x" * 500)]
    line = tl.render_timeline(segments)
    assert len(line.splitlines()[-1]) < 120


def test_malformed_sidecar_rows_are_skipped_not_fatal(tl):
    """A recorder killed mid-write leaves a truncated final row.

    One bad line must not cost the user the timeline for a take they cannot
    re-perform.
    """
    rows = [
        {"at": 0.0, "app": "Ableton Live", "title": "Set"},
        {"at": "not-a-number", "app": "Chrome"},
        {"app": "NoTimestamp"},
        None,
        {"at": 30.0, "app": "Chrome", "title": ""},
    ]
    samples = tl.load_samples(rows)
    assert [s.app for s in samples] == ["Ableton Live", "Chrome"]


def test_out_of_order_samples_are_sorted_before_segmenting(tl):
    rows = [(50.0, "C", ""), (0.0, "A", ""), (20.0, "B", "")]
    segments = tl.segments_from_samples(_samples(tl, rows), total_duration=80.0)
    assert [s.app for s in segments] == ["A", "B", "C"]


def test_summary_ranks_apps_by_dwell(tl):
    rows = [(0.0, "Ableton Live", ""), (60.0, "Chrome", ""), (70.0, "Ableton Live", "")]
    segments = tl.segments_from_samples(_samples(tl, rows), total_duration=120.0)
    summary = tl.summarize(segments)
    assert summary["apps"][0][0] == "Ableton Live"
    assert summary["switches"] == len(segments) - 1


def test_timeline_for_handles_persisted_rows_end_to_end(tl):
    rows = [{"at": 0.0, "app": "Ableton Live", "title": "Set 1"}, {"at": 45.0, "app": "Chrome", "title": "docs"}]
    block, summary = tl.timeline_for(rows, total_duration=90.0)
    assert "Ableton Live" in block
    assert summary is not None and summary["segments"] == 2


def test_timeline_for_with_no_rows_reports_nothing_rather_than_an_empty_header(tl):
    block, summary = tl.timeline_for([], total_duration=90.0)
    assert block == ""
    assert summary is None


# ── Clock reconciliation ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rev():
    return _load("review")


def test_capture_scale_is_the_ratio_of_recorded_to_real_time(rev):
    meta = {"wall_seconds": 21.1, "video_seconds": 9.1}
    assert rev.capture_scale_for(meta, 9.1) == pytest.approx(9.1 / 21.1)


def test_capture_scale_is_neutral_without_meta(rev):
    """An imported file (Game Bar, OBS) has no meta and needs no correction."""
    assert rev.capture_scale_for(None, 60.0) == 1.0
    assert rev.capture_scale_for({}, 60.0) == 1.0


def test_capture_scale_never_stretches_past_the_clip(rev):
    """A file longer than its own session is nonsense; clamp rather than trust.

    Trusting a >1 ratio would push timeline rows past the end of the footage,
    which reads to the model as events that never appear.
    """
    assert rev.capture_scale_for({"wall_seconds": 10.0, "video_seconds": 30.0}, 30.0) == 1.0


@pytest.mark.parametrize(
    "meta",
    [
        {"wall_seconds": 0, "video_seconds": 10},
        {"wall_seconds": 10, "video_seconds": 0},
        {"wall_seconds": -5, "video_seconds": 10},
        {"wall_seconds": "bad", "video_seconds": 10},
        {"video_seconds": 10},
    ],
)
def test_nonsense_meta_falls_back_to_neutral(rev, meta):
    assert rev.capture_scale_for(meta, 10.0) == 1.0


def test_prepared_output_is_named_off_the_source_take(rev):
    """The transcode must be identifiable as derived, and never clobber a take."""
    dest = rev.prepared_path(Path("/takes/20260824-solo.mp4"))
    assert dest.name.endswith(".prepared.mp4")
    assert "20260824-solo" in dest.name
    assert dest != Path("/takes/20260824-solo.mp4")


def test_retimed_clips_tell_the_model_the_footage_is_retimed(rev):
    """A model that doesn't know a clip is 2x will misreport every timestamp."""
    prep_mod = _load("prepare")
    prompt = rev.build_prompt(
        "how was my timing?",
        duration_s=600.0,
        spec=prep_mod.PrepareSpec(speed=2.0),
    )
    assert "2x" in prompt
    assert "CLIP time" in prompt


def test_unretimed_clips_get_no_confusing_retiming_note(rev):
    prep_mod = _load("prepare")
    prompt = rev.build_prompt("wyt?", duration_s=600.0, spec=prep_mod.PrepareSpec(speed=1.0))
    assert "retimed" not in prompt


def test_prompt_carries_the_window_timeline(rev):
    prompt = rev.build_prompt("wyt?", timeline_block="Window timeline (clip time):\n00:00  Ableton Live")
    assert "Ableton Live" in prompt


def test_empty_question_still_asks_something_useful(rev):
    """A bare `hermes watch review` must not send an empty prompt."""
    assert rev.build_prompt("").strip()


# ── Registration ──────────────────────────────────────────────────────────
#
# Loaded through the real PluginManager against an isolated HERMES_HOME, so
# these assert what a user actually gets rather than what the source says.

def _enable_watch(hermes_home: Path) -> None:
    import yaml

    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["watch"]}}), encoding="utf-8"
    )


def test_watch_is_discovered_but_not_loaded_without_opt_in(tmp_path, monkeypatch):
    """A capture plugin must never activate itself.

    Screen recording is the last thing that should arrive switched on, so the
    bundled default is discovered-and-inert.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import plugins as pmod

    manager = pmod.PluginManager()
    manager.discover_and_load()

    assert "watch" not in manager._cli_commands


def test_enabling_watch_registers_the_cli_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _enable_watch(tmp_path)
    from hermes_cli import plugins as pmod

    manager = pmod.PluginManager()
    manager.discover_and_load()

    assert "watch" in manager._cli_commands, "hermes watch did not register"
    entry = manager._cli_commands["watch"]
    assert entry["plugin"] == "watch"
    assert entry["handler_fn"] is not None


def test_registered_parser_exposes_the_record_lifecycle(tmp_path, monkeypatch):
    """start/stop/status/list/review/cost must all be reachable as subcommands."""
    import argparse

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _enable_watch(tmp_path)
    from hermes_cli import plugins as pmod

    manager = pmod.PluginManager()
    manager.discover_and_load()

    parser = argparse.ArgumentParser(prog="hermes watch")
    manager._cli_commands["watch"]["setup_fn"](parser)
    help_text = parser.format_help()
    for verb in ("start", "stop", "status", "list", "review", "cost"):
        assert verb in help_text


def test_bare_invocation_prints_usage_and_fails_cleanly(tmp_path, monkeypatch, capsys):
    """`hermes watch` with no subcommand must not traceback."""
    import argparse

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _enable_watch(tmp_path)
    from hermes_cli import plugins as pmod

    manager = pmod.PluginManager()
    manager.discover_and_load()
    entry = manager._cli_commands["watch"]

    parser = argparse.ArgumentParser(prog="hermes watch")
    entry["setup_fn"](parser)
    rc = entry["handler_fn"](parser.parse_args([]))

    assert rc == 1
    assert "start" in capsys.readouterr().out


def test_enabling_watch_registers_the_slash_command(tmp_path, monkeypatch):
    """The GUI surface: /watch must reach the desktop and TUI palettes.

    Both discover plugin slash commands through this registry, and the desktop
    surfaces non-builtin commands as extensions, so registering here is the
    whole integration — no core file needs to know the plugin exists.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _enable_watch(tmp_path)
    from hermes_cli import plugins as pmod

    manager = pmod.PluginManager()
    manager.discover_and_load()

    assert "watch" in manager._plugin_commands, "/watch did not register"
    entry = manager._plugin_commands["watch"]
    assert entry["plugin"] == "watch"
    assert callable(entry["handler"])
    # An args_hint is what gives Discord/desktop pickers an argument field.
    assert "live" in entry["args_hint"]


def test_the_slash_handler_answers_without_touching_the_screen(tmp_path, monkeypatch):
    """Bare /watch must explain itself rather than starting a capture."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _enable_watch(tmp_path)
    from hermes_cli import plugins as pmod

    manager = pmod.PluginManager()
    manager.discover_and_load()
    out = manager._plugin_commands["watch"]["handler"]("")

    assert "/watch live" in out
