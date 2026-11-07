"""Tests for the frame loop — the domain-agnostic change gate.

Backed by REAL fixtures rather than synthetic buffers. ``fixtures/watch_frames``
holds twelve 33x32 grayscale thumbnails extracted by ffmpeg from rendered
plugin-GUI frames (a filter-response curve plus level meters): frames 0-3 are
static, 4-9 sweep the cutoff slowly, 10 changes preset, 11 is static again.

That fixture exists because the first version of this module was tuned against
a synthetic panel with a 6-pixel knob pointer, which no real GUI resembles. At
9x8 (the conventional dHash size) a sweep moved 1-2 bits of 64 — below the
brightness-noise floor, so the gate would never have woken at all. The numbers
asserted here are measured against the realistic frames.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "watch_frames"

STATIC = (0, 1, 2, 3)
SWEEP = (4, 5, 6, 7, 8, 9)
PRESET = 10


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


@pytest.fixture(scope="module")
def gui(fr):
    """The rendered GUI sequence as Frames, one per second."""
    out = []
    for index in range(12):
        thumb = (FIXTURES / f"gui{index:02d}.gray").read_bytes()
        out.append(fr.Frame(at=float(index), jpeg=b"jpeg-bytes", thumb=thumb))
    return out


# ══ Fixture sanity ════════════════════════════════════════════════════════

def test_fixtures_match_the_hash_geometry(fr, gui):
    """A thumbnail of the wrong size silently hashes to 0 and disables the gate."""
    assert len(gui) == 12
    for frame in gui:
        assert len(frame.thumb) == fr.HASH_WIDTH * fr.HASH_HEIGHT


def test_geometry_is_fine_enough_to_resolve_gui_movement(fr):
    """Regression guard on the sizing decision.

    64-bit dHash (the conventional size) could not see a filter sweep at all.
    Dropping back below ~1000 bits would silently reintroduce that.
    """
    assert fr.HASH_BITS >= 1000


# ══ Change detection on real frames ═══════════════════════════════════════

def test_static_frames_report_no_change(fr, gui):
    for earlier, later in zip(STATIC, STATIC[1:]):
        assert fr.visual_change(gui[earlier], gui[later]) == 0.0


def test_identical_frames_report_no_change(fr, gui):
    assert fr.visual_change(gui[5], gui[5]) == 0.0


def test_the_first_frame_of_a_session_is_entirely_new(fr, gui):
    assert fr.visual_change(None, gui[0]) == 1.0


def test_a_slow_filter_sweep_is_caught_frame_to_frame(fr, gui):
    """The case the whole module exists for, and the one 9x8 dHash missed.

    A cutoff crawling open is the most valuable thing to notice in synth work
    and the hardest to see: each step is small. Every frame of the sweep must
    clear the threshold.
    """
    for earlier, later in zip(SWEEP, SWEEP[1:]):
        change = fr.visual_change(gui[earlier], gui[later])
        assert change >= fr.DEFAULT_CHANGE_THRESHOLD, f"f{earlier}->f{later} = {change}"


def test_accumulated_sweep_change_exceeds_any_single_step(fr, gui):
    """Accumulation is why `FrameRing.novelty` looks several frames back."""
    total = fr.visual_change(gui[SWEEP[0]], gui[SWEEP[-1]])
    steps = [fr.visual_change(gui[a], gui[b]) for a, b in zip(SWEEP, SWEEP[1:])]
    assert total > max(steps)


def test_changed_agrees_with_the_threshold(fr, gui):
    assert fr.changed(gui[5], gui[6])
    assert not fr.changed(gui[1], gui[2])


# ══ Noise rejection ══════════════════════════════════════════════════════

def test_a_brightness_shift_is_not_a_change(fr, gui):
    """A window dimming on focus loss is not the user doing something.

    Measured at ~0.049 before flooring — close enough to the 0.08 threshold that
    without the floor, alt-tabbing would wake the decider and spend a model call
    to be told nothing happened.
    """
    dimmed = fr.Frame(at=99.0, thumb=bytes(max(0, b - 40) for b in gui[0].thumb))
    assert fr.visual_change(gui[0], dimmed) == 0.0


def test_a_brighter_panel_is_also_not_a_change(fr, gui):
    brighter = fr.Frame(at=99.0, thumb=bytes(min(255, b + 35) for b in gui[0].thumb))
    assert fr.visual_change(gui[0], brighter) == 0.0


def test_the_noise_floor_sits_below_the_change_threshold(fr):
    """If these cross, either noise wakes the gate or real change is floored."""
    assert fr.NOISE_FLOOR < fr.DEFAULT_CHANGE_THRESHOLD


def test_an_undersized_thumbnail_hashes_to_zero_rather_than_crashing(fr):
    """A truncated grab must degrade, not take the loop down mid-session."""
    assert fr.dhash(b"\x00" * 10) == 0


# ══ Localisation ══════════════════════════════════════════════════════════

def test_region_changes_cover_the_whole_frame(fr, gui):
    scores = fr.region_changes(gui[5], gui[8])
    assert len(scores) == 6
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_region_changes_locate_the_sweep(fr, gui):
    """Which tile moved is how "one knob" is told from "new preset"."""
    scores = fr.region_changes(gui[5], gui[8])
    assert max(scores) > 0.0
    assert fr.describe_regions(scores) != "no localised movement"


def test_static_frames_have_no_localised_movement(fr, gui):
    assert fr.describe_regions(fr.region_changes(gui[1], gui[2])) == "no localised movement"


def test_region_description_is_words_not_numbers(fr, gui):
    described = fr.describe_regions(fr.region_changes(gui[5], gui[8]))
    assert any(word in described for word in ("top", "bottom", "whole"))


def test_region_changes_degrade_on_a_short_buffer(fr, gui):
    assert fr.region_changes(gui[0], fr.Frame(at=1.0, thumb=b"\x00" * 5)) == []


# ══ The ring ══════════════════════════════════════════════════════════════

def test_the_ring_is_bounded(fr, gui):
    """An unbounded buffer is the context-growth bug every reference impl ships."""
    ring = fr.FrameRing(capacity=4)
    for frame in gui:
        ring.add(frame)
    assert len(ring.frames) == 4
    assert ring.latest is gui[-1]


def test_add_returns_the_change_against_the_previous_frame(fr, gui):
    ring = fr.FrameRing()
    assert ring.add(gui[0]) == 1.0
    assert ring.add(gui[1]) == 0.0
    assert ring.add(gui[5]) > 0.0


def test_recent_frames_are_oldest_first(fr, gui):
    """Reversed order silently inverts every 'then what happened' judgement."""
    ring = fr.FrameRing()
    for frame in gui:
        ring.add(frame)
    recent = ring.recent(3)
    assert [f.at for f in recent] == sorted(f.at for f in recent)
    assert recent[-1] is gui[-1]


def test_ring_novelty_catches_accumulated_drift(fr, gui):
    """Looking several frames back is what sees a slow sweep as movement."""
    ring = fr.FrameRing()
    for frame in gui[: SWEEP[-1] + 1]:
        ring.add(frame)
    assert ring.novelty(lookback=4) >= fr.DEFAULT_CHANGE_THRESHOLD


def test_novelty_fires_on_the_tick_a_sweep_starts(fr, gui):
    """The measured bug: endpoint comparison blinds the gate at sweep onset.

    A sweep beginning now still has static frames behind it, so
    ``change(oldest, newest)`` reads ~0 and the tick is gated — suppressing
    exactly the moment worth looking at. Nine minutes of sound design with ten
    sweeps produced 159 'unchanged' verdicts and one utterance before this.
    """
    ring = fr.FrameRing()
    for frame in gui[: STATIC[-1] + 1]:
        ring.add(frame)
    ring.add(gui[SWEEP[0] + 1])  # first genuinely moving frame
    assert ring.novelty(lookback=4) >= fr.DEFAULT_CHANGE_THRESHOLD


def test_novelty_is_the_max_pairwise_change_not_the_endpoints(fr, gui):
    """Movement in the middle of the window must not be invisible.

    A ring holding static -> moved -> back-to-similar has near-zero endpoint
    difference while definitely containing movement.
    """
    ring = fr.FrameRing()
    ring.add(gui[1])
    ring.add(gui[9])
    ring.add(gui[1])
    assert ring.novelty(lookback=3) > 0.0


def test_ring_novelty_is_quiet_while_nothing_moves(fr, gui):
    ring = fr.FrameRing()
    for frame in gui[: STATIC[-1] + 1]:
        ring.add(frame)
    assert ring.novelty(lookback=3) == 0.0


def test_a_single_frame_is_wholly_novel(fr, gui):
    ring = fr.FrameRing()
    ring.add(gui[0])
    assert ring.novelty() == 1.0


def test_span_selects_by_time(fr, gui):
    ring = fr.FrameRing(capacity=12)
    for frame in gui:
        ring.add(frame)
    assert len(ring.span(3.0, now=11.0)) == 4


# ══ Encoding ══════════════════════════════════════════════════════════════

def test_frames_encode_as_data_urls(fr, gui):
    urls = fr.as_data_urls(gui[:3])
    assert len(urls) == 3
    assert all(u.startswith("data:image/jpeg;base64,") for u in urls)


def test_frames_without_jpeg_bytes_are_skipped(fr):
    assert fr.as_data_urls([fr.Frame(at=0.0, thumb=b"\x00" * 100)]) == []


def test_frame_block_uses_relative_negative_offsets(fr, gui):
    """'three seconds ago' is the useful framing when asking about now."""
    block = fr.frame_block(gui[8:12], now=11.0)
    assert "-3.0s" in block
    assert "+0.0s" in block


def test_frame_block_carries_the_region_hint(fr, gui):
    block = fr.frame_block(gui[:3], now=2.0, regions="movement in top-centre")
    assert "top-centre" in block


def test_empty_frame_block_renders_nothing(fr):
    assert fr.frame_block([], now=0.0) == ""


# ══ ffmpeg argv ═══════════════════════════════════════════════════════════

def test_grab_produces_both_outputs_from_one_pass(fr):
    """Two invocations would hash a different moment than the one sent.

    A change could then be detected in a frame the model never sees, and vice
    versa — the gate and the evidence must be the same instant.
    """
    args = fr.grab_args(["-f", "x11grab", "-i", ":0"], jpeg_path="out.jpg", thumb_path="out.gray")
    assert args.count("ffmpeg") == 1
    assert "out.jpg" in args
    assert "out.gray" in args
    assert args.count("-i") == 1


def test_grab_inherits_the_hosts_capture_args(fr):
    """The loop knows nothing about per-OS capture; the plan supplies it."""
    args = fr.grab_args(["-f", "gdigrab", "-i", "desktop"])
    assert "gdigrab" in args
    assert "desktop" in args


def test_grab_thumbnail_matches_the_hash_geometry(fr):
    """A scale filter that drifts from HASH_WIDTH/HEIGHT breaks every hash."""
    args = fr.grab_args(["-i", "x"], thumb_path="t.gray")
    joined = " ".join(args)
    assert f"scale={fr.HASH_WIDTH}:{fr.HASH_HEIGHT}" in joined
    assert "gray" in joined


def test_grab_takes_exactly_one_frame(fr):
    args = fr.grab_args(["-i", "x"])
    assert "-frames:v" in args
    assert args[args.index("-frames:v") + 1] == "1"


def test_grab_without_a_thumb_path_only_emits_the_jpeg(fr):
    args = fr.grab_args(["-i", "x"], jpeg_path="only.jpg")
    assert "only.jpg" in args
    assert "rawvideo" not in args


# ══ The signal contract ══════════════════════════════════════════════════
#
# The cross-scale bug these replaced (frame novelty vs a keypress-shaped
# threshold, 159 gated sweeps) is now structurally impossible: this module
# calibrates its own salience before the decider ever sees a number.

def test_screen_signal_is_calibrated_to_zero_when_static(fr, gui):
    ring = fr.FrameRing()
    for frame in gui[: STATIC[-1] + 1]:
        ring.add(frame)
    signal = fr.screen_signal(ring, now=float(STATIC[-1]))
    assert signal.name == "screen"
    assert signal.salience == 0.0


def test_screen_signal_is_salient_during_a_sweep(fr, gui):
    """The case the whole module exists for, expressed in the shared unit."""
    ring = fr.FrameRing()
    for frame in gui[: SWEEP[-1] + 1]:
        ring.add(frame)
    signal = fr.screen_signal(ring, now=float(SWEEP[-1]))
    assert signal.salience > 0.0


def test_screen_salience_clears_the_deciders_one_threshold(fr, gui):
    """The actual contract: a real sweep must wake the loop, unaided.

    Before calibration, real GUI movement (0.08-0.15 Hamming) never reached the
    decider's 0.34 default and every sweep was gated.
    """
    decider = _load("decider")
    ring = fr.FrameRing()
    for frame in gui[: SWEEP[-1] + 1]:
        ring.add(frame)
    assert fr.screen_signal(ring, now=float(SWEEP[-1])).salience >= decider.MIN_SALIENCE


def test_brightness_noise_stays_below_the_threshold(fr, gui):
    """A window dimming must not wake the loop."""
    decider = _load("decider")
    ring = fr.FrameRing()
    ring.add(gui[0])
    ring.add(fr.Frame(at=1.0, thumb=bytes(max(0, b - 40) for b in gui[0].thumb)))
    assert fr.screen_signal(ring, now=1.0).salience < decider.MIN_SALIENCE


def test_screen_signal_carries_the_frames_and_the_region_hint(fr, gui):
    ring = fr.FrameRing()
    for frame in gui[: SWEEP[-1] + 1]:
        ring.add(frame)
    signal = fr.screen_signal(ring, now=float(SWEEP[-1]))
    assert "Frames" in signal.block
    assert "raw_change" in signal.detail


def test_screen_signal_never_asks_to_wait(fr, gui):
    """The screen has no opinion about interruption; other sources do."""
    ring = fr.FrameRing()
    ring.add(gui[0])
    assert fr.screen_signal(ring, now=0.0).can_speak is True
