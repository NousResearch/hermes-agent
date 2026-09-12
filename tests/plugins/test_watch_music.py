"""Tests for the music track and the interruption (hold) gate.

The music track's contract is that timing judgements are correct and that
"steady" is distinguishable from "rushing" — get that wrong and the coach is
confidently wrong, which is worse than silent. The hold gate's contract is that
an utterance produced mid-phrase is DELIVERED LATER rather than dropped.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(module_name: str):
    path = REPO_ROOT / "plugins" / "watch" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"_watch_{module_name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mus():
    return _load("music")


@pytest.fixture(scope="module")
def dec():
    return _load("decider")


def _perfect(mus, bpm=120.0, count=32, start=0.0, pitch=60, velocity=80):
    beat = 60.0 / bpm
    return [
        mus.Onset(at=start + i * beat, pitch=pitch, velocity=velocity)
        for i in range(count)
    ]


def _drifting(mus, bpm=120.0, count=32, accel=0.0, jitter=0.0, seed=3):
    """Onsets whose tempo CHANGES by ``accel`` (fraction) across the take.

    Steady drift, not a constant offset: the interval shrinks progressively, so
    the second half really is a different tempo from the first. That is what
    rushing means.
    """
    import random

    rng = random.Random(seed)
    beat = 60.0 / bpm
    out, at = [], 0.0
    for i in range(count):
        progress = i / max(1, count - 1)
        step = beat * (1.0 - accel * progress)
        wobble = rng.uniform(-jitter, jitter) if jitter else 0.0
        out.append(mus.Onset(at=at + wobble, pitch=60, velocity=80))
        at += step
    return out


def _offset_tempo(mus, bpm=120.0, count=32, factor=1.0):
    """A steady performance at ``bpm * factor`` — a tempo CHOICE, not a fault."""
    return _perfect(mus, bpm=bpm * factor, count=count)


# ══ Tempo ═════════════════════════════════════════════════════════════════

def test_tempo_is_recovered_from_an_even_performance(mus):
    assert mus.estimate_tempo(_perfect(mus, bpm=120.0)) == pytest.approx(120.0, abs=1.0)


@pytest.mark.parametrize("bpm", [72.0, 96.0, 140.0, 174.0])
def test_tempo_is_recovered_across_the_usable_range(mus, bpm):
    assert mus.estimate_tempo(_perfect(mus, bpm=bpm)) == pytest.approx(bpm, abs=2.0)


def test_too_few_notes_yields_no_tempo(mus):
    """Two notes are an interval, not a tempo. Reporting a BPM there is noise."""
    assert mus.estimate_tempo(_perfect(mus, count=3)) is None
    assert mus.estimate_tempo([]) is None


def test_mixed_note_lengths_resolve_to_one_pulse(mus):
    """A passage of quarters and eighths is one tempo, not the average of two.

    Without folding, the median interval lands between the two note lengths and
    reports a tempo that is in neither — confidently wrong feedback.
    """
    beat = 0.5  # 120 BPM
    times, at = [], 0.0
    for i in range(24):
        times.append(at)
        at += beat if i % 3 else beat / 2
    onsets = [mus.Onset(at=t, pitch=60) for t in times]
    assert mus.estimate_tempo(onsets) == pytest.approx(120.0, abs=6.0)


def test_a_long_pause_does_not_drag_the_tempo(mus):
    """The median is used precisely so one rest cannot rewrite the pulse."""
    onsets = _perfect(mus, bpm=120.0, count=16)
    tail = _perfect(mus, bpm=120.0, count=16, start=onsets[-1].at + 8.0)
    assert mus.estimate_tempo(onsets + tail) == pytest.approx(120.0, abs=2.0)


# ══ Timing verdict ════════════════════════════════════════════════════════

def test_an_even_performance_reads_as_steady(mus):
    verdict = mus.timing_verdict(_perfect(mus, bpm=120.0))
    assert verdict["verdict"] == "steady"
    assert verdict["audible"] is False


def test_steady_playing_is_not_worth_interrupting(mus):
    """`audible` is the gate: measurable is not the same as worth saying."""
    assert mus.timing_verdict(_perfect(mus, bpm=100.0))["audible"] is False


def test_rushing_means_the_tempo_is_increasing(mus):
    """Rushing is a CHANGE in pulse, not a fast pulse.

    The first implementation measured mean deviation from a fixed grid, which
    both aliases past ±0.5 beats (steady drift averages back to zero) and
    mislabels a deliberate tempo choice as a fault.
    """
    verdict = mus.timing_verdict(_drifting(mus, bpm=120.0, count=48, accel=0.12))
    assert verdict["verdict"] == "rushing"
    assert verdict["tempo_drift"] > 0
    assert verdict["audible"] is True


def test_dragging_means_the_tempo_is_decreasing(mus):
    verdict = mus.timing_verdict(_drifting(mus, bpm=120.0, count=48, accel=-0.12))
    assert verdict["verdict"] == "dragging"
    assert verdict["tempo_drift"] < 0


def test_a_steady_but_different_tempo_is_not_a_fault(mus):
    """Playing 8% fast with no click is a tempo choice, not rushing.

    Telling a player they are early on every note when they are simply playing
    at their own steady tempo is confidently wrong feedback.
    """
    verdict = mus.timing_verdict(_offset_tempo(mus, bpm=120.0, factor=1.08))
    assert verdict["verdict"] == "steady"
    assert verdict["audible"] is False


def test_a_reference_tempo_turns_an_offset_into_a_fault(mus):
    """With a click or backing track, sitting off it IS the note to give."""
    fast = _offset_tempo(mus, bpm=120.0, factor=1.08)
    verdict = mus.timing_verdict(fast, bpm=120.0)
    assert verdict["verdict"] == "rushing"
    assert verdict["reference_error"] > 0
    assert verdict["audible"] is True


def test_a_reference_tempo_detects_sitting_behind_it(mus):
    slow = _offset_tempo(mus, bpm=120.0, factor=0.92)
    verdict = mus.timing_verdict(slow, bpm=120.0)
    assert verdict["verdict"] == "dragging"
    assert verdict["reference_error"] < 0


def test_matching_the_reference_is_steady(mus):
    assert mus.timing_verdict(_perfect(mus, bpm=120.0), bpm=120.0)["verdict"] == "steady"


def test_erratic_timing_reads_as_inconsistent_not_steady(mus):
    """Jitter around a stable pulse: the pulse is fine, the notes are untidy."""
    verdict = mus.timing_verdict(_drifting(mus, bpm=120.0, count=40, jitter=0.09))
    assert verdict["verdict"] == "inconsistent"
    assert verdict["audible"] is True


def test_grid_deviations_measure_jitter_and_alias_by_design(mus):
    """Documented limitation, asserted so nobody 'fixes' it into drift detection.

    Snapping to the nearest grid line wraps past ±0.5 beats, so steady drift
    sweeps the range and averages to ~0. That is why `tempo_drift` exists.
    """
    deviations = mus.grid_deviations(_drifting(mus, bpm=120.0, count=48, accel=0.12), 120.0)
    assert all(-0.5 <= d <= 0.5 for d in deviations)


# ══ Phrases — never judge across a rest ═══════════════════════════════════

def test_a_window_is_split_at_its_rests(mus):
    first = _perfect(mus, bpm=120.0, count=10)
    second = _perfect(mus, bpm=120.0, count=14, start=first[-1].at + 3.0)
    groups = mus.phrases(first + second)
    assert [len(g) for g in groups] == [10, 14]


def test_staccato_playing_is_one_phrase(mus):
    assert len(mus.phrases(_perfect(mus, bpm=140.0, count=30))) == 1


def test_the_longest_phrase_is_what_gets_judged(mus):
    short = _perfect(mus, bpm=120.0, count=8)
    long = _perfect(mus, bpm=120.0, count=40, start=short[-1].at + 3.0)
    assert len(mus.longest_phrase(short + long)) == 40


def test_a_rest_inside_the_window_does_not_fake_a_tempo_verdict(mus):
    """The measured bug: a pause counted as a slow interval and inverted the call.

    Live readout before this fix, on a take that was rushing at +14%:
    "dragging (tempo drift -3.6%, jitter 0.411 beats)". The 4 s rest inside the
    window was being treated as a note length — so the median tempo was wrong,
    the two drift halves straddled different passages, and grid jitter aliased.
    """
    rushing = _drifting(mus, bpm=120.0, count=40, accel=0.14)
    after = _perfect(mus, bpm=120.0, count=12, start=rushing[-1].at + 4.0)

    spanning = mus.timing_verdict(rushing + after)
    # The pause must not drive the numbers: jitter stays in the range real
    # playing produces, nowhere near the 0.41 aliasing signature.
    assert spanning["spread_beats"] < 0.35


def test_drift_is_measured_inside_one_phrase(mus):
    """Comparing halves either side of a rest compares two different passages."""
    steady_a = _perfect(mus, bpm=100.0, count=20)
    steady_b = _perfect(mus, bpm=150.0, count=20, start=steady_a[-1].at + 4.0)
    # Two steady phrases at different tempos are not one rushing performance.
    drift = mus.tempo_drift(steady_a + steady_b)
    assert drift is None or abs(drift) < 0.1


def test_rushing_inside_a_single_phrase_is_still_caught(mus):
    """The fix must not cost sensitivity to the thing it exists to report."""
    verdict = mus.timing_verdict(_drifting(mus, bpm=120.0, count=48, accel=0.12))
    assert verdict["verdict"] == "rushing"


def test_empty_input_has_no_phrases(mus):
    assert mus.phrases([]) == []
    assert mus.longest_phrase([]) == []


def test_deviations_are_in_beats_not_seconds(mus):
    """The same absolute wobble is a bigger fraction of a fast beat."""
    wobble = 0.04  # seconds
    slow = mus.grid_deviations(_drifting(mus, bpm=60.0, count=40, jitter=wobble), 60.0)
    fast = mus.grid_deviations(_drifting(mus, bpm=180.0, count=40, jitter=wobble), 180.0)
    import statistics as st

    assert st.pstdev(fast) > st.pstdev(slow)


def test_the_grid_is_anchored_on_the_first_note(mus):
    """A performance that starts late is not thereby 'dragging'.

    Anchoring on the recorder's clock would report a constant offset that says
    nothing about the playing.
    """
    late = _perfect(mus, bpm=120.0, start=7.31)
    assert mus.timing_verdict(late, bpm=120.0)["verdict"] == "steady"


def test_verdict_degrades_gracefully_without_enough_notes(mus):
    verdict = mus.timing_verdict(_perfect(mus, count=2))
    assert verdict["audible"] is False
    assert verdict["tempo"] is None


# ══ Rests — the delivery windows ══════════════════════════════════════════

def test_rests_are_found_between_phrases(mus):
    phrase = _perfect(mus, bpm=120.0, count=8)
    later = _perfect(mus, bpm=120.0, count=8, start=phrase[-1].at + 3.0)
    found = mus.rests(phrase + later)
    assert len(found) == 1
    assert found[0][1] - found[0][0] >= 3.0


def test_staccato_gaps_are_not_rests(mus):
    """Short gaps are articulation, not a place to start talking."""
    assert mus.rests(_perfect(mus, bpm=120.0, count=16)) == []


def test_in_rest_is_answerable_in_real_time(mus):
    """One-sided: if the last note was long enough ago, they have stopped."""
    onsets = _perfect(mus, bpm=120.0, count=8)
    end = onsets[-1].at
    assert not mus.in_rest(onsets, at=end + 0.2)
    assert mus.in_rest(onsets, at=end + 2.0)


def test_silence_before_the_first_note_counts_as_a_rest(mus):
    assert mus.in_rest(_perfect(mus, start=30.0), at=1.0)


# ══ Dynamics and novelty ══════════════════════════════════════════════════

def test_flat_dynamics_are_detected(mus):
    """Invisible in both the video and the timing, and a very common problem."""
    flat = mus.dynamics(_perfect(mus, velocity=80))
    assert flat is not None and flat["spread"] == 0.0
    assert "flat" in mus.render_music_block(_perfect(mus, velocity=80))


def test_dynamics_absent_when_the_source_has_no_velocity(mus):
    """Audio onsets carry timing only — don't invent dynamics for them."""
    assert mus.dynamics([mus.Onset(at=float(i)) for i in range(20)]) is None


def test_a_tempo_change_scores_high_musical_novelty(mus):
    baseline = _perfect(mus, bpm=100.0, count=20)
    recent = _perfect(mus, bpm=150.0, count=20, start=100.0)
    assert mus.musical_novelty(recent, baseline) >= 0.34


def test_the_same_groove_scores_low_musical_novelty(mus):
    """Repeating a groove is not news — same contract as the key track."""
    baseline = _perfect(mus, bpm=120.0, count=20)
    recent = _perfect(mus, bpm=120.0, count=20, start=60.0)
    assert mus.musical_novelty(recent, baseline) < 0.34


def test_a_register_jump_scores_high_musical_novelty(mus):
    baseline = _perfect(mus, bpm=120.0, count=20, pitch=48)
    recent = _perfect(mus, bpm=120.0, count=20, start=60.0, pitch=72)
    assert mus.musical_novelty(recent, baseline) >= 0.34


def test_stopping_scores_high_musical_novelty(mus):
    assert mus.musical_novelty([], _perfect(mus, count=20)) == 1.0


# ══ The rendered block ════════════════════════════════════════════════════

def test_block_reports_derived_facts_not_raw_onsets(mus):
    """200 timestamps make the model do arithmetic it is bad at."""
    block = mus.render_music_block(_drifting(mus, bpm=120.0, count=48, accel=0.12))
    assert "BPM" in block
    assert "rushing" in block
    assert "drift" in block
    assert block.count("\n") < 8


def test_block_scales_rest_times_by_clip_speed(mus):
    phrase = _perfect(mus, bpm=120.0, count=8)
    later = _perfect(mus, bpm=120.0, count=8, start=phrase[-1].at + 4.0)
    block = mus.render_music_block(phrase + later, bpm=120.0, speed=2.0)
    assert "rests" in block


def test_empty_performance_renders_nothing(mus):
    assert mus.render_music_block([]) == ""


# ══ The hold gate, driven by the music source ═════════════════════════════
#
# The decider knows nothing about music. `music.notes_signal` reports
# `can_speak=False` mid-phrase, and the generic hold gate does the rest.

def test_notes_signal_reports_a_calibrated_salience(mus):
    """Sources calibrate themselves; the decider sees one 0..1 number."""
    onsets = _drifting(mus, bpm=120.0, count=48, accel=0.12)
    signal = mus.notes_signal(onsets, now=onsets[-1].at)
    assert signal.name == "notes"
    assert 0.0 <= signal.salience <= 1.0
    assert "BPM" in signal.block


def test_a_persistent_fault_is_salient_even_with_no_change(mus):
    """Steady-but-wrong is the most useful thing to say in a practice session.

    A player holding a rushing groove scores zero novelty by construction —
    nothing is changing — so a novelty-only source would suppress exactly the
    observation they asked for.
    """
    rushing = _drifting(mus, bpm=120.0, count=48, accel=0.12)
    signal = mus.notes_signal(rushing, now=rushing[-1].at)
    assert signal.salience >= mus.FAULT_SALIENCE


def test_steady_playing_is_not_salient(mus):
    steady = _perfect(mus, bpm=120.0, count=48)
    signal = mus.notes_signal(steady, now=steady[-1].at)
    assert signal.salience < mus.FAULT_SALIENCE


def test_the_source_says_wait_mid_phrase(mus):
    onsets = _perfect(mus, bpm=120.0, count=16)
    mid = mus.notes_signal(onsets, now=onsets[-1].at - 0.2)
    assert mid.can_speak is False


def test_the_source_says_go_in_a_rest(mus):
    onsets = _perfect(mus, bpm=120.0, count=16)
    resting = mus.notes_signal(onsets, now=onsets[-1].at + 3.0)
    assert resting.can_speak is True


def test_mid_phrase_holds_and_the_rest_delivers(dec, mus):
    """End to end, with no music-specific code in the decider."""
    onsets = _perfect(mus, bpm=120.0, count=16)
    end = onsets[-1].at
    state = dec.DeciderState()

    mid = dec.decide(
        state,
        [mus.notes_signal(onsets, now=end - 0.5)],
        at=end - 0.5,
        brief="timing",
        ask=lambda _s, _u: "Your left hand is dragging.",
    )
    assert mid.reason == "held"

    after = dec.decide(
        state,
        [mus.notes_signal(onsets, now=end + 3.0)],
        at=end + 3.0,
        brief="timing",
    )
    assert after.spoke
    assert after.text == "Your left hand is dragging."


def test_note_rate_is_compared_per_second_not_per_count(mus):
    """Window spans differ constantly; counts make that a musical event.

    The baseline is clipped at session start and a rest shortens whichever
    window holds it, so 40 notes against 8 is routine. Measured before the fix:
    salience 1.0 for metronome-steady playing.
    """
    long_window = _perfect(mus, bpm=120.0, count=40)
    short_window = _perfect(mus, bpm=120.0, count=8, start=100.0)
    assert mus.musical_novelty(short_window, long_window) < 0.2


def test_a_genuine_rate_change_is_still_caught(mus):
    """The fix must not cost sensitivity to real rate changes."""
    slow = _perfect(mus, bpm=60.0, count=20)
    fast = _perfect(mus, bpm=180.0, count=20, start=100.0)
    assert mus.musical_novelty(fast, slow) >= 0.34
