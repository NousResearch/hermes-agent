"""Musical timing — the input track for someone playing, not clicking.

Two things differ from the game case and both matter.

**MIDI is not a privacy risk.** A note number is not a letter; a performance
cannot be read as a sentence. So the allowlist/text-mode machinery in ``inputs``
does not apply here and note data is recorded in full. What IS sensitive is
audio from a room microphone, which is why the audio path only ever yields
derived features (onset times, RMS) and never a recording the model could
transcribe.

**Timing is the whole question.** "Was I in time" is a comparison between when a
note landed and when the grid says it should have. A 1 fps frame cannot see
that; an onset timestamp at millisecond resolution can, which makes this track
the primary evidence for a music session in the same way the key track is for a
rotation. Tempo is estimated from the performance itself rather than asked for,
because a player who does not know their own tempo is exactly the player asking
for feedback.

Everything here is pure functions over onset times, so it works identically for
MIDI note-ons, ffmpeg ``silencedetect`` boundaries, or a librosa onset envelope —
whatever the caller can produce.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type hints only
    from plugins.watch.signals import Signal

#: Below this many onsets there is no rhythm to analyze — two notes define an
#: interval, not a tempo, and reporting a BPM from them is noise dressed as data.
MIN_ONSETS_FOR_TEMPO = 6

#: Silence longer than this reads as a musical rest rather than articulation.
#: Shorter gaps are staccato or a chord change; this is where a phrase ended.
REST_SECONDS = 1.2

#: Timing deviation past this fraction of a beat is audible to a listener.
#: ~50 ms at 120 BPM. Under it, "you were early" is a measurement, not a note
#: worth interrupting someone for.
AUDIBLE_DEVIATION_BEATS = 0.1

#: Relative tempo change worth mentioning. 3% is roughly the point where a
#: listener notices a piece speeding up; below it, every human performance
#: "drifts" and reporting it would make the coach a metronome pedant.
TEMPO_DRIFT_AUDIBLE = 0.03


@dataclass(frozen=True)
class Onset:
    """One attack — a MIDI note-on or a detected audio onset.

    Attributes:
        at: Seconds since recording started.
        pitch: MIDI note number when known. ``None`` for audio onsets, where
            only the timing is available.
        velocity: 0..127 when known; dynamics are part of "how did that sound".
    """

    at: float
    pitch: Optional[int] = None
    velocity: Optional[int] = None


def intervals(onsets: Iterable[Onset]) -> list[float]:
    """Inter-onset intervals — the raw material for every timing judgement."""
    times = sorted(o.at for o in onsets)
    return [round(b - a, 4) for a, b in zip(times, times[1:])]


def estimate_tempo(onsets: Iterable[Onset]) -> Optional[float]:
    """BPM implied by the performance, or None when there isn't enough to say.

    Uses the MEDIAN interval, not the mean: a single long pause between phrases
    would drag a mean into nonsense, and the median is exactly the "typical note
    length" a listener hears as the pulse. Intervals are folded toward the
    median first so a passage mixing eighths and quarters still resolves to one
    pulse rather than averaging into a tempo that is in neither.
    """
    gaps = [g for g in intervals(onsets) if g > 0]
    if len(gaps) < MIN_ONSETS_FOR_TEMPO - 1:
        return None

    base = statistics.median(gaps)
    if base <= 0:
        return None

    folded = []
    for gap in gaps:
        ratio = gap / base
        # Collapse obvious multiples/divisions onto the pulse (a half note is
        # the same tempo as a quarter, played longer).
        for divisor in (4, 3, 2, 1):
            if abs(ratio - divisor) < 0.25:
                folded.append(gap / divisor)
                break
        else:
            if abs(ratio - 0.5) < 0.12:
                folded.append(gap * 2)
            elif abs(ratio - 0.25) < 0.08:
                folded.append(gap * 4)
            else:
                folded.append(gap)

    pulse = statistics.median(folded)
    return round(60.0 / pulse, 1) if pulse > 0 else None


def grid_deviations(onsets: Iterable[Onset], bpm: float) -> list[float]:
    """Signed deviation of each onset from the nearest grid line, in BEATS.

    Negative is early, positive is late. Beats rather than seconds because
    "20 ms late" means something different at 80 BPM than at 180, and the player
    experiences the beat-relative version.

    IMPORTANT — this measures JITTER, not drift, and it aliases. Snapping to the
    nearest grid line means a deviation beyond ±0.5 beats wraps around to the
    next line, so a performance with steady accumulating drift produces
    deviations that sweep the full range and average to roughly zero. That is a
    property of grid-relative measurement, not a bug to paper over: use
    ``tempo_drift`` for "are they speeding up" and this for "are they tight
    around their own pulse".

    The grid is anchored on the FIRST onset rather than on zero: a performance
    that starts half a beat after the recording began is not thereby late.
    """
    times = sorted(o.at for o in onsets)
    if len(times) < 2 or bpm <= 0:
        return []

    beat = 60.0 / bpm
    origin = times[0]
    out = []
    for moment in times:
        beats_elapsed = (moment - origin) / beat
        nearest = round(beats_elapsed)
        out.append(round(beats_elapsed - nearest, 4))
    return out


def phrases(onsets: Iterable[Onset], min_gap: float = REST_SECONDS) -> list[list[Onset]]:
    """Split a performance at its rests.

    Every timing judgement has to be made INSIDE a phrase. A window that spans a
    rest contains one interval that is not a note length at all — it is a pause —
    and that interval poisons everything downstream: it inflates the median (so
    the tempo estimate is wrong), it makes the two halves of ``tempo_drift``
    incomparable (they are different passages), and it drives grid jitter into
    aliasing.

    Observed live: a 20 s window spanning one 4 s rest during a take that was
    rushing at +14% reported "dragging, -3.6% drift, jitter 0.411 beats". Every
    number in that sentence was an artefact of the pause.
    """
    ordered = sorted(onsets, key=lambda o: o.at)
    if not ordered:
        return []

    groups: list[list[Onset]] = [[ordered[0]]]
    for previous, current in zip(ordered, ordered[1:]):
        if current.at - previous.at >= min_gap:
            groups.append([current])
        else:
            groups[-1].append(current)
    return groups


def longest_phrase(onsets: Iterable[Onset], min_gap: float = REST_SECONDS) -> list[Onset]:
    """The most substantial continuous passage in a window.

    What to judge when a window spans several phrases: the longest one is the
    one the player was actually working on, and it is the only span where an
    interval is guaranteed to be a note length.
    """
    groups = phrases(onsets, min_gap=min_gap)
    return max(groups, key=len) if groups else []


def tempo_drift(onsets: Iterable[Onset], min_gap: float = REST_SECONDS) -> Optional[float]:
    """Relative tempo change across the performance. Negative = slowing down.

    This is what "rushing" and "dragging" actually mean when there is no click
    track to compare against. A player who is consistently 5% fast is not
    rushing — they are playing at a different tempo, and telling them they are
    early on every note is wrong. A player whose second half is 5% faster than
    their first half IS rushing, and that is a real note worth giving.

    Measured within a single phrase: comparing halves either side of a rest
    compares two different passages and reports their difference as drift.
    Halves rather than a regression on intervals because a single fumbled note
    skews a slope fit far more than it moves a median.
    """
    passage = longest_phrase(onsets, min_gap=min_gap)
    if len(passage) < MIN_ONSETS_FOR_TEMPO * 2:
        return None

    middle = len(passage) // 2
    early = estimate_tempo(passage[:middle])
    late = estimate_tempo(passage[middle:])
    if not early or not late or early <= 0:
        return None
    return round((late - early) / early, 4)


def timing_verdict(
    onsets: Iterable[Onset],
    bpm: Optional[float] = None,
    min_gap: float = REST_SECONDS,
) -> dict:
    """Rushing, dragging, inconsistent or steady — with the numbers behind it.

    Judged on the longest continuous PHRASE in the input, never across a rest
    (see ``phrases``). Three distinct faults, deliberately not collapsed,
    because the advice for each is different:

    * **rushing / dragging** — the pulse is CHANGING (``tempo_drift``). "You
      speed up through the second half."
    * **off the reference** — a ``bpm`` was supplied (a click, a backing track)
      and the player's own pulse differs from it. "You're sitting 4% behind the
      track."
    * **inconsistent** — the pulse is stable but individual notes scatter around
      it (grid jitter). "Your pulse is fine, the notes are untidy."

    Without a reference ``bpm``, a steady-but-different tempo is NOT a fault —
    it is just the tempo they chose, and reporting it as rushing is confidently
    wrong feedback.
    """
    passage = longest_phrase(onsets, min_gap=min_gap)
    own_tempo = estimate_tempo(passage)
    if own_tempo is None:
        return {"tempo": None, "verdict": "not enough notes", "audible": False}

    drift = tempo_drift(passage, min_gap=min_gap)
    # Jitter is always measured against the player's OWN pulse, so a deliberate
    # tempo choice never shows up as untidiness.
    deviations = grid_deviations(passage, own_tempo)
    spread = statistics.pstdev(deviations) if len(deviations) > 1 else 0.0

    reference_error = None
    if bpm is not None and bpm > 0:
        reference_error = round((own_tempo - bpm) / bpm, 4)

    verdict = "steady"
    if drift is not None and abs(drift) >= TEMPO_DRIFT_AUDIBLE:
        verdict = "rushing" if drift > 0 else "dragging"
    elif reference_error is not None and abs(reference_error) >= TEMPO_DRIFT_AUDIBLE:
        # Faster than the reference reads as rushing, slower as dragging.
        verdict = "rushing" if reference_error > 0 else "dragging"
    elif spread >= AUDIBLE_DEVIATION_BEATS:
        verdict = "inconsistent"

    result = {
        "tempo": own_tempo,
        "verdict": verdict,
        "spread_beats": round(spread, 3),
        "audible": verdict != "steady",
        "notes": len(passage),
    }
    if drift is not None:
        result["tempo_drift"] = drift
    if reference_error is not None:
        result["reference_error"] = reference_error
    return result


def rests(onsets: Iterable[Onset], min_gap: float = REST_SECONDS) -> list[tuple[float, float]]:
    """``(start, end)`` spans where nothing was played.

    The delivery windows. Speaking during a phrase is destructive in a way it
    never is in a game — the person is producing the thing being judged, and an
    interruption ruins the take rather than merely annoying them.
    """
    times = sorted(o.at for o in onsets)
    return [
        (round(a, 3), round(b, 3))
        for a, b in zip(times, times[1:])
        if b - a >= min_gap
    ]


def in_rest(onsets: Iterable[Onset], at: float, min_gap: float = REST_SECONDS) -> bool:
    """Whether ``at`` falls inside a rest — i.e. whether it is safe to speak now.

    Note this is answerable in real time with a one-sided test: if the last
    onset was more than ``min_gap`` ago, the player has stopped, and we do not
    need to know when they will start again.
    """
    times = sorted(o.at for o in onsets if o.at <= at)
    if not times:
        return True
    return at - times[-1] >= min_gap


def dynamics(onsets: Iterable[Onset]) -> Optional[dict]:
    """Velocity spread, when the source provides it.

    Flat dynamics is one of the most common and least-noticed problems in a
    take, and it is invisible in both the video and the timing.
    """
    velocities = [o.velocity for o in onsets if o.velocity is not None]
    if len(velocities) < MIN_ONSETS_FOR_TEMPO:
        return None
    return {
        "mean": round(statistics.fmean(velocities), 1),
        "spread": round(statistics.pstdev(velocities), 1) if len(velocities) > 1 else 0.0,
        "range": [min(velocities), max(velocities)],
    }


def pitch_range(onsets: Iterable[Onset]) -> Optional[tuple[int, int]]:
    pitches = [o.pitch for o in onsets if o.pitch is not None]
    return (min(pitches), max(pitches)) if pitches else None


def musical_novelty(recent: Iterable[Onset], baseline: Iterable[Onset]) -> float:
    """0..1 change score, the musical analogue of ``inputs.novelty_score``.

    Same contract, different features: a player repeating a groove is not news,
    while a tempo shift, a register jump, or stopping is. Tempo change is
    weighted first because it is the thing a practising musician is usually
    working on.

    Note rate is compared as notes-per-SECOND, never as raw counts. The two
    windows routinely cover different spans — the baseline is clipped at session
    start, and a rest shortens whichever window contains it — so comparing
    counts reports a change that is entirely an artefact of window length.
    Measured: 40 notes in a full recent window against 8 in a clipped baseline
    scored 1.0 for a performance that was metronome-steady.
    """
    recent_list, baseline_list = list(recent), list(baseline)
    if not baseline_list:
        return 1.0
    if not recent_list:
        return 1.0

    scores = []

    recent_tempo = estimate_tempo(recent_list)
    baseline_tempo = estimate_tempo(baseline_list)
    if recent_tempo and baseline_tempo:
        scores.append(min(1.0, abs(recent_tempo - baseline_tempo) / baseline_tempo * 4))

    recent_pitches = pitch_range(recent_list)
    baseline_pitches = pitch_range(baseline_list)
    if recent_pitches and baseline_pitches:
        # An octave of movement in the centre of the range is a real change.
        recent_centre = statistics.fmean(recent_pitches)
        baseline_centre = statistics.fmean(baseline_pitches)
        scores.append(min(1.0, abs(recent_centre - baseline_centre) / 12))

    recent_rate = _note_rate(recent_list)
    baseline_rate = _note_rate(baseline_list)
    if recent_rate is not None and baseline_rate is not None and baseline_rate > 0:
        scores.append(min(1.0, abs(recent_rate - baseline_rate) / baseline_rate))

    return round(max(scores) if scores else 0.0, 3)


def _note_rate(onsets: Sequence[Onset]) -> Optional[float]:
    """Notes per second across the span the notes actually occupy.

    ``None`` when a single note gives no span to divide by — one note is not a
    rate, and treating it as one is how a window boundary becomes a musical
    event.
    """
    if len(onsets) < 2:
        return None
    span = onsets[-1].at - onsets[0].at
    return len(onsets) / span if span > 0 else None


def render_music_block(
    onsets: Iterable[Onset],
    *,
    bpm: Optional[float] = None,
    speed: float = 1.0,
) -> str:
    """The text block handed to the model for a music session.

    Derived facts, not raw onsets: "112 BPM, rushing by 0.08 of a beat, flat
    dynamics" is a handful of tokens and directly answerable, where a list of
    200 timestamps makes the model do arithmetic it is bad at.
    """
    ordered = sorted(onsets, key=lambda o: o.at)
    if not ordered:
        return ""

    verdict = timing_verdict(ordered, bpm)
    lines = ["Performance track (derived from note timing):"]

    if verdict["tempo"]:
        detail = f"tempo ~{verdict['tempo']} BPM, {verdict['notes']} notes, timing: {verdict['verdict']}"
        if "tempo_drift" in verdict:
            detail += f" (tempo drift {verdict['tempo_drift']:+.1%}"
            detail += f", jitter {verdict['spread_beats']:.3f} beats)"
        if "reference_error" in verdict:
            detail += f" — {verdict['reference_error']:+.1%} vs the {bpm:g} BPM reference"
        lines.append(detail)
    else:
        lines.append(f"{len(ordered)} notes, too few to estimate tempo")

    dyn = dynamics(ordered)
    if dyn:
        flat = " (flat — little dynamic variation)" if dyn["spread"] < 8 else ""
        lines.append(f"velocity mean {dyn['mean']}, spread {dyn['spread']}{flat}")

    span = pitch_range(ordered)
    if span:
        lines.append(f"pitch range {span[0]}-{span[1]}")

    quiet = rests(ordered)
    if quiet:
        scaled = [(round(a / speed, 1), round(b / speed, 1)) for a, b in quiet[-3:]]
        lines.append(f"recent rests (clip time): {scaled}")

    return "\n".join(lines)


# ── The signal contract ───────────────────────────────────────────────────

#: Salience calibration for the performance track. ``musical_novelty`` is a
#: composite of tempo change, register shift and note-rate change: repeating a
#: groove sits under 0.2, while a tempo shift or a register jump runs 0.4+.
SALIENCE_FLOOR = 0.2
SALIENCE_CEILING = 0.6

#: Extra salience granted when the timing verdict is an AUDIBLE fault. A player
#: holding a steady groove scores no novelty by construction — nothing is
#: changing — but "you have been rushing for the last thirty seconds" is exactly
#: what they asked to be told. Without this the most useful observation in a
#: practice session is the one the gate suppresses.
FAULT_SALIENCE = 0.7


def notes_signal(
    onsets: Iterable[Onset],
    *,
    now: float,
    window_seconds: float = 20.0,
    baseline_seconds: float = 60.0,
    bpm: Optional[float] = None,
) -> "Signal":
    """This track's opinion about the current moment, as a calibrated Signal.

    Two things make it salient: a CHANGE (they switched groove or register) or a
    persistent audible FAULT (they are rushing). The second is why this source
    cannot just report novelty — steady playing is zero novelty and may still be
    steadily wrong.

    ``can_speak`` is False mid-phrase. Talking over a musician ruins the take
    being judged, which is worse than annoying: the thing the feedback is about
    gets destroyed by the feedback.
    """
    from plugins.watch.signals import Signal, calibrate

    ordered = sorted(onsets, key=lambda o: o.at)
    recent = [o for o in ordered if now - window_seconds <= o.at <= now]
    baseline = [
        o for o in ordered
        if now - baseline_seconds <= o.at < now - window_seconds
    ]

    raw = musical_novelty(recent, baseline)
    salience = calibrate(raw, SALIENCE_FLOOR, SALIENCE_CEILING)

    verdict = timing_verdict(recent, bpm)
    if verdict.get("audible"):
        salience = max(salience, FAULT_SALIENCE)

    return Signal(
        name="notes",
        salience=salience,
        block=render_music_block(recent, bpm=bpm),
        can_speak=in_rest(ordered, at=now),
        detail={
            "raw_novelty": raw,
            "verdict": verdict.get("verdict"),
            "notes": len(recent),
        },
    )
