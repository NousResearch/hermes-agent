"""The signal contract — one unit, one threshold, pluggable sources.

Every source of "something is happening" reports the same thing: a **salience**
in 0..1 where 0 means "at or below my own noise floor" and 1 means
"unambiguously an event". Calibration happens INSIDE the module that owns the
signal, because only that module knows what its numbers mean.

This exists because the alternative was measured and it fails. Three sources
grew three novelty scores on three incompatible scales — a Hamming fraction
(frames: real movement 0.08-0.15), a distribution distance (keys: steady play
near 0, a rotation switch near 1), and a tempo/pitch composite (notes) — all fed
into one ``novelty`` parameter with one threshold. Frame movement never reached
a keypress-shaped threshold of 0.34, so a nine-minute sound-design session
containing ten filter sweeps produced 159 "unchanged" verdicts and exactly one
utterance. The first attempt at a fix was a documentation comment warning
callers to pass a different threshold per source, which is a landmine with a
sign on it.

With salience the decider knows nothing about Hamming distances, key
distributions or BPM. It takes signals, compares one number against one
threshold, and concatenates the text blocks. Adding MIDI CC or audio onsets
later is a new source implementing this contract, not a new scale for the gate
to get wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class Signal:
    """One source's opinion about this moment.

    Attributes:
        name: Short identifier — ``"screen"``, ``"keys"``, ``"notes"``. Appears
            in the decision log so a replay shows WHICH source woke the loop.
        salience: 0..1, calibrated by the source. 0 = at its noise floor,
            1 = unambiguous event. This is the only number the decider reads.
        block: Text handed to the model. The source decides what is worth
            saying about itself; the decider just assembles.
        can_speak: The source's opinion on whether now is a safe moment to
            interrupt. A musician mid-phrase says False. Sources with no opinion
            leave it True.
        detail: Raw diagnostics for the log — the pre-calibration numbers, so a
            replay can re-tune the calibration without re-recording.
    """

    name: str
    salience: float
    block: str = ""
    can_speak: bool = True
    detail: dict = field(default_factory=dict)


def calibrate(raw: float, floor: float, ceiling: float) -> float:
    """Map a source's own units onto 0..1 salience.

    ``floor`` is where the source's noise lives — below it, nothing is
    reported. ``ceiling`` is where the source is certain. Both are measured
    per-source; this function is the only place the mapping happens so the
    arithmetic can't drift between sources.
    """
    if ceiling <= floor:
        return 1.0 if raw > floor else 0.0
    if raw <= floor:
        return 0.0
    return round(min(1.0, (raw - floor) / (ceiling - floor)), 4)


def strongest(signals: Iterable[Signal]) -> Optional[Signal]:
    """The source with the most to say, or None when there are no signals.

    Max rather than mean or sum: any ONE source noticing something is enough to
    look. Averaging would let three quiet sources drown out the one that saw
    the filter sweep, and summing would make the wake threshold depend on how
    many sources happen to be enabled.
    """
    ordered = list(signals)
    return max(ordered, key=lambda s: s.salience) if ordered else None


def salience_of(signals: Iterable[Signal]) -> float:
    """Combined salience — the number the gate compares."""
    best = strongest(signals)
    return best.salience if best else 0.0


def can_speak(signals: Iterable[Signal]) -> bool:
    """Whether EVERY source considers this a safe moment to interrupt.

    Unanimous, not majority: one source knowing the person is mid-phrase is
    enough to wait, and being wrong costs a few seconds of delay while being
    wrong the other way ruins the take being judged.
    """
    return all(signal.can_speak for signal in signals)


def blocks(signals: Iterable[Signal]) -> str:
    """The sources' text, in the order given, skipping empty ones."""
    return "\n\n".join(s.block for s in signals if s.block)


def summarize(signals: Sequence[Signal]) -> dict:
    """Per-source salience, for the decision log."""
    return {s.name: round(s.salience, 3) for s in signals}
