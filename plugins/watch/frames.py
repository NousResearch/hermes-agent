"""The frame loop — grab, hash, judge whether anything changed.

This is the general mechanism, and the per-domain tracks (keys, notes) are
enrichment on top of it. The reason is simple: whatever the person is doing, the
screen shows it. A synth's filter sweep, an automation lane being drawn, a
health bar dropping, a DAW's playhead — all of it is pixels, and a perceptual
hash over a thumbnail answers "did anything change" without knowing what any of
those things are. Building a bespoke signal extractor per domain (MIDI CC,
keypresses, game telemetry) is how this turns into ten half-features instead of
one that works.

Two products per grab, from one ffmpeg pass:

* a **JPEG** at model resolution, which is what actually gets sent;
* a **thumbnail** — raw 8-bit grayscale, a few dozen bytes — which never leaves
  the machine and exists only to be hashed.

Everything here is pure functions over bytes, so the whole gate is testable
without a screen, a GPU, or an image library. No Pillow, no numpy: a dHash over
a 9x8 grayscale buffer is 72 comparisons, which is cheaper than importing
anything that would do it for us.
"""

from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type hints only
    from plugins.watch.signals import Signal

#: Thumbnail geometry for hashing. 33x32 gives 32x32 = 1024 horizontal
#: comparisons.
#:
#: Sized from measurement, not convention. The usual dHash shape is 9x8 (64
#: bits), which is built for whole-image dedupe and is far too coarse for this
#: job: against rendered plugin-GUI frames, a filter sweep moved 1-2 bits of 64
#: — indistinguishable from noise, so the gate would never have woken. 1024
#: bits resolves a curve redraw or a fader move while still being a 1 KB buffer
#: and a pure-Python hash.
HASH_WIDTH = 33
HASH_HEIGHT = 32
HASH_BITS = (HASH_WIDTH - 1) * HASH_HEIGHT

#: Hamming distance, as a fraction of HASH_BITS, above which the screen counts
#: as changed.
#:
#: Measured against the rendered GUI fixtures (filter curve + level meters):
#: the SLOWEST step of a cutoff sweep reads 0.079, the fastest 0.107, and a
#: whole-panel brightness shift — which is NOT a user action — reads 0.049.
#: 0.07 sits between the noise floor and the weakest real signal. It was
#: briefly 0.08, which cut the slowest sweep frame (0.0791) by a hair: a
#: threshold with no margin against the exact signal it exists to catch is a
#: missed feature, and missing a slow sweep is the expensive direction of error
#: here — a spurious wake costs one cheap call, a missed sweep costs the whole
#: observation.
DEFAULT_CHANGE_THRESHOLD = 0.07

#: Change below this is treated as no change at all, whatever its cause.
#: The brightness floor: gamma shifts, focus dimming and JPEG noise all live
#: under here, and gradient hashing does not fully cancel them.
NOISE_FLOOR = 0.055

#: How many frames the ring keeps. The decider sees the last few, not the
#: session: a rolling window is the whole point, and an unbounded ring is the
#: context-growth bug every streaming reference implementation ships with.
DEFAULT_RING = 12

#: Frames handed to the model per decision. More than a handful stops adding
#: information and starts adding cost — at 66 tok/frame (media_resolution low)
#: six frames is ~400 tokens, which is the right order next to a 300-token
#: prompt.
DEFAULT_SEND = 6


@dataclass(frozen=True)
class Frame:
    """One captured moment.

    Attributes:
        at: Seconds since the loop started.
        jpeg: Encoded frame at model resolution. The only part that is ever
            transmitted.
        thumb: Raw grayscale bytes, ``HASH_WIDTH * HASH_HEIGHT`` long. Local
            only — it exists to be hashed and is useless as an image.
    """

    at: float
    jpeg: bytes = b""
    thumb: bytes = b""

    @property
    def phash(self) -> int:
        return dhash(self.thumb)


# ── Hashing ───────────────────────────────────────────────────────────────

def dhash(thumb: bytes, width: int = HASH_WIDTH, height: int = HASH_HEIGHT) -> int:
    """Difference hash of a raw grayscale buffer.

    One bit per horizontally-adjacent pixel pair: is the left brighter than the
    right. Gradients rather than absolute values, so the hash is stable under
    the brightness and gamma shifts that a window losing focus, a fade, or a
    monitor's auto-dim produce — none of which are the user doing something.
    """
    if len(thumb) < width * height:
        return 0

    bits = 0
    index = 0
    for row in range(height):
        base = row * width
        for column in range(width - 1):
            left = thumb[base + column]
            right = thumb[base + column + 1]
            if left > right:
                bits |= 1 << index
            index += 1
    return bits


def hamming(a: int, b: int) -> int:
    """Differing bit count between two hashes."""
    return bin(a ^ b).count("1")


def visual_change(previous: Optional[Frame], current: Frame) -> float:
    """0..1 change between consecutive frames.

    The free novelty gate, and the domain-agnostic one: it does not know or care
    whether the moving thing is a filter cutoff, a health bar, or a playhead.
    A missing previous frame scores 1.0 — the first frame of a session is
    entirely new.

    Sub-``NOISE_FLOOR`` results are floored to zero. Gradient hashing suppresses
    brightness shifts but does not eliminate them (measured ~0.049 for a
    whole-panel dim), and reporting that as change would let a window losing
    focus wake the decider — which then spends a model call to be told nothing
    happened.
    """
    if previous is None:
        return 1.0
    raw = hamming(previous.phash, current.phash) / HASH_BITS
    return 0.0 if raw < NOISE_FLOOR else round(raw, 4)


def changed(previous: Optional[Frame], current: Frame,
            threshold: float = DEFAULT_CHANGE_THRESHOLD) -> bool:
    return visual_change(previous, current) >= threshold


def region_changes(
    previous: Frame,
    current: Frame,
    *,
    columns: int = 3,
    rows: int = 2,
    width: int = HASH_WIDTH,
    height: int = HASH_HEIGHT,
) -> list[float]:
    """Per-tile change, reading order.

    Cheap localisation: on a synth, the difference between "they turned one
    knob" and "they switched preset" is whether change is confined to one tile
    or spread across all of them. The decider can be told which, in a handful of
    tokens, without any UI understanding.
    """
    if len(previous.thumb) < width * height or len(current.thumb) < width * height:
        return []

    out: list[float] = []
    for tile_row in range(rows):
        for tile_col in range(columns):
            y0 = tile_row * height // rows
            y1 = (tile_row + 1) * height // rows
            x0 = tile_col * width // columns
            x1 = (tile_col + 1) * width // columns

            differing = 0
            total = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    offset = y * width + x
                    # Compare gradients, not levels, for the same reason dhash
                    # does: brightness shifts are not events.
                    if x + 1 < x1:
                        before = previous.thumb[offset] > previous.thumb[offset + 1]
                        after = current.thumb[offset] > current.thumb[offset + 1]
                        differing += int(before != after)
                        total += 1
            out.append(round(differing / total, 3) if total else 0.0)
    return out


def describe_regions(scores: Sequence[float], threshold: float = 0.15) -> str:
    """Where on screen the movement was, in words the model can use."""
    if not scores:
        return ""
    names = ["top-left", "top-centre", "top-right", "bottom-left", "bottom-centre", "bottom-right"]
    hot = [names[i] for i, score in enumerate(scores) if i < len(names) and score >= threshold]
    if not hot:
        return "no localised movement"
    if len(hot) >= 5:
        return "movement across the whole screen"
    return "movement in " + ", ".join(hot)


# ── The ring ──────────────────────────────────────────────────────────────

@dataclass
class FrameRing:
    """Bounded rolling buffer of recent frames.

    Bounded is the point. Every streaming reference implementation that keeps a
    growing embedding/KV buffer eventually dies of its own context; a ring that
    forgets on purpose is what makes an hour-long session possible at constant
    cost.
    """

    capacity: int = DEFAULT_RING
    frames: deque = field(default_factory=deque)

    def add(self, frame: Frame) -> float:
        """Append and return the visual change against the previous frame."""
        delta = visual_change(self.latest, frame)
        self.frames.append(frame)
        while len(self.frames) > self.capacity:
            self.frames.popleft()
        return delta

    @property
    def latest(self) -> Optional[Frame]:
        return self.frames[-1] if self.frames else None

    def recent(self, count: int = DEFAULT_SEND) -> list[Frame]:
        """The last ``count`` frames, oldest first.

        Oldest first because the model reads a sequence as time moving forward,
        and handing it the reverse quietly inverts every "then what happened"
        judgement.
        """
        return list(self.frames)[-count:]

    def span(self, seconds: float, now: float) -> list[Frame]:
        return [f for f in self.frames if now - seconds <= f.at <= now]

    def novelty(self, lookback: int = 4) -> float:
        """How much has moved recently — the value the decider gates on.

        The MAXIMUM pairwise change across the last ``lookback + 1`` frames, not
        the endpoint-to-endpoint difference. Endpoints were the first
        implementation and they fail in both directions:

        * A sweep that STARTS on this tick still has static frames behind it, so
          an endpoint comparison reads ~0 and the gate suppresses the very
          moment worth looking at. Measured: 159 ``unchanged`` verdicts across a
          nine-minute sound-design session that contained ten separate sweeps,
          with exactly one utterance getting through.
        * A sweep that ENDS mid-ring reads high for several ticks after the
          movement stopped, so the loop keeps paying to be told it is over.

        Taking the max over the window answers "did anything move in the last
        few seconds", which is the actual question, and it inherits
        ``visual_change``'s noise floor so a static window is exactly 0.0.
        """
        window = list(self.frames)[-(lookback + 1):]
        if len(window) < 2:
            return 1.0
        return max(
            visual_change(earlier, later)
            for earlier, later in zip(window, window[1:])
        )


# ── Encoding for the model ────────────────────────────────────────────────

def as_data_urls(frames: Iterable[Frame]) -> list[str]:
    """JPEG frames as data URLs, in the shape both major APIs accept.

    Gemini Live takes ``image/jpeg`` blobs; OpenAI Realtime takes an
    ``input_image`` content part with a base64 data URL. Same bytes either way,
    so the encoding lives here and the transport decides how to wrap it.
    """
    return [
        "data:image/jpeg;base64," + base64.b64encode(f.jpeg).decode("ascii")
        for f in frames
        if f.jpeg
    ]


def frame_block(frames: Sequence[Frame], *, now: float, regions: str = "") -> str:
    """Text that accompanies the frames: when each was taken, and what moved.

    Timestamps are relative and negative ("-3.0s"), because the model is being
    asked about the present and "three seconds ago" is the useful framing.
    """
    if not frames:
        return ""
    offsets = ", ".join(f"{f.at - now:+.1f}s" for f in frames)
    lines = [f"Frames ({len(frames)}, oldest first): {offsets}"]
    if regions:
        lines.append(regions)
    return "\n".join(lines)


# ── The signal contract ───────────────────────────────────────────────────

#: Salience calibration for screen movement, from the measured GUI fixtures:
#: a whole-panel brightness shift (not a user action) reads 0.049, the weakest
#: real sweep step 0.079, the strongest 0.15. So the floor is just above
#: brightness noise and the ceiling is where movement is unmistakable. These
#: numbers live here — in the module that owns the signal — which is the whole
#: point of the contract.
SALIENCE_FLOOR = 0.055
SALIENCE_CEILING = 0.15


def screen_signal(
    ring: "FrameRing",
    *,
    now: float,
    lookback: int = 4,
    send: int = DEFAULT_SEND,
) -> "Signal":
    """This module's opinion about the current moment, as a calibrated Signal.

    The primary source. Whatever the person is doing — sweeping a filter,
    drawing automation, taking a pull — it is on screen, so this works with no
    per-domain capture and no knowledge of what any of those things are.
    """
    from plugins.watch.signals import Signal, calibrate

    raw = ring.novelty(lookback=lookback)
    recent = ring.recent(send)
    regions = (
        describe_regions(region_changes(recent[0], recent[-1]))
        if len(recent) > 1
        else ""
    )

    return Signal(
        name="screen",
        salience=calibrate(raw, SALIENCE_FLOOR, SALIENCE_CEILING),
        block=frame_block(recent, now=now, regions=regions),
        detail={"raw_change": raw, "frames": len(recent)},
    )


# ── ffmpeg argv ───────────────────────────────────────────────────────────

def grab_args(
    input_args: Sequence[str],
    *,
    width: int = 640,
    jpeg_path: str = "-",
    thumb_path: Optional[str] = None,
) -> list[str]:
    """One-shot grab producing a model-resolution JPEG and a hash thumbnail.

    Both outputs come from ONE ffmpeg invocation. Two invocations per tick would
    double the capture cost and, worse, hash a different moment than the one
    sent to the model — so a change could be detected in a frame the model never
    sees, and vice versa.

    ``input_args`` is whatever ``capture.capture_plan`` produced for this host,
    so the frame loop inherits per-OS capture without knowing anything about it.

    ``-frames:v 1`` is emitted BEFORE EACH OUTPUT, not once. ffmpeg's frame
    limit is a per-output option: a single leading copy bounds only the first
    file and the second capture runs until killed. Measured: every grab hit its
    timeout and the live loop completed 13 seconds with zero ticks, which looks
    exactly like a quiet session rather than a broken one.
    """
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    args += list(input_args)

    # Model-resolution JPEG.
    args += [
        "-frames:v", "1",
        "-vf", f"scale={width}:-2:flags=bicubic",
        "-f", "image2", "-c:v", "mjpeg", jpeg_path,
    ]

    if thumb_path is not None:
        # Tiny grayscale, raw. No encoder, no container: the bytes ARE the
        # pixels, which is what makes the pure-Python hash possible.
        args += [
            "-frames:v", "1",
            "-vf", f"scale={HASH_WIDTH}:{HASH_HEIGHT}:flags=bilinear,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", thumb_path,
        ]

    return args


#: Input frame rate for a one-shot grab. Deliberately NOT the loop's 1 fps: the
#: input rate is how long the capture device waits between frames, so grabbing
#: at 1 fps makes every grab block for a second before it can deliver anything.
#: The loop's cadence is the loop's business; the grab should return at once.
GRAB_FRAMERATE = 15.0

