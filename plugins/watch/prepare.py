"""Fit a recording into what a video-capable model will actually accept.

Two budgets constrain a take, they are NOT the same budget, and conflating them
is the whole reason this module exists:

  * BYTES — ``video_analyze`` base64s the file inline and hard-caps the payload
    at 50 MiB, i.e. ~37 MiB of file. Resolution, CRF, output fps and
    ``mpdecimate`` all move this number.
  * TOKENS — Gemini-family models sample the *container timeline* at 1 fps and
    bill 258 tok/frame (66 with ``media_resolution: low``) plus 32 tok/s of
    audio. That is per second of DURATION, so shrinking the picture buys
    nothing here. Only trimming or speeding the clip up moves this number.

The practical consequence, stated once because it is counter-intuitive:
dropping duplicate frames makes the FILE smaller and costs exactly the same to
analyze. Speeding a clip up 2x halves the token bill and halves the effective
sample rate (one frame per two seconds of real time). Slowing it down doubles
both. `effective_fps_for` and `estimate_tokens` are the pair that makes that
trade legible before anyone spends money on it.

Everything here is a pure function over a spec — no ffmpeg process, no
filesystem — so the ladder can be tested without a capture device or a host OS
to fake.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional

# ── Provider-side constants ────────────────────────────────────────────────
# Mirrors tools/vision_tools.py::_MAX_VIDEO_BASE64_BYTES. Base64 inflates by
# 4/3, so the file ceiling is the payload ceiling scaled down, minus room for
# the data-URL prefix and the JSON envelope around it.
MAX_BASE64_BYTES = 50 * 1024 * 1024
BASE64_OVERHEAD = 4 / 3
_ENVELOPE_SLACK_BYTES = 256 * 1024

#: Largest file we will hand to ``video_analyze`` (~37 MiB).
MAX_FILE_BYTES = int((MAX_BASE64_BYTES - _ENVELOPE_SLACK_BYTES) / BASE64_OVERHEAD)

#: Gemini's documented tokenizer rates. Frames are billed per second of
#: timeline at the model's own 1 fps sampling, NOT per frame in our container.
TOKENS_PER_FRAME_DEFAULT = 258
TOKENS_PER_FRAME_LOW = 66
TOKENS_PER_AUDIO_SECOND = 32
PROVIDER_SAMPLE_FPS = 1.0

#: atempo only accepts 0.5–100 per filter instance, and stacking more than a
#: couple of them audibly smears the result. Past this we refuse rather than
#: silently mangle a musician's take.
_MAX_SPEED = 8.0
_MIN_SPEED = 0.25


@dataclass(frozen=True)
class PrepareSpec:
    """How to transcode one take before analysis.

    Attributes:
        width: Output width in pixels; height follows the source aspect
            (``-2`` keeps it even for h264). ``None`` keeps the source size.
        fps: Output frame rate. ``None`` keeps the source rate and lets
            ``mpdecimate`` do the thinning instead.
        crf: x264 quality (lower is better/bigger). 23 is ffmpeg's default;
            the ladder walks up from there.
        speed: Playback multiplier. ``2.0`` = twice as fast = half the
            duration = half the tokens = one sampled frame per 2 s of real
            time. ``0.5`` = half speed = double tokens, 2 effective fps.
        audio: Keep the audio track. For music this is the channel that
            matters and it is 8x cheaper per second than the picture.
        decimate: Drop visually-duplicate frames (``mpdecimate``). Only
            meaningful when ``fps`` is None — an fps filter resamples the
            timeline and reintroduces the duplicates this would have removed.
        media_resolution: ``"default"`` or ``"low"``; only affects the token
            ESTIMATE, since the flag itself is set by the caller at request
            time, not by the transcode.
    """

    width: Optional[int] = 640
    fps: Optional[float] = 1.0
    crf: int = 28
    speed: float = 1.0
    audio: bool = True
    decimate: bool = False
    media_resolution: str = "default"

    def validate(self) -> None:
        if self.width is not None and self.width < 64:
            raise ValueError(f"width must be >= 64 px (got {self.width})")
        if self.fps is not None and not 0 < self.fps <= 60:
            raise ValueError(f"fps must be in (0, 60] (got {self.fps})")
        if not 0 <= self.crf <= 51:
            raise ValueError(f"crf must be in [0, 51] (got {self.crf})")
        if not _MIN_SPEED <= self.speed <= _MAX_SPEED:
            raise ValueError(
                f"speed must be in [{_MIN_SPEED}, {_MAX_SPEED}] (got {self.speed})"
            )
        if self.media_resolution not in {"default", "low"}:
            raise ValueError(
                f"media_resolution must be 'default' or 'low' (got {self.media_resolution!r})"
            )


# ── Cost / capability math ────────────────────────────────────────────────

def effective_fps_for(spec: PrepareSpec) -> float:
    """Frames of REAL TIME the model gets to see, per second.

    The model samples our container at 1 fps whatever we put in it, so the only
    way to show it more of a fast passage is to stretch that passage across
    more container seconds. Half speed means two of its samples land inside one
    real second: 2 effective fps.
    """
    return PROVIDER_SAMPLE_FPS / spec.speed


def estimate_tokens(source_duration_s: float, spec: PrepareSpec) -> int:
    """Input tokens a prepared take will cost, on Gemini-family rates.

    ``source_duration_s`` is the length of the RAW take; the spec's speed is
    what turns it into billed timeline seconds.
    """
    if source_duration_s <= 0:
        return 0

    billed_seconds = source_duration_s / spec.speed
    per_frame = (
        TOKENS_PER_FRAME_LOW
        if spec.media_resolution == "low"
        else TOKENS_PER_FRAME_DEFAULT
    )
    sampled_frames = billed_seconds * PROVIDER_SAMPLE_FPS
    tokens = sampled_frames * per_frame
    if spec.audio:
        tokens += billed_seconds * TOKENS_PER_AUDIO_SECOND
    return int(round(tokens))


# ── The filter chain ──────────────────────────────────────────────────────

def _atempo_chain(speed: float) -> list[str]:
    """``atempo`` stages that multiply out to *speed*.

    One instance is capped at 2x in practice before artefacts get obvious, so a
    larger factor is split into equal stages rather than pushed through a
    single filter.
    """
    if abs(speed - 1.0) < 1e-9:
        return []

    stages: list[float] = []
    remaining = speed
    while remaining > 2.0 + 1e-9:
        stages.append(2.0)
        remaining /= 2.0
    while remaining < 0.5 - 1e-9:
        stages.append(0.5)
        remaining /= 0.5
    stages.append(remaining)
    return [f"atempo={stage:g}" for stage in stages]


def video_filters(spec: PrepareSpec) -> list[str]:
    """Video filter stages, in application order.

    Order is load-bearing and was verified against real ffmpeg output:

    1. ``setpts`` FIRST. Retiming has to happen before any frame-rate
       resampling, because ``fps`` fixes the stream's output cadence and a
       later ``setpts`` then squeezes frames into slots the encoder has already
       committed to — it drops them, and the resulting duration is neither the
       source's nor ``source/speed``. Measured: a 20 s take at ``fps=1`` then
       ``setpts=PTS/2`` came out 12 s with 12 frames instead of 10 s with 10.
       Since ``render_timeline`` divides its stamps by ``speed`` and
       ``estimate_tokens`` bills ``duration/speed``, that mismatch silently
       misdates every event the model is asked about.
    2. ``fps`` second, resampling the ALREADY-RETIMED timeline, so the output
       cadence is exactly what was asked for on the clip the model will see.
    3. ``scale`` last, so the scaler only touches surviving frames.
    """
    stages: list[str] = []

    if abs(spec.speed - 1.0) > 1e-9:
        stages.append(f"setpts=PTS/{spec.speed:g}")

    # mpdecimate and fps are mutually exclusive on purpose — see the module
    # docstring. Honouring both would resample the timeline right back to a
    # fixed cadence and undo the decimation.
    if spec.fps is not None:
        stages.append(f"fps={spec.fps:g}")
    elif spec.decimate:
        stages.append("mpdecimate")

    if spec.width is not None:
        stages.append(f"scale={spec.width}:-2:flags=bicubic")

    return stages


def audio_filters(spec: PrepareSpec) -> list[str]:
    return _atempo_chain(spec.speed) if spec.audio else []


def prepare_args(source: str, dest: str, spec: PrepareSpec) -> list[str]:
    """Full ffmpeg argv for one transcode pass.

    ``-y`` is deliberate: prepare is idempotent and re-running it after a
    ladder step must overwrite its own previous attempt rather than block on a
    prompt no one is there to answer.
    """
    spec.validate()

    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", source]

    vf = video_filters(spec)
    if vf:
        args += ["-vf", ",".join(vf)]

    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(spec.crf), "-pix_fmt", "yuv420p"]

    if spec.audio:
        af = audio_filters(spec)
        if af:
            args += ["-af", ",".join(af)]
        # Mono at 64k: the provider downmixes to single-channel 1 Kbps anyway,
        # so stereo is bytes spent on something that is discarded upstream.
        args += ["-c:a", "aac", "-b:a", "64k", "-ac", "1"]
    else:
        args += ["-an"]

    # +faststart so a partially-read file still parses, and a clip that gets
    # uploaded rather than analyzed locally starts playing without the whole
    # moov atom.
    args += ["-movflags", "+faststart", dest]
    return args


# ── The fit ladder ────────────────────────────────────────────────────────

#: Ordered rungs, cheapest-looking first. Each rung only says what it CHANGES;
#: the caller's spec supplies everything else, so a user who asked for
#: audio-only or half speed keeps that through the whole descent.
#:
#: Quality is spent before information: CRF and resolution come first because a
#: blurrier frame still shows the model what happened, whereas dropping the
#: frame rate or the audio removes something it could have reasoned about.
FIT_LADDER: tuple[dict, ...] = (
    {},
    {"crf": 32},
    {"width": 480, "crf": 32},
    {"width": 480, "crf": 36},
    {"width": 384, "crf": 38},
    {"width": 320, "crf": 40, "fps": 0.5},
)


def ladder_specs(base: PrepareSpec, ladder: Iterable[dict] = FIT_LADDER) -> list[PrepareSpec]:
    """The descent to try, as concrete specs.

    A rung that would UPGRADE the caller's request is clamped, not applied: if
    someone explicitly asked for 320 px, rung 3's ``width: 480`` must not widen
    it back out and re-inflate the file we are trying to shrink.
    """
    out: list[PrepareSpec] = []
    for rung in ladder:
        changes = dict(rung)
        if "width" in changes and base.width is not None:
            changes["width"] = min(changes["width"], base.width)
        if "crf" in changes:
            changes["crf"] = max(changes["crf"], base.crf)
        if "fps" in changes and base.fps is not None:
            changes["fps"] = min(changes["fps"], base.fps)
        candidate = replace(base, **changes)
        if candidate not in out:
            out.append(candidate)
    return out


def fits(size_bytes: int) -> bool:
    return 0 < size_bytes <= MAX_FILE_BYTES


def suggest_speedup(size_bytes: int, duration_s: float) -> Optional[float]:
    """Speed factor that would bring an over-cap take under it, if any.

    The last resort after the ladder bottoms out, and the honest one: at this
    point the clip is simply too long for an inline payload, so the fix is to
    show the model less of the timeline rather than a worse picture of all of
    it. ``None`` means even 8x wouldn't do it — trim or segment instead.
    """
    if fits(size_bytes) or duration_s <= 0 or size_bytes <= 0:
        return None
    needed = size_bytes / MAX_FILE_BYTES
    # Round up to a tidy quarter-step so the number is repeatable rather than
    # a float nobody can retype.
    factor = min(_MAX_SPEED, (int(needed * 4) + 1) / 4)
    return factor if factor <= _MAX_SPEED and needed <= _MAX_SPEED else None


def describe_cost(source_duration_s: float, spec: PrepareSpec) -> dict:
    """One-line explanation of what a take will cost and show, for the CLI."""
    return {
        "billed_seconds": round(source_duration_s / spec.speed, 2),
        "effective_fps": round(effective_fps_for(spec), 3),
        "estimated_tokens": estimate_tokens(source_duration_s, spec),
        "media_resolution": spec.media_resolution,
        "with_audio": spec.audio,
    }
