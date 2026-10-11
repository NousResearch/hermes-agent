"""Review — take a recorded file and get an opinion on it.

The pipeline is: probe the take, pick a prepare spec, walk the fit ladder until
the file is under the inline-payload cap, attach the window timeline as text,
and hand the whole thing to the built-in ``video_analyze`` tool.

The one design decision worth defending: this does NOT implement its own model
call. ``video_analyze`` already resolves the provider through
``agent/auxiliary_client.py`` (per-task overrides, payment fallback, vision
routing), base64s the file, and sniffs the mime type. Re-implementing that here
would fork the provider ladder and drift from it. What this module adds is the
part ``video_analyze`` has no business knowing: how to make a screen recording
small enough and what other tracks belong in the prompt.

Provider caveat, surfaced rather than hidden: video input is not universal.
Gemini-family and several open VL models accept a video part; Claude does not
accept video at all. Sending an mp4 to a model that cannot read one produces a
confusing refusal, so ``model`` is a parameter and the CLI names a video-capable
default instead of inheriting whatever the main chat happens to be using.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Optional

from plugins.watch import prepare as prep
from plugins.watch import recorder as rec
from plugins.watch import timeline as tl

logger = logging.getLogger(__name__)

#: Video-capable default. Named explicitly because the main conversation's
#: model is frequently NOT video-capable, and inheriting it is the single most
#: likely way for a review to fail in a way the user cannot diagnose.
DEFAULT_MODEL = "google/gemini-3-flash-preview"

_PROBE_TIMEOUT = 30
_PREPARE_TIMEOUT = 1800


class ReviewError(RuntimeError):
    """A failure worth showing the user verbatim."""


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def probe(path: Path) -> dict:
    """Duration, size, and stream shape for a take.

    Duration drives the token estimate and the speed suggestion, so a take we
    cannot probe is one we cannot cost — better to say so than to guess and
    surprise someone with a bill.
    """
    if not path.is_file():
        raise ReviewError(f"No such file: {path}")
    if not ffprobe_available():
        raise ReviewError("ffprobe not found on PATH (install ffmpeg).")

    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration,size",
                "-show_entries", "stream=codec_type,width,height,r_frame_rate",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ReviewError(f"ffprobe failed: {(exc.stderr or '').strip()[-300:]}") from exc
    except subprocess.SubprocessError as exc:
        raise ReviewError(f"ffprobe failed: {exc}") from exc

    try:
        data = json.loads(out.stdout)
    except ValueError as exc:
        raise ReviewError("ffprobe returned unparseable output") from exc

    fmt = data.get("format", {})
    streams = data.get("stream", data.get("streams", []) or [])
    kinds = {s.get("codec_type") for s in streams}

    try:
        duration = float(fmt.get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0

    return {
        "duration_seconds": duration,
        "size_bytes": path.stat().st_size,
        "has_audio": "audio" in kinds,
        "has_video": "video" in kinds,
    }


def prepared_path(source: Path) -> Path:
    """Where the transcode lands — beside the take, clearly derived from it."""
    return rec.sidecar(source, "prepared.mp4")


def run_prepare(source: Path, dest: Path, spec: prep.PrepareSpec) -> int:
    """Transcode once; return the output size in bytes."""
    args = prep.prepare_args(str(source), str(dest), spec)
    try:
        subprocess.run(args, capture_output=True, text=True, timeout=_PREPARE_TIMEOUT, check=True)
    except subprocess.CalledProcessError as exc:
        raise ReviewError(f"ffmpeg prepare failed: {(exc.stderr or '').strip()[-400:]}") from exc
    except subprocess.SubprocessError as exc:
        raise ReviewError(f"ffmpeg prepare failed: {exc}") from exc

    if not dest.is_file():
        raise ReviewError("ffmpeg reported success but wrote no file")
    return dest.stat().st_size


def fit_take(
    source: Path,
    base: prep.PrepareSpec,
    *,
    duration_s: float,
    on_step=None,
) -> tuple[Path, prep.PrepareSpec, int]:
    """Transcode down the ladder until the take fits the inline cap.

    Returns the prepared path, the spec that produced it, and its size. Raises
    when even the bottom rung is too big — with the speedup that WOULD fit, so
    the message is actionable rather than just a refusal.
    """
    dest = prepared_path(source)
    last_size = 0
    last_spec = base

    for spec in prep.ladder_specs(base):
        size = run_prepare(source, dest, spec)
        last_size, last_spec = size, spec
        if on_step is not None:
            on_step(spec, size)
        if prep.fits(size):
            return dest, spec, size

    suggestion = prep.suggest_speedup(last_size, duration_s)
    cap_mb = prep.MAX_FILE_BYTES / 1048576
    if suggestion is not None:
        raise ReviewError(
            f"Take is still {last_size / 1048576:.1f} MB after the full quality "
            f"ladder (cap {cap_mb:.1f} MB). "
            f"Retry with --speed {suggestion:g} to fit, or trim it with "
            f"--start/--duration."
        )
    raise ReviewError(
        f"Take is {last_size / 1048576:.1f} MB and too long to send inline "
        f"(cap {cap_mb:.1f} MB). Trim it with --start/--duration and review the "
        f"segments separately."
    )


def build_prompt(
    question: str,
    *,
    timeline_block: str = "",
    duration_s: float = 0.0,
    spec: Optional[prep.PrepareSpec] = None,
) -> str:
    """Assemble the text half of the request.

    The timeline and the retiming note are both here rather than left implicit,
    because a model told "this clip is 2x real time" reads its own timestamps
    correctly, and one that is not will confidently misreport when things
    happened.
    """
    parts = [question.strip() or "What do you think of this? Be specific and concrete."]

    if spec is not None and abs(spec.speed - 1.0) > 1e-9:
        real = duration_s
        clip = duration_s / spec.speed if spec.speed else duration_s
        parts.append(
            f"Note: this clip is retimed {spec.speed:g}x — {real:.0f}s of real "
            f"time compressed into {clip:.0f}s of footage. Timestamps below and "
            f"in the video are CLIP time."
        )

    if timeline_block:
        parts.append(timeline_block)

    return "\n\n".join(parts)


def timeline_block_for(
    timeline_path: Optional[Path],
    duration_s: float,
    *,
    include_titles: bool = True,
    speed: float = 1.0,
    capture_scale: float = 1.0,
) -> tuple[str, Optional[dict]]:
    """Read the JSONL sidecar and render it, or return empty on any problem.

    A missing or corrupt timeline degrades the review; it never fails it. The
    video is the primary artifact and a take is still worth an opinion without
    the second track.

    ``duration_s`` here is WALL-CLOCK, because that is the clock the sidecar's
    stamps are on; ``capture_scale`` maps the result onto the file's timeline.
    """
    if timeline_path is None or not Path(timeline_path).is_file():
        return "", None

    rows = []
    try:
        with Path(timeline_path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError as exc:
        logger.debug("watch: could not read timeline (%s)", exc)
        return "", None

    return tl.timeline_for(
        rows,
        duration_s,
        include_titles=include_titles,
        speed=speed,
        capture_scale=capture_scale,
    )


def capture_scale_for(meta: Optional[dict], video_seconds: float) -> float:
    """How much to compress wall-clock stamps to land on the file's timeline.

    ``video_seconds / wall_seconds``, from the meta sidecar written at stop
    time. 1.0 whenever there is nothing to correct — no meta (an imported
    file), a recorder that kept up, or numbers that don't make sense. Clamped to
    (0, 1]: a file LONGER than its own recording session is not a thing, and
    trusting such a number would stretch the timeline past the end of the clip.
    """
    if not meta:
        return 1.0
    try:
        wall = float(meta.get("wall_seconds") or 0.0)
        recorded = float(meta.get("video_seconds") or video_seconds or 0.0)
    except (TypeError, ValueError):
        return 1.0
    if wall <= 0 or recorded <= 0:
        return 1.0
    return min(1.0, recorded / wall)


async def analyze(
    video_path: Path,
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Hand the prepared clip to the built-in video tool.

    Imported lazily so the plugin's pure modules stay importable (and testable)
    without pulling the whole vision/auxiliary-client tree.
    """
    from tools.vision_tools import video_analyze_tool

    raw = await video_analyze_tool(
        video_url=str(video_path),
        user_prompt=prompt,
        model=model,
    )
    try:
        return json.loads(raw)
    except ValueError:
        return {"success": False, "error": "video_analyze returned unparseable output"}


def plan_review(
    source: Path,
    *,
    spec: prep.PrepareSpec,
    timeline_path: Optional[Path] = None,
    include_titles: bool = True,
    meta: Optional[dict] = None,
) -> dict:
    """Everything decidable BEFORE spending money, as one dict.

    Split out from the doing so the CLI can show a cost estimate and let the
    user back out — the whole point of surfacing tokens per take is that
    somebody gets to say no.

    Two clocks are in play and they are used for different things: COST comes
    off the file's own duration (that is what the provider bills), while the
    window timeline is stamped in wall-clock and gets mapped onto the file via
    ``capture_scale``. Using one number for both is how the timeline ends up
    pointing at the wrong moments.
    """
    info = probe(source)
    duration = info["duration_seconds"]
    effective = replace(spec, audio=spec.audio and info["has_audio"])
    scale = capture_scale_for(meta, duration)
    wall_duration = duration / scale if scale > 0 else duration

    block, summary = timeline_block_for(
        timeline_path,
        wall_duration,
        include_titles=include_titles,
        speed=effective.speed,
        capture_scale=scale,
    )

    return {
        "probe": info,
        "spec": effective,
        "cost": prep.describe_cost(duration, effective),
        "capture_scale": scale,
        "wall_seconds": wall_duration,
        "timeline_block": block,
        "timeline_summary": summary,
    }
