"""The window timeline — what app was in front, and when.

This is the track a native screen recorder cannot give you, and the reason
recording through Hermes is worth doing rather than uploading an .mp4 from Game
Bar. The desktop app already polls the OS window list every 1.5 s while the HUD
is open (``apps/desktop/electron/hud-game-overlay.ts`` on top of
``window-below.ts``); a recording that is running at the same time can keep the
transitions and hand them to the model as TEXT.

Why text rather than trusting the pixels: a title bar is a few hundred
tokens of picture that the model has to OCR and may get wrong, whereas
"04:12 Ableton Live -> Chrome" is about ten tokens it cannot misread. For a
music take the segment boundaries are the DAW window changing; for a game they
are the death screen. Either way the model reasons better over a timeline it
was handed than one it had to infer.

Privacy: window TITLES are the sharpest thing here — they carry document names,
URLs, and message previews. The HUD's own window-below tool treats titles as
metadata-only and never captures pixels; this module goes one step further and
lets titles be dropped entirely (``include_titles=False``) while keeping the app
names, because "which app" is the part the timeline actually needs.

Pure functions over sample lists — no polling, no IPC. The caller owns the
clock, which is also what makes the offsets testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

#: Below this, a transition is someone alt-tabbing through a stack, not a
#: change of activity. Filtering them keeps a 40-line timeline from becoming a
#: 400-line one that buries the real segments.
MIN_DWELL_SECONDS = 2.0

#: Titles can be arbitrarily long (a full file path, a whole tweet). Truncate
#: rather than let one row dominate the block.
MAX_TITLE_CHARS = 60


@dataclass(frozen=True)
class WindowSample:
    """One observation of the frontmost window.

    Attributes:
        at: Seconds since recording started. The caller stamps this from the
            same clock the recording uses, so an offset here lines up with the
            same offset in the video.
        app: Owning application name, as the OS reports it.
        title: Window title, possibly empty (macOS withholds it without the
            Screen Recording permission, and several apps never set one).
    """

    at: float
    app: str
    title: str = ""


@dataclass(frozen=True)
class TimelineSegment:
    """A contiguous stretch where one app was in front."""

    start: float
    end: float
    app: str
    title: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def _clean_title(title: str) -> str:
    flat = " ".join(title.split())
    if len(flat) <= MAX_TITLE_CHARS:
        return flat
    return flat[: MAX_TITLE_CHARS - 1] + "…"


def segments_from_samples(
    samples: Iterable[WindowSample],
    total_duration: float,
    min_dwell: float = MIN_DWELL_SECONDS,
) -> list[TimelineSegment]:
    """Collapse a poll stream into segments.

    Two things happen here, and the order matters. First adjacent samples of
    the same app are merged — the poll fires every 1.5 s and 40 minutes of
    Ableton is one segment, not 1,600 rows. Then segments shorter than
    ``min_dwell`` are dropped and their time is absorbed by the PREVIOUS
    segment, so a flick through a window switcher does not shatter a long
    stretch of work into three pieces.

    Title changes within one app do NOT split a segment: a DAW rewriting its
    title bar on every save would otherwise produce a new row every few
    seconds. The first title seen for the app wins, since that is the one that
    identifies what was being worked on.
    """
    ordered = sorted(samples, key=lambda s: s.at)
    if not ordered:
        return []

    merged: list[TimelineSegment] = []
    for sample in ordered:
        if merged and merged[-1].app == sample.app:
            continue
        if merged:
            previous = merged[-1]
            merged[-1] = TimelineSegment(
                start=previous.start,
                end=sample.at,
                app=previous.app,
                title=previous.title,
            )
        merged.append(
            TimelineSegment(
                start=sample.at,
                end=total_duration,
                app=sample.app,
                title=_clean_title(sample.title),
            )
        )

    if merged:
        last = merged[-1]
        merged[-1] = TimelineSegment(
            start=last.start,
            end=max(total_duration, last.start),
            app=last.app,
            title=last.title,
        )

    kept: list[TimelineSegment] = []
    for segment in merged:
        if segment.duration < min_dwell and kept:
            previous = kept[-1]
            kept[-1] = TimelineSegment(
                start=previous.start,
                end=segment.end,
                app=previous.app,
                title=previous.title,
            )
            continue
        kept.append(segment)

    return kept


def _stamp(seconds: float) -> str:
    """``MM:SS``, or ``H:MM:SS`` past an hour — the format Gemini documents for
    referring to a moment in a video, so the model can tie a timeline row to the
    footage without being taught a new convention."""
    total = int(round(max(0.0, seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_timeline(
    segments: Iterable[TimelineSegment],
    *,
    include_titles: bool = True,
    speed: float = 1.0,
    capture_scale: float = 1.0,
) -> str:
    """Render segments as the text block that rides along with the clip.

    Two independent retimings sit between a wall-clock observation and the
    moment the model sees it, and BOTH have to be applied or every stamp is
    wrong in a way that is invisible on inspection:

    * ``capture_scale`` — the capture backend's own shortfall. A recorder that
      cannot keep up writes fewer frames than real time, so a 21 s session can
      produce a 9 s file (measured, x11grab under load). The window samples are
      stamped in wall-clock seconds, so they must be compressed onto the file's
      shorter timeline by ``video_seconds / wall_seconds``.
    * ``speed`` — the deliberate retiming applied by `prepare`. A 2x clip puts
      a 4:00 event at 2:00.

    Applied in that order, since capture shortfall happens first (at record
    time) and prepare's retiming acts on the file that produced.
    """
    rows = []
    for segment in segments:
        label = segment.app
        if include_titles and segment.title:
            # Truncated HERE, not only where segments are built: rendering is
            # the boundary where row width matters, and a segment can reach it
            # without passing through `segments_from_samples` (a pinned or
            # caller-constructed one). Clamping at the output covers every path
            # into it rather than the one the sample stream happens to take.
            label = f"{segment.app} — {_clean_title(segment.title)}"
        rows.append(f"{_stamp(segment.start * capture_scale / speed)}  {label}")

    if not rows:
        return ""

    return "Window timeline (clip time):\n" + "\n".join(rows)


def summarize(segments: Iterable[TimelineSegment]) -> dict:
    """Aggregate dwell per app, for the CLI to print after a take."""
    totals: dict[str, float] = {}
    ordered = list(segments)
    for segment in ordered:
        totals[segment.app] = totals.get(segment.app, 0.0) + segment.duration

    return {
        "apps": sorted(totals.items(), key=lambda kv: kv[1], reverse=True),
        "segments": len(ordered),
        "switches": max(0, len(ordered) - 1),
    }


def load_samples(rows: Iterable[dict]) -> list[WindowSample]:
    """Parse persisted poll rows, skipping anything malformed.

    The sidecar is written by a long-running recorder that can be killed
    mid-line, so a truncated final row is expected rather than exceptional —
    one bad row must not cost the user the whole timeline for a take they
    cannot re-record.
    """
    out: list[WindowSample] = []
    for row in rows:
        try:
            out.append(
                WindowSample(
                    at=float(row["at"]),
                    app=str(row["app"]),
                    title=str(row.get("title", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def timeline_for(
    rows: Iterable[dict],
    total_duration: float,
    *,
    include_titles: bool = True,
    speed: float = 1.0,
    capture_scale: float = 1.0,
    min_dwell: float = MIN_DWELL_SECONDS,
) -> tuple[str, Optional[dict]]:
    """Persisted rows straight to (rendered block, summary).

    ``total_duration`` is WALL-CLOCK seconds, matching the stamps in ``rows``;
    ``capture_scale`` maps those onto the file's own timeline at render time.
    Segmenting in wall time keeps ``min_dwell`` meaningful — "was this app up
    for two real seconds" is the question, not "for two frames of a file that
    may have been captured at half rate".
    """
    samples = load_samples(rows)
    if not samples:
        return "", None
    segments = segments_from_samples(samples, total_duration, min_dwell=min_dwell)
    return (
        render_timeline(
            segments,
            include_titles=include_titles,
            speed=speed,
            capture_scale=capture_scale,
        ),
        summarize(segments),
    )
