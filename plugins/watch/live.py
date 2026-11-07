"""The live loop — grab, gate, ask, deliver, log. On a clock, for real.

This is the driver. Everything it needs already exists: ``capture`` knows how to
address this host's screen, ``frames`` turns that into a calibrated screen
signal, ``inputs``/``music`` add resolution the frames cannot have, ``decider``
decides, and ``auxiliary_client`` talks to the model. This module owns the clock
and the plumbing between them, and nothing else.

Three properties it must have, each learned from measurement rather than taste:

* **Self-clocking, drop-to-latest.** The next grab is scheduled only after the
  current tick finishes. A queue would let latency grow without bound and start
  delivering commentary about something two minutes old — the same reason
  Google's own Live API console self-schedules at 0.5 fps rather than buffering.
* **Every tick logged.** The decision log is the artifact that makes a brief
  tunable: record once, replay against different policies forever, spending
  nothing.
* **Bounded everything.** The frame ring forgets, the utterance history is
  capped, the event tracks are trimmed to the window. An hour-long session costs
  what a one-minute session costs per tick.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from plugins.watch import capture as cap
from plugins.watch import frames as fr
from plugins.watch import inputs as inp
from plugins.watch import music as mus
from plugins.watch import decider as dec
from plugins.watch.recorder import watch_dir
from plugins.watch.signals import Signal

logger = logging.getLogger(__name__)

#: Seconds between grabs. One is the natural rate: providers sample video at
#: 1 fps anyway, so a faster clock buys nothing the model can see, and a slower
#: one starts missing gestures.
DEFAULT_INTERVAL = 1.0

#: A grab that takes longer than this is abandoned. Screen capture stalling for
#: two seconds means the machine is busy — usually the exact moment the user is
#: doing something interesting — and a blocked grab must not stall the clock.
GRAB_TIMEOUT = 2.0

#: Model-resolution frame width. 640 is plenty for judging a GUI at 66 tokens
#: per frame; the thumbnail that drives the gate is separate and tiny.
FRAME_WIDTH = 640

#: Consecutive failed grabs before concluding capture is broken rather than the
#: screen being quiet. Those two states are indistinguishable from the outside —
#: both produce zero ticks — so a loop that never grabs anything must say why
#: instead of running to completion looking healthy.
MAX_CONSECUTIVE_FAILURES = 5


@dataclass
class LiveSession:
    """Everything one live run needs to keep between ticks."""

    brief: str
    ring: fr.FrameRing = field(default_factory=fr.FrameRing)
    state: dec.DeciderState = field(default_factory=dec.DeciderState)
    policy: dec.Policy = field(default_factory=dec.Policy)
    #: Optional enrichment tracks. Empty unless a provider is supplying them.
    key_events: list = field(default_factory=list)
    onsets: list = field(default_factory=list)
    started_at: float = 0.0
    ticks: int = 0
    log_path: Optional[Path] = None

    def elapsed(self, now: Optional[float] = None) -> float:
        return (now if now is not None else time.time()) - self.started_at


# ── Grabbing ──────────────────────────────────────────────────────────────

def grab_once(
    input_args: Sequence[str],
    *,
    at: float,
    width: int = FRAME_WIDTH,
    timeout: float = GRAB_TIMEOUT,
) -> Optional[fr.Frame]:
    """One frame plus its hash thumbnail, or None if the grab failed.

    Both outputs come from a single ffmpeg pass so the bytes that are hashed and
    the bytes that reach the model are the same instant. Failure returns None
    rather than raising: a dropped frame is a missing tick, not a dead session,
    and screen capture fails transiently for reasons (a fullscreen transition, a
    display sleep) that resolve on their own.
    """
    with tempfile.TemporaryDirectory(prefix="hermes-watch-") as scratch:
        jpeg_path = Path(scratch) / "frame.jpg"
        thumb_path = Path(scratch) / "frame.gray"
        args = fr.grab_args(
            input_args,
            width=width,
            jpeg_path=str(jpeg_path),
            thumb_path=str(thumb_path),
        )
        try:
            subprocess.run(
                args, capture_output=True, timeout=timeout, check=True
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("watch live: grab failed (%s)", exc)
            return None

        if not jpeg_path.is_file() or not thumb_path.is_file():
            return None

        return fr.Frame(
            at=at,
            jpeg=jpeg_path.read_bytes(),
            thumb=thumb_path.read_bytes(),
        )


# ── The model call ────────────────────────────────────────────────────────

def make_asker(
    session: LiveSession,
    *,
    model: Optional[str] = None,
    send: int = fr.DEFAULT_SEND,
) -> Callable[[str, str], str]:
    """Build the ``ask`` the decider calls when a tick clears every free gate.

    A direct ``auxiliary_client`` call, NOT an agent turn. An agent turn would
    pay the full system prompt and tool schemas on every decision — the whole
    point of the free gates is that the paid path stays small.

    The conversation is one message: frames plus text. There is no accumulating
    history beyond the last few utterances the prompt carries explicitly, so a
    three-hour session costs the same per call as the first one.
    """
    from agent.auxiliary_client import call_llm

    resolved = model or _default_model()

    def ask(system_prompt: str, user_prompt: str) -> str:
        content: list[dict] = [{"type": "text", "text": user_prompt}]
        for url in fr.as_data_urls(session.ring.recent(send)):
            content.append({"type": "image_url", "image_url": {"url": url}})

        try:
            response = call_llm(
                task="watch",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                model=resolved,
                temperature=0.2,
                # A decision is one sentence or the literal NO REPLY. Capping
                # this is a latency control as much as a cost one: the loop is
                # on a one-second clock and cannot wait on a paragraph.
                max_tokens=60,
                timeout=20.0,
            )
        except Exception as exc:  # pragma: no cover — network path
            logger.debug("watch live: model call failed (%s)", exc)
            return dec.NO_REPLY

        return _first_text(response)

    return ask


def _default_model() -> str:
    """A video-capable default, overridable by config.

    Pinned rather than inherited: the conversation's model is frequently NOT
    vision-capable, and a live loop that silently returns NO REPLY forever
    because the model cannot see images is indistinguishable from a quiet
    session.
    """
    try:
        from hermes_cli.config import cfg_get, load_config

        configured = cfg_get(load_config(), "auxiliary", "watch", "model", default=None)
        if configured:
            return str(configured)
    except Exception:
        pass
    return "google/gemini-3-flash-preview"


def _first_text(response) -> str:
    """Pull the text out of whatever shape the client returned."""
    try:
        from agent.auxiliary_client import extract_content_or_reasoning

        return (extract_content_or_reasoning(response) or "").strip()
    except Exception:
        pass
    try:
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return ""


# ── Signals for one tick ──────────────────────────────────────────────────

def tick_signals(session: LiveSession, *, now: float) -> list[Signal]:
    """Every source's opinion about this moment.

    The screen is always present — it is the mechanism. The other tracks appear
    only when something is feeding them, and each one is enrichment: higher
    resolution than a 1 fps frame can carry, plus an opinion about whether now
    is a safe moment to speak.
    """
    signals = [fr.screen_signal(session.ring, now=now)]

    if session.key_events:
        signals.append(
            inp.keys_signal(session.key_events, now=now, duration_s=session.ticks)
        )

    if session.onsets:
        signals.append(mus.notes_signal(session.onsets, now=now))

    return signals


def trim_tracks(session: LiveSession, *, now: float, keep: float = 120.0) -> None:
    """Drop track events older than the widest window any source looks at.

    Without this the enrichment tracks grow for the life of the session — the
    exact unbounded-context bug the frame ring exists to avoid, arriving through
    a side door.
    """
    cutoff = now - keep
    if session.key_events:
        session.key_events[:] = [e for e in session.key_events if e.at >= cutoff]
    if session.onsets:
        session.onsets[:] = [o for o in session.onsets if o.at >= cutoff]


# ── The loop ──────────────────────────────────────────────────────────────

def run_tick(
    session: LiveSession,
    input_args: Sequence[str],
    *,
    ask: Optional[Callable[[str, str], str]],
    now: Optional[float] = None,
    on_speak: Optional[Callable[[dec.Decision], None]] = None,
) -> Optional[dec.Decision]:
    """One iteration: grab, build signals, decide, deliver.

    Returns the Decision, or None when the grab failed (a missing tick).
    """
    moment = now if now is not None else session.elapsed()

    frame = grab_once(input_args, at=moment)
    if frame is None:
        return None

    session.ring.add(frame)
    session.ticks += 1
    trim_tracks(session, now=moment)

    decision = dec.decide(
        session.state,
        tick_signals(session, now=moment),
        at=moment,
        brief=session.brief,
        ask=ask,
        policy=session.policy,
    )

    if decision.spoke and on_speak is not None:
        on_speak(decision)

    return decision


def run_live(
    *,
    brief: str,
    duration: Optional[float] = None,
    interval: float = DEFAULT_INTERVAL,
    model: Optional[str] = None,
    policy: Optional[dec.Policy] = None,
    platform: Optional[str] = None,
    audio_device: Optional[str] = None,
    screen_index: int = 1,
    on_speak: Optional[Callable[[dec.Decision], None]] = None,
    stop: Optional[threading.Event] = None,
    ask: Optional[Callable[[str, str], str]] = None,
    key_provider: Optional[Callable[[], Sequence]] = None,
    note_provider: Optional[Callable[[], Sequence]] = None,
) -> dict:
    """Watch until stopped. Returns a summary plus the path to the decision log.

    ``ask`` is injectable so the whole loop can be exercised without a network;
    left unset it is a real ``auxiliary_client`` call. Same for the enrichment
    providers, which return whatever key/note events have accumulated since the
    last tick.
    """
    host = platform or sys.platform
    # The capture plan's framerate is the DEVICE's rate, not the loop's. A
    # one-shot grab at 1 fps blocks for a full second waiting for its frame;
    # the loop's cadence is enforced by the loop, not by starving the device.
    plan = cap.capture_plan(
        host,
        fps=fr.GRAB_FRAMERATE,
        audio_device=audio_device,
        screen_index=screen_index,
        display=os.environ.get("DISPLAY"),
        wayland_display=os.environ.get("WAYLAND_DISPLAY"),
    )
    if plan.blocked:
        return {"success": False, "error": plan.blocked}

    if shutil.which("ffmpeg") is None:
        return {"success": False, "error": "ffmpeg not found on PATH."}

    # Video only: the loop grabs single frames, and an audio input on a
    # one-frame grab makes ffmpeg wait for an audio packet that never comes.
    input_args = [a for a in plan.args]
    session = LiveSession(brief=brief, policy=policy or dec.Policy())
    session.started_at = time.time()
    session.log_path = _new_log_path(brief)

    asker = ask or make_asker(session, model=model)
    stopper = stop or threading.Event()
    spoken: list[dict] = []
    failures = 0

    def deliver(decision: dec.Decision) -> None:
        entry = {"at": round(decision.at, 1), "text": decision.text}
        spoken.append(entry)
        if on_speak is not None:
            on_speak(decision)

    while not stopper.is_set():
        started = time.time()
        moment = session.elapsed(started)

        if duration is not None and moment > duration:
            break

        if key_provider is not None:
            session.key_events.extend(key_provider())
        if note_provider is not None:
            session.onsets.extend(note_provider())

        if run_tick(session, input_args, ask=asker, now=moment, on_speak=deliver) is None:
            failures += 1
            # A capture that fails EVERY time is a broken configuration, not a
            # quiet screen, and the two are indistinguishable from the outside:
            # a loop that grabs nothing reports zero ticks and looks exactly
            # like a session where nothing happened. Bail with the reason
            # instead of running silently to the end.
            if failures >= MAX_CONSECUTIVE_FAILURES and session.ticks == 0:
                return {
                    "success": False,
                    "error": (
                        f"Screen capture failed {failures} times in a row and "
                        f"never produced a frame. Check that ffmpeg can grab "
                        f"this display: ffmpeg -f {input_args[1] if len(input_args) > 1 else '?'} "
                        f"-i {input_args[-1]} -frames:v 1 /tmp/test.jpg"
                    ),
                }
        else:
            failures = 0

        # Self-clocking: the next grab is scheduled from the END of this tick,
        # so a slow model call or a stalled capture costs a frame rather than
        # accumulating a backlog of stale moments.
        stopper.wait(max(0.0, interval - (time.time() - started)))

    write_log(session)
    stats = dec.replay_stats(session.state.log)
    return {
        "success": True,
        "brief": brief,
        "seconds": round(session.elapsed(), 1),
        "spoken": spoken,
        "log_path": str(session.log_path) if session.log_path else None,
        **stats,
    }


# ── The decision log ──────────────────────────────────────────────────────

def _new_log_path(brief: str) -> Path:
    directory = watch_dir() / "live"
    directory.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "-" for c in brief.lower())[:40].strip("-")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return directory / f"{stamp}{'-' + slug if slug else ''}.jsonl"


def write_log(session: LiveSession) -> Optional[Path]:
    """Persist the tick-by-tick decisions.

    JSONL with a header line carrying the brief and policy, so a replay knows
    what settings produced the trace it is comparing against.
    """
    if session.log_path is None:
        return None

    try:
        with session.log_path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "brief": session.brief,
                        "ticks": session.ticks,
                        "policy": {
                            "refractory": session.policy.refractory,
                            "call_cooldown": session.policy.call_cooldown,
                            "hold_timeout": session.policy.hold_timeout,
                            "min_salience": session.policy.min_salience,
                        },
                    }
                )
                + "\n"
            )
            for row in session.state.log:
                handle.write(json.dumps(row) + "\n")
    except OSError as exc:
        logger.debug("watch live: could not write log (%s)", exc)
        return None

    return session.log_path


def read_log(path: Path) -> tuple[dict, list[dict]]:
    """Load a decision log as ``(header, rows)``."""
    header: dict = {}
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if index == 0 and "brief" in parsed:
                header = parsed
            else:
                rows.append(parsed)
    return header, rows


def replay(path: Path, policy: Optional[dec.Policy] = None) -> dict:
    """Re-run a recorded session's SIGNALS against a different policy.

    The payoff for logging every tick: tuning costs nothing after the first
    recording. The salience each source reported is in the log, so the free
    gates can be re-evaluated exactly. Model calls are not re-issued — a tick
    that would newly clear the gates is counted as a call rather than made,
    which is what makes a sweep across a dozen policies free.
    """
    header, rows = read_log(path)
    rules = policy or dec.Policy()

    state = dec.DeciderState()
    would_call = 0
    for row in rows:
        salience = max(row.get("signals", {}).values(), default=0.0)
        at = float(row.get("at", 0.0))

        if salience < rules.min_salience:
            continue
        if state.last_spoke_at is not None and at - state.last_spoke_at < rules.refractory:
            continue
        if state.last_call_at is not None and at - state.last_call_at < rules.call_cooldown:
            continue

        would_call += 1
        state.note_call(at)
        # The recorded answer is what the model said at this moment; reusing it
        # is the closest honest approximation available offline.
        text = row.get("text") or ""
        if text and not dec.is_repetition(text, state.history, threshold=rules.similarity):
            state.remember(text, at, limit=rules.history)

    return {
        "brief": header.get("brief", ""),
        "ticks": len(rows),
        "model_calls": would_call,
        "call_rate": round(would_call / len(rows), 3) if rows else 0.0,
        "spoke": len(state.history),
        "policy": {
            "refractory": rules.refractory,
            "call_cooldown": rules.call_cooldown,
            "min_salience": rules.min_salience,
        },
    }
