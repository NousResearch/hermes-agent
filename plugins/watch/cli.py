"""CLI for the watch plugin — ``hermes watch <subcommand>``.

  start   — begin recording the screen (+ audio, + window timeline)
  stop    — finalize the take
  status  — is something recording, and how big is it
  list    — recorded takes
  review  — prepare a take and ask a video-capable model about it
  cost    — what a take WOULD cost to review, without spending anything

``cost`` exists because the token bill for video is not intuitive (a minute of
footage is ~18k tokens, an hour is ~1M) and the only honest way to offer that is
to let someone see the number before the call, not after.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Optional

from plugins.watch import decider as dec
from plugins.watch import live
from plugins.watch import prepare as prep
from plugins.watch import recorder as rec
from plugins.watch import review as rev


def _mb(size_bytes: float) -> str:
    return f"{size_bytes / 1048576:.1f} MB"


def register_cli(subparser: argparse.ArgumentParser) -> None:
    """Build the ``hermes watch`` argparse tree."""
    subs = subparser.add_subparsers(dest="watch_command")

    start_p = subs.add_parser("start", help="Start recording the screen")
    start_p.add_argument("--label", default="", help="short name for the take")
    start_p.add_argument("--fps", type=float, default=15.0, help="capture frame rate (default 15)")
    start_p.add_argument(
        "--audio-device", default=None,
        help="Windows: exact dshow device name. Linux: pulse source name. macOS: avfoundation index.",
    )
    start_p.add_argument(
        "--screen-index", type=int, default=1,
        help="macOS only: avfoundation screen device index (default 1)",
    )

    subs.add_parser("stop", help="Stop recording and finalize the take")
    subs.add_parser("status", help="Show whether a recording is running")

    list_p = subs.add_parser("list", help="List recorded takes")
    list_p.add_argument("--limit", type=int, default=20)

    for name, helptext in (
        ("review", "Prepare a take and ask a model about it"),
        ("cost", "Estimate what reviewing a take would cost (no API call)"),
    ):
        p = subs.add_parser(name, help=helptext)
        p.add_argument(
            "take", nargs="?", default=None,
            help="path to a take (default: the most recent one)",
        )
        p.add_argument(
            "-q", "--question", default="",
            help="what to ask about the take",
        )
        p.add_argument("--model", default=rev.DEFAULT_MODEL, help="video-capable model")
        p.add_argument("--width", type=int, default=640, help="output width (default 640)")
        p.add_argument("--fps", dest="out_fps", type=float, default=1.0,
                       help="output frame rate; providers sample at 1 fps (default 1)")
        p.add_argument("--crf", type=int, default=28, help="x264 quality floor (default 28)")
        p.add_argument(
            "--speed", type=float, default=1.0,
            help="retime: 2 = twice as fast (half the tokens, one sample per 2s of real time)",
        )
        p.add_argument("--no-audio", action="store_true", help="drop the audio track")
        p.add_argument(
            "--low-res", action="store_true",
            help="assume media_resolution=low for the estimate (66 tok/frame)",
        )
        p.add_argument(
            "--no-titles", action="store_true",
            help="timeline keeps app names but drops window titles",
        )

    live_p = subs.add_parser("live", help="Watch continuously and comment when it matters")
    live_p.add_argument(
        "-b", "--brief", default="",
        help='what to watch for, e.g. "my rotation timing" or "how I use my synths"',
    )
    live_p.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    live_p.add_argument("--interval", type=float, default=live.DEFAULT_INTERVAL,
                        help="seconds between frames (default 1)")
    live_p.add_argument("--model", default=None, help="video-capable model (default: config)")
    live_p.add_argument("--refractory", type=float, default=dec.DEFAULT_REFRACTORY,
                        help="quiet period after speaking (default 12s)")
    live_p.add_argument("--cooldown", type=float, default=dec.DEFAULT_CALL_COOLDOWN,
                        help="minimum seconds between model calls (default 8s)")
    live_p.add_argument("--min-salience", type=float, default=dec.MIN_SALIENCE,
                        help="how much must change before asking (0-1, default 0.25)")
    live_p.add_argument("--screen-index", type=int, default=1, help="macOS avfoundation screen index")
    live_p.add_argument("--quiet", action="store_true", help="only print what it says")

    replay_p = subs.add_parser(
        "replay", help="Re-run a recorded session against different settings, free"
    )
    replay_p.add_argument("log", nargs="?", default=None,
                          help="decision log (default: the most recent)")
    replay_p.add_argument("--refractory", type=float, default=None)
    replay_p.add_argument("--cooldown", type=float, default=None)
    replay_p.add_argument("--min-salience", type=float, default=None)
    replay_p.add_argument("--sweep", action="store_true",
                          help="try a range of salience thresholds and compare")


def _resolve_take(argument: Optional[str]) -> Path:
    if argument:
        return Path(argument).expanduser()
    recent = rec.takes(limit=1)
    if not recent:
        raise rev.ReviewError("No takes recorded yet. Run: hermes watch start")
    return Path(recent[0]["path"])


def _spec_from_args(args) -> prep.PrepareSpec:
    return prep.PrepareSpec(
        width=args.width,
        fps=args.out_fps,
        crf=args.crf,
        speed=args.speed,
        audio=not args.no_audio,
        media_resolution="low" if args.low_res else "default",
    )


def _timeline_for(take: Path) -> Optional[Path]:
    candidate = rec.timeline_path_for(take)
    return candidate if candidate.is_file() else None


def _plan_for(args, take: Path) -> dict:
    return rev.plan_review(
        take,
        spec=_spec_from_args(args),
        timeline_path=_timeline_for(take),
        include_titles=not args.no_titles,
        meta=rec.read_meta(take),
    )


def _print_plan(plan: dict) -> None:
    probe = plan["probe"]
    cost = plan["cost"]
    print(f"  duration      {probe['duration_seconds']:.1f}s  ({_mb(probe['size_bytes'])} on disk)")
    print(f"  billed as     {cost['billed_seconds']:.1f}s of timeline")
    print(f"  effective fps {cost['effective_fps']:g} (frames of real time the model sees per second)")
    print(f"  audio         {'yes' if cost['with_audio'] else 'no'}")
    print(f"  est. tokens   ~{cost['estimated_tokens']:,}")

    scale = plan.get("capture_scale", 1.0)
    if scale < 0.98:
        # Worth saying out loud: the capture backend dropped frames, so the
        # file covers more real time than its duration suggests and the
        # timeline has been compressed to match.
        print(
            f"  capture lag   {plan['wall_seconds']:.0f}s of real time in "
            f"{probe['duration_seconds']:.0f}s of video (×{scale:.2f}) — timeline rescaled"
        )

    summary = plan.get("timeline_summary")
    if summary:
        apps = ", ".join(f"{app} {secs:.0f}s" for app, secs in summary["apps"][:4])
        print(f"  window track  {summary['segments']} segments, {summary['switches']} switches — {apps}")
    else:
        print("  window track  none (no timeline sidecar for this take)")


def _cmd_start(args) -> int:
    result = rec.start(
        label=args.label,
        fps=args.fps,
        audio_device=args.audio_device,
        screen_index=args.screen_index,
    )
    if not result.get("success"):
        print(f"Could not start: {result.get('error')}", file=sys.stderr)
        return 1

    print(f"Recording → {result['video_path']}")
    print(f"  audio: {'yes' if result['audio'] else 'no'}")
    for note in result.get("notes", []):
        print(f"  note: {note}")
    print("Stop with: hermes watch stop")
    return 0


def _cmd_stop(_args) -> int:
    result = rec.stop()
    if not result.get("success"):
        print(result.get("error"), file=sys.stderr)
        return 1
    print(f"Take saved → {result['video_path']}")
    print(f"  {result['duration_seconds']:.1f}s of video, {_mb(result['size_bytes'])}")
    lag = result.get("capture_lag")
    if lag:
        print(
            f"  note: {result['wall_seconds']:.0f}s of real time — the capture "
            f"backend dropped {lag:.0f}s worth of frames (lower --fps, or close "
            f"something heavy). The timeline is rescaled to match the video."
        )
    if result.get("timeline_path"):
        print(f"  window timeline: {result['timeline_path']}")
    print("Review with: hermes watch review -q \"what do you think?\"")
    return 0


def _cmd_status(_args) -> int:
    state = rec.status()
    if not state.get("recording"):
        if state.get("note"):
            print(f"Not recording ({state['note']}; partial file at {state.get('video_path')})")
        else:
            print("Not recording.")
        return 0
    print(f"Recording {state['elapsed_seconds']:.0f}s → {state['video_path']}")
    print(f"  {_mb(state['size_bytes'])}, {state['fps']:g} fps, audio {'on' if state['audio'] else 'off'}")
    return 0


def _cmd_list(args) -> int:
    rows = rec.takes(limit=args.limit)
    if not rows:
        print("No takes recorded yet.")
        return 0
    for row in rows:
        marker = "+timeline" if row["has_timeline"] else ""
        print(f"{_mb(row['size_bytes']):>10}  {row['path']}  {marker}")
    return 0


def _cmd_cost(args) -> int:
    take = _resolve_take(args.take)
    plan = _plan_for(args, take)
    print(f"{take}")
    _print_plan(plan)
    if not args.low_res:
        print("  (--low-res would cut frame tokens ~4x)")
    return 0


def _cmd_review(args) -> int:
    take = _resolve_take(args.take)
    plan = _plan_for(args, take)
    print(f"{take}")
    _print_plan(plan)

    def on_step(spec: prep.PrepareSpec, size: int) -> None:
        verdict = "fits" if prep.fits(size) else "too big"
        print(f"  prepare  {spec.width or 'src'}px crf{spec.crf} → {_mb(size)} ({verdict})")

    prepared, spec, _size = rev.fit_take(
        take,
        plan["spec"],
        duration_s=plan["probe"]["duration_seconds"],
        on_step=on_step,
    )

    prompt = rev.build_prompt(
        args.question,
        timeline_block=plan["timeline_block"],
        duration_s=plan["probe"]["duration_seconds"],
        spec=spec,
    )

    print(f"\nAsking {args.model}…\n")
    result = asyncio.run(rev.analyze(prepared, prompt, model=args.model))
    if not result.get("success"):
        print(f"Review failed: {result.get('error', 'unknown error')}", file=sys.stderr)
        return 1
    print(result.get("analysis", "").strip())
    return 0


def _cmd_live(args) -> int:
    brief = args.brief or "anything notable about how they are performing"
    policy = dec.Policy(
        refractory=args.refractory,
        call_cooldown=args.cooldown,
        min_salience=args.min_salience,
    )

    if not args.quiet:
        print(f"Watching for: {brief}")
        print("Ctrl-C to stop.\n")

    def announce(decision) -> None:
        stamp = f"{int(decision.at) // 60}:{int(decision.at) % 60:02d}"
        held = " (held for a gap)" if decision.deferred else ""
        print(f"  {stamp}  {decision.text}{held}", flush=True)

    stop = threading.Event()

    def on_sigint(_signum, _frame) -> None:
        stop.set()

    previous = signal.signal(signal.SIGINT, on_sigint)
    try:
        result = live.run_live(
            brief=brief,
            duration=args.duration,
            interval=args.interval,
            model=args.model,
            policy=policy,
            screen_index=args.screen_index,
            on_speak=announce,
            stop=stop,
        )
    finally:
        signal.signal(signal.SIGINT, previous)

    if not result.get("success"):
        print(f"\nCould not start: {result.get('error')}", file=sys.stderr)
        return 1

    print(
        f"\n{result['seconds']:.0f}s watched, {result['ticks']} frames, "
        f"{result['model_calls']} model calls ({result['call_rate'] * 100:.1f}%), "
        f"{result['spoke']} said"
    )
    if result.get("suppressed_as_repetition"):
        print(f"  {result['suppressed_as_repetition']} repeats suppressed")
    if result.get("log_path"):
        print(f"  decision log: {result['log_path']}")
        print("  tune it for free with: hermes watch replay --sweep")
    return 0


def _resolve_log(argument: Optional[str]) -> Path:
    if argument:
        return Path(argument).expanduser()
    directory = rec.watch_dir() / "live"
    logs = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        raise rev.ReviewError("No live sessions recorded yet. Run: hermes watch live")
    return logs[0]


def _cmd_replay(args) -> int:
    path = _resolve_log(args.log)
    header, rows = live.read_log(path)
    print(f"{path.name}  ({len(rows)} ticks, brief: {header.get('brief', '—')})\n")

    if args.sweep:
        # The whole point of logging every tick: a dozen policies cost nothing
        # once the session is on disk.
        print("  min_salience   calls   call%   said")
        for threshold in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
            stats = live.replay(path, dec.Policy(min_salience=threshold))
            print(
                f"  {threshold:>12.2f}  {stats['model_calls']:>6}  "
                f"{stats['call_rate'] * 100:>5.1f}%  {stats['spoke']:>4}"
            )
        return 0

    base = dec.Policy()
    policy = dec.Policy(
        refractory=args.refractory if args.refractory is not None else base.refractory,
        call_cooldown=args.cooldown if args.cooldown is not None else base.call_cooldown,
        min_salience=args.min_salience if args.min_salience is not None else base.min_salience,
    )
    stats = live.replay(path, policy)
    print(f"  model calls {stats['model_calls']} ({stats['call_rate'] * 100:.1f}%)")
    print(f"  would say   {stats['spoke']}")
    print(f"  settings    {stats['policy']}")
    return 0


_HANDLERS = {
    "start": _cmd_start,
    "stop": _cmd_stop,
    "status": _cmd_status,
    "list": _cmd_list,
    "review": _cmd_review,
    "cost": _cmd_cost,
    "live": _cmd_live,
    "replay": _cmd_replay,
}


def watch_command(args) -> int:
    """Dispatch ``hermes watch <subcommand>``."""
    name = getattr(args, "watch_command", None)
    if not name:
        print("Usage: hermes watch {start|stop|status|list|review|cost|live|replay}")
        print("  hermes watch start --label solo-take")
        print("  hermes watch stop")
        print("  hermes watch review -q \"how was my timing?\"")
        print("  hermes watch live -b \"how I use my synths\"")
        print("  hermes watch replay --sweep")
        return 1

    handler = _HANDLERS.get(name)
    if handler is None:
        print(f"Unknown subcommand: {name}", file=sys.stderr)
        return 1

    try:
        return handler(args)
    except rev.ReviewError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
