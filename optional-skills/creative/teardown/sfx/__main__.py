"""CLI: render a cue sheet, or list the available generators."""

import argparse
import importlib.util
import inspect
import json
import os
import sys

from . import synth
from .mix import Mix


def list_generators():
    for name, fn in inspect.getmembers(synth, inspect.isfunction):
        if fn.__module__ != synth.__name__ or name.startswith("_"):
            continue
        doc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
        print(f"{name}{inspect.signature(fn)}\n    {doc}")


def load_cues(path):
    spec = importlib.util.spec_from_file_location("cues", path)
    cues = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cues)
    if not hasattr(cues, "score"):
        sys.exit(f"{path} needs a score(mix, tl) function")
    return cues


def main():
    ap = argparse.ArgumentParser(prog="python run_sfx.py", description="Render a cue sheet to a mastered WAV.")
    ap.add_argument("cues", nargs="?", help="cue sheet: a .py file with score(mix, tl)")
    ap.add_argument("--timeline", help="timeline JSON (default: the cue sheet's TIMELINE, else timeline.json)")
    ap.add_argument("--out", help="output WAV (default: the cue sheet's OUT, else teardown-out/<cue>.wav, or $TEARDOWN_OUT)")
    ap.add_argument("--list", action="store_true", help="list sound generators and exit")
    args = ap.parse_args()

    if args.list:
        list_generators()
        return
    if not args.cues:
        ap.error("pass a cue sheet, or --list")

    cues = load_cues(args.cues)
    # A relative TIMELINE in the cue sheet is read next to the cue sheet, so it runs from any folder.
    timeline = args.timeline or os.path.join(os.path.dirname(os.path.abspath(args.cues)), getattr(cues, "TIMELINE", "timeline.json"))
    stem = os.path.splitext(os.path.basename(args.cues))[0]
    out = args.out or getattr(cues, "OUT", os.path.join(os.environ.get("TEARDOWN_OUT", "teardown-out"), f"{stem}.wav"))
    with open(timeline, encoding="utf8") as f:
        tl = json.load(f)

    mix = Mix(tl["duration"], seed=getattr(cues, "SEED", 11))
    cues.score(mix, tl)
    stats = mix.master(target_lufs=getattr(cues, "TARGET_LUFS", -14.0))
    mix.write(out)
    print(
        f"wrote {out}: {tl['duration']}s, {mix.count} sounds, "
        f"{stats['lufs']:.1f} LUFS, true peak {stats['peak_db']:.1f} dBFS, drive {stats['drive']}"
    )


if __name__ == "__main__":
    main()
