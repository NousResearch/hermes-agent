#!/usr/bin/env python3
"""Run the Squish CLI on a local video and print its JSON contract.

Thin wrapper around ``npx -y @getsquish/squish`` (Apache-2.0,
github.com/getsquish/squish). The subprocess receives an argument list --
never a shell string -- so paths containing whitespace stay intact.
Stdlib only; no network access of its own (npx handles the package fetch).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE = "@getsquish/squish"
CONTRACT = "squish-cli-v0"
DENSITIES = ("3x3", "4x4", "5x5", "6x6")


def build_command(
    video: str,
    density: str | None = None,
    start: str | None = None,
    end: str | None = None,
    out: str | None = None,
) -> list[str]:
    """Build the npx argument list. The video path is one argv element."""
    if density is not None and density not in DENSITIES:
        raise ValueError(f"--density must be one of: {', '.join(DENSITIES)}")
    cmd = ["npx", "-y", PACKAGE, video, "--json"]
    if density is not None:
        cmd += ["--density", density]
    if start is not None:
        cmd += ["--start", start]
    if end is not None:
        cmd += ["--end", end]
    if out is not None:
        cmd += ["--out", out]
    return cmd


def parse_contract(stdout: str) -> dict:
    """Validate the CLI's JSON output against the frozen contract."""
    payload = json.loads(stdout)
    if payload.get("contract") != CONTRACT:
        raise ValueError(
            f"unexpected contract {payload.get('contract')!r} (want {CONTRACT!r})"
        )
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("output contains no contact sheets (files[] is empty)")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Turn a local video into timestamped contact sheets."
    )
    parser.add_argument("video", help="path to a local video file")
    parser.add_argument("--density", help="grid density: 3x3 (default), 4x4, 5x5, 6x6")
    parser.add_argument("--start", help="window start: seconds (90) or timecode (1:30)")
    parser.add_argument("--end", help="window end: seconds or timecode")
    parser.add_argument("--out", help="directory where sheets are written")
    args = parser.parse_args(argv)

    if shutil.which("npx") is None:
        print(
            "error: `npx` not found -- install Node.js >= 20 "
            "(ffmpeg is also required).",
            file=sys.stderr,
        )
        return 1
    video = Path(args.video)
    if not video.is_file():
        print(f"error: no such video file: {video}", file=sys.stderr)
        return 1

    try:
        cmd = build_command(str(video), args.density, args.start, args.end, args.out)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode or 1

    try:
        payload = parse_contract(proc.stdout)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    json.dump(payload, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
