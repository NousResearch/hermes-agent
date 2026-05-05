#!/usr/bin/env python3
"""ADB visual harness for Hermes Android emulator validation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_PACKAGE = "com.nousresearch.hermesagent"


def adb_path() -> str:
    sdk = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
    if sdk:
        candidate = Path(sdk) / "platform-tools" / ("adb.exe" if os.name == "nt" else "adb")
        if candidate.is_file():
            return str(candidate)
    return "adb"


def adb_args(serial: str | None, *args: str) -> list[str]:
    base = [adb_path()]
    if serial:
        base += ["-s", serial]
    return base + list(args)


def run_adb(serial: str | None, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        adb_args(serial, *args),
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def devices(_: argparse.Namespace) -> int:
    result = subprocess.run([adb_path(), "devices", "-l"], text=True, check=False)
    return result.returncode


def screenshot(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        adb_args(args.serial, "exec-out", "screencap", "-p"),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        return proc.returncode
    out.write_bytes(proc.stdout)
    print(out)
    return 0


def tap(args: argparse.Namespace) -> int:
    run_adb(args.serial, "shell", "input", "tap", str(args.x), str(args.y))
    return 0


def swipe(args: argparse.Namespace) -> int:
    run_adb(
        args.serial,
        "shell",
        "input",
        "swipe",
        str(args.x1),
        str(args.y1),
        str(args.x2),
        str(args.y2),
        str(args.duration_ms),
    )
    return 0


def text(args: argparse.Namespace) -> int:
    payload = args.text.replace("%", "%s").replace(" ", "%s")
    run_adb(args.serial, "shell", "input", "text", payload)
    return 0


def keyevent(args: argparse.Namespace) -> int:
    run_adb(args.serial, "shell", "input", "keyevent", args.key)
    return 0


def launch(args: argparse.Namespace) -> int:
    result = run_adb(
        args.serial,
        "shell",
        "monkey",
        "-p",
        args.package,
        "-c",
        "android.intent.category.LAUNCHER",
        "1",
        check=False,
    )
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    combined = result.stdout + result.stderr
    if "No activities found" in combined:
        return 1
    return result.returncode


def set_size(args: argparse.Namespace) -> int:
    run_adb(args.serial, "shell", "wm", "size", args.size)
    if args.density:
        run_adb(args.serial, "shell", "wm", "density", str(args.density))
    return 0


def reset_size(args: argparse.Namespace) -> int:
    run_adb(args.serial, "shell", "wm", "size", "reset")
    run_adb(args.serial, "shell", "wm", "density", "reset")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--serial", help="ADB device serial. Omit when exactly one device is attached.")
    sub = root.add_subparsers(dest="command", required=True)

    sub.add_parser("devices").set_defaults(func=devices)

    screenshot_parser = sub.add_parser("screenshot")
    screenshot_parser.add_argument("--out", required=True, help="PNG output path on the host.")
    screenshot_parser.set_defaults(func=screenshot)

    tap_parser = sub.add_parser("tap")
    tap_parser.add_argument("x", type=int)
    tap_parser.add_argument("y", type=int)
    tap_parser.set_defaults(func=tap)

    swipe_parser = sub.add_parser("swipe")
    swipe_parser.add_argument("x1", type=int)
    swipe_parser.add_argument("y1", type=int)
    swipe_parser.add_argument("x2", type=int)
    swipe_parser.add_argument("y2", type=int)
    swipe_parser.add_argument("--duration-ms", type=int, default=300)
    swipe_parser.set_defaults(func=swipe)

    text_parser = sub.add_parser("text")
    text_parser.add_argument("text")
    text_parser.set_defaults(func=text)

    key_parser = sub.add_parser("keyevent")
    key_parser.add_argument("key", help="Android keyevent name or number, for example BACK or 4.")
    key_parser.set_defaults(func=keyevent)

    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--package", default=DEFAULT_PACKAGE)
    launch_parser.set_defaults(func=launch)

    size_parser = sub.add_parser("set-size")
    size_parser.add_argument("size", help="Resolution such as 1920x1080.")
    size_parser.add_argument("--density", type=int, help="Optional density, for example 240.")
    size_parser.set_defaults(func=set_size)

    reset_parser = sub.add_parser("reset-size")
    reset_parser.set_defaults(func=reset_size)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return args.func(args)
    except subprocess.CalledProcessError as error:
        if error.stdout:
            sys.stdout.write(error.stdout)
        if error.stderr:
            sys.stderr.write(error.stderr)
        return error.returncode


if __name__ == "__main__":
    raise SystemExit(main())
