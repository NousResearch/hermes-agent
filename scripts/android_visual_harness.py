#!/usr/bin/env python3
"""ADB visual harness for Hermes Android emulator validation."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_PACKAGE = "com.nousresearch.hermesagent"
DEFAULT_READY_TEXT = "Message Hermes|Settings|Hermes"
UI_DUMP_REMOTE_PATH = "/sdcard/window_dump.xml"


def adb_path() -> str:
    sdk = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
    if sdk:
        candidate = Path(sdk) / "platform-tools" / ("adb.exe" if os.name == "nt" else "adb")
        if candidate.is_file():
            return str(candidate)
    if os.name == "nt":
        default_sdk = Path(
            r"C:\Users\Ady\Documents\Codex\2026-05-02\c-users-ady-downloads-hermes-android\_android_sdk"
        )
        candidate = default_sdk / "platform-tools" / "adb.exe"
        if candidate.is_file():
            return str(candidate)
    resolved = shutil.which("adb.exe" if os.name == "nt" else "adb")
    if resolved:
        return resolved
    raise FileNotFoundError(
        "adb was not found. Set ANDROID_HOME or ANDROID_SDK_ROOT, or put adb on PATH."
    )


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


def write_screenshot(serial: str | None, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        adb_args(serial, "exec-out", "screencap", "-p"),
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
    run_adb(args.serial, "shell", f"input text {shlex.quote(payload)}")
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


def wait_for_focus(serial: str | None, package: str, timeout_ms: int) -> bool:
    deadline = time.monotonic() + (timeout_ms / 1000)
    last_focus = ""
    while time.monotonic() <= deadline:
        result = run_adb(serial, "shell", "dumpsys", "window", check=False)
        combined = result.stdout + result.stderr
        focus_lines = [
            line.strip()
            for line in combined.splitlines()
            if "mCurrentFocus" in line or "mFocusedApp" in line
        ]
        if focus_lines:
            last_focus = " | ".join(focus_lines)
        if any("mCurrentFocus" in line and package in line for line in focus_lines):
            return True
        time.sleep(1)
    sys.stderr.write(f"Timed out waiting for focused window from {package}. Last focus: {last_focus}\n")
    return False


def read_ui_xml(serial: str | None) -> str:
    run_adb(serial, "shell", "rm", "-f", UI_DUMP_REMOTE_PATH, check=False)
    dump_result = run_adb(serial, "shell", "uiautomator", "dump", UI_DUMP_REMOTE_PATH, check=False)
    if dump_result.returncode != 0:
        return ""
    cat_result = run_adb(serial, "exec-out", "cat", UI_DUMP_REMOTE_PATH, check=False)
    if cat_result.returncode != 0:
        return ""
    xml = cat_result.stdout
    if not xml.lstrip().startswith("<?xml"):
        return ""
    return xml


def center_from_bounds(bounds: str) -> tuple[int, int] | None:
    try:
        left_top, right_bottom = bounds.split("][", 1)
        left, top = [int(part) for part in left_top.strip("[]").split(",", 1)]
        right, bottom = [int(part) for part in right_bottom.strip("[]").split(",", 1)]
        return ((left + right) // 2, (top + bottom) // 2)
    except (AttributeError, ValueError):
        return None


def tap_first_ui_text(serial: str | None, xml: str, labels: tuple[str, ...]) -> bool:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return False
    for node in root.iter("node"):
        text = node.attrib.get("text", "")
        content_description = node.attrib.get("content-desc", "")
        if text in labels or content_description in labels:
            center = center_from_bounds(node.attrib.get("bounds", ""))
            if center is None:
                continue
            run_adb(serial, "shell", "input", "tap", str(center[0]), str(center[1]), check=False)
            return True
    return False


def continue_past_anr_dialog(serial: str | None, xml: str) -> bool:
    if "isn&apos;t responding" not in xml and "isn't responding" not in xml:
        return False
    if "Wait" not in xml:
        return False
    if tap_first_ui_text(serial, xml, ("Wait",)):
        print("Dismissed Android ANR dialog with Wait")
        return True
    return False


def wait_for_ui_text(serial: str | None, ready_text: str, timeout_ms: int) -> bool:
    if not ready_text:
        return True
    ready_texts = tuple(text.strip() for text in ready_text.split("|") if text.strip())
    if not ready_texts:
        return True
    deadline = time.monotonic() + (timeout_ms / 1000)
    while time.monotonic() <= deadline:
        xml = read_ui_xml(serial)
        if any(text in xml for text in ready_texts):
            return True
        if xml and continue_past_anr_dialog(serial, xml):
            time.sleep(1)
            continue
        time.sleep(2)
    sys.stderr.write(f"Timed out waiting for UI text: {ready_text}\n")
    return False


def set_size(args: argparse.Namespace) -> int:
    run_adb(args.serial, "shell", "wm", "size", args.size)
    if args.density:
        run_adb(args.serial, "shell", "wm", "density", str(args.density))
    return 0


def reset_size(args: argparse.Namespace) -> int:
    run_adb(args.serial, "shell", "wm", "size", "reset")
    run_adb(args.serial, "shell", "wm", "density", "reset")
    return 0


def dump_ui(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    xml = read_ui_xml(args.serial)
    if not xml:
        sys.stderr.write(f"uiautomator dump did not produce XML at {UI_DUMP_REMOTE_PATH}\n")
        return 1
    out.write_text(xml, encoding="utf-8")
    print(out)
    return 0


def wide_capture(args: argparse.Namespace) -> int:
    if args.size:
        run_adb(args.serial, "shell", "wm", "size", args.size)
    if args.density:
        run_adb(args.serial, "shell", "wm", "density", str(args.density))
    try:
        if not args.no_launch:
            launch_result = launch(argparse.Namespace(serial=args.serial, package=args.package))
            if launch_result != 0:
                return launch_result
            if args.ready_timeout_ms and not wait_for_focus(
                args.serial,
                args.package,
                args.ready_timeout_ms,
            ):
                return 1
            if args.ready_timeout_ms and not wait_for_ui_text(
                args.serial,
                args.ready_text,
                args.ready_timeout_ms,
            ):
                return 1
        time.sleep(args.wait_ms / 1000)
        return write_screenshot(args.serial, Path(args.out))
    finally:
        if not args.keep_size:
            reset_size(argparse.Namespace(serial=args.serial))


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

    click_parser = sub.add_parser("click")
    click_parser.add_argument("x", type=int)
    click_parser.add_argument("y", type=int)
    click_parser.set_defaults(func=tap)

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

    dump_parser = sub.add_parser("dump-ui")
    dump_parser.add_argument("--out", required=True, help="Host XML output path.")
    dump_parser.set_defaults(func=dump_ui)

    wide_parser = sub.add_parser("wide-capture")
    wide_parser.add_argument("--out", required=True, help="Host PNG output path.")
    wide_parser.add_argument("--package", default=DEFAULT_PACKAGE)
    wide_parser.add_argument("--size", default="1920x1080", help="Temporary emulator resolution.")
    wide_parser.add_argument("--density", type=int, default=240, help="Temporary emulator density.")
    wide_parser.add_argument("--wait-ms", type=int, default=8000, help="Wait after launch before capture.")
    wide_parser.add_argument(
        "--ready-timeout-ms",
        type=int,
        default=90000,
        help="Maximum time to wait for the package to own the focused window.",
    )
    wide_parser.add_argument(
        "--ready-text",
        default=DEFAULT_READY_TEXT,
        help="Pipe-separated UI text alternatives that may appear before capture; pass an empty value to skip this check.",
    )
    wide_parser.add_argument("--no-launch", action="store_true", help="Capture current screen without launching Hermes.")
    wide_parser.add_argument("--keep-size", action="store_true", help="Do not reset wm size/density after capture.")
    wide_parser.set_defaults(func=wide_capture)
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
