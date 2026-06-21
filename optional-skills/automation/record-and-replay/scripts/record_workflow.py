#!/usr/bin/env python3
"""Record & Replay — Cadillac recording script for Hermes Agent.

Captures real input events (mouse clicks, keystrokes, scrolls, drags) via
CGEventTap, synchronized with periodic screenshots and accessibility tree
snapshots. Produces a structured recording that the vision model can turn
into a replayable skill.

Usage:
    python3 record_workflow.py [--interval 1.0] [--output-dir PATH]

Requires:
    - macOS (uses Quartz CGEventTap + screencapture)
    - pyobjc (pip install pyobjc-framework-Quartz)
    - cua-driver on $PATH (for AX tree snapshots)
    - Accessibility + Screen Recording permissions granted to Terminal

Output structure:
    ~/.hermes/recordings/<timestamp>/
    ├── recording.json          # Event log + metadata
    ├── screenshots/
    │   ├── 0001.png           # Numbered screenshots
    │   ├── 0002.png
    │   └── ...
    ├── ax_trees/
    │   ├── 0001.txt           # AX tree at each screenshot
    │   ├── 0002.txt
    │   └── ...
    └── events/
        └── events.jsonl       # One JSON event per line (streaming)

Event types captured:
    - mouse_down / mouse_up (with coordinates, button, modifier flags)
    - key_down / key_up (with key code, character, modifier flags)
    - scroll (with direction, amount, coordinates)
    - mouse_dragged (with from/to coordinates)
    - app_activated (when frontmost app changes)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ── Quartz imports (pyobjc) ──────────────────────────────────────────────────

try:
    import Quartz
    from Quartz import (
        CGEventTapCreate,
        CGEventTapEnable,
        CGEventMaskBit,
        kCGSessionEventTap,
        kCGHeadInsertEventTap,
        kCGEventTapOptionListenOnly,
        kCGEventLeftMouseDown,
        kCGEventLeftMouseUp,
        kCGEventRightMouseDown,
        kCGEventRightMouseUp,
        kCGEventOtherMouseDown,
        kCGEventOtherMouseUp,
        kCGEventMouseMoved,
        kCGEventLeftMouseDragged,
        kCGEventRightMouseDragged,
        kCGEventOtherMouseDragged,
        kCGEventScrollWheel,
        kCGEventKeyDown,
        kCGEventKeyUp,
        kCGEventFlagsChanged,
        CFMachPortCreateRunLoopSource,
        CFRunLoopAddSource,
        CFRunLoopGetCurrent,
        CFRunLoopRun,
        CFRunLoopStop,
        kCFRunLoopCommonModes,
    )
    from CoreFoundation import CFMachPortRef

    QUARTZ_AVAILABLE = True
except ImportError:
    QUARTZ_AVAILABLE = False


# ── Event mask ───────────────────────────────────────────────────────────────

EVENT_MASK = (
    CGEventMaskBit(kCGEventLeftMouseDown)
    | CGEventMaskBit(kCGEventLeftMouseUp)
    | CGEventMaskBit(kCGEventRightMouseDown)
    | CGEventMaskBit(kCGEventRightMouseUp)
    | CGEventMaskBit(kCGEventOtherMouseDown)
    | CGEventMaskBit(kCGEventOtherMouseUp)
    | CGEventMaskBit(kCGEventMouseMoved)
    | CGEventMaskBit(kCGEventLeftMouseDragged)
    | CGEventMaskBit(kCGEventRightMouseDragged)
    | CGEventMaskBit(kCGEventOtherMouseDragged)
    | CGEventMaskBit(kCGEventScrollWheel)
    | CGEventMaskBit(kCGEventKeyDown)
    | CGEventMaskBit(kCGEventKeyUp)
    | CGEventMaskBit(kCGEventFlagsChanged)
    if QUARTZ_AVAILABLE
    else 0
)

# Modifier flag bits
CGM_FLAG_MASK = {
    "caps_lock": 0x10000,
    "shift": 0x20000,
    "control": 0x40000,
    "option": 0x80000,
    "command": 0x100000,
    "numeric_pad": 0x200000,
    "help": 0x400000,
    "function": 0x800000,
}


def decode_flags(flags: int) -> list[str]:
    """Decode CGEvent modifier flags into a list of modifier names."""
    mods = []
    for name, bit in CGM_FLAG_MASK.items():
        if flags & bit:
            mods.append(name)
    return mods


def get_frontmost_app() -> dict:
    """Get the frontmost app name, bundle ID, and PID."""
    try:
        ws = Quartz.NSWorkspace.sharedWorkspace()
        app = ws.frontmostApplication()
        return {
            "name": app.localizedName() if app else "Unknown",
            "bundle_id": app.bundleIdentifier() if app else "",
            "pid": app.processIdentifier() if app else 0,
        }
    except Exception:
        return {"name": "Unknown", "bundle_id": "", "pid": 0}


def get_mouse_location() -> tuple[float, float]:
    """Get current mouse cursor position (global coordinates)."""
    try:
        event = Quartz.CGEventCreate(None)
        loc = Quartz.CGEventGetLocation(event)
        return (loc.x, loc.y)
    except Exception:
        return (0.0, 0.0)


def keycode_to_char(keycode: int, flags: int) -> str:
    """Convert a CGKeyCode to a human-readable character when possible."""
    # Common key codes mapping
    special_keys = {
        0: "a", 1: "s", 2: "d", 3: "f", 4: "h", 5: "g", 6: "z", 7: "x",
        8: "c", 9: "v", 11: "b", 12: "q", 13: "w", 14: "e", 15: "r",
        16: "y", 17: "t", 18: "1", 19: "2", 20: "3", 21: "4", 22: "5",
        23: "6", 24: "=", 25: "9", 26: "7", 27: "-", 28: "8", 29: "0",
        30: "]", 31: "o", 32: "u", 33: "[", 34: "i", 35: "p",
        36: "return", 37: "l", 38: "j", 39: "'", 40: "k", 41: ";",
        42: "\\", 43: ",", 44: "/", 45: "n", 46: "m", 47: ".",
        48: "tab", 49: "space", 50: "`", 51: "delete", 53: "escape",
        55: "cmd", 56: "shift", 57: "caps_lock", 58: "option", 59: "control",
        60: "right_shift", 61: "right_option", 62: "right_control",
        63: "fn", 65: " keypad_decimal", 67: "keypad_*",
        69: "keypad_+", 71: "keypad_clear", 75: "keypad_/",
        76: "keypad_enter", 78: "keypad_-", 81: "keypad_=",
        82: "keypad_0", 83: "keypad_1", 84: "keypad_2", 85: "keypad_3",
        86: "keypad_4", 87: "keypad_5", 88: "keypad_6", 89: "keypad_7",
        91: "keypad_8", 92: "keypad_9",
        96: "f5", 97: "f6", 98: "f7", 99: "f3", 100: "f8",
        101: "f9", 109: "f10", 103: "f11", 111: "f12",
        105: "f13", 107: "f14", 113: "f15", 106: "f16",
        122: "f1", 120: "f2", 99: "f3", 118: "f4",
        123: "left_arrow", 124: "right_arrow", 125: "down_arrow", 126: "up_arrow",
    }
    return special_keys.get(keycode, f"keycode_{keycode}")


def capture_screenshot(output_path: str) -> bool:
    """Capture a screenshot to the given path using screencapture."""
    try:
        result = subprocess.run(
            ["screencapture", "-x", "-t", "png", output_path],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False


def capture_ax_tree(output_path: str, app_name: str | None = None) -> bool:
    """Capture the accessibility tree using cua-driver."""
    try:
        cmd = ["cua-driver", "mcp"]
        # We use a simpler approach: call cua-driver's CLI directly
        # for a window state dump
        result = subprocess.run(
            ["cua-driver", "get_window_state"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            with open(output_path, "w") as f:
                f.write(result.stdout)
            return True
    except FileNotFoundError:
        # cua-driver not installed — write a placeholder
        pass
    except Exception:
        pass

    # Fallback: try osascript for basic window info
    try:
        script = """
        tell application "System Events"
            set frontApp to first application process whose frontmost is true
            set appName to name of frontApp
            set windowList to ""
            repeat with w in windows of frontApp
                set windowList to windowList & name of w & "\\n"
            end repeat
            return appName & "\\n" & windowList
        end tell
        """
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=5,
        )
        with open(output_path, "w") as f:
            f.write(f"# AX Tree (osascript fallback)\n\n")
            f.write(f"Frontmost app: {result.stdout.strip() if result.returncode == 0 else 'Unknown'}\n")
        return True
    except Exception:
        with open(output_path, "w") as f:
            f.write("# AX Tree unavailable (cua-driver not installed)\n")
        return False


# ── Recording state ──────────────────────────────────────────────────────────

class RecordingState:
    def __init__(self, output_dir: Path, interval: float):
        self.output_dir = output_dir
        self.interval = interval
        self.screenshot_dir = output_dir / "screenshots"
        self.ax_dir = output_dir / "ax_trees"
        self.events_dir = output_dir / "events"
        self.events_file = self.events_dir / "events.jsonl"
        self.start_time = time.time()
        self.event_count = 0
        self.screenshot_count = 0
        self.last_app = None
        self.recording = True
        self.events_lock = threading.Lock()
        self._events_fh = None

    def setup(self):
        """Create output directories."""
        for d in [self.screenshot_dir, self.ax_dir, self.events_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self._events_fh = open(self.events_file, "w", buffering=1)  # line-buffered

    def log_event(self, event: dict):
        """Write an event to the JSONL file."""
        event["timestamp"] = time.time() - self.start_time
        event["wall_time"] = datetime.now().isoformat()
        with self.events_lock:
            if self._events_fh:
                self._events_fh.write(json.dumps(event) + "\n")
                self.event_count += 1

    def capture_snapshot(self):
        """Take a screenshot + AX tree snapshot."""
        num = self.screenshot_count + 1
        shot_path = str(self.screenshot_dir / f"{num:04d}.png")
        ax_path = str(self.ax_dir / f"{num:04d}.txt")

        # Capture in parallel
        shot_ok = capture_screenshot(shot_path)
        app_info = get_frontmost_app()
        ax_ok = capture_ax_tree(ax_path, app_info.get("name"))

        mouse_x, mouse_y = get_mouse_location()

        self.log_event({
            "type": "snapshot",
            "screenshot": f"screenshots/{num:04d}.png" if shot_ok else None,
            "ax_tree": f"ax_trees/{num:04d}.txt" if ax_ok else None,
            "frontmost_app": app_info,
            "mouse_position": [mouse_x, mouse_y],
            "screenshot_number": num,
        })

        self.screenshot_count = num

        # Check for app switch
        current_app = app_info.get("name", "")
        if self.last_app and current_app != self.last_app:
            self.log_event({
                "type": "app_switch",
                "from": self.last_app,
                "to": current_app,
            })
        self.last_app = current_app

    def close(self):
        """Close file handles and write metadata."""
        if self._events_fh:
            self._events_fh.close()

        metadata = {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "duration_seconds": time.time() - self.start_time,
            "event_count": self.event_count,
            "screenshot_count": self.screenshot_count,
            "interval": self.interval,
            "platform": "macos",
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


# ── CGEventTap callback ──────────────────────────────────────────────────────

_state: RecordingState | None = None


def event_callback(proxy, event_type, event, refcon):
    """CGEventTap callback — runs on the CoreFoundation run loop thread."""
    global _state
    if _state is None or not _state.recording:
        return event

    try:
        flags = Quartz.CGEventGetFlags(event)
        modifiers = decode_flags(flags)
        mouse_loc = Quartz.CGEventGetLocation(event)

        evt = {
            "type": "",
            "modifiers": modifiers,
            "flags": flags,
        }

        if event_type in (kCGEventLeftMouseDown, kCGEventLeftMouseUp,
                          kCGEventRightMouseDown, kCGEventRightMouseUp,
                          kCGEventOtherMouseDown, kCGEventOtherMouseUp):
            button_map = {
                kCGEventLeftMouseDown: ("mouse_down", "left"),
                kCGEventLeftMouseUp: ("mouse_up", "left"),
                kCGEventRightMouseDown: ("mouse_down", "right"),
                kCGEventRightMouseUp: ("mouse_up", "right"),
                kCGEventOtherMouseDown: ("mouse_down", "middle"),
                kCGEventOtherMouseUp: ("mouse_up", "middle"),
            }
            evt_type, button = button_map.get(event_type, ("mouse_event", "unknown"))
            evt["type"] = evt_type
            evt["button"] = button
            evt["position"] = [mouse_loc.x, mouse_loc.y]

        elif event_type in (kCGEventMouseMoved, kCGEventLeftMouseDragged,
                            kCGEventRightMouseDragged, kCGEventOtherMouseDragged):
            if event_type == kCGEventMouseMoved:
                evt["type"] = "mouse_moved"
            else:
                evt["type"] = "mouse_dragged"
            evt["position"] = [mouse_loc.x, mouse_loc.y]

        elif event_type == kCGEventScrollWheel:
            delta_y = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGScrollWheelEventDeltaAxis1
            )
            delta_x = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGScrollWheelEventDeltaAxis2
            )
            evt["type"] = "scroll"
            evt["position"] = [mouse_loc.x, mouse_loc.y]
            evt["delta_y"] = delta_y
            evt["delta_x"] = delta_x
            evt["direction"] = "down" if delta_y > 0 else "up" if delta_y < 0 else "none"

        elif event_type in (kCGEventKeyDown, kCGEventKeyUp):
            keycode = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventKeycode
            )
            char = keycode_to_char(keycode, flags)
            evt["type"] = "key_down" if event_type == kCGEventKeyDown else "key_up"
            evt["keycode"] = keycode
            evt["key"] = char
            # Try to get the actual character via keyboard layout
            try:
                chars = Quartz.CGEventKeyboardGetUnicodeString(event, 100)
                if chars:
                    evt["character"] = chars[:10]  # limit length
            except Exception:
                pass

        elif event_type == kCGEventFlagsChanged:
            evt["type"] = "flags_changed"
            evt["modifiers"] = modifiers

        else:
            evt["type"] = f"unknown_{event_type}"

        if evt["type"]:
            _state.log_event(evt)

    except Exception as e:
        # Don't let callback errors crash the run loop
        if _state:
            _state.log_event({"type": "error", "error": str(e)})

    return event


# ── Snapshot thread ──────────────────────────────────────────────────────────

def snapshot_loop(state: RecordingState):
    """Background thread that captures periodic screenshots + AX trees."""
    while state.recording:
        state.capture_snapshot()
        time.sleep(state.interval)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Record macOS workflow for Hermes Record & Replay"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Screenshot interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ~/.hermes/recordings/<timestamp>)",
    )
    parser.add_argument(
        "--no-ax",
        action="store_true",
        help="Skip AX tree capture (faster, less detail)",
    )
    args = parser.parse_args()

    if not QUARTZ_AVAILABLE:
        print("ERROR: pyobjc-framework-Quartz is required.", file=sys.stderr)
        print("Install with: pip install pyobjc-framework-Quartz", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(hermes_home) / "recordings" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    global _state
    _state = RecordingState(output_dir, args.interval)
    _state.setup()

    # Create the CGEventTap
    tap = CGEventTapCreate(
        kCGSessionEventTap,
        kCGHeadInsertEventTap,
        kCGEventTapOptionListenOnly,  # listen-only: don't block or modify events
        EVENT_MASK,
        event_callback,
        None,
    )

    if tap is None:
        print(
            "ERROR: Failed to create CGEventTap. You need to grant:",
            file=sys.stderr,
        )
        print("  System Settings > Privacy & Security > Accessibility", file=sys.stderr)
        print("  System Settings > Privacy & Security > Screen Recording", file=sys.stderr)
        print("  (for your Terminal app)", file=sys.stderr)
        sys.exit(1)

    # Create a run loop source and add it to the current run loop
    source = CFMachPortCreateRunLoopSource(None, tap, 0)
    run_loop = CFRunLoopGetCurrent()
    CFRunLoopAddSource(run_loop, source, kCFRunLoopCommonModes)
    CGEventTapEnable(tap, True)

    # Start the snapshot thread
    snap_thread = threading.Thread(target=snapshot_loop, args=(_state,), daemon=True)
    snap_thread.start()

    # Handle stop signals — use a flag file + run-loop timer so signals
    # work even when CFRunLoop blocks the main thread.
    stop_flag = output_dir / ".stop"

    def signal_handler(sig, frame):
        print("\n⏹  Stopping recording...", file=sys.stderr)
        _state.recording = False
        # Write flag file as backup
        stop_flag.touch()
        CFRunLoopStop(run_loop)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Background thread that checks for stop flag file (every 0.5s).
    # This lets external processes stop the recording by creating the flag
    # file, and also catches signal-induced _state.recording = False.
    def stop_watcher():
        while _state.recording:
            if stop_flag.exists():
                _state.recording = False
                CFRunLoopStop(run_loop)
                return
            time.sleep(0.5)
        # If _state.recording was set False by signal, stop the run loop
        CFRunLoopStop(run_loop)

    stop_thread = threading.Thread(target=stop_watcher, daemon=True)
    stop_thread.start()

    print(f"🔴 Recording started", file=sys.stderr)
    print(f"   Output: {output_dir}", file=sys.stderr)
    print(f"   Interval: {args.interval}s", file=sys.stderr)
    print(f"   Events: {output_dir / 'events' / 'events.jsonl'}", file=sys.stderr)
    print(f"   Stop: touch {stop_flag}  or  kill -INT {os.getpid()}", file=sys.stderr)
    print(f"", file=sys.stderr)

    # Run the event loop (blocks until stopped)
    CFRunLoopRun()

    # Cleanup
    _state.recording = False
    snap_thread.join(timeout=3)
    _state.close()

    # Remove stop flag
    if stop_flag.exists():
        stop_flag.unlink()

    print(f"\n✅ Recording saved to: {output_dir}", file=sys.stderr)
    print(f"   Events captured: {_state.event_count}", file=sys.stderr)
    print(f"   Screenshots: {_state.screenshot_count}", file=sys.stderr)
    print(f"   Duration: {time.time() - _state.start_time:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
