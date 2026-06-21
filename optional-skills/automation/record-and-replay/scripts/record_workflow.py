#!/usr/bin/env python3
"""Record & Replay — Cross-platform recording script for Hermes Agent.

Captures real input events (mouse clicks, keystrokes, scrolls, drags) globally
across ALL applications, synchronized with periodic screenshots and
accessibility tree snapshots. Produces a structured recording that the vision
model can turn into a replayable skill.

Works on macOS, Linux, and Windows.

Usage:
    python3 record_workflow.py [--interval 1.0] [--output-dir PATH]

Platform requirements:
    macOS:
        pip install pyobjc-framework-Quartz
        cua-driver (for AX trees) — optional, falls back to osascript
        Permissions: Accessibility + Screen Recording for Terminal

    Linux (X11):
        pip install pynput mss
        cua-driver or pyatspi (for AT trees) — optional, falls back to xdotool
        Install: sudo apt install xdotool scrot (or gnome-screenshot)

    Windows:
        pip install pynput mss pygetwindow uiautomation
        No special permissions needed (run as admin for elevated windows)

Output structure:
    ~/.hermes/recordings/<timestamp>/
    ├── metadata.json
    ├── events/
    │   └── events.jsonl       # One JSON event per line (streaming)
    ├── screenshots/
    │   ├── 0001.png
    │   └── ...
    └── ax_trees/
        ├── 0001.txt
        └── ...

Event types captured (all platforms):
    - mouse_down / mouse_up (with coordinates, button, modifier flags)
    - key_down / key_up (with key code, character, modifier flags)
    - scroll (with direction, amount, coordinates)
    - mouse_dragged (with from/to coordinates)
    - app_activated (when frontmost app changes)
    - snapshot (periodic screenshot + AX tree + mouse position)
    - app_switch (when frontmost app changes)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ── Platform detection ───────────────────────────────────────────────────────

PLATFORM = platform.system().lower()  # 'darwin', 'linux', 'windows'
IS_MACOS = PLATFORM == "darwin"
IS_LINUX = PLATFORM == "linux"
IS_WINDOWS = PLATFORM == "windows"


# ── Platform-specific imports ────────────────────────────────────────────────

# macOS: Quartz (CGEventTap)
if IS_MACOS:
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
        QUARTZ_AVAILABLE = True
    except ImportError:
        QUARTZ_AVAILABLE = False
else:
    QUARTZ_AVAILABLE = False

# Cross-platform: pynput (Linux + Windows input capture)
try:
    from pynput import mouse as pynput_mouse
    from pynput import keyboard as pynput_keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

# Cross-platform: mss (fast screenshots, works on all 3 platforms)
try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False


# ── Modifier flag decoding (macOS) ───────────────────────────────────────────

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


# ── macOS keycode mapping ────────────────────────────────────────────────────

MAC_KEYCODE_MAP = {
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
    63: "fn", 123: "left_arrow", 124: "right_arrow",
    125: "down_arrow", 126: "up_arrow",
    122: "f1", 120: "f2", 99: "f3", 118: "f4",
    96: "f5", 97: "f6", 98: "f7", 100: "f8",
    101: "f9", 109: "f10", 103: "f11", 111: "f12",
}


def mac_keycode_to_char(keycode: int) -> str:
    return MAC_KEYCODE_MAP.get(keycode, f"keycode_{keycode}")


# ── Frontmost app detection ──────────────────────────────────────────────────

def get_frontmost_app() -> dict:
    """Get the frontmost app name, bundle ID/WM_CLASS, and PID."""
    if IS_MACOS:
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

    elif IS_LINUX:
        try:
            # Try xdotool for active window info
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                win_name = result.stdout.strip()
                # Get WM_CLASS
                result2 = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowclassname"],
                    capture_output=True, text=True, timeout=2,
                )
                wm_class = result2.stdout.strip() if result2.returncode == 0 else ""
                # Get PID
                result3 = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowpid"],
                    capture_output=True, text=True, timeout=2,
                )
                pid = int(result3.stdout.strip()) if result3.returncode == 0 and result3.stdout.strip().isdigit() else 0
                return {"name": win_name, "bundle_id": wm_class, "pid": pid}
        except (FileNotFoundError, Exception):
            pass
        return {"name": "Unknown", "bundle_id": "", "pid": 0}

    elif IS_WINDOWS:
        try:
            import win32gui
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                title = win32gui.GetWindowText(hwnd)
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                return {"name": title or "Unknown", "bundle_id": "", "pid": pid}
        except Exception:
            pass
        return {"name": "Unknown", "bundle_id": "", "pid": 0}

    return {"name": "Unknown", "bundle_id": "", "pid": 0}


def get_mouse_location() -> tuple[float, float]:
    """Get current mouse cursor position (global coordinates)."""
    if IS_MACOS:
        try:
            event = Quartz.CGEventCreate(None)
            loc = Quartz.CGEventGetLocation(event)
            return (loc.x, loc.y)
        except Exception:
            return (0.0, 0.0)
    elif IS_LINUX:
        try:
            result = subprocess.run(
                ["xdotool", "getmouselocation"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                # Output: "x:123 y:456 screen:0 window:789"
                parts = result.stdout.strip().split()
                x = float(parts[0].split(":")[1]) if len(parts) > 0 else 0
                y = float(parts[1].split(":")[1]) if len(parts) > 1 else 0
                return (x, y)
        except (FileNotFoundError, Exception):
            pass
        return (0.0, 0.0)
    elif IS_WINDOWS:
        try:
            import win32api
            pos = win32api.GetCursorPos()
            return (float(pos[0]), float(pos[1]))
        except Exception:
            return (0.0, 0.0)
    return (0.0, 0.0)


# ── Screenshot capture ───────────────────────────────────────────────────────

# Shared MSS instance (created on first use)
_mss_instance = None


def capture_screenshot(output_path: str) -> bool:
    """Capture a screenshot to the given path."""
    # Try MSS first (cross-platform, fast)
    if MSS_AVAILABLE:
        global _mss_instance
        try:
            if _mss_instance is None:
                _mss_instance = mss.mss()
            monitor = _mss_instance.monitors[0]  # all monitors
            shot = _mss_instance.grab(monitor)
            mss.tools.to_png(shot.rgb, shot.size, output=output_path)
            return os.path.exists(output_path)
        except Exception:
            pass

    # Fallback to platform-specific tools
    if IS_MACOS:
        try:
            result = subprocess.run(
                ["screencapture", "-x", "-t", "png", output_path],
                capture_output=True, timeout=5,
            )
            return result.returncode == 0 and os.path.exists(output_path)
        except Exception:
            return False

    elif IS_LINUX:
        for cmd in [
            ["scrot", "-o", output_path],
            ["gnome-screenshot", "-f", output_path],
            ["import", "-window", "root", output_path],  # ImageMagick
        ]:
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=5)
                if result.returncode == 0 and os.path.exists(output_path):
                    return True
            except (FileNotFoundError, Exception):
                continue
        return False

    elif IS_WINDOWS:
        try:
            import PIL.ImageGrab
            img = PIL.ImageGrab.grab()
            img.save(output_path, "PNG")
            return os.path.exists(output_path)
        except Exception:
            return False

    return False


# ── AX / AT tree capture ─────────────────────────────────────────────────────

def capture_ax_tree(output_path: str, app_name: str | None = None) -> bool:
    """Capture the accessibility tree."""
    if IS_MACOS:
        # Try cua-driver
        try:
            result = subprocess.run(
                ["cua-driver", "get_window_state"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout:
                with open(output_path, "w") as f:
                    f.write(result.stdout)
                return True
        except (FileNotFoundError, Exception):
            pass

        # Fallback: osascript
        try:
            script = (
                'tell application "System Events"\n'
                'set frontApp to first application process whose frontmost is true\n'
                'set appName to name of frontApp\n'
                'set windowList to ""\n'
                'repeat with w in windows of frontApp\n'
                'set windowList to windowList & name of w & "\\n"\n'
                'end repeat\n'
                'return appName & "\\n" & windowList\n'
                'end tell'
            )
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=5,
            )
            with open(output_path, "w") as f:
                f.write(f"# AX Tree (osascript fallback)\n\n")
                f.write(f"Frontmost app: {result.stdout.strip() if result.returncode == 0 else 'Unknown'}\n")
            return True
        except Exception:
            pass

    elif IS_LINUX:
        # Try pyatspi (GNOME/AT-SPI2)
        try:
            import pyatspi
            desktop = pyatspi.Registry.getDesktop(0)
            with open(output_path, "w") as f:
                f.write(f"# AT Tree (pyatspi)\n\n")
                for i in range(desktop.childCount):
                    app = desktop[i]
                    f.write(f"App: {app.name} (role: {app.getRoleName()})\n")
                    if app.childCount > 0:
                        for j in range(min(app.childCount, 5)):  # top 5 windows
                            child = app[j]
                            f.write(f"  Window: {child.name} (role: {child.getRoleName()})\n")
            return True
        except ImportError:
            pass
        except Exception:
            pass

        # Fallback: xdotool for basic window info
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=2,
            )
            result2 = subprocess.run(
                ["xdotool", "search", "", "getwindowname"],
                capture_output=True, text=True, timeout=2,
            )
            with open(output_path, "w") as f:
                f.write("# AT Tree (xdotool fallback)\n\n")
                f.write(f"Active window: {result.stdout.strip() if result.returncode == 0 else 'Unknown'}\n")
                if result2.returncode == 0:
                    f.write(f"Windows:\n{result2.stdout}")
            return True
        except Exception:
            pass

    elif IS_WINDOWS:
        # Try uiautomation
        try:
            import uiautomation as uia
            root = uia.GetRootControl()
            with open(output_path, "w") as f:
                f.write("# UI Tree (uiautomation)\n\n")
                f.write(f"Desktop children:\n")
                for child in root.GetChildren():
                    f.write(f"  {child.Name} (type: {child.ControlTypeName})\n")
            return True
        except ImportError:
            pass
        except Exception:
            pass

    # Final fallback
    with open(output_path, "w") as f:
        f.write(f"# AX Tree unavailable on {PLATFORM}\n")
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
        for d in [self.screenshot_dir, self.ax_dir, self.events_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self._events_fh = open(self.events_file, "w", buffering=1)

    def log_event(self, event: dict):
        event["timestamp"] = time.time() - self.start_time
        event["wall_time"] = datetime.now().isoformat()
        with self.events_lock:
            if self._events_fh:
                self._events_fh.write(json.dumps(event) + "\n")
                self.event_count += 1

    def capture_snapshot(self):
        num = self.screenshot_count + 1
        shot_path = str(self.screenshot_dir / f"{num:04d}.png")
        ax_path = str(self.ax_dir / f"{num:04d}.txt")

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

        current_app = app_info.get("name", "")
        if self.last_app and current_app != self.last_app:
            self.log_event({
                "type": "app_switch",
                "from": self.last_app,
                "to": current_app,
            })
        self.last_app = current_app

    def close(self):
        if self._events_fh:
            self._events_fh.close()
        metadata = {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "duration_seconds": time.time() - self.start_time,
            "event_count": self.event_count,
            "screenshot_count": self.screenshot_count,
            "interval": self.interval,
            "platform": PLATFORM,
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


# ── macOS: CGEventTap recording ─────────────────────────────────────────────

_state: RecordingState | None = None


def mac_event_callback(proxy, event_type, event, refcon):
    """CGEventTap callback — runs on the CoreFoundation run loop thread."""
    global _state
    if _state is None or not _state.recording:
        return event

    try:
        flags = Quartz.CGEventGetFlags(event)
        modifiers = decode_flags(flags)
        mouse_loc = Quartz.CGEventGetLocation(event)
        evt = {"type": "", "modifiers": modifiers, "flags": flags}

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
            evt["type"] = "mouse_moved" if event_type == kCGEventMouseMoved else "mouse_dragged"
            evt["position"] = [mouse_loc.x, mouse_loc.y]

        elif event_type == kCGEventScrollWheel:
            delta_y = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGScrollWheelEventDeltaAxis1)
            delta_x = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGScrollWheelEventDeltaAxis2)
            evt["type"] = "scroll"
            evt["position"] = [mouse_loc.x, mouse_loc.y]
            evt["delta_y"] = delta_y
            evt["delta_x"] = delta_x
            evt["direction"] = "down" if delta_y > 0 else "up" if delta_y < 0 else "none"

        elif event_type in (kCGEventKeyDown, kCGEventKeyUp):
            keycode = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventKeycode)
            char = mac_keycode_to_char(keycode)
            evt["type"] = "key_down" if event_type == kCGEventKeyDown else "key_up"
            evt["keycode"] = keycode
            evt["key"] = char
            try:
                chars = Quartz.CGEventKeyboardGetUnicodeString(event, 100)
                if chars:
                    evt["character"] = chars[:10]
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
        if _state:
            _state.log_event({"type": "error", "error": str(e)})

    return event


def run_macos_recording(state: RecordingState):
    """Run recording on macOS using CGEventTap."""
    if not QUARTZ_AVAILABLE:
        print("ERROR: pyobjc-framework-Quartz is required on macOS.", file=sys.stderr)
        print("Install with: pip install pyobjc-framework-Quartz", file=sys.stderr)
        sys.exit(1)

    event_mask = (
        CGEventMaskBit(kCGEventLeftMouseDown) | CGEventMaskBit(kCGEventLeftMouseUp)
        | CGEventMaskBit(kCGEventRightMouseDown) | CGEventMaskBit(kCGEventRightMouseUp)
        | CGEventMaskBit(kCGEventOtherMouseDown) | CGEventMaskBit(kCGEventOtherMouseUp)
        | CGEventMaskBit(kCGEventMouseMoved) | CGEventMaskBit(kCGEventLeftMouseDragged)
        | CGEventMaskBit(kCGEventRightMouseDragged) | CGEventMaskBit(kCGEventOtherMouseDragged)
        | CGEventMaskBit(kCGEventScrollWheel)
        | CGEventMaskBit(kCGEventKeyDown) | CGEventMaskBit(kCGEventKeyUp)
        | CGEventMaskBit(kCGEventFlagsChanged)
    )

    tap = CGEventTapCreate(
        kCGSessionEventTap, kCGHeadInsertEventTap,
        kCGEventTapOptionListenOnly,  # listen-only: don't block or modify events
        event_mask, mac_event_callback, None,
    )

    if tap is None:
        print("ERROR: Failed to create CGEventTap.", file=sys.stderr)
        print("Grant: System Settings > Privacy & Security > Accessibility", file=sys.stderr)
        print("       System Settings > Privacy & Security > Screen Recording", file=sys.stderr)
        sys.exit(1)

    source = CFMachPortCreateRunLoopSource(None, tap, 0)
    run_loop = CFRunLoopGetCurrent()
    CFRunLoopAddSource(run_loop, source, kCFRunLoopCommonModes)
    CGEventTapEnable(tap, True)

    # Snapshot thread
    snap_thread = threading.Thread(target=snapshot_loop, args=(state,), daemon=True)
    snap_thread.start()

    # Stop watcher thread
    stop_flag = state.output_dir / ".stop"

    def stop_watcher():
        while state.recording:
            if stop_flag.exists():
                state.recording = False
                CFRunLoopStop(run_loop)
                return
            time.sleep(0.5)
        CFRunLoopStop(run_loop)

    stop_thread = threading.Thread(target=stop_watcher, daemon=True)
    stop_thread.start()

    # Signal handler
    def signal_handler(sig, frame):
        print("\n⏹  Stopping recording...", file=sys.stderr)
        state.recording = False
        stop_flag.touch()
        CFRunLoopStop(run_loop)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"🔴 Recording started (macOS / CGEventTap)", file=sys.stderr)
    print(f"   Stop: touch {stop_flag}  or  kill -INT {os.getpid()}", file=sys.stderr)

    CFRunLoopRun()

    state.recording = False
    snap_thread.join(timeout=3)
    state.close()
    if stop_flag.exists():
        stop_flag.unlink()


# ── Linux/Windows: pynput recording ─────────────────────────────────────────

def run_pynput_recording(state: RecordingState):
    """Run recording on Linux or Windows using pynput."""
    if not PYNPUT_AVAILABLE:
        print(f"ERROR: pynput is required on {PLATFORM}.", file=sys.stderr)
        print("Install with: pip install pynput mss", file=sys.stderr)
        sys.exit(1)

    # Track current modifiers for key combos
    current_mods = set()

    # ── Mouse listener ──
    def on_click(x, y, button, pressed):
        if not state.recording:
            return False
        btn_name = {pynput_mouse.Button.left: "left",
                    pynput_mouse.Button.right: "right",
                    pynput_mouse.Button.middle: "middle"}.get(button, str(button))
        state.log_event({
            "type": "mouse_down" if pressed else "mouse_up",
            "button": btn_name,
            "position": [float(x), float(y)],
            "modifiers": list(current_mods),
        })

    def on_scroll(x, y, dx, dy):
        if not state.recording:
            return False
        state.log_event({
            "type": "scroll",
            "position": [float(x), float(y)],
            "delta_x": int(dx),
            "delta_y": int(dy),
            "direction": "down" if dy > 0 else "up" if dy < 0 else "none",
            "modifiers": list(current_mods),
        })

    _last_drag_pos = [None]
    _dragging = [False]

    def on_move(x, y):
        if not state.recording:
            return False
        # Only log drags, not every mouse move (too noisy)
        if _dragging[0] and _last_drag_pos[0]:
            state.log_event({
                "type": "mouse_dragged",
                "from": list(_last_drag_pos[0]),
                "to": [float(x), float(y)],
                "modifiers": list(current_mods),
            })
        _last_drag_pos[0] = (float(x), float(y))

    # ── Keyboard listener ──
    mod_keys = {
        pynput_keyboard.Key.shift, pynput_keyboard.Key.shift_l, pynput_keyboard.Key.shift_r,
        pynput_keyboard.Key.ctrl, pynput_keyboard.Key.ctrl_l, pynput_keyboard.Key.ctrl_r,
        pynput_keyboard.Key.alt, pynput_keyboard.Key.alt_l, pynput_keyboard.Key.alt_r,
        pynput_keyboard.Key.cmd, pynput_keyboard.Key.cmd_l, pynput_keyboard.Key.cmd_r,
        pynput_keyboard.Key.space,
    }

    mod_names = {
        pynput_keyboard.Key.shift: "shift", pynput_keyboard.Key.shift_l: "shift",
        pynput_keyboard.Key.shift_r: "shift",
        pynput_keyboard.Key.ctrl: "control", pynput_keyboard.Key.ctrl_l: "control",
        pynput_keyboard.Key.ctrl_r: "control",
        pynput_keyboard.Key.alt: "option", pynput_keyboard.Key.alt_l: "option",
        pynput_keyboard.Key.alt_r: "option",
        pynput_keyboard.Key.cmd: "command", pynput_keyboard.Key.cmd_l: "command",
        pynput_keyboard.Key.cmd_r: "command",
    }

    def on_press(key):
        if not state.recording:
            return False

        # Track modifiers
        if key in mod_keys:
            mod_name = mod_names.get(key, str(key))
            current_mods.add(mod_name)
            return  # Don't log standalone modifier presses

        # Get key name
        try:
            char = key.char
            key_name = char
        except AttributeError:
            key_name = str(key).replace("Key.", "")
            char = None

        state.log_event({
            "type": "key_down",
            "key": key_name,
            "character": char,
            "modifiers": list(current_mods),
        })

        # Track drag start (mouse button held + movement)
        if key == pynput_mouse.Button.left:
            _dragging[0] = True

    def on_release(key):
        if not state.recording:
            return False

        if key in mod_keys:
            mod_name = mod_names.get(key, str(key))
            current_mods.discard(mod_name)
            return

        try:
            char = key.char
            key_name = char
        except AttributeError:
            key_name = str(key).replace("Key.", "")
            char = None

        state.log_event({
            "type": "key_up",
            "key": key_name,
            "character": char,
            "modifiers": list(current_mods),
        })

    # Start listeners
    mouse_listener = pynput_mouse.Listener(
        on_click=on_click, on_scroll=on_scroll, on_move=on_move,
    )
    keyboard_listener = pynput_keyboard.Listener(
        on_press=on_press, on_release=on_release,
    )

    # Snapshot thread
    snap_thread = threading.Thread(target=snapshot_loop, args=(state,), daemon=True)

    # Stop watcher
    stop_flag = state.output_dir / ".stop"

    def stop_watcher():
        while state.recording:
            if stop_flag.exists():
                state.recording = False
                mouse_listener.stop()
                keyboard_listener.stop()
                return
            time.sleep(0.5)

    stop_thread = threading.Thread(target=stop_watcher, daemon=True)

    # Signal handler
    def signal_handler(sig, frame):
        print("\n⏹  Stopping recording...", file=sys.stderr)
        state.recording = False
        stop_flag.touch()
        mouse_listener.stop()
        keyboard_listener.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start everything
    mouse_listener.start()
    keyboard_listener.start()
    snap_thread.start()
    stop_thread.start()

    print(f"🔴 Recording started ({PLATFORM} / pynput)", file=sys.stderr)
    print(f"   Stop: touch {stop_flag}  or  kill -INT {os.getpid()}", file=sys.stderr)

    # Block until recording stops
    try:
        while state.recording:
            time.sleep(0.2)
    except KeyboardInterrupt:
        state.recording = False

    mouse_listener.stop()
    keyboard_listener.stop()
    snap_thread.join(timeout=3)
    state.close()

    if stop_flag.exists():
        stop_flag.unlink()


# ── Snapshot thread ──────────────────────────────────────────────────────────

def snapshot_loop(state: RecordingState):
    """Background thread that captures periodic screenshots + AX trees."""
    while state.recording:
        state.capture_snapshot()
        time.sleep(state.interval)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=f"Record {PLATFORM} workflow for Hermes Record & Replay"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Screenshot interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: ~/.hermes/recordings/<timestamp>)",
    )
    parser.add_argument(
        "--no-ax", action="store_true",
        help="Skip AX tree capture (faster, less detail)",
    )
    args = parser.parse_args()

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

    start_time = time.time()

    # Run platform-specific recording
    if IS_MACOS:
        run_macos_recording(_state)
    elif IS_LINUX or IS_WINDOWS:
        run_pynput_recording(_state)
    else:
        print(f"ERROR: Unsupported platform: {PLATFORM}", file=sys.stderr)
        sys.exit(1)

    print(f"\n✅ Recording saved to: {output_dir}", file=sys.stderr)
    print(f"   Events captured: {_state.event_count}", file=sys.stderr)
    print(f"   Screenshots: {_state.screenshot_count}", file=sys.stderr)
    print(f"   Duration: {time.time() - start_time:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
