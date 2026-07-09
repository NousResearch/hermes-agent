from __future__ import annotations

import ctypes
import os
import time
from ctypes import POINTER, Structure, Union, WinDLL, c_long, c_int, c_uint, c_ulong, c_ushort
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


user32 = WinDLL("user32", use_last_error=True)
kernel32 = WinDLL("kernel32", use_last_error=True)


def _collect_machine_vitals() -> dict:
    try:
        import psutil  # type: ignore  # optional

        return {
            "cpu_percent": float(psutil.cpu_percent(interval=None)),
            "memory_percent": float(psutil.virtual_memory().percent),
            "source": "psutil",
        }
    except Exception:
        pass
    try:
        load1, load5, load15 = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
        return {
            "cpu_percent": load1 * 100.0,
            "memory_percent": 0.0,
            "source": "os-loadavg-fallback",
        }
    except Exception:
        pass
    return {"cpu_percent": 0.0, "memory_percent": 0.0, "source": "none"}


class _MOUSEINPUT(Structure):
    _fields_ = [
        ("dx", c_long),
        ("dy", c_long),
        ("mouseData", c_ulong),
        ("dwFlags", c_ulong),
        ("time", c_ulong),
        ("dwExtraInfo", ctypes.c_ulonglong),
    ]


class _KEYBDINPUT(Structure):
    _fields_ = [
        ("wVk", c_ushort),
        ("wScan", c_ushort),
        ("dwFlags", c_ulong),
        ("time", c_ulong),
        ("dwExtraInfo", ctypes.c_ulonglong),
    ]


class _HARDWAREINPUT(Structure):
    _fields_ = [
        ("uMsg", c_ulong),
        ("wParamL", c_ushort),
        ("wParamH", c_ushort),
    ]


class _INPUT_UNION(Union):
    _fields_ = [
        ("mi", _MOUSEINPUT),
        ("ki", _KEYBDINPUT),
        ("hi", _HARDWAREINPUT),
    ]


class _INPUT(Structure):
    _fields_ = [("type", c_ulong), ("union", _INPUT_UNION)]


INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_LWIN = 0x5B
VK_F5 = 0x74
VK_F11 = 0x7A
VK_F12 = 0x7B
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_LMENU = 0xA4
VK_RMENU = 0xA5

KEY_ALIASES = {
    "enter": VK_RETURN,
    "return": VK_RETURN,
    "esc": VK_ESCAPE,
    "escape": VK_ESCAPE,
    "space": VK_SPACE,
    "tab": VK_TAB,
    "backspace": VK_BACK,
    "shift": VK_SHIFT,
    "ctrl": VK_CONTROL,
    "control": VK_CONTROL,
    "alt": VK_MENU,
    "left": VK_LEFT,
    "up": VK_UP,
    "right": VK_RIGHT,
    "down": VK_DOWN,
    "win": VK_LWIN,
    "lwin": VK_LWIN,
    "rwin": 0x5C,
    "f5": VK_F5,
    "f11": VK_F11,
    "f12": VK_F12,
    "lctrl": VK_LCONTROL,
    "rctrl": VK_RCONTROL,
    "lshift": VK_LSHIFT,
    "rshift": VK_RSHIFT,
    "lalt": VK_LMENU,
    "ralt": VK_RMENU,
}


class ActionKind(Enum):
    MOVE = "move"
    CLICK = "click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    SCROLL = "scroll"
    TYPE = "type"
    PRESS = "press"
    HOTKEY = "hotkey"
    SCREENSHOT = "screenshot"
    FULLSCREEN = "fullscreen"
    FOCUS = "focus"


@dataclass
class ActionResult:
    ok: bool
    action: str
    rc: int = 0
    stdout: str = ""
    stderr: str = ""
    machine: dict = field(default_factory=dict)
    surface: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "rc": self.rc,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "action": self.action,
            "machine": self.machine,
            "surface": self.surface,
        }


@dataclass
class NativeBackend:
    max_x: int = 0
    max_y: int = 0
    _last_move: Tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        self.max_x = max(1, int(user32.GetSystemMetrics(0)))
        self.max_y = max(1, int(user32.GetSystemMetrics(1)))

    def _normalize(self, x: int, y: int) -> Tuple[int, int]:
        return (
            max(0, min(x, self.max_x)),
            max(0, min(y, self.max_y)),
        )

    def move(self, x: int, y: int, duration: float = 0.0) -> ActionResult:
        target_x, target_y = self._normalize(x, y)
        start = self._last_move
        start_time = time.time()
        steps = max(1, int(duration * 10))
        if steps <= 1 or duration <= 0:
            user32.SetCursorPos(target_x, target_y)
            self._last_move = (target_x, target_y)
            return ActionResult(
                ok=True,
                action="move",
                stdout=f"{target_x},{target_y}",
                surface={"kind": "computer_use", "backend": "native", "action": "move"},
            )
        for idx in range(steps + 1):
            frac = idx / steps
            cur_x = int(start[0] + (target_x - start[0]) * frac)
            cur_y = int(start[1] + (target_y - start[1]) * frac)
            user32.SetCursorPos(cur_x, cur_y)
            time.sleep(duration / steps)
        user32.SetCursorPos(target_x, target_y)
        self._last_move = (target_x, target_y)
        return ActionResult(
            ok=True,
            action="move",
            stdout=f"{target_x},{target_y}",
            surface={"kind": "computer_use", "backend": "native", "action": "move"},
        )

    @staticmethod
    def _send_mouse(flags: int, x: int = 0, y: int = 0, data: int = 0) -> bool:
        inp = _INPUT(
            type=c_ulong(INPUT_MOUSE),
            union=_INPUT_UNION(
                mi=_MOUSEINPUT(
                    dx=c_long(x),
                    dy=c_long(y),
                    mouseData=c_ulong(data),
                    dwFlags=c_ulong(flags),
                    time=c_ulong(0),
                    dwExtraInfo=ctypes.c_ulonglong(0),
                )
            ),
        )
        sent = user32.SendInput(c_ulong(1), ctypes.byref(inp), ctypes.sizeof(_INPUT))
        return sent == 1

    def click(self, button: str = "left", count: int = 1) -> ActionResult:
        if button == "right":
            flags = [MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP]
        elif button == "middle":
            mid = 0x0010
            flags = [mid | 0x0001, mid]
        else:
            flags = [MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP]
        for _ in range(max(1, count)):
            if not self._send_mouse(flags[0]):
                return ActionResult(ok=False, action="click", rc=1, stderr="SendInput down failed")
            time.sleep(0.01)
            if not self._send_mouse(flags[1]):
                return ActionResult(ok=False, action="click", rc=2, stderr="SendInput up failed")
            time.sleep(0.02 if count > 1 else 0.0)
        return ActionResult(
            ok=True,
            action="click",
            stdout=f"{button}:{count}",
            surface={"kind": "computer_use", "backend": "native", "action": "click", "button": button, "count": count},
        )

    def scroll(self, clicks: int) -> ActionResult:
        ok = self._send_mouse(MOUSEEVENTF_WHEEL, data=c_ulong(max(-1, min(1, clicks)) * 120).value)
        return ActionResult(
            ok=bool(ok),
            action="scroll",
            stdout=str(clicks),
            rc=0 if ok else 1,
            surface={"kind": "computer_use", "backend": "native", "action": "scroll"},
        )

    def _vk(self, key: str) -> int:
        if len(key) == 1:
            return ctypes.windll.user32.VkKeyScanW(ord(key)) & 0xFF
        vk = KEY_ALIASES.get(key.lower())
        if vk is None:
            vk = int(str(key), 0)
        return vk

    def press(self, key: str) -> ActionResult:
        vk = self._vk(key)
        ok_down = user32.keybd_event(c_ushort(vk), c_ushort(0), c_ulong(0), c_ulong(0))
        time.sleep(0.02)
        ok_up = user32.keybd_event(c_ushort(vk), c_ushort(0), c_ulong(KEYEVENTF_KEYUP), c_ulong(0))
        ok = ok_down is not None and ok_up is not None
        return ActionResult(
            ok=bool(ok),
            action="press",
            stdout=key,
            rc=0 if ok else 1,
            surface={"kind": "computer_use", "backend": "native", "action": "press", "key": key},
        )

    def type_text(self, text: str, cpm: int = 900) -> ActionResult:
        delay = 60.0 / max(1, cpm)
        for ch in text:
            if ch == "\n":
                self.press("enter")
                continue
            if ch == "\t":
                self.press("tab")
                continue
            inp = _INPUT(
                type=c_ulong(INPUT_KEYBOARD),
                union=_INPUT_UNION(
                    ki=_KEYBDINPUT(
                        wVk=c_ushort(0),
                        wScan=c_ushort(ord(ch)),
                        dwFlags=c_ulong(KEYEVENTF_UNICODE),
                        time=c_ulong(0),
                        dwExtraInfo=ctypes.c_ulonglong(0),
                    )
                ),
            )
            user32.SendInput(c_ulong(1), ctypes.byref(inp), ctypes.sizeof(_INPUT))
            time.sleep(delay)
        return ActionResult(
            ok=True,
            action="type",
            stdout=text,
            surface={"kind": "computer_use", "backend": "native", "action": "type"},
        )

    def hotkey(self, keys: Tuple[str, ...]) -> ActionResult:
        vks = [self._vk(key) for key in keys]
        for vk in vks:
            user32.keybd_event(c_ushort(vk), c_ushort(0), c_ulong(0), c_ulong(0))
            time.sleep(0.01)
        for vk in reversed(vks):
            user32.keybd_event(c_ushort(vk), c_ushort(0), c_ulong(KEYEVENTF_KEYUP), c_ulong(0))
            time.sleep(0.01)
        return ActionResult(
            ok=True,
            action="hotkey",
            stdout="+".join(keys),
            surface={"kind": "computer_use", "backend": "native", "action": "hotkey", "keys": list(keys)},
        )

    def screenshot(self, path: str = "") -> ActionResult:
        try:
            import mss  # type: ignore  # optional
        except Exception as exc:
            return ActionResult(ok=False, action="screenshot", rc=1, stderr=f"mss unavailable: {exc}")
        if not path:
            path = str(kernel32.GetTempPathW(260)) + f"hermes_screenshot_{int(time.time())}.png"
        with mss.mss() as sct:
            sct.shot(output=path)
        return ActionResult(
            ok=True,
            action="screenshot",
            stdout=path,
            surface={"kind": "computer_use", "backend": "native", "action": "screenshot"},
        )


@dataclass
class ComputerUseCommandCenter:
    """Victus superagent computer-use surface."""
    backend: str = "native"
    max_steps: int = 32
    dry_run: bool = False
    action_delay: float = 0.05
    _backend: Optional[NativeBackend] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._backend is None:
            self._backend = NativeBackend()

    def execute(self, action: ActionKind, payload: dict) -> ActionResult:
        if self.dry_run:
            surface = {
                "kind": "computer_use",
                "backend": self.backend,
                "mode": "dry_run",
                "action": action.value,
            }
            for key in ["button", "count", "x", "y", "clicks", "text", "title", "duration", "keys"]:
                if key in payload:
                    surface[key] = payload[key]
            return ActionResult(
                ok=True,
                action=action.value,
                stdout=str(payload),
                surface=surface,
            )
        machine = _collect_machine_vitals()
        try:
            if action == ActionKind.MOVE:
                x = int(payload.get("x") or 0)
                y = int(payload.get("y") or 0)
                result = self._backend.move(x, y, duration=float(payload.get("duration") or 0.0))
            elif action == ActionKind.CLICK:
                result = self._backend.click(
                    button=str(payload.get("button") or "left"),
                    count=int(payload.get("count") or 1),
                )
            elif action == ActionKind.RIGHT_CLICK:
                result = self._backend.click(button="right")
            elif action == ActionKind.DOUBLE_CLICK:
                result = self._backend.click(count=2)
            elif action == ActionKind.SCROLL:
                result = self._backend.scroll(int(payload.get("clicks") or 1))
            elif action == ActionKind.TYPE:
                result = self._backend.type_text(str(payload.get("text") or ""), cpm=int(payload.get("cpm") or 900))
            elif action == ActionKind.PRESS:
                result = self._backend.press(str(payload.get("key") or "enter"))
            elif action == ActionKind.HOTKEY:
                keys = tuple(str(k) for k in (payload.get("keys") or ["ctrl", "c"]))
                result = self._backend.hotkey(keys)
            elif action == ActionKind.FOCUS:
                title = str(payload.get("title") or "")
                result = self._focus_window(title)
            elif action == ActionKind.FULLSCREEN:
                result = self._toggle_fullscreen()
            else:
                return ActionResult(ok=False, action=action.value, rc=2, stderr=f"unsupported action: {action.value}")
        except Exception as exc:
            return ActionResult(ok=False, action=action.value, rc=3, stderr=f"{exc.__class__.__name__}: {exc}")
        result.machine = machine
        return result

    def _focus_window(self, title: str) -> ActionResult:
        hwnd = user32.FindWindowW(None, title)
        if not hwnd:
            return ActionResult(ok=False, action="focus", rc=1, stderr=f"window not found: {title}")
        user32.SetForegroundWindow(hwnd)
        return ActionResult(ok=True, action="focus", stdout=title, surface={"kind": "computer_use", "backend": self.backend, "action": "focus", "title": title})

    def _toggle_fullscreen(self) -> ActionResult:
        user32.keybd_event(c_ushort(VK_F11), c_ushort(0), c_ulong(0), c_ulong(0))
        time.sleep(0.05)
        user32.keybd_event(c_ushort(VK_F11), c_ushort(0), c_ulong(KEYEVENTF_KEYUP), c_ulong(0))
        return ActionResult(ok=True, action="fullscreen", surface={"kind": "computer_use", "backend": self.backend, "action": "fullscreen"})

    def dispatch(self, raw: str) -> dict:
        raw = (raw or "").strip()
        if not raw.startswith("computer://"):
            return self._default_response("invalid computer command")
        body = raw[len("computer://"):].strip()
        action_str = (body.split()[0] if body else "").lower()
        try:
            action = ActionKind(action_str)
        except ValueError:
            return self._default_response(f"unknown action: {action_str}")
        rest = body[len(action_str):].strip()
        payload = self._parse_action_payload(action, rest)
        if action in {ActionKind.CLICK, ActionKind.RIGHT_CLICK, ActionKind.DOUBLE_CLICK, ActionKind.MOVE, ActionKind.SCROLL} and "x" not in payload and "y" not in payload:
            return self._default_response("requires x/y or target")
        result = self.execute(action, payload)
        output = result.to_dict()
        if result.ok and not result.stdout:
            output["stdout"] = self._build_stdout(action, payload)
        return output

    def _build_stdout(self, action: ActionKind, payload: dict) -> str:
        if action in {ActionKind.MOVE, ActionKind.CLICK, ActionKind.RIGHT_CLICK, ActionKind.DOUBLE_CLICK, ActionKind.SCROLL}:
            return f"{payload.get('x', '?')},{payload.get('y', '?')}"
        if action == ActionKind.PRESS:
            return str(payload.get("key") or "")
        if action == ActionKind.TYPE:
            return str(payload.get("text") or "")
        if action == ActionKind.FOCUS:
            return str(payload.get("title") or "")
        return action.value

    def _default_response(self, stderr: str) -> dict:
        return ActionResult(ok=False, action="error", rc=2, stderr=stderr).to_dict()

    def _parse_action_payload(self, action: ActionKind, rest: str) -> dict:
        payload: dict[str, object] = {"action": action.value}
        if not rest:
            return payload
        if action in {ActionKind.MOVE, ActionKind.CLICK, ActionKind.RIGHT_CLICK, ActionKind.DOUBLE_CLICK, ActionKind.SCROLL}:
            coords = [part.strip() for part in rest.split(",") if part.strip()]
            if coords:
                first = coords[0].lower()
                if first in {"left", "right", "middle"}:
                    payload["button"] = first
                    if len(coords) > 1:
                        try:
                            payload["x"] = int(coords[1])
                        except Exception:
                            pass
                    if len(coords) > 2:
                        try:
                            payload["y"] = int(coords[2])
                        except Exception:
                            pass
                    if action == ActionKind.SCROLL:
                        payload["clicks"] = int(payload.get("y") or 1)
                else:
                    try:
                        payload["x"] = int(coords[0])
                    except Exception:
                        pass
                    if len(coords) > 1:
                        try:
                            payload["y"] = int(coords[1])
                        except Exception:
                            pass
                    if action == ActionKind.SCROLL:
                        payload["clicks"] = int(payload.get("x") or 1)
        else:
            payload["text"] = rest
        return payload
