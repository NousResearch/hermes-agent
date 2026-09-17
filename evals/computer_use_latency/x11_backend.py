"""Real-X11 test backend for the computer_use `sequence` latency eval (eval-only, never shipped).

Drives REAL pixels and input on Xvfb: mss screenshots, XTest pointer/keyboard events, Xlib focus,
against tkinter apps created in-process. It is NOT cua-driver: no AX tree, no pid-scoped event
posting, no screenshot-dedup or aux-vision routing inside the backend itself. The A/B/C comparison
stays honest because every arm goes through the identical tool-layer path
(``handle_computer_use`` -> ``_dispatch`` -> backend); only the orchestration differs, which is
exactly what the eval measures. Absolute milliseconds are Xvfb-box-specific; the arm deltas are
the evidence.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.backend import ActionResult, CaptureResult, ComputerUseBackend, UIElement

# X11 keysym names for typeable punctuation (letters/digits derive from the char itself).
_PUNCT_KEYS = {
    " ": ("space", False), ".": ("period", False), ",": ("comma", False),
    "-": ("minus", False), "_": ("minus", True), "@": ("at", True),
    ":": ("colon", False), "/": ("slash", False),
}
# key-action combo token -> X11 keysym name.
_KEY_NAMES = {
    "return": "Return", "enter": "Return", "tab": "Tab", "escape": "Escape",
    "space": "space", "backspace": "BackSpace", "delete": "Delete",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "ctrl": "Control_L", "control": "Control_L", "shift": "Shift_L",
    "alt": "Alt_L", "super": "Super_L", "cmd": "Super_L", "meta": "Super_L",
}


def _char_keysym(ch: str) -> Tuple[str, bool]:
    """(keysym name, shift?) for a typeable char."""
    if ch.isalpha():
        return ch.upper(), ch.isupper()
    if ch.isdigit():
        return ch, False
    try:
        return _PUNCT_KEYS[ch]
    except KeyError:
        raise ValueError(f"eval backend cannot type {ch!r}")


def _require_display() -> None:
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("evals/computer_use_latency needs DISPLAY (run under Xvfb, e.g. DISPLAY=:99)")


@dataclass
class EvalWidget:
    widget: Any  # tkinter widget
    role: str    # AX-style role surfaced to the policy
    label: str


@dataclass
class EvalApp:
    name: str
    root: Any  # tkinter Tk
    widgets: List[EvalWidget] = field(default_factory=list)


class X11EvalBackend(ComputerUseBackend):
    """Minimal real backend: screenshots via mss, input via XTest, focus via Xlib."""

    def __init__(self) -> None:
        self._display = None
        self._apps: Dict[str, EvalApp] = {}
        self._elements: List[UIElement] = []
        self._last_app: Optional[str] = None
        self._last_target: Dict[str, Any] = {}
        # Exact counters for the eval's capture metrics.
        self.captures_taken = 0
        self.images_produced = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        _require_display()
        from Xlib.display import Display
        if self._display is None:
            self._display = Display()

    def stop(self) -> None:
        if self._display is not None:
            self._display.close()
            self._display = None

    def is_available(self) -> bool:
        return self._display is not None

    # -- app registration (eval harness only) -------------------------------
    def add_app(self, app: EvalApp) -> None:
        self._apps[app.name] = app

    def pump(self) -> None:
        for app in self._apps.values():
            app.root.update()

    def find_element(self, role: str = "", label_contains: str = "") -> int:
        """1-based index of the first element of the last capture matching role/label (what the
        scripted policy reads instead of parsing pixels — same data the capture returned)."""
        for el in self._elements:
            if role and el.role != role:
                continue
            if label_contains and label_contains not in el.label:
                continue
            return el.index
        raise LookupError(f"no element role={role!r} label~={label_contains!r} in last capture")

    # -- input helpers -------------------------------------------------------
    def _keycode(self, name: str) -> int:
        from Xlib import XK
        kc = self._display.keysym_to_keycode(XK.string_to_keysym(name))
        if not kc:
            raise ValueError(f"no keycode for keysym {name!r}")
        return kc

    def _press_key(self, name: str, shift: bool = False) -> None:
        from Xlib import X
        from Xlib.ext import xtest
        d = self._display
        if shift:
            xtest.fake_input(d, X.KeyPress, self._keycode("Shift_L"))
        xtest.fake_input(d, X.KeyPress, self._keycode(name))
        xtest.fake_input(d, X.KeyRelease, self._keycode(name))
        if shift:
            xtest.fake_input(d, X.KeyRelease, self._keycode("Shift_L"))
        d.sync()

    def _click_at(self, x: int, y: int, button: int = 1, count: int = 1) -> None:
        from Xlib import X
        from Xlib.ext import xtest
        d = self._display
        d.screen().root.warp_pointer(x, y)
        d.sync()
        time.sleep(0.05)
        for _ in range(count):
            xtest.fake_input(d, X.ButtonPress, button)
            d.sync()
            time.sleep(0.05)
            xtest.fake_input(d, X.ButtonRelease, button)
            d.sync()
        self.pump()

    def _resolve_point(self, element: Optional[int], x: Optional[int],
                       y: Optional[int]) -> Tuple[int, int]:
        if element is not None:
            for el in self._elements:
                if el.index == element:
                    bx, by, bw, bh = el.bounds
                    return bx + bw // 2, by + bh // 2
            raise LookupError(f"unknown element #{element}")
        if x is not None and y is not None:
            return int(x), int(y)
        raise ValueError("click needs element or x/y")

    # -- ComputerUseBackend ---------------------------------------------------
    def capture(self, mode: str = "som", app: Optional[str] = None,
                pid: Optional[int] = None, window_id: Optional[int] = None) -> CaptureResult:
        _require_display()
        import mss as _mss
        name = app or self._last_app or next(iter(self._apps), None)
        if name is None or name not in self._apps:
            raise RuntimeError(f"eval backend has no app {name!r}")
        root = self._apps[name].root
        gx, gy, gw, gh = root.winfo_rootx(), root.winfo_rooty(), root.winfo_width(), root.winfo_height()
        with _mss.mss() as sct:
            shot = sct.grab({"left": gx, "top": gy, "width": gw, "height": gh})
            png = _mss.tools.to_png(shot.rgb, shot.size)
        elements: List[UIElement] = []
        if mode in ("ax", "som"):
            elements = self._enumerate(name)
            self._elements = elements
        if mode == "som":
            png = _draw_badges(png, elements, gx, gy)
        self.captures_taken += 1
        png_b64 = None
        if mode != "ax":
            png_b64 = base64.b64encode(png).decode()
            self.images_produced += 1
        self._last_app = name
        self._last_target = {"pid": os.getpid(), "window_id": root.winfo_id()}
        return CaptureResult(mode=mode, width=gw, height=gh, png_b64=png_b64,
                             elements=elements, app=name, window_title=name,
                             png_bytes_len=len(png))

    def _enumerate(self, name: str) -> List[UIElement]:
        out = []
        for i, ew in enumerate(self._apps[name].widgets, start=1):
            w = ew.widget
            x, y = w.winfo_rootx(), w.winfo_rooty()
            out.append(UIElement(index=i, role=ew.role, label=ew.label,
                                 bounds=(x, y, w.winfo_width(), w.winfo_height()),
                                 app=name, pid=os.getpid()))
        return out

    def click(self, *, element: Optional[int] = None, x: Optional[int] = None,
              y: Optional[int] = None, button: str = "left", count: int = 1,
              **_: Any) -> ActionResult:
        try:
            px, py = self._resolve_point(element, x, y)
        except (LookupError, ValueError) as e:
            return ActionResult(ok=False, action="click", code="unknown_element", message=str(e))
        self._click_at(px, py, button={"left": 1, "middle": 2, "right": 3}.get(button, 1),
                       count=max(1, count or 1))
        time.sleep(0.15)
        self.pump()
        # Honest verdict: the backend posts events but never semantically confirms the effect.
        return ActionResult(ok=True, action="click", effect="unverifiable")

    def type_text(self, text: str, **_: Any) -> ActionResult:
        try:
            for ch in text:
                name, shift = _char_keysym(ch)
                self._press_key(name, shift)
        except ValueError as e:
            return ActionResult(ok=False, action="type", code="untypable_char", message=str(e))
        time.sleep(0.1)
        self.pump()
        return ActionResult(ok=True, action="type", effect="unverifiable")

    def key(self, keys: str, **_: Any) -> ActionResult:
        parts = [p.strip().lower() for p in keys.replace("+", " ").replace("-", " ").split() if p.strip()]
        if not parts:
            return ActionResult(ok=False, action="key", code="empty_combo", message="no keys given")
        *mods, main = parts
        try:
            mod_names = [_KEY_NAMES[m] for m in mods]
            main_name = _KEY_NAMES.get(main, main.upper() if len(main) == 1 else main)
            from Xlib import X
            from Xlib.ext import xtest
            d = self._display
            for m in mod_names:
                xtest.fake_input(d, X.KeyPress, self._keycode(m))
            self._press_key(main_name)
            for m in reversed(mod_names):
                xtest.fake_input(d, X.KeyRelease, self._keycode(m))
            d.sync()
        except (ValueError, KeyError) as e:
            return ActionResult(ok=False, action="key", code="bad_combo", message=str(e))
        time.sleep(0.1)
        self.pump()
        return ActionResult(ok=True, action="key", effect="unverifiable")

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        if app not in self._apps:
            return ActionResult(ok=False, action="focus_app", code="unknown_app",
                                message=f"no eval app {app!r}")
        from Xlib import X
        win_id = self._apps[app].root.winfo_id()
        win = self._display.create_resource_object("window", win_id)
        win.set_input_focus(X.RevertToParent, X.CurrentTime)
        if raise_window:
            win.configure(stack_mode=X.Above)
        self._display.sync()
        self._last_app = app
        return ActionResult(ok=True, action="focus_app", effect="unverifiable")

    def drag(self, *, from_element: Optional[int] = None, to_element: Optional[int] = None,
             from_coordinate: Optional[list] = None, to_coordinate: Optional[list] = None,
             **_: Any) -> ActionResult:
        try:
            x0, y0 = self._resolve_point(from_element,
                                         from_coordinate[0] if from_coordinate else None,
                                         from_coordinate[1] if from_coordinate else None)
            x1, y1 = self._resolve_point(to_element,
                                         to_coordinate[0] if to_coordinate else None,
                                         to_coordinate[1] if to_coordinate else None)
        except (LookupError, ValueError) as e:
            return ActionResult(ok=False, action="drag", code="unknown_element", message=str(e))
        from Xlib import X
        from Xlib.ext import xtest
        d = self._display
        d.screen().root.warp_pointer(x0, y0)
        d.sync()
        xtest.fake_input(d, X.ButtonPress, 1)
        d.sync()
        for i in range(1, 6):
            d.screen().root.warp_pointer(x0 + (x1 - x0) * i // 5, y0 + (y1 - y0) * i // 5)
            d.sync()
            time.sleep(0.02)
        xtest.fake_input(d, X.ButtonRelease, 1)
        d.sync()
        self.pump()
        return ActionResult(ok=True, action="drag", effect="unverifiable")

    def scroll(self, *, direction: str = "down", amount: int = 3,
               element: Optional[int] = None, x: Optional[int] = None,
               y: Optional[int] = None, **_: Any) -> ActionResult:
        try:
            px, py = self._resolve_point(element, x, y)
        except (LookupError, ValueError):
            name = self._last_app or next(iter(self._apps))
            r = self._apps[name].root
            px, py = r.winfo_rootx() + r.winfo_width() // 2, r.winfo_rooty() + r.winfo_height() // 2
        from Xlib import X
        from Xlib.ext import xtest
        d = self._display
        d.screen().root.warp_pointer(px, py)
        d.sync()
        button = {"up": 4, "down": 5, "left": 6, "right": 7}.get(direction, 5)
        for _ in range(max(1, amount or 1)):
            xtest.fake_input(d, X.ButtonPress, button)
            d.sync()
            xtest.fake_input(d, X.ButtonRelease, button)
            d.sync()
        self.pump()
        return ActionResult(ok=True, action="scroll", effect="unverifiable")

    def set_value(self, value: str, element: Optional[int] = None, **_: Any) -> ActionResult:
        target = None
        for el in self._elements:
            if el.index == element:
                target = self._apps[el.app].widgets[el.index - 1].widget
        if target is None:
            return ActionResult(ok=False, action="set_value", code="unknown_element",
                                message=f"no element #{element}")
        try:
            import tkinter as tk
            if isinstance(target, tk.Entry):
                target.delete(0, tk.END)
                target.insert(0, value)
            elif isinstance(target, tk.Text):
                target.delete("1.0", tk.END)
                target.insert("1.0", value)
            else:
                return ActionResult(ok=False, action="set_value", code="unsupported",
                                    message=f"eval backend cannot set_value on {type(target).__name__}")
        except Exception as e:
            return ActionResult(ok=False, action="set_value", code="set_failed", message=str(e))
        self.pump()
        return ActionResult(ok=True, action="set_value", effect="unverifiable")

    def list_apps(self) -> List[str]:
        return sorted(self._apps)


def _draw_badges(png: bytes, elements: List[UIElement], gx: int, gy: int) -> bytes:
    """Numbered SOM overlays. Element bounds are root-relative; the grab starts at (gx, gy)."""
    import io
    from PIL import Image, ImageDraw
    img = Image.open(io.BytesIO(png)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for el in elements:
        x, y, w, h = el.bounds[0] - gx, el.bounds[1] - gy, el.bounds[2], el.bounds[3]
        draw.rectangle([x, y, x + 20, y + 20], fill=(220, 30, 30), outline=(255, 255, 255))
        draw.text((x + 5, y + 2), str(el.index), fill=(255, 255, 255))
        draw.rectangle([x, y, x + w, y + h], outline=(220, 30, 30), width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
