"""
Core module — window discovery, control operations with retry, structured output.

All public functions return a ``Result`` dict with at least ``{"success": bool}``.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Callable

from pywinauto import Desktop, Application
from pywinauto.findwindows import ElementNotFoundError
import pywinauto.timings

from .utils.screenshot import capture as _capture
from .utils.config import load_config

log = logging.getLogger("win-gui-test")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

Result = dict[str, Any]


def _find_windows(title_fragment: str) -> list:
    """Return all visible UIA windows whose title contains *title_fragment*."""
    desktop = Desktop(backend="uia")
    return [w for w in desktop.windows() if w.window_text() and title_fragment in w.window_text()]


def _retry(
    fn: Callable[[], Any],
    retries: int = 3,
    delay: float = 1.0,
    exc_types: tuple = (ElementNotFoundError, pywinauto.timings.TimeoutError, RuntimeError),
) -> Any:
    """Call *fn* up to *retries* times, sleeping *delay* s between attempts."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except exc_types as exc:
            last_exc = exc
            log.debug("Retry %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _first_window(title: str, timeout: float = 10) -> Any:
    """Return the first matching window, raising if none found within *timeout* s."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        wins = _find_windows(title)
        if wins:
            return wins[0]
        time.sleep(0.3)
    raise ElementNotFoundError(f"Window containing '{title}' not found after {timeout}s")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def list_windows() -> Result:
    """List all visible desktop windows."""
    desktop = Desktop(backend="uia")
    windows = []
    for w in desktop.windows():
        try:
            if w.is_visible() and w.window_text():
                r = w.rectangle()
                windows.append(
                    {
                        "title": w.window_text(),
                        "class": w.class_name(),
                        "rect": f"({r.left},{r.top}) {r.width()}x{r.height()}",
                    }
                )
        except Exception:
            continue
    return {"success": True, "windows": windows, "count": len(windows)}


def list_elements(title: str, timeout: float = 10) -> Result:
    """List all UIA descendants (controls) of the first matching window."""
    try:
        w = _retry(lambda: _first_window(title, timeout), retries=1)
    except ElementNotFoundError as exc:
        return {"success": False, "error": str(exc)}
    elements = []
    for e in w.descendants():
        try:
            info = e.element_info
            if info.name and info.control_type:
                try:
                    r = e.rectangle
                    if callable(r):
                        r = r()
                    rect_str = f"({r.left},{r.top}) {r.width()}x{r.height()}"
                except Exception:
                    rect_str = ""
                elements.append(
                    {
                        "type": info.control_type,
                        "name": info.name.strip()[:80],
                        "rect": rect_str,
                    }
                )
        except Exception:
            continue
    return {"success": True, "window": title, "elements": elements, "count": len(elements)}


def screenshot(
    title: str | None = None,
    output_dir: str = "",
    filename: str | None = None,
    fallback: bool = True,
) -> Result:
    """Take a screenshot, optionally focusing *title* first.

    Returns dict with ``filepath`` on success.
    """
    out_dir = output_dir or load_config().get("screenshot_dir", os.path.expanduser("~/Desktop/gui_screenshots"))
    ts = time.strftime("%Y%m%d_%H%M%S")
    label = (title or "screen").replace(" ", "_")[:40]
    name = filename or f"{label}_{ts}.png"
    filepath = os.path.join(out_dir, name)

    result_path = _capture(filepath, fallback=fallback, window_title=title)
    if result_path:
        return {"success": True, "filepath": result_path, "window": title}
    return {"success": False, "error": "All screenshot methods failed"}


def click(title: str, target: str, timeout: float = 10) -> Result:
    """Click a control by name within the first matching window."""
    try:
        w = _retry(lambda: _first_window(title, timeout), retries=1)
    except ElementNotFoundError as exc:
        return {"success": False, "error": str(exc)}

    def _do() -> bool:
        w.set_focus()
        time.sleep(0.2)
        # direct child_window
        try:
            btn = w.child_window(title=target)
            btn.click_input()
            return True
        except Exception:
            pass
        # fallback — scan descendants
        for e in w.descendants():
            try:
                info = e.element_info
                if info.name and target in info.name:
                    e.click_input()
                    return True
            except Exception:
                continue
        raise ElementNotFoundError(f"No control containing '{target}' found")

    try:
        _retry(_do, retries=load_config().get("retry", {}).get("count", 3))
        return {"success": True, "action": "click", "target": target}
    except Exception as exc:
        return {"success": False, "action": "click", "target": target, "error": str(exc)}


def click_coords(title: str, x: int, y: int, timeout: float = 10) -> Result:
    """Click at absolute screen coordinates after focusing the window."""
    try:
        w = _retry(lambda: _first_window(title, timeout), retries=1)
    except ElementNotFoundError as exc:
        return {"success": False, "error": str(exc)}
    try:
        w.set_focus()
        time.sleep(0.2)
        from pywinauto.mouse import click as _mouse_click

        _mouse_click(coords=(x, y))
        return {"success": True, "action": "click_coords", "x": x, "y": y}
    except Exception as exc:
        return {"success": False, "action": "click_coords", "x": x, "y": y, "error": str(exc)}


def send_keys(title: str, keys: str, timeout: float = 10) -> Result:
    """Send keystrokes to the focused window."""
    try:
        w = _retry(lambda: _first_window(title, timeout), retries=1)
    except ElementNotFoundError as exc:
        return {"success": False, "error": str(exc)}
    try:
        w.set_focus()
        time.sleep(0.2)
        from pywinauto.keyboard import send_keys as _send

        _send(keys)
        return {"success": True, "action": "send_keys", "keys": keys}
    except Exception as exc:
        return {"success": False, "action": "send_keys", "keys": keys, "error": str(exc)}


def scroll(title: str, target: str = "", dy: int = -3, timeout: float = 10) -> Result:
    """Scroll a control (or the window itself) by *dy* wheel clicks."""
    try:
        w = _retry(lambda: _first_window(title, timeout), retries=1)
    except ElementNotFoundError as exc:
        return {"success": False, "error": str(exc)}
    try:
        w.set_focus()
        time.sleep(0.2)
        from pywinauto.mouse import scroll as _scroll

        cx, cy = w.rectangle().center()
        if target:
            for e in w.descendants():
                try:
                    info = e.element_info
                    if info.name and target in info.name:
                        r = e.rectangle()
                        cx, cy = r.center()
                        break
                except Exception:
                    continue
        _scroll(coords=(cx, cy), wheel_dist=dy)
        return {"success": True, "action": "scroll", "target": target or title, "dy": dy}
    except Exception as exc:
        return {"success": False, "action": "scroll", "target": target, "error": str(exc)}


def get_rect(title: str, target: str, timeout: float = 10) -> Result:
    """Get precise rectangle of a control by name."""
    try:
        w = _retry(lambda: _first_window(title, timeout), retries=1)
    except ElementNotFoundError as exc:
        return {"success": False, "error": str(exc)}
    for e in w.descendants():
        try:
            info = e.element_info
            if info.name and target in info.name:
                r = e.rectangle()
                return {
                    "success": True,
                    "name": info.name.strip()[:60],
                    "type": info.control_type,
                    "rect": f"({r.left},{r.top}) {r.width()}x{r.height()}",
                    "left": r.left, "top": r.top,
                    "width": r.width(), "height": r.height(),
                    "center_x": r.left + r.width() // 2,
                    "center_y": r.top + r.height() // 2,
                }
        except Exception:
            continue
    return {"success": False, "error": f"Control '{target}' not found"}


def launch(app_path: str, wait: float = 2.0) -> Result:
    """Launch an application (shell=True)."""
    import subprocess

    try:
        subprocess.Popen(app_path, shell=True)
        if wait:
            time.sleep(wait)
        return {"success": True, "action": "launch", "app": app_path}
    except Exception as exc:
        return {"success": False, "action": "launch", "app": app_path, "error": str(exc)}
