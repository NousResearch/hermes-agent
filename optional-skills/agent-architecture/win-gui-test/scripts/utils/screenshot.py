"""Screenshot with automatic fallback (mss → PIL.ImageGrab)."""

import os
import time
import logging
from typing import Optional

log = logging.getLogger(__name__)


def _screenshot_mss(filepath: str) -> bool:
    """Capture all-monitors screenshot via mss (fast, but fails on lock-screen)."""
    try:
        import mss
        from PIL import Image

        with mss.mss() as sct:
            monitor = sct.monitors[0]
            img = Image.frombytes("RGB", (monitor["width"], monitor["height"]), sct.grab(monitor).rgb)
            img.save(filepath)
        return True
    except Exception as exc:
        log.warning("mss screenshot failed: %s", exc)
        return False


def _screenshot_pil(filepath: str) -> bool:
    """Fallback: PIL ImageGrab (works on lock-screen but single-monitor)."""
    try:
        from PIL import ImageGrab

        img = ImageGrab.grab()
        img.save(filepath)
        return True
    except Exception as exc:
        log.error("PIL ImageGrab also failed: %s", exc)
        return False


def capture(
    filepath: str,
    fallback: bool = True,
    window_title: str | None = None,
) -> str | None:
    """Take a screenshot, returning the filepath or None on failure.

    Args:
        filepath: Destination path (dir must exist).
        fallback: Whether to try PIL.ImageGrab if mss fails.
        window_title: If given, try to focus the window first.

    Returns:
        The absolute filepath on success, or None.
    """
    if window_title:
        try:
            from pywinauto import Desktop

            desktop = Desktop(backend="uia")
            for w in desktop.windows():
                try:
                    if w.window_text() and window_title in w.window_text():
                        w.set_focus()
                        time.sleep(0.3)
                        break
                except Exception:
                    continue
        except Exception as exc:
            log.debug("Could not focus window '%s': %s", window_title, exc)

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    if _screenshot_mss(filepath):
        return os.path.abspath(filepath)

    if fallback and _screenshot_pil(filepath):
        return os.path.abspath(filepath)

    return None
