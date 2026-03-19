"""
Global Hotkey Handler for Cosmos Desktop.
"""

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False


class GlobalHotkey:
    """
    Register and handle global hotkeys.

    Uses the 'keyboard' library for cross-platform support.
    """

    def __init__(self, hotkey: str, callback: Callable):
        """
        Initialize global hotkey.

        Args:
            hotkey: Hotkey string (e.g., "ctrl+shift+f")
            callback: Function to call when hotkey is pressed
        """
        self.hotkey = hotkey
        self.callback = callback
        self._running = False
        self._hotkey_id = None

    def start(self):
        """Start listening for the hotkey."""
        if not KEYBOARD_AVAILABLE:
            logger.warning("keyboard library not available - hotkeys disabled")
            return

        try:
            self._hotkey_id = keyboard.add_hotkey(
                self.hotkey,
                self._on_hotkey,
                suppress=False
            )
            self._running = True
            logger.info(f"Global hotkey registered: {self.hotkey}")
        except Exception as e:
            logger.error(f"Failed to register hotkey: {e}")

    def stop(self):
        """Stop listening for the hotkey."""
        if not KEYBOARD_AVAILABLE:
            return

        self._running = False

        if self._hotkey_id is not None:
            try:
                keyboard.remove_hotkey(self._hotkey_id)
                self._hotkey_id = None
            except Exception as e:
                logger.error(f"Failed to remove hotkey: {e}")

    def _on_hotkey(self):
        """Handle hotkey press."""
        if self.callback:
            try:
                self.callback()
            except Exception as e:
                logger.error(f"Hotkey callback error: {e}")

    def update_hotkey(self, new_hotkey: str):
        """Update the hotkey binding."""
        self.stop()
        self.hotkey = new_hotkey
        self.start()
