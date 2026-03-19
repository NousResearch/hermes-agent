"""
Cosmos Desktop Application Entry Point.
"""

import sys
import logging
from typing import Optional

try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QIcon
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False
    QApplication = object

logger = logging.getLogger(__name__)


class CosmosApp(QApplication if PYSIDE_AVAILABLE else object):
    """
    Main Cosmos desktop application.

    Manages:
    - Main window
    - System tray
    - Global hotkeys
    - Background services
    """

    def __init__(self, argv=None):
        if not PYSIDE_AVAILABLE:
            raise ImportError("PySide6 is required for the desktop app: pip install PySide6")

        super().__init__(argv or sys.argv)

        # Application metadata
        self.setApplicationName("Cosmos")
        self.setApplicationDisplayName("Cosmos AI")
        self.setOrganizationName("Cosmos")
        self.setOrganizationDomain("cosmos.ai")

        # Initialize components
        self._init_theme()
        self._init_core()
        self._init_ui()
        self._init_hotkeys()
        self._init_tray()

    def _init_theme(self):
        """Initialize application theme."""
        try:
            import darkdetect
            self.is_dark_mode = darkdetect.isDark()
        except ImportError:
            self.is_dark_mode = False

        # Apply stylesheet based on theme
        from .themes import get_stylesheet
        self.setStyleSheet(get_stylesheet(self.is_dark_mode))

    def _init_core(self):
        """Initialize Cosmos core systems."""
        self.cosmos_core = None
        try:
            # Lazy load core to avoid circular imports
            pass
        except Exception as e:
            logger.warning(f"Core initialization deferred: {e}")

    def _init_ui(self):
        """Initialize main window."""
        from .main_window import MainWindow
        self.main_window = MainWindow(self)
        self.main_window.show()

    def _init_hotkeys(self):
        """Initialize global hotkeys."""
        try:
            from .hotkeys import GlobalHotkey
            self.hotkey = GlobalHotkey("ctrl+shift+f", self.toggle_window)
            self.hotkey.start()
        except Exception as e:
            logger.warning(f"Hotkey registration failed: {e}")

    def _init_tray(self):
        """Initialize system tray."""
        from .system_tray import SystemTrayIcon
        self.tray = SystemTrayIcon(self)
        self.tray.show()

    def toggle_window(self):
        """Toggle main window visibility."""
        if self.main_window.isVisible():
            self.main_window.hide()
        else:
            self.main_window.show()
            self.main_window.activateWindow()
            self.main_window.raise_()

    def show_quick_command(self):
        """Show quick command dialog."""
        from .quick_command import QuickCommandDialog
        dialog = QuickCommandDialog(self.main_window)
        dialog.exec()

    def cleanup(self):
        """Cleanup before exit."""
        try:
            if hasattr(self, 'hotkey'):
                self.hotkey.stop()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def main():
    """Entry point for desktop app."""
    app = CosmosApp(sys.argv)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
