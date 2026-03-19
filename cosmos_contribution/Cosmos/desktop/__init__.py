"""
Cosmos Windows Desktop Interface.

A full-featured desktop application with:
- System tray integration
- Global hotkey (Ctrl+Shift+F)
- Chat interface
- Task management
- Memory browser
- Settings

Requires: PySide6, keyboard, darkdetect
"""

from .app import CosmosApp
from .main_window import MainWindow

__all__ = ['CosmosApp', 'MainWindow']


def launch_desktop():
    """Launch the desktop application."""
    import sys
    app = CosmosApp(sys.argv)
    return app.exec()
