"""
Cosmos System Tray Integration.
"""

try:
    from PySide6.QtWidgets import QSystemTrayIcon, QMenu
    from PySide6.QtGui import QIcon, QAction
    from PySide6.QtCore import Qt
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False
    QSystemTrayIcon = object


class SystemTrayIcon(QSystemTrayIcon if PYSIDE_AVAILABLE else object):
    """System tray icon for background operation."""

    def __init__(self, app):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__()
        self.app = app

        # Set icon (using a placeholder - would use actual icon file)
        # self.setIcon(QIcon(":/icons/cosmos.png"))
        self.setToolTip("Cosmos AI")

        # Create context menu
        self._create_menu()

        # Connect signals
        self.activated.connect(self._on_activated)

    def _create_menu(self):
        """Create the tray context menu."""
        menu = QMenu()

        # Show/Hide
        show_action = QAction("Open Cosmos", menu)
        show_action.triggered.connect(self._show_main)
        menu.addAction(show_action)

        # Quick command
        quick_action = QAction("Quick Command...", menu)
        quick_action.triggered.connect(self._quick_command)
        menu.addAction(quick_action)

        menu.addSeparator()

        # Status
        status_action = QAction("Status: Running", menu)
        status_action.setEnabled(False)
        menu.addAction(status_action)

        menu.addSeparator()

        # Settings
        settings_action = QAction("Settings", menu)
        settings_action.triggered.connect(self._show_settings)
        menu.addAction(settings_action)

        # Exit
        exit_action = QAction("Exit", menu)
        exit_action.triggered.connect(self._exit)
        menu.addAction(exit_action)

        self.setContextMenu(menu)

    def _on_activated(self, reason):
        """Handle tray icon activation."""
        if reason == QSystemTrayIcon.DoubleClick:
            self._show_main()

    def _show_main(self):
        """Show the main window."""
        if self.app and self.app.main_window:
            self.app.main_window.show()
            self.app.main_window.activateWindow()
            self.app.main_window.raise_()

    def _quick_command(self):
        """Open quick command dialog."""
        if self.app:
            self.app.show_quick_command()

    def _show_settings(self):
        """Show settings."""
        self._show_main()
        if self.app and self.app.main_window:
            # Navigate to settings page (index 5)
            self.app.main_window.sidebar.setCurrentRow(5)

    def _exit(self):
        """Exit the application."""
        if self.app:
            self.app.cleanup()
            self.app.quit()
