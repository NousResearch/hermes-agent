"""
Cosmos Main Window.
"""

import logging
from typing import Optional

try:
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QStackedWidget, QListWidget, QListWidgetItem, QSplitter
    )
    from PySide6.QtCore import Qt, QSize
    from PySide6.QtGui import QIcon
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False
    QMainWindow = object

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow if PYSIDE_AVAILABLE else object):
    """
    Main application window with sidebar navigation.

    Layout:
    ┌─────────┬──────────────────────┐
    │ Sidebar │    Content Area      │
    │ [Chat]  │                      │
    │ [Tasks] │    (Stacked Widget)  │
    │ [Memory]│                      │
    │ [Tools] │                      │
    │ [Logs]  │                      │
    │ [Gear]  │                      │
    └─────────┴──────────────────────┘
    """

    def __init__(self, app=None):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__()
        self.app = app

        self.setWindowTitle("Cosmos AI")
        self.setMinimumSize(900, 600)
        self.resize(1200, 800)

        self._setup_ui()
        self._setup_menu()

    def _setup_ui(self):
        """Setup the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Splitter for resizable sidebar
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Sidebar
        self.sidebar = self._create_sidebar()
        splitter.addWidget(self.sidebar)

        # Content area (stacked widget for multiple views)
        self.content_stack = QStackedWidget()
        splitter.addWidget(self.content_stack)

        # Add content pages
        self._add_content_pages()

        # Set initial sizes (sidebar: 200px)
        splitter.setSizes([200, 1000])

        # Connect sidebar selection
        self.sidebar.currentRowChanged.connect(self._on_sidebar_changed)

    def _create_sidebar(self) -> QListWidget:
        """Create the sidebar navigation."""
        sidebar = QListWidget()
        sidebar.setMaximumWidth(250)
        sidebar.setMinimumWidth(150)

        # Navigation items
        items = [
            ("Chat", "💬"),
            ("Tasks", "📋"),
            ("Memory", "🧠"),
            ("Tools", "🔧"),
            ("Logs", "📜"),
            ("Settings", "⚙️"),
        ]

        for name, icon in items:
            item = QListWidgetItem(f"{icon}  {name}")
            item.setSizeHint(QSize(0, 50))
            sidebar.addItem(item)

        sidebar.setCurrentRow(0)
        return sidebar

    def _add_content_pages(self):
        """Add content pages to the stack."""
        # Chat page
        from .chat_widget import ChatWidget
        self.chat_widget = ChatWidget(self)
        self.content_stack.addWidget(self.chat_widget)

        # Tasks page
        from .task_widget import TaskWidget
        self.task_widget = TaskWidget(self)
        self.content_stack.addWidget(self.task_widget)

        # Memory page
        from .memory_widget import MemoryWidget
        self.memory_widget = MemoryWidget(self)
        self.content_stack.addWidget(self.memory_widget)

        # Tools page (placeholder)
        tools_placeholder = QWidget()
        self.content_stack.addWidget(tools_placeholder)

        # Logs page (placeholder)
        logs_placeholder = QWidget()
        self.content_stack.addWidget(logs_placeholder)

        # Settings page
        from .settings_dialog import SettingsWidget
        self.settings_widget = SettingsWidget(self)
        self.content_stack.addWidget(self.settings_widget)

    def _on_sidebar_changed(self, index: int):
        """Handle sidebar selection change."""
        self.content_stack.setCurrentIndex(index)

    def _setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction("New Chat", self._new_chat)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        # View menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction("Toggle Sidebar", self._toggle_sidebar)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction("About", self._show_about)

    def _new_chat(self):
        """Start a new chat."""
        if hasattr(self, 'chat_widget'):
            self.chat_widget.clear_chat()
        self.sidebar.setCurrentRow(0)

    def _toggle_sidebar(self):
        """Toggle sidebar visibility."""
        self.sidebar.setVisible(not self.sidebar.isVisible())

    def _show_about(self):
        """Show about dialog."""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "About Cosmos",
            "Cosmos AI Desktop\n\n"
            "Your intelligent AI companion.\n\n"
            "Version 2.9.4"
        )

    def closeEvent(self, event):
        """Handle window close - minimize to tray instead."""
        if self.app and hasattr(self.app, 'tray'):
            event.ignore()
            self.hide()
            self.app.tray.showMessage(
                "Cosmos",
                "Running in background. Click tray icon to reopen.",
                1000
            )
        else:
            event.accept()
