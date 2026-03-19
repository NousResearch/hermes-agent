"""
Cosmos Memory Browser Widget.
"""

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
        QPushButton, QTextEdit, QLabel, QListWidget
    )
    from PySide6.QtCore import Qt
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False
    QWidget = object


class MemoryWidget(QWidget if PYSIDE_AVAILABLE else object):
    """Widget for browsing and searching Cosmos's memory."""

    def __init__(self, parent=None):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the memory UI."""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Memory System")
        header.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(header)

        # Search bar
        search_layout = QHBoxLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search memories...")
        self.search_input.returnPressed.connect(self._search)
        search_layout.addWidget(self.search_input)

        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self._search)
        search_layout.addWidget(search_btn)

        layout.addLayout(search_layout)

        # Results list
        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self._show_memory)
        layout.addWidget(self.results_list)

        # Memory detail view
        self.detail_view = QTextEdit()
        self.detail_view.setReadOnly(True)
        self.detail_view.setMaximumHeight(200)
        layout.addWidget(self.detail_view)

        # Stats label
        self.stats_label = QLabel("Memory stats: Loading...")
        layout.addWidget(self.stats_label)

        # Load initial stats
        self._load_stats()

    def _search(self):
        """Search memories."""
        query = self.search_input.text().strip()
        if not query:
            return

        self.results_list.clear()

        try:
            from Cosmos.memory.memory_system import get_memory_system
            memory = get_memory_system()

            if memory:
                results = memory.search(query, top_k=20)
                for result in results:
                    content = result.get("content", result.get("text", ""))[:100]
                    self.results_list.addItem(content)

        except Exception as e:
            self.results_list.addItem(f"Error: {e}")

    def _show_memory(self, item):
        """Show full memory content."""
        self.detail_view.setText(item.text())

    def _load_stats(self):
        """Load memory system statistics."""
        try:
            from Cosmos.memory.memory_system import get_memory_system
            memory = get_memory_system()

            if memory:
                stats = memory.get_stats()
                self.stats_label.setText(
                    f"Total memories: {stats.get('total', 0)} | "
                    f"Recent: {stats.get('recent', 0)}"
                )
            else:
                self.stats_label.setText("Memory system not initialized")

        except Exception as e:
            self.stats_label.setText(f"Stats unavailable: {e}")
