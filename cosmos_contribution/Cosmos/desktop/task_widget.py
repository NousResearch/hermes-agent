"""
Cosmos Task Management Widget.
"""

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
        QTableWidgetItem, QPushButton, QLabel, QHeaderView
    )
    from PySide6.QtCore import Qt
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False
    QWidget = object


class TaskWidget(QWidget if PYSIDE_AVAILABLE else object):
    """Widget for viewing and managing evolution tasks."""

    def __init__(self, parent=None):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self._setup_ui()
        self._load_tasks()

    def _setup_ui(self):
        """Setup the task UI."""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Evolution Tasks")
        header.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(header)

        # Task table
        self.task_table = QTableWidget()
        self.task_table.setColumnCount(5)
        self.task_table.setHorizontalHeaderLabels([
            "ID", "Type", "Description", "Agent", "Status"
        ])

        # Make columns stretch
        header = self.task_table.horizontalHeader()
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        layout.addWidget(self.task_table)

        # Buttons
        button_layout = QHBoxLayout()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_tasks)
        button_layout.addWidget(refresh_btn)

        add_btn = QPushButton("Add Task")
        add_btn.clicked.connect(self._add_task)
        button_layout.addWidget(add_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

    def _load_tasks(self):
        """Load tasks from the evolution loop."""
        self.task_table.setRowCount(0)

        try:
            from Cosmos.core.agent_spawner import get_spawner
            spawner = get_spawner()
            if spawner:
                tasks = spawner.get_pending_tasks() + spawner.get_active_tasks()

                for i, task in enumerate(tasks):
                    self.task_table.insertRow(i)
                    self.task_table.setItem(i, 0, QTableWidgetItem(str(task.id)))
                    self.task_table.setItem(i, 1, QTableWidgetItem(task.task_type.name))
                    self.task_table.setItem(i, 2, QTableWidgetItem(task.description[:50]))
                    self.task_table.setItem(i, 3, QTableWidgetItem(task.assigned_to or "Unassigned"))
                    self.task_table.setItem(i, 4, QTableWidgetItem(task.status.name))

        except Exception as e:
            # Add placeholder row
            self.task_table.insertRow(0)
            self.task_table.setItem(0, 2, QTableWidgetItem(f"Error loading tasks: {e}"))

    def _add_task(self):
        """Open dialog to add a new task."""
        from PySide6.QtWidgets import QInputDialog

        desc, ok = QInputDialog.getText(self, "Add Task", "Task description:")
        if ok and desc:
            try:
                from Cosmos.core.agent_spawner import get_spawner, TaskType
                spawner = get_spawner()
                if spawner:
                    spawner.add_task(
                        task_type=TaskType.DEVELOPMENT,
                        description=desc,
                    )
                    self._load_tasks()
            except Exception as e:
                pass
