"""
Cosmos Desktop Themes.
"""


def get_stylesheet(dark_mode: bool = False) -> str:
    """
    Get application stylesheet.

    Args:
        dark_mode: Whether to use dark mode

    Returns:
        Qt stylesheet string
    """
    if dark_mode:
        return DARK_THEME
    return LIGHT_THEME


LIGHT_THEME = """
QMainWindow {
    background-color: #f5f5f5;
}

QWidget {
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}

QListWidget {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
}

QListWidget::item {
    padding: 8px;
    border-bottom: 1px solid #f0f0f0;
}

QListWidget::item:selected {
    background-color: #e3f2fd;
    color: #1976d2;
}

QListWidget::item:hover {
    background-color: #f5f5f5;
}

QPushButton {
    background-color: #1976d2;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
}

QPushButton:hover {
    background-color: #1565c0;
}

QPushButton:pressed {
    background-color: #0d47a1;
}

QLineEdit, QTextEdit {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 8px;
}

QLineEdit:focus, QTextEdit:focus {
    border-color: #1976d2;
}

QGroupBox {
    font-weight: bold;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 16px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}

QTableWidget {
    background-color: #ffffff;
    gridline-color: #f0f0f0;
}

QHeaderView::section {
    background-color: #f5f5f5;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #e0e0e0;
}
"""


DARK_THEME = """
QMainWindow {
    background-color: #1e1e1e;
}

QWidget {
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
    color: #e0e0e0;
}

QListWidget {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
}

QListWidget::item {
    padding: 8px;
    border-bottom: 1px solid #3d3d3d;
}

QListWidget::item:selected {
    background-color: #0d47a1;
    color: white;
}

QListWidget::item:hover {
    background-color: #3d3d3d;
}

QPushButton {
    background-color: #0d47a1;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
}

QPushButton:hover {
    background-color: #1565c0;
}

QPushButton:pressed {
    background-color: #1976d2;
}

QLineEdit, QTextEdit {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 8px;
    color: #e0e0e0;
}

QLineEdit:focus, QTextEdit:focus {
    border-color: #1976d2;
}

QGroupBox {
    font-weight: bold;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 16px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}

QTableWidget {
    background-color: #2d2d2d;
    gridline-color: #3d3d3d;
}

QHeaderView::section {
    background-color: #3d3d3d;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #4d4d4d;
}
"""
