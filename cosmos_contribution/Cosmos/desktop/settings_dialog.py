"""
Cosmos Settings Widget.
"""

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QFormLayout, QLineEdit,
        QCheckBox, QPushButton, QLabel, QGroupBox,
        QSpinBox, QComboBox, QMessageBox
    )
    from PySide6.QtCore import Qt
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False
    QWidget = object


class SettingsWidget(QWidget if PYSIDE_AVAILABLE else object):
    """Settings configuration widget."""

    def __init__(self, parent=None):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the settings UI."""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Settings")
        header.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(header)

        # General settings group
        general_group = QGroupBox("General")
        general_layout = QFormLayout(general_group)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["System", "Light", "Dark"])
        general_layout.addRow("Theme:", self.theme_combo)

        self.hotkey_edit = QLineEdit("Ctrl+Shift+F")
        general_layout.addRow("Global Hotkey:", self.hotkey_edit)

        self.start_minimized = QCheckBox()
        general_layout.addRow("Start Minimized:", self.start_minimized)

        layout.addWidget(general_group)

        # API Keys group
        api_group = QGroupBox("API Keys")
        api_layout = QFormLayout(api_group)

        self.bankr_key = QLineEdit()
        self.bankr_key.setEchoMode(QLineEdit.Password)
        self.bankr_key.setPlaceholderText("bk_...")
        api_layout.addRow("Bankr API Key:", self.bankr_key)

        self.openai_key = QLineEdit()
        self.openai_key.setEchoMode(QLineEdit.Password)
        api_layout.addRow("OpenAI Key:", self.openai_key)

        layout.addWidget(api_group)

        # Trading settings group
        trading_group = QGroupBox("Trading")
        trading_layout = QFormLayout(trading_group)

        self.trading_enabled = QCheckBox()
        trading_layout.addRow("Enable Trading:", self.trading_enabled)

        self.max_trade = QSpinBox()
        self.max_trade.setRange(1, 10000)
        self.max_trade.setValue(100)
        self.max_trade.setSuffix(" USD")
        trading_layout.addRow("Max Trade:", self.max_trade)

        self.default_chain = QComboBox()
        self.default_chain.addItems(["base", "ethereum", "solana", "polygon"])
        trading_layout.addRow("Default Chain:", self.default_chain)

        layout.addWidget(trading_group)

        # Buttons
        button_layout = QVBoxLayout()

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)
        layout.addStretch()

    def _save_settings(self):
        """Save settings to config file."""
        import os

        # Set environment variables (temporary)
        if self.bankr_key.text():
            os.environ["BANKR_API_KEY"] = self.bankr_key.text()

        if self.openai_key.text():
            os.environ["OPENAI_API_KEY"] = self.openai_key.text()

        os.environ["BANKR_TRADING_ENABLED"] = str(self.trading_enabled.isChecked()).lower()
        os.environ["BANKR_MAX_TRADE_USD"] = str(self.max_trade.value())
        os.environ["BANKR_DEFAULT_CHAIN"] = self.default_chain.currentText()

        QMessageBox.information(self, "Settings", "Settings saved!")
