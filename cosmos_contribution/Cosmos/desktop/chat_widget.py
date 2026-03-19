"""
Cosmos Chat Widget.
"""

import logging
import asyncio
from typing import Optional, List

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
        QLineEdit, QPushButton, QScrollArea, QLabel, QFrame
    )
    from PySide6.QtCore import Qt, Signal, QThread
    from PySide6.QtGui import QFont
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False
    QWidget = object

logger = logging.getLogger(__name__)


class ChatWidget(QWidget if PYSIDE_AVAILABLE else object):
    """
    Chat interface for interacting with Cosmos.
    """

    def __init__(self, parent=None):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self.messages: List[dict] = []
        self._setup_ui()

    def _setup_ui(self):
        """Setup the chat UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 11))
        layout.addWidget(self.chat_display)

        # Input area
        input_layout = QHBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask Cosmos anything... (or use 'Hey Farn, ...')")
        self.input_field.setMinimumHeight(40)
        self.input_field.returnPressed.connect(self._on_send)
        input_layout.addWidget(self.input_field)

        self.send_button = QPushButton("Send")
        self.send_button.setMinimumHeight(40)
        self.send_button.setMinimumWidth(80)
        self.send_button.clicked.connect(self._on_send)
        input_layout.addWidget(self.send_button)

        layout.addLayout(input_layout)

        # Initial message
        self._add_message("assistant", "Hello! I'm Cosmos, your AI assistant. How can I help you today?")

    def _on_send(self):
        """Handle send button click."""
        text = self.input_field.text().strip()
        if not text:
            return

        self.input_field.clear()
        self._add_message("user", text)

        # Process asynchronously
        self._process_message(text)

    def _add_message(self, role: str, content: str):
        """Add a message to the display."""
        self.messages.append({"role": role, "content": content})

        # Format message
        if role == "user":
            prefix = "<b>You:</b>"
            color = "#2196F3"
        else:
            prefix = "<b>Cosmos:</b>"
            color = "#4CAF50"

        html = f'<p style="color: {color}; margin: 10px 0;">{prefix}<br/>{content}</p>'
        self.chat_display.append(html)

    def _process_message(self, text: str):
        """Process user message through NLP system."""
        try:
            # Use the NLP command router
            from Cosmos.core.nlp import process_command

            # Run async processing
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(process_command(text))
            loop.close()

            response = result.get("response", "I processed your request.")
            self._add_message("assistant", response)

        except ImportError:
            # NLP module not available, use simple response
            self._add_message(
                "assistant",
                f"I understand you said: '{text}'. The NLP module is loading..."
            )
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            self._add_message("assistant", f"Sorry, I encountered an error: {str(e)}")

    def clear_chat(self):
        """Clear the chat history."""
        self.messages = []
        self.chat_display.clear()
        self._add_message("assistant", "Chat cleared. How can I help you?")
