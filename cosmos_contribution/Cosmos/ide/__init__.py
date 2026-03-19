"""
Cosmos IDE Module.

Web-based IDE with Monaco Editor and integrated terminal.

Features:
- Code editing with syntax highlighting
- File explorer
- Integrated terminal (xterm.js)
- Git integration
- AI assistant panel
"""

from .app import CosmosIDE

__all__ = ['CosmosIDE']
