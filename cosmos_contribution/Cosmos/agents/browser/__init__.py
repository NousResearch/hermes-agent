"""
Cosmos Agentic Browser Module.

Provides autonomous web browsing capabilities using Browser-Use with Playwright.
Can navigate, fill forms, click buttons, extract data, and complete multi-step web tasks.

Requires: browser-use, playwright
Note: Requires Python 3.11+
"""

from .agent import CosmosBrowserAgent, BrowserResult
from .controller import BrowserController
from .stealth import StealthBrowser

__all__ = [
    'CosmosBrowserAgent',
    'BrowserResult',
    'BrowserController',
    'StealthBrowser',
    'get_browser_agent',
]

_agent: "CosmosBrowserAgent" = None


def get_browser_agent() -> "CosmosBrowserAgent":
    """Get or create the global browser agent."""
    global _agent
    if _agent is None:
        _agent = CosmosBrowserAgent()
    return _agent
