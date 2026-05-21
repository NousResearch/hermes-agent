"""Hermes Git Projects plugin.

This standalone plugin is primarily a dashboard extension. It intentionally
registers no chat tools: project import, issue logging, source-control actions,
and suggested-skill management live under the dashboard plugin API and UI.
"""

from __future__ import annotations


def register(ctx) -> None:
    """No-op CLI/gateway registration hook for plugin discovery."""
    return None
