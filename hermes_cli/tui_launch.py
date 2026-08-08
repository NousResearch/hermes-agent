"""Shared failure type for preparing the Node TUI runtime."""

from __future__ import annotations


class TuiLaunchError(RuntimeError):
    """Actionable TUI preparation failure for non-CLI callers.

    The ordinary CLI surface still prints the message and exits with status 1.
    Long-lived callers such as the dashboard raise this exception instead so
    the real npm/node/workspace failure can be logged and returned over the
    WebSocket rather than collapsing to the opaque string ``SystemExit(1)``.
    """


__all__ = ["TuiLaunchError"]
