#!/usr/bin/env python3
"""Simple safe print that tries rich, falls back to regular print."""

try:
    from rich import print as rich_print
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def safe_print(*args, **kwargs):
    """Try rich.print, fall back to regular print if it fails."""
    if RICH_AVAILABLE:
        try:
            rich_print(*args, **kwargs)
            return
        except Exception:
            pass
    # Fallback to regular print
    print(*args, **kwargs)
