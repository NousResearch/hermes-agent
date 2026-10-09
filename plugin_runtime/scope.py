"""Hermes-home scoping shared by plugin runtime operations."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@contextmanager
def plugin_home_scope(home: Path):
    """Bind plugin runtime work to the manager's immutable Hermes home."""
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)
