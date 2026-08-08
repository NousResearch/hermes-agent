"""plan-secretary plugin: human-confirmed, session-scoped task plans.

Capture assistant future-commitments (precise actor+action+object filter),
confirm with a due time, remind in the originating session, and resolve via
parallel/defer/replace decisions. See README.md for the full design.
"""
from __future__ import annotations

from . import core

__all__ = ["core"]
