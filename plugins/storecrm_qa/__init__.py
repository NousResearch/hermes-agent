"""Hermes-owned StoreCRM QA control plane plugin.

This package intentionally does not register a core model tool.  The first
surface is a local CLI module plus importable store primitives for later
orchestrators.
"""

from __future__ import annotations

from .store import StoreCRMQAStore, default_db_path

__all__ = ["StoreCRMQAStore", "default_db_path", "register"]


def register(ctx) -> None:
    """Plugin loader entrypoint.

    The QA control plane is currently exercised through
    ``python -m plugins.storecrm_qa.cli`` and direct Python imports.  Keeping
    registration inert avoids adding any always-on model tool footprint.
    """

    return None
