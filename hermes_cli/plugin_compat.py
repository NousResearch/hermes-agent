"""Frozen exports for ``hermes update`` processes that predate the plugin compat layer's removal.

The Sep 2026 decomposition kept old import paths alive for external plugins until 2026-09-14; that
layer, and the scanner that reported plugins still using it, is gone. An updater already running from
an older checkout still lazy-imports these three names for its post-update notice after the swap
(``tests/compat/old_updater_surface.json``), so they stay as inert stubs with nothing left to report.
"""
from __future__ import annotations

from typing import Any


def compat_report(manifests: Any = None, *, force: bool = False) -> dict[str, list[Any]]:
    return {}


def removal_in_effect(today: Any = None) -> bool:
    return True


def summary_lines(report: Any, *, today: Any = None) -> list[str]:
    return []
