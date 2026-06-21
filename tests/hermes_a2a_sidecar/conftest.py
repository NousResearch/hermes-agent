"""Test fixtures for the A2A sidecar suite.

Some sibling test modules install lightweight stub ``google`` /
``google.*`` modules into ``sys.modules`` at import time (e.g. fake
``google.oauth2`` for Google Workspace tests). Those stubs are plain
``types.ModuleType`` objects without a ``__path__``, so once they leak into the
shared pytest process they shadow the real ``google`` namespace package and
break ``import google.protobuf`` — which the official ``a2a-sdk`` (and this
sidecar) depend on.

This conftest runs just before the modules in this directory are collected.
It drops any ``google`` / ``google.*`` entry that is not a real namespace
package, then re-imports the genuine ``google.protobuf`` so the A2A SDK loads
cleanly regardless of collection order.
"""

from __future__ import annotations

import importlib
import sys


def _repair_google_namespace() -> None:
    google_mod = sys.modules.get("google")
    needs_repair = google_mod is not None and not hasattr(google_mod, "__path__")
    if not needs_repair:
        # Even if the top-level package is fine, a stubbed submodule may shadow
        # google.protobuf. Probe and only purge on failure.
        try:
            importlib.import_module("google.protobuf.json_format")
            return
        except Exception:
            needs_repair = True
    if not needs_repair:
        return
    for name in [m for m in list(sys.modules) if m == "google" or m.startswith("google.")]:
        mod = sys.modules.get(name)
        # Keep real packages (they expose __path__); drop bare stubs.
        if mod is not None and not hasattr(mod, "__path__"):
            del sys.modules[name]
    importlib.invalidate_caches()
    importlib.import_module("google.protobuf.json_format")


_repair_google_namespace()
