"""Test-only in-memory reference backend for the contract suites.

This package is a test fixture, never a product backend: it exists so every contract suite exercises real behavior against a real assembled backend. It never ships in the ``persistence`` package and is exempt from that package's no-backend-implementation restriction.

Extension is append-only with no shared-file edits: add ONE capability module in this package (for example ``conversation/sessions.py``) that defines the repository and calls ``register_repository`` at import time. Every non-package module except ``backend.py`` is auto-imported below, including modules in subpackages, so registration is wiring enough; nobody edits ``backend.py``, this file, or another capability's module.
"""

from __future__ import annotations

import importlib
import pkgutil

from .backend import InMemoryBackend, open_reference_backend, register_repository

# Auto-import every repository module so each capability's register_repository call
# runs; sorted for deterministic registration order.
for _module_info in sorted(pkgutil.walk_packages(__path__, prefix=f"{__name__}."), key=lambda m: m.name):
    if not _module_info.ispkg and _module_info.name != f"{__name__}.backend":
        importlib.import_module(_module_info.name)

__all__ = ["InMemoryBackend", "open_reference_backend", "register_repository"]
