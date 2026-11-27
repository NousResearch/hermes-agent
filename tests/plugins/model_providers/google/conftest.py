"""Ensure the bundled Google model-provider package is discovered and traversable for mock.patch/monkeypatch."""

from __future__ import annotations

import sys
from types import ModuleType

import plugins
import providers

providers.get_provider_profile("gemini")

_google_pkg = sys.modules.get("plugins.model_providers.google")
if _google_pkg is not None:
    _mp = sys.modules.get("plugins.model_providers")
    if _mp is None:
        _mp = ModuleType("plugins.model_providers")
        sys.modules["plugins.model_providers"] = _mp
    _mp.google = _google_pkg
    plugins.model_providers = _mp
