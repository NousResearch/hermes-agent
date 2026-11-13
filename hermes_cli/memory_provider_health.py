"""Best-effort health details for loaded memory providers.

Core never imports a plugin's third-party dependencies directly. These helpers
read state a *loaded* provider already exposes, so ``hermes doctor`` /
``hermes memory status`` can surface a degraded-but-"available" provider —
the holographic plugin reports ``is_available() == True`` while silently
running FTS-only when numpy is missing (#136707).
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Optional


def _provider_package_modules(provider, leaf: str) -> list[ModuleType]:
    """Every loaded module named ``<provider-package>.leaf`` for *provider*.

    The provider class lives in the package ``__init__`` (bundled
    ``plugins.memory.holographic`` or the synthetic user-plugin namespace),
    so its HRR kernel is ``<that package>.holographic``. Every class in the
    MRO is considered (test doubles sometimes subclass the provider from a
    different module), and already-loaded modules are checked as well:
    synthetic namespaces can make the qualified name differ from
    ``import_module``'s resolution."""
    roots = []
    for cls in type(provider).__mro__:
        module = cls.__module__
        if module and module not in roots and not module.startswith("tests"):
            roots.append(module)
    names: list[str] = []
    for root in roots:
        qualified = f"{root}.{leaf}"
        names.append(qualified)
        names.extend(n for n in sys.modules
                     if n != qualified and n.startswith(root + ".") and n.endswith("." + leaf))
    found: list[ModuleType] = []
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            try:
                module = importlib.import_module(name)
            except Exception:
                continue
        if hasattr(module, "_HAS_NUMPY") and module not in found:
            found.append(module)
    return found


def holographic_hrr_state(provider) -> Optional[dict]:
    """HRR health of a loaded holographic provider, or None when unknowable.

    ``{"numpy": True, "numpy_version": "2.x"}`` when the HRR kernel imported
    numpy (semantic HRR recall active), ``{"numpy": False}`` when the kernel
    is running its FTS-only fallback. None means the loaded plugin does not
    expose the kernel flag (older/other build) — callers stay silent rather
    than assert a state they cannot verify.
    """
    kernels = _provider_package_modules(provider, "holographic")
    if not kernels:
        return None
    kernel = kernels[0]
    if not bool(getattr(kernel, "_HAS_NUMPY", False)):
        return {"numpy": False}
    state = {"numpy": True}
    try:
        import numpy  # the kernel imported it successfully; this hits sys.modules

        version = getattr(numpy, "__version__", None)
        if version:
            state["numpy_version"] = version
    except Exception:
        pass
    return state
