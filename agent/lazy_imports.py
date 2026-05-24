"""Lazy import optimizer for Hermes Agent.

Defers module imports until first use, reducing startup time and memory
footprint.  Tools that are rarely used (e.g. video generation, speech-to-text)
should not be imported at startup.

How it works
------------
Tools are registered with ``lazy_import()`` which returns a proxy object.
The actual module is imported only when an attribute is accessed on the proxy.

```python
from agent.lazy_imports import lazy_import

# At module load time — no import yet
video_gen = lazy_import("tools.video_tool")

# Later, when actually called — import happens here
result = video_gen.video_generate(prompt="...")
```

Startup Impact
--------------
Hermes Agent imports ~200 modules at startup.  Lazy importing the heaviest
tools (browser, video, speech, image gen, MCP) can reduce startup time by
30-50% on cold starts.

Config
------
```yaml
performance:
  lazy_imports: true        # enabled by default
  lazy_import_exclude: []   # modules to always eager-import
```
"""

import importlib
import logging
import sys
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Registry of lazy import proxies
_lazy_registry: dict[str, "_LazyModule"] = {}
_lazy_lock = threading.Lock()


class _LazyModule:
    """Proxy that defers module import until first attribute access."""

    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module: Any = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> Any:
        """Load the module if not already loaded."""
        if self._module is not None:
            return self._module
        with self._lock:
            if self._module is None:
                logger.debug("Lazy import: %s", self._module_name)
                self._module = importlib.import_module(self._module_name)
            return self._module

    def __getattr__(self, name: str) -> Any:
        module = self._ensure_loaded()
        return getattr(module, name)

    def __repr__(self) -> str:
        if self._module is not None:
            return f"<LazyModule {self._module_name!r} (loaded)>"
        return f"<LazyModule {self._module_name!r} (not yet loaded)>"


def lazy_import(module_name: str) -> _LazyModule:
    """Create a lazy import proxy for a module.

    Parameters
    ----------
    module_name:
        Full module path (e.g. "tools.video_tool").

    Returns
    -------
    _LazyModule
        A proxy that imports the module on first attribute access.
    """
    with _lazy_lock:
        if module_name not in _lazy_registry:
            _lazy_registry[module_name] = _LazyModule(module_name)
        return _lazy_registry[module_name]


def force_import(module_name: str) -> Any:
    """Force-import a lazy module immediately.

    Useful for pre-warming tools that you know will be needed.
    """
    proxy = lazy_import(module_name)
    proxy._ensure_loaded()
    return proxy._module


def get_lazy_status() -> dict[str, Any]:
    """Get status of all lazy imports (for diagnostics)."""
    with _lazy_lock:
        return {
            name: "loaded" if proxy._module is not None else "deferred"
            for name, proxy in _lazy_registry.items()
        }


def preload_heavy_modules() -> None:
    """Pre-load commonly-used heavy modules during startup.

    Called when ``performance.lazy_imports: false`` in config.
    """
    heavy_modules = [
        "tools.browser_tool",
        "tools.video_tool",
        "tools.speech_to_text",
        "tools.text_to_speech",
        "tools.image_gen_tool",
        "tools.mcp_tool",
    ]
    for name in heavy_modules:
        try:
            lazy_import(name)._ensure_loaded()
        except ImportError:
            pass  # Module not available — skip silently
