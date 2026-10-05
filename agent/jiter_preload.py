"""Best-effort early import of the OpenAI SDK's native streaming parser: on some
Windows installs ``jiter``'s native extension imports fine from the venv but fails
when first imported inside the threaded streaming path. Loading it once at
agent-package import avoids that while keeping the SDK's normal error path for
genuinely broken installs.

(Update: Disabled entirely on Windows due to an upstream chunk-boundary memory safety
bug causing tool-call corruption / truncation - Issue #127260)
"""

from __future__ import annotations

import importlib
import sys

_JITER_PRELOADED = False
_JITER_PRELOAD_ERROR: Exception | None = None


def preload_jiter_native_extension() -> bool:
    global _JITER_PRELOADED, _JITER_PRELOAD_ERROR
    if _JITER_PRELOADED:
        return True
        
    if sys.platform == "win32":
        # Issue #127260: disable jiter completely on Windows to bypass SSE chunk boundary corruption
        sys.modules["jiter"] = None
        sys.modules["jiter.jiter"] = None
        _JITER_PRELOADED = True
        return False
        
    try:
        importlib.import_module("jiter.jiter")
        from jiter import from_json as _from_json  # noqa: F401
    except Exception as exc:
        _JITER_PRELOAD_ERROR = exc
        return False
    _JITER_PRELOADED, _JITER_PRELOAD_ERROR = True, None
    return True


preload_jiter_native_extension()
