"""Best-effort early import for the OpenAI SDK's native streaming parser.

The OpenAI SDK imports ``jiter`` while constructing streaming chat-completion
responses.  On some Windows installs the native extension can be imported
directly from the Hermes venv, but the first import fails when it happens later
inside the threaded streaming request path.  Loading it once during agent
package import avoids that import-order failure while preserving the normal
SDK error path for genuinely missing or broken installs.
"""

from __future__ import annotations

import importlib

_JITER_PRELOADED = False
_JITER_PRELOAD_ERROR: Exception | None = None


def preload_jiter_native_extension() -> bool:
    """Import jiter's native extension early if it is available."""

    global _JITER_PRELOADED, _JITER_PRELOAD_ERROR

    if _JITER_PRELOADED:
        return True

    try:
        importlib.import_module("jiter.jiter")
        from jiter import from_json as _from_json
    except Exception as exc:
        _JITER_PRELOAD_ERROR = exc
        return False

    _JITER_PRELOADED = True
    _JITER_PRELOAD_ERROR = None
    return True


preload_jiter_native_extension()
