"""Best-effort early import of the OpenAI SDK's native streaming parser: on some
Windows installs ``jiter``'s native extension imports fine from the venv but fails
when first imported inside the threaded streaming path. Loading it once at
agent-package import avoids that while keeping the SDK's normal error path for
genuinely broken installs."""

from __future__ import annotations

import importlib

_JITER_PRELOADED = False
_JITER_PRELOAD_ERROR: Exception | None = None


def preload_jiter_native_extension() -> bool:
    global _JITER_PRELOADED, _JITER_PRELOAD_ERROR
    if _JITER_PRELOADED:
        return True
    try:
        importlib.import_module("jiter.jiter")
        from jiter import from_json as _from_json  # noqa: F401
    except Exception as exc:
        _JITER_PRELOAD_ERROR = exc
        return False
    _JITER_PRELOADED, _JITER_PRELOAD_ERROR = True, None
    return True


preload_jiter_native_extension()
