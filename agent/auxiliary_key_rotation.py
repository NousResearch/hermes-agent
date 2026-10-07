"""Rotate the published auxiliary main-runtime key after a credential refresh.

Split from ``agent.auxiliary_client`` (facade size cap). It reads the facade's runtime-main state
at call time, so the late import below always sees the current legacy mirror values.
"""

from __future__ import annotations

from typing import Any


def rotate_runtime_main_api_key(old: Any, new: Any) -> None:
    """Swap a revoked main key for its replacement in the published runtime, IN PLACE.

    A refresh can run inside a request worker's copied Context, where rebinding the ContextVar
    would stay invisible to the turn thread; the published dict is shared, so mutating it is not.
    """
    from agent.auxiliary_client import (
        _RUNTIME_MAIN_COMPAT_SNAPSHOT, _MAIN_RUNTIME_FIELDS, _RUNTIME_MAIN_CONTEXT, _normalize_api_key,
        _publish_runtime_main_mirrors,
    )

    runtime = _RUNTIME_MAIN_CONTEXT.get()
    if isinstance(runtime, dict) and runtime.get("api_key") == old:
        # Only the runtime the mirrors were published from may republish them: a scoped runtime (or another
        # session's) that happens to share the old key must not overwrite them with its own route.
        published = tuple(runtime.get(field, "") for field in _MAIN_RUNTIME_FIELDS) == _RUNTIME_MAIN_COMPAT_SNAPSHOT
        runtime["api_key"] = _normalize_api_key(new)
        if published:
            _publish_runtime_main_mirrors(tuple(runtime.get(field, "") for field in _MAIN_RUNTIME_FIELDS))
