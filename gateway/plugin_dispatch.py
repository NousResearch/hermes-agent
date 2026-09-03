"""Signature-tolerant dispatch for third-party platform plugins.

Bundled adapters register zero-arg ``setup_fn`` and accept
``connect(*, is_reconnect=False)``. External plugins (Keet, older forks)
still declare ``setup_fn(config)`` or ``connect(self)`` without the keyword.
Calling them with the current Hermes contract raises TypeError and aborts
``hermes gateway setup`` / gateway start (#97065).

Inspect the callable and pass only the parameters it actually accepts.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable


def invoke_setup_fn(setup_fn: Callable[..., Any], config: Any = None) -> Any:
    """Call a platform ``setup_fn``, supplying ``config`` only if required."""
    if setup_fn is None:
        return None
    try:
        signature = inspect.signature(setup_fn)
    except (TypeError, ValueError):
        return setup_fn()

    required = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and parameter.default is inspect.Parameter.empty
        and parameter.name != "self"
    ]
    if not required:
        return setup_fn()

    target = required[0]
    if target.kind == inspect.Parameter.KEYWORD_ONLY:
        return setup_fn(**{target.name: config})
    return setup_fn(config)


def connect_adapter(adapter: Any, *, is_reconnect: bool = False) -> Any:
    """Call ``adapter.connect``, omitting ``is_reconnect`` when unsupported."""
    connect = adapter.connect
    try:
        signature = inspect.signature(connect)
        accepts_reconnect = "is_reconnect" in signature.parameters
    except (TypeError, ValueError):
        accepts_reconnect = False
    if accepts_reconnect:
        return connect(is_reconnect=is_reconnect)
    return connect()
