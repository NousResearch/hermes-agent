"""Signature-tolerant dispatch for third-party platform plugins (#97065).

Core's documented plugin contract is ``PlatformEntry.setup_fn() -> None`` and
``BasePlatformAdapter.connect(*, is_reconnect: bool = False)``. Plugins written
against a looser/older contract register ``setup_fn(config)``
(keet-platform) or override ``connect()`` without the keyword, so calling them
with the documented shape raised a TypeError: ``hermes gateway setup`` died with
``_setup_fn() missing 1 required positional argument: 'config'`` and the adapter
never left ``retrying``.

Both call sites therefore inspect the callable first and pass only what it
actually accepts. Plugins on the documented contract are called exactly as
before — a zero-argument ``setup_fn`` never even runs its config factory.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

_CALLABLE_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)


def invoke_setup_fn(setup_fn: Callable[..., Any], config_factory: Callable[[], Any]) -> Any:
    """Run a platform ``setup_fn``, supplying config only when its signature requires it.

    ``setup_fn()`` (the documented contract) is called with no arguments;
    a legacy ``setup_fn(config)`` / ``setup_fn(*, config)`` receives whatever
    ``config_factory()`` returns. ``config_factory`` runs at most once and only
    when an argument is genuinely needed, so the documented path stays free of
    the config load. A signature that cannot be introspected falls back to the
    documented zero-argument call.
    """
    try:
        parameters = list(inspect.signature(setup_fn).parameters.values())
    except (TypeError, ValueError):
        return setup_fn()
    required = [
        p for p in parameters
        if p.default is inspect.Parameter.empty and p.kind in _CALLABLE_KINDS
    ]
    if not required:
        return setup_fn()
    first = required[0]
    if first.kind is inspect.Parameter.KEYWORD_ONLY:
        return setup_fn(**{first.name: config_factory()})
    return setup_fn(config_factory())


def connect_adapter(adapter: Any, *, is_reconnect: bool = False) -> Any:
    """Call ``adapter.connect``, forwarding ``is_reconnect`` only when it is accepted.

    Returns whatever ``connect()`` returns (an awaitable for async adapters).
    A signature that cannot be introspected is treated as not accepting the
    keyword, mirroring ``BasePlatformAdapter._accepts_kwarg(..., unknown=False)``
    — omitting a defaulted keyword is safe, passing an unknown one raises.
    """
    connect = adapter.connect
    try:
        parameters = inspect.signature(connect).parameters
    except (TypeError, ValueError):
        accepts = False
    else:
        accepts = "is_reconnect" in parameters or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )
    return connect(is_reconnect=is_reconnect) if accepts else connect()
