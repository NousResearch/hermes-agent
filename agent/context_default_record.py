"""Which context-window resolutions ended on a static default, recorded by the resolver itself.

``get_model_context_length`` has many sources (config, endpoint metadata, /models, provider catalogs,
local probes, Bedrock's table and probe). Only the resolver knows whether its answer came from one of
them or from a default (the generic 256K, Bedrock's 128K), so the "fallback" label reads this record
instead of re-deriving the answer. Keyed by profile home: a multiplexed gateway resolves for several
profiles in one process.
"""
from __future__ import annotations

import contextvars
import functools
import inspect
from typing import Callable, Optional

# Set while a recorded resolution runs; the default branches flip it via mark_context_defaulted().
_DEFAULTED: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar("context_defaulted", default=None)
_RECORDS: dict[tuple[str, str, str], int] = {}  # (profile home, model, base_url) -> the defaulted window
_MAX_RECORDS = 512


def _key(model, base_url) -> tuple[str, str, str]:
    from hermes_constants import hermes_home_key
    return hermes_home_key(), str(model or "").strip(), str(base_url or "").strip().rstrip("/")


def mark_context_defaulted() -> None:
    """Called by a resolver branch that returns a static default instead of a known window."""
    flag = _DEFAULTED.get()
    if flag is not None:
        flag[0] = True


def records_context_defaults(resolve: Callable[..., int]) -> Callable[..., int]:
    """Wrap the resolver: remember a defaulted answer for its (model, base_url), forget it otherwise."""
    signature = inspect.signature(resolve)

    @functools.wraps(resolve)
    def wrapper(*args, **kwargs):
        flag = [False]
        token = _DEFAULTED.set(flag)
        try:
            context_length = resolve(*args, **kwargs)
        finally:
            _DEFAULTED.reset(token)
        call = signature.bind_partial(*args, **kwargs).arguments
        key = _key(call.get("model"), call.get("base_url"))
        if flag[0]:
            if len(_RECORDS) >= _MAX_RECORDS:
                _RECORDS.clear()
            _RECORDS[key] = context_length
        else:
            _RECORDS.pop(key, None)
        return context_length
    return wrapper


def was_context_defaulted(context_length, model, base_url="") -> bool:
    """True when the last resolution of ``model`` on ``base_url`` in this profile defaulted to ``context_length``."""
    return _RECORDS.get(_key(model, base_url)) == context_length
