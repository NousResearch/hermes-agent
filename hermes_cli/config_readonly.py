"""Read-only views of the config: the raw on-disk view and the loaded merged view.

Split out of :mod:`hermes_cli.config` (code-health size ratchet); ``hermes_cli.config``
re-exports ``read_raw_config_readonly``/``load_config_readonly``, so
``from hermes_cli.config import load_config_readonly`` keeps working for every caller.

The loaded view also serves a context-scoped projection: inside
``readonly_config_scope(project)`` each ``load_config_readonly()`` returns ``project`` applied to
a deepcopy of the cached config, while the cache itself and the writable
``load_config()``/``save_config()`` path keep the persisted view.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Optional

_readonly_projection: ContextVar[Optional[Callable[[dict[str, Any]], dict[str, Any]]]] = ContextVar(
    "hermes_readonly_projection", default=None
)


@contextmanager
def readonly_config_scope(project: Callable[[dict[str, Any]], dict[str, Any]]):
    """Project a copy of the cached config for read-only consumers inside the scope.
    ``project`` receives a deepcopy and returns the dict ``load_config_readonly()`` serves; the
    shared cache is never touched. ``load_config()``/``save_config()`` keep the persisted view on
    purpose — writers must never see or persist runtime-projected facts — so consumers that should
    see the projection must read via ``load_config_readonly()``."""
    token = _readonly_projection.set(project)
    try:
        yield
    finally:
        _readonly_projection.reset(token)


def read_raw_config_readonly() -> dict[str, Any]:
    """``read_raw_config()`` without the per-call deepcopy, for callers that ONLY READ.
    **Mutating the result corrupts the in-process cache for every subsequent caller.** Meant for
    per-turn policy checks that were paying a full config deepcopy 2-3x per agent turn."""
    from hermes_cli.config import _read_raw_config_impl

    return _read_raw_config_impl(want_deepcopy=False)


def load_config_readonly() -> dict[str, Any]:
    """``load_config()`` without the defensive deepcopy (~half of the 265us cache-hit cost).
    **Mutating the returned dict (or any nested structure) corrupts the in-process cache for every
    subsequent caller** — only for code paths that never write to the result.
    Inside ``readonly_config_scope()`` the served dict is that scope's projection instead."""
    from hermes_cli.config import _load_config_impl

    cfg = _load_config_impl(want_deepcopy=False)
    project = _readonly_projection.get()
    if project is None:
        return cfg
    return project(copy.deepcopy(cfg))
