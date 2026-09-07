"""Coalesce concurrent synchronous HTTP reads before threadpool admission."""
import asyncio
import inspect
from functools import partial, wraps
from weakref import WeakKeyDictionary
from typing import get_type_hints

from anyio.to_thread import run_sync
from hermes_constants import get_hermes_home


def _freeze(value):
    if isinstance(value, dict):
        return (type(value), frozenset((_freeze(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(map(_freeze, value)))
    if isinstance(value, (set, frozenset)):
        return (type(value), frozenset(map(_freeze, value)))
    return (type(value), value)


def coalesced_read(func, *, thread_runner=run_sync):
    """Return an async, in-flight-only wrapper around a synchronous read.

    One worker per decorated function/event loop; identical home + bound
    arguments share its result. Arguments must be hashable or built-in
    containers of hashable values and must not be mutated during the read.
    A custom async thread_runner must await the supplied zero-argument callable
    through a context-preserving threadpool, not abandon running work.
    """
    signature = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)
    signature = signature.replace(
        parameters=[p.replace(annotation=hints.get(p.name, p.annotation))
                    for p in signature.parameters.values()],
        return_annotation=hints.get("return", signature.return_annotation),
    )
    states = WeakKeyDictionary()

    @wraps(func)
    async def wrapped(*args, **kwargs):
        loop = asyncio.get_running_loop()
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        key = (str(get_hermes_home()), _freeze(bound.arguments))
        hash(key)
        pending, admission = states.setdefault(loop, ({}, asyncio.Semaphore(1)))
        task = pending.get(key)
        if task is None:
            async def execute():
                try:
                    async with admission:
                        return await thread_runner(partial(func, *args, **kwargs))
                finally:
                    pending.pop(key, None)
                    if not pending:
                        states.pop(loop, None)
            task = loop.create_task(execute())
            pending[key] = task
            # Observe failures even if every HTTP caller has disconnected.
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        # A disconnected caller must not release admission for a live worker.
        return await asyncio.shield(task)

    wrapped.__signature__ = signature
    wrapped.__annotations__ = hints
    return wrapped