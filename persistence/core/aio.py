"""The awaitable form of every verb: ``await store.aio.<repository>.<verb>(...)``.

Every verb is defined once, as a plain call; event-loop code awaits the same verb through ``store.aio``, with the same meaning, records and errors. The backend owns execution: an assembled store exposes ``executor``, and the default view here runs the plain verb there with the caller's context copied, handing the result or the exception back to the awaiting coroutine. A backend with a native asynchronous driver supplies its own ``aio`` instead.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
from concurrent.futures import Executor
from typing import Any, Callable

__all__ = ["AwaitableRepository", "AwaitableStore"]


class AwaitableRepository:
    """The awaitable form of one repository: each public verb, awaited, runs the plain verb on the store's executor."""

    def __init__(self, repository: Any, executor: Executor) -> None:
        self._repository = repository
        self._executor = executor

    def __getattr__(self, name: str) -> Callable[..., Any]:
        verb = getattr(self._repository, name)
        if name.startswith("_") or not callable(verb):
            raise AttributeError(name)

        @functools.wraps(verb)
        async def awaitable(*args: Any, **kwargs: Any) -> Any:
            context = contextvars.copy_context()
            call = functools.partial(context.run, _complete, verb, args, kwargs)
            return await asyncio.get_running_loop().run_in_executor(self._executor, call)

        return awaitable


def _complete(verb: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    """Run the verb to completion on the executor: a verb that yields rows lazily is drained there, never on the loop."""
    result = verb(*args, **kwargs)
    return tuple(result) if inspect.isgenerator(result) else result


class AwaitableStore:
    """The default ``aio`` view of an assembled store: ``store.aio.<repository>`` is that repository's awaitable form on ``store.executor``."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def __getattr__(self, name: str) -> AwaitableRepository:
        if name.startswith("_"):
            raise AttributeError(name)
        return AwaitableRepository(getattr(self._store, name), self._store.executor)
