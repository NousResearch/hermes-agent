"""The awaitable form of every verb: ``await store.aio.<repository>.<verb>(...)``.

Every verb is defined once, as a plain call; event-loop code awaits the same verb through ``store.aio``, with the same meaning, records and errors. The backend owns execution: an assembled store exposes ``executor``, and the default view here runs the plain verb there with the caller's context copied, handing the result or the exception back to the awaiting coroutine. A backend with a native asynchronous driver supplies its own ``aio`` instead.

Two kinds of verb have no awaitable form. A verb whose protocol declares a context manager as its return (``messages.read_snapshot``): the block would be entered on the loop thread while each read inside it ran on an executor thread. A verb whose protocol declares an iterator as its return (``messages.terminal_tool_call_payloads``): awaited whole it would hold every row at once and could not stop early. Event-loop code runs either through the plain verb off the loop.
"""

from __future__ import annotations

import asyncio
import collections.abc
import contextlib
import contextvars
import functools
import inspect
import typing
from concurrent.futures import Executor
from typing import Any, Callable, Optional

__all__ = ["AwaitableRepository", "AwaitableStore"]

# Declared returns that have no awaitable form, with the reason the refusal names.
_NO_AWAITABLE_FORM = {
    contextlib.AbstractContextManager: "returns a context manager",
    collections.abc.Iterator: "streams its rows",
}


class AwaitableRepository:
    """The awaitable form of one repository: each public verb, awaited, runs the plain verb on the store's executor. A verb ``protocol`` declares as returning a context manager or an iterator is refused."""

    def __init__(self, repository: Any, executor: Executor, protocol: Optional[type] = None) -> None:
        self._repository = repository
        self._executor = executor
        self._protocol = protocol

    def __getattr__(self, name: str) -> Callable[..., Any]:
        verb = getattr(self._repository, name)
        if name.startswith("_") or not callable(verb):
            raise AttributeError(name)
        if self._protocol is not None and (reason := _refusal(self._protocol, name)):
            raise AttributeError(f"{name} {reason} and has no awaitable form: run the plain verb off the event loop")

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


@functools.cache
def _refusal(protocol: type, name: str) -> Optional[str]:
    """Why ``protocol``'s ``name`` has no awaitable form, read from its return annotation alone (evaluated in the protocol's module; its other annotations may name ``TYPE_CHECKING``-only types), or ``None``."""
    verb = getattr(protocol, name, None)
    if not callable(verb):
        return None
    returns = inspect.get_annotations(verb).get("return")
    if isinstance(returns, str):
        returns = eval(returns, verb.__globals__)
    return _NO_AWAITABLE_FORM.get(typing.get_origin(returns))


@functools.cache
def _repository_protocols() -> dict[str, Any]:
    """``{attribute: declared protocol}`` of ``PersistenceStore``."""
    from .backend import PersistenceStore  # backend.py imports this module

    return typing.get_type_hints(PersistenceStore)


class AwaitableStore:
    """The default ``aio`` view of an assembled store: ``store.aio.<repository>`` is that repository's awaitable form on ``store.executor``."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def __getattr__(self, name: str) -> AwaitableRepository:
        if name.startswith("_"):
            raise AttributeError(name)
        return AwaitableRepository(getattr(self._store, name), self._store.executor, _repository_protocols().get(name))
