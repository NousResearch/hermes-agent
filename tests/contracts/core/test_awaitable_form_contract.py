"""Contract cases for the awaitable form: ``await store.aio.<repository>.<verb>(...)`` means exactly what the plain call means."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from persistence import AwaitableRepository, ExportTooLarge, StoreContention, TransientStoreError
from persistence.conversation.sessions import NewSession


def _awaited(awaitable):
    return asyncio.run(awaitable)


_REPOSITORIES = (
    "sessions", "messages", "lineage", "search", "usage", "maintenance", "state_meta", "profile_identity", "routing",
    "health", "admissions", "runtime", "bindings", "coordination", "worker_executions", "hosted_rooms",
    "hosted_room_peers", "hosted_room_replicas", "hosted_room_driver", "hosted_room_policy", "hosted_room_attachments",
    "usage_analytics",
)


def test_every_repository_has_its_awaitable_form(backend, scope_a):
    store = backend.open(scope_a)
    unfolded = getattr(backend, "unfolded_capabilities", frozenset())
    for name in (name for name in _REPOSITORIES if name not in unfolded):
        repository = getattr(store, name)
        verbs = [v for v in dir(repository) if not v.startswith("_") and callable(getattr(repository, v))]
        assert verbs and all(callable(getattr(getattr(store.aio, name), v)) for v in verbs)


def test_both_forms_give_identical_results(backend, scope_a):
    store_a = backend.open(scope_a)
    _awaited(store_a.aio.state_meta.set("aio:key", "one"))
    store_a.state_meta.set("aio:other", "two")
    assert _awaited(store_a.aio.state_meta.get("aio:other")) == store_a.state_meta.get("aio:other") == "two"
    assert store_a.state_meta.get("aio:key") == "one"
    assert _awaited(store_a.aio.state_meta.list_prefix("aio:")) == store_a.state_meta.list_prefix("aio:")
    _awaited(store_a.aio.sessions.create(NewSession(session_id="aio-s", source="cli")))
    assert _awaited(store_a.aio.sessions.get("aio-s")) == store_a.sessions.get("aio-s")
    assert _awaited(store_a.aio.health.probe()) == store_a.health.probe()


def test_both_forms_raise_the_same_error(backend, scope_a):
    store_a = backend.open(scope_a)
    with pytest.raises(ValueError) as plain:
        store_a.state_meta.get("")
    with pytest.raises(ValueError) as awaited:
        _awaited(store_a.aio.state_meta.get(""))
    assert type(awaited.value) is type(plain.value) and awaited.value.args == plain.value.args


class _Probe:
    """A stand-in repository recording where and with which context its verbs ran."""

    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker", default="unset")

    def where(self) -> tuple[int, str]:
        return threading.get_ident(), self.marker.get()

    def rows(self):
        yield from (1, 2, 3)

    def refuse(self) -> None:
        raise ExportTooLarge("s", 9, 3)

    def overloaded(self) -> None:
        raise StoreContention("write waited its whole patience")


def test_the_default_view_runs_off_the_loop_thread_with_the_callers_context():
    executor_store = ThreadPoolExecutor(max_workers=1)
    try:
        repository = AwaitableRepository(_Probe(), executor_store)

        async def scenario():
            _Probe.marker.set("caller")
            thread, marker = await repository.where()
            return threading.get_ident(), thread, marker, await repository.rows()

        loop_thread, verb_thread, marker, rows = asyncio.run(scenario())
        assert verb_thread != loop_thread and marker == "caller"
        assert rows == (1, 2, 3)
        with pytest.raises(ExportTooLarge) as refused:
            asyncio.run(repository.refuse())
        assert (refused.value.session_id, refused.value.message_count, refused.value.limit) == ("s", 9, 3)
        with pytest.raises(AttributeError):
            repository._private
    finally:
        executor_store.shutdown(wait=True)


def test_overload_is_a_transient_store_error_with_reason_busy():
    executor_store = ThreadPoolExecutor(max_workers=1)
    try:
        with pytest.raises(TransientStoreError) as overloaded:
            asyncio.run(AwaitableRepository(_Probe(), executor_store).overloaded())
        assert overloaded.value.reason == "busy"
        assert TransientStoreError("disk i/o error").reason is None
    finally:
        executor_store.shutdown(wait=True)
