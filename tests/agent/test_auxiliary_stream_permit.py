"""Permit ownership on streams, including close-before-first-next (#78576)."""
import gc
import threading
import weakref
from unittest.mock import Mock

import pytest

from agent import auxiliary_client as aux


class Stream:
    def __init__(self, failure=None):
        self.items = iter(["chunk"])
        self.failure = failure
        self.close = Mock()

    def __iter__(self):
        return self

    def __next__(self):
        if self.failure:
            raise self.failure
        return next(self.items)


def owned(stream):
    semaphore = threading.BoundedSemaphore(1)
    semaphore.acquire()
    return aux._release_sync_semaphore_after_stream(stream, semaphore), semaphore


def assert_released_once(semaphore):
    assert semaphore.acquire(blocking=False)
    assert not semaphore.acquire(blocking=False)
    semaphore.release()


def test_close_before_iteration_releases_and_closes_once():
    source = Stream()
    stream, semaphore = owned(source)
    stream.close()
    stream.close()
    assert_released_once(semaphore)
    source.close.assert_called_once_with()
    with pytest.raises(StopIteration):
        next(stream)


@pytest.mark.parametrize("failure", [None, RuntimeError("stream failed")])
def test_iteration_completion_or_error_releases_once(failure):
    source = Stream(failure)
    stream, semaphore = owned(source)
    if failure:
        with pytest.raises(RuntimeError, match="stream failed"):
            next(stream)
    else:
        assert next(stream) == "chunk"
        assert not semaphore.acquire(blocking=False)
        with pytest.raises(StopIteration):
            next(stream)
    stream.close()
    source.close.assert_called_once_with()
    assert_released_once(semaphore)


def test_close_failure_still_releases_once():
    source = Stream()
    source.close.side_effect = RuntimeError("close failed")
    stream, semaphore = owned(source)
    with pytest.raises(RuntimeError, match="close failed"):
        stream.close()
    stream.close()
    assert_released_once(semaphore)


def test_iter_failure_still_releases():
    class Broken(Stream):
        def __iter__(self):
            raise ValueError("iter failed")
    source = Broken()
    stream, semaphore = owned(source)
    with pytest.raises(ValueError, match="iter failed"):
        next(stream)
    source.close.assert_called_once_with()
    assert_released_once(semaphore)


def test_public_call_transfers_permit_until_close(monkeypatch):
    semaphore = threading.BoundedSemaphore(1)
    source = Stream()
    monkeypatch.setattr(aux, "_acquire_sync_aux_semaphore", lambda task: semaphore)
    monkeypatch.setattr(aux, "_call_llm_impl", lambda **kwargs: source)
    stream = aux.call_llm(task="compression", messages=[], stream=True)
    assert not semaphore.acquire(blocking=False)
    stream.close()
    assert_released_once(semaphore)
    source.close.assert_called_once_with()


@pytest.mark.parametrize("started", [False, True])
def test_abandoned_stream_releases_permit(started):
    source = Stream()
    stream, semaphore = owned(source)
    if started:
        assert next(stream) == "chunk"
    reference = weakref.ref(stream)
    del stream
    gc.collect()
    assert reference() is None
    source.close.assert_called_once_with()
    assert_released_once(semaphore)
