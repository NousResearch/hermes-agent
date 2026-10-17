"""Transfer a synchronous auxiliary permit to its stream's lifetime."""
import threading
from typing import Any


class SemaphoreOwnedStream:
    """Unlike a generator finally, close works before the first iteration."""

    def __init__(self, stream: Any, semaphore: threading.BoundedSemaphore):
        self._stream = stream
        self._iterator = None
        self._semaphore = semaphore
        self._closed = False
        self._close_lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            if self._iterator is None:
                self._iterator = iter(self._stream)
            return next(self._iterator)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        try:
            close = getattr(self._stream, "close", None)
            if callable(close):
                close()
        finally:
            self._semaphore.release()
