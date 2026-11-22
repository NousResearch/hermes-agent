"""Refresh coordination and cross-process locking for ``provider_models_cache.json``.

Parallel prefetch workers, SWR refresh threads and other Hermes processes (dashboard, desktop,
CLI) all rewrite the one cache file; each write must re-read it under this lock or a slow writer
puts back the snapshot it loaded before its fetch, dropping rows written meanwhile.
"""

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from agent.memory_provider import spawn_context_thread

logger = logging.getLogger(__name__)

_thread_lock = threading.Lock()
_holder = threading.local()
_swr_refresh_lock = threading.Lock()
_swr_refresh_inflight: dict[str | tuple[str, str], threading.Event] = {}


def _swr_refresh_key(cache_key: str) -> str | tuple[str, str]:
    # Each routed profile has its own disk cache and credentials.
    from hermes_constants import get_hermes_home_override, hermes_home_key
    return cache_key if get_hermes_home_override() is None else (hermes_home_key(), cache_key)


def wait_for_provider_models_refresh(cache_key: str, *, timeout: float) -> bool:
    """Wait for this profile's in-flight catalog refresh, if any. Call off the event loop."""
    key = _swr_refresh_key(cache_key)
    with _swr_refresh_lock:
        done = _swr_refresh_inflight.get(key)
    return done is None or done.wait(timeout)


def start_background_cache_refresh(cache_key: str, refresh: Callable[[], None]) -> None:
    """Run one refresh per profile and key, carrying its scope into the worker."""
    key = _swr_refresh_key(cache_key)
    with _swr_refresh_lock:
        if key in _swr_refresh_inflight:
            return
        done = _swr_refresh_inflight[key] = threading.Event()

    def run() -> None:
        try:
            refresh()
        finally:
            with _swr_refresh_lock:
                _swr_refresh_inflight.pop(key, None)
                done.set()

    spawn_context_thread(run, name=f"model-cache-swr-{cache_key}").start()


@contextmanager
def provider_models_cache_lock(path: Path):
    from hermes_cli.auth import _file_lock
    with _thread_lock, _file_lock(path.with_name(path.name + ".lock"), _holder, 10.0,
                                  f"Timed out waiting for the provider models cache lock ({path})"):
        yield


def write_under_cache_lock(path: Path, write: Callable[[], None]) -> None:
    """Run ``write`` (re-read, set one row, save) under the thread + cross-process lock.

    The file is a cache: an unwritable home or a wedged lock skips the write instead of failing
    the model listing that triggered it."""
    try:
        with provider_models_cache_lock(path):
            write()
    except (OSError, TimeoutError):
        logger.debug("provider models cache write skipped (%s)", path, exc_info=True)
