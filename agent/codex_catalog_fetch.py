"""Patient Codex account-catalog fetch: bounded foreground wait, background completion.

``/backend-api/codex/models`` answers in well under a second for most ChatGPT accounts and in
~30 s for some. Every caller (picker open, setup wizard, ``/model`` switch, the context-window
probe) used to wait the whole HTTP timeout and then fall back to the curated list, so a slow
account lost its account-gated rows (Astra) on every surface and paid the stall each time.

Here the caller waits at most ``foreground_timeout`` for the request; a slow answer keeps running
on a daemon thread for up to the fetch's own HTTP timeout and is handed to ``on_complete`` so the
caller's cache serves it on the next read. One request per ``key`` is in flight at a time; a
second caller joins the wait instead of starting another probe.
"""
from __future__ import annotations

import contextvars
import logging
import threading
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

CatalogResult = Tuple[List[Any], Optional[int]]  # (entries, last HTTP status), as fetch_codex_catalog_entries

_lock = threading.Lock()
_inflight: dict[str, "_Probe"] = {}


class _Probe:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.result: Optional[CatalogResult] = None
        self.error: Optional[BaseException] = None
        self.on_complete: list[Callable[[CatalogResult], Any]] = []


def fetch_catalog_patiently(
    key: str,
    fetch: Callable[[], CatalogResult],
    *,
    foreground_timeout: float,
    on_complete: Optional[Callable[[CatalogResult], Any]] = None,
) -> Optional[CatalogResult]:
    """Run ``fetch`` (which carries its own, long HTTP timeout) and return its result when it
    lands within ``foreground_timeout``. Otherwise return ``None`` now and, if the request later
    succeeds with a non-empty catalog, call ``on_complete(result)`` once from the worker thread.
    A failed fetch yields ``([], None)`` so the caller's fallback path runs; failures never
    reach ``on_complete``.
    """
    with _lock:
        probe = _inflight.get(key)
        started = probe is None
        if probe is None:
            probe = _inflight[key] = _Probe()

    if started:
        ctx = contextvars.copy_context()  # the worker reads the calling profile's credentials/caches

        def _run() -> None:
            try:
                probe.result = fetch()
            except BaseException as exc:  # noqa: BLE001 - surfaced to foreground waiters via probe.error
                probe.error = exc
            finally:
                with _lock:
                    _inflight.pop(key, None)
                    callbacks, probe.on_complete = probe.on_complete, []
                    probe.done.set()
            if probe.result is not None and probe.result[0]:
                for callback in callbacks:
                    try:
                        callback(probe.result)
                    except Exception:
                        logger.debug("Codex catalog completion callback failed", exc_info=True)

        threading.Thread(target=lambda: ctx.run(_run), daemon=True, name="codex-catalog-fetch").start()

    if not probe.done.wait(foreground_timeout):
        with _lock:
            if not probe.done.is_set():  # still running: hand the answer to the cache when it lands
                if on_complete is not None:
                    probe.on_complete.append(on_complete)
                logger.debug("Codex catalog request still running after %.1fs; serving the fallback and "
                             "finishing in the background", foreground_timeout)
                return None
    if probe.error is not None:
        logger.debug("Codex catalog request failed: %s", probe.error)
        return [], None
    return probe.result
