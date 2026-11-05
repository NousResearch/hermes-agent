"""Thread races for the contract suites: verbs are synchronous, so contenders race on real threads."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


def run_concurrently(*calls: Callable[[], Any], return_exceptions: bool = False) -> list[Any]:
    """Run each zero-argument call on its own thread, released together by a barrier; results in call order.

    With ``return_exceptions`` a call's exception is its result; otherwise the first one in call order is raised after every call finished.
    """
    if not calls:
        return []
    barrier = threading.Barrier(len(calls))

    def run(call: Callable[[], Any]) -> tuple[bool, Any]:
        barrier.wait(timeout=30.0)
        try:
            return True, call()
        except BaseException as exc:
            return False, exc

    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        outcomes = list(pool.map(run, calls))
    if not return_exceptions:
        for ok, value in outcomes:
            if not ok:
                raise value
    return [value for _, value in outcomes]
