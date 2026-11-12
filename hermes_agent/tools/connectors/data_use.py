"""Data-use requirements of connector results, collected while one tool call dispatches.

The executor binds a sink around each dispatch and connector dispatch adds to it, so a batch's entries
union into the wrapper's sink and the requirement never rides the model-visible result string.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Iterable, Iterator, Optional

_sink: contextvars.ContextVar[Optional[set[str]]] = contextvars.ContextVar("connector_data_use", default=None)


@contextlib.contextmanager
def collect_data_use() -> Iterator[set[str]]:
    sink: set[str] = set()
    token = _sink.set(sink)
    try:
        yield sink
    finally:
        _sink.reset(token)


def record_data_use(requires: Iterable[str]) -> None:
    """No-op outside an agent dispatch (execute_code RPC, MCP serve): no message is built there."""
    sink = _sink.get()
    if sink is not None:
        sink.update(requires)
