"""Request-local trusted runtime values for agent tool subprocesses.

Values are bound by authenticated gateway entry points and bridged only into
subprocess environments. They are never copied into prompts or global process
environment state.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Mapping

_runtime_env: ContextVar[dict[str, str]] = ContextVar(
    "hermes_request_runtime_env", default={}
)


def get_runtime_env() -> dict[str, str]:
    """Return a copy of the runtime values bound to the current request."""
    return dict(_runtime_env.get())


@contextmanager
def bind_runtime_env(values: Mapping[str, str]) -> Iterator[None]:
    """Bind trusted runtime values for one request and restore on exit."""
    token = _runtime_env.set(dict(values))
    try:
        yield
    finally:
        _runtime_env.reset(token)
