"""Profile-local coordination for plugin installation writers and clone snapshots.

This is a cooperative lock, not a filesystem sandbox: manual edits, third-party
installers and general config editors do not participate. Keep the lock file in
the home (outside the replaceable plugins tree), and never copy it to a clone.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import wraps
import os
from pathlib import Path
import threading

from hermes_constants import assert_named_profile_home_live, get_hermes_home

_ownership = threading.local()


def installation_lock_held(home: Path | None = None) -> bool:
    """Whether this thread owns a transaction, optionally for one specific home."""
    held = getattr(_ownership, "held", ())
    if home is not None:
        return (os.getpid(), str(home.resolve())) in held
    return any(pid == os.getpid() for pid, _ in held)


class PluginDiscoveryLock:
    """Manager-local reentrant gate, always installation → discovery.

    Keep the order at the lock itself: eager/deferred loading, unload callbacks,
    and the memory-provider collector all enter the same manager gate. Binding
    the immutable home here also covers callers outside a discovery home scope.
    """

    def __init__(self, home: Path):
        self._home = home
        self._lock = threading.RLock()
        self._entries = threading.local()

    def __enter__(self):
        with ExitStack() as stack:
            stack.enter_context(plugin_installation_lock(self._home))
            stack.enter_context(self._lock)
            entries = getattr(self._entries, "stack", None)
            if entries is None:
                entries = self._entries.stack = []
            entries.append(stack.pop_all())
        return self

    def __exit__(self, *exc):
        return self._entries.stack.pop().__exit__(*exc)


@contextmanager
def plugin_installation_lock(home: Path | None = None):
    """Serialize a whole installation transaction; same-thread nesting is safe.

    Explicit homes allow clone readers to lock their source without changing the
    active profile. Nesting distinct homes is rejected before waiting so A→B
    and B→A cannot deadlock. Acquire before discovery, then config/state locks.
    Ownership is thread-local, not a ContextVar: copied contexts in worker
    threads must still acquire the thread/OS lock themselves.
    """
    from hermes_cli.plugins_state import _locked_plugin_state

    home = Path(home if home is not None else get_hermes_home()).resolve()
    key = (os.getpid(), str(home))
    held = getattr(_ownership, "held", None)
    if held is None:
        held = _ownership.held = set()
    if key in held:
        yield
        return
    if any(pid == os.getpid() for pid, _ in held):
        raise RuntimeError("Cannot nest plugin installation locks for distinct homes")
    assert_named_profile_home_live(home)
    with _locked_plugin_state(home / "plugin-installation"):
        held.add(key)
        try:
            yield
        finally:
            held.remove(key)


def installation_transaction(operation):
    """Hold the active home's lock across an entire synchronous plugin operation.

    The CLI, dashboard and pack fan-out compose these entry points; nesting keeps
    their code, provenance, housekeeping and activation writes in one boundary
    while direct calls to the shared cores receive the same protection.
    """
    @wraps(operation)
    def locked(*args, **kwargs):
        with plugin_installation_lock():
            return operation(*args, **kwargs)
    return locked
