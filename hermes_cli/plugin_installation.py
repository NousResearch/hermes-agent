"""Profile-local coordination for plugin installation writers and clone snapshots.

This is a cooperative lock, not a filesystem sandbox: manual edits, third-party
installers and general config editors do not participate. The lock file lives in
the home (outside the replaceable plugins tree) and is never copied to a clone.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import wraps
import os
from pathlib import Path
import threading

from hermes_constants import assert_named_profile_home_live, get_hermes_home, get_process_hermes_home

_ownership = threading.local()


def installation_lock_held(home: Path | None = None) -> bool:
    """Whether this thread owns a transaction, optionally for one specific home."""
    held = getattr(_ownership, "held", ())
    if home is not None:
        return (os.getpid(), str(home.resolve())) in held
    return any(pid == os.getpid() for pid, _ in held)


def is_launch_home(home: Path) -> bool:
    """The launch profile keeps env-only settings that no file can rebuild; other homes rebuild from files."""
    return Path(home).resolve() == get_process_hermes_home().resolve()


class PluginDiscoveryLock:
    """Manager-local reentrant gate that always takes installation before discovery."""

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
    """Serialize a whole installation transaction; same-thread nesting is safe, nesting
    distinct homes is rejected before waiting so A→B and B→A cannot deadlock."""
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


@contextmanager
def management_scope():
    """Installation lock plus the active home's terminal policy, for one dashboard management call.
    Request scoping binds home and secrets only; provider code must never see the ambient policy."""
    from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope
    from tui_gateway.launch_profile_policy import launch_terminal_env

    home = Path(get_hermes_home()).resolve()
    with plugin_installation_lock(home):
        overlay = launch_terminal_env() if is_launch_home(home) else None
        token = install_profile_terminal_scope(home, env_overlay=overlay)
        try:
            yield home
        finally:
            reset_terminal_scope(token)


def installation_transaction(operation):
    """Hold the active home's lock across a whole synchronous plugin operation."""
    @wraps(operation)
    def locked(*args, **kwargs):
        with plugin_installation_lock():
            return operation(*args, **kwargs)
    return locked
