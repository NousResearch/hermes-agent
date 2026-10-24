"""Cross-process lock for one ``config.yaml`` read-modify-write cycle.

Every config writer reads the file, changes its copy, then atomically replaces the file. Two
processes doing that at once (gateway ``/model`` + ``hermes config set``, Desktop backend + CLI)
both start from the same snapshot, and the loser's replace silently drops the winner's keys:
two writers adding 60 keys each kept only 60 of 120 (#96571). ``_CONFIG_LOCK`` is per process and
cannot help. This is the ``config.yaml`` twin of the ``.env`` lock (#77187), on the same helper.

Lock order: this lock may be held while taking the ``.env`` lock, never the reverse.
"""

from __future__ import annotations

import functools
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_HOLDERS: dict[str, threading.local] = {}
_HOLDERS_GUARD = threading.Lock()


def _holder(path: Path) -> threading.local:
    """Per-path reentrancy tracker: multiplexed profiles each lock their own file."""
    with _HOLDERS_GUARD:
        return _HOLDERS.setdefault(str(path), threading.local())


@contextmanager
def config_write_lock(config_path: str | Path) -> Iterator[None]:
    """Hold ``_CONFIG_LOCK`` then ``<config>.lock`` from the fresh read through the atomic replace.

    Reentrant per thread. The in-process lock is always taken first: existing callers already hold
    ``_CONFIG_LOCK`` when they save, so the opposite order would deadlock two threads."""
    from hermes_cli.auth import _file_lock
    from hermes_cli.config import _CONFIG_LOCK
    from hermes_constants import mkdir_under_hermes_home

    path = Path(config_path)
    mkdir_under_hermes_home(path.parent)  # the lock file must not resurrect a deleted profile
    with _CONFIG_LOCK, _file_lock(
        path.with_name(path.name + ".lock"),
        _holder(path),
        15.0,
        f"Timed out waiting for the config file lock ({path})",
    ):
        yield


def holding_config_write_lock(fn: Any) -> Any:
    """Run *fn* under the active profile's config lock (for whole read-modify-write commands)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from hermes_cli.config import get_config_path, is_managed

        if is_managed():  # read-only install: let fn print its refusal instead of failing on the lock file
            return fn(*args, **kwargs)
        with config_write_lock(get_config_path()):
            return fn(*args, **kwargs)

    return wrapper
