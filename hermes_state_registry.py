"""Compatibility module for outside plugins that still import ``hermes_state_registry``; removed in v0.27.0.

``acquire`` opens a ``hermes_state.SessionDB`` on the storage interface and ``release`` closes it; nothing is shared
between callers. The first ``acquire`` in a process emits the ``hermes_state`` stand-in's one deprecation warning,
unless another stand-in name has already emitted it.
"""

import warnings as _warnings

import hermes_state as _hermes_state


def acquire(db_path=None):
    """A ``hermes_state.SessionDB`` on the store of ``db_path`` (a ``state.db``), or of the current profile."""
    if _hermes_state._first_warning():
        _warnings.warn(
            f"hermes_state_registry.acquire is deprecated and is removed in {_hermes_state._REMOVAL_RELEASE}; use "
            f"persistence_assembly.open_persistence(profile_scope()) instead ({_hermes_state._MIGRATION_GUIDE}).",
            DeprecationWarning, stacklevel=2)
    return _hermes_state._SessionDB(db_path)


def release(db) -> bool:
    """Close ``db``: ``True``, as for any ``SessionDB`` ``acquire`` lent, and ``False`` for None."""
    if db is None:
        return False
    db.close()
    return True


def release_or_close(db) -> None:
    """Close ``db``."""
    release(db)
