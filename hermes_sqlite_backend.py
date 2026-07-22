"""Early sqlite backend selection for Hermes entry points.

When the optional ``modern-sqlite`` extra is installed, Hermes should prefer
``pysqlite3`` before any long-lived entry point binds stdlib ``sqlite3``.
That has to happen *before* the first ``import sqlite3`` in the process;
once another module has already imported stdlib ``sqlite3``, existing module
references keep pointing at it even if ``sys.modules`` is replaced later.

The helper is intentionally tiny and side-effect free until called. Entry
points such as ``gateway.run`` import this module and call
``select_sqlite_backend()`` immediately before their own ``import sqlite3``.
``hermes_state`` does the same so direct imports of that module still get the
same behavior outside the normal entry-point path.
"""

from __future__ import annotations

import sys


def select_sqlite_backend() -> bool:
    """Route future ``import sqlite3`` calls through ``pysqlite3`` when enabled.

    Returns True only when the alias was installed in this call. Returns False
    when ``pysqlite3`` is unavailable (the default install) or when ``sqlite3``
    was already imported too early for a safe swap.
    """
    if "sqlite3" in sys.modules:
        return False

    try:
        import pysqlite3  # type: ignore[import-not-found]
    except ImportError:
        return False

    sys.modules["sqlite3"] = pysqlite3
    dbapi2 = getattr(pysqlite3, "dbapi2", None)
    if dbapi2 is not None:
        sys.modules["sqlite3.dbapi2"] = dbapi2
    return True
