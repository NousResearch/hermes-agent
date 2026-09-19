"""Database routing shared by the small cron stores."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from hermes_constants import get_hermes_home


def uses_postgres(*, sqlite_override: object = None) -> bool:
    """Use the profile backend unless a test explicitly selected a SQLite file."""
    if sqlite_override is not None:
        return False
    from hermes_state_backend import resolve_database_settings

    home = get_hermes_home().resolve()
    return resolve_database_settings(home / "state.db").backend == "postgres"


@contextmanager
def postgres_transaction(*, store: str, write: bool) -> Iterator[object]:
    """Borrow the profile pool and keep one cron operation in one transaction."""
    from hermes_state_registry import acquire, release_or_close

    database = acquire(get_hermes_home().resolve() / "state.db")
    try:
        if database.backend != "postgres":
            raise RuntimeError("PostgreSQL cron storage requires database.backend: postgres")
        context = (
            database._write_ctx(lock_scope=f"cron.{store}")
            if write
            else database._read_ctx()
        )
        with context as connection:
            yield connection
    finally:
        release_or_close(database)


def is_postgres_connection(connection: object) -> bool:
    return getattr(connection, "backend", "sqlite") == "postgres"
