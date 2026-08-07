"""Hermetic environment for ad-hoc Kanban regression and E2E harnesses.

This module deliberately imports no Kanban code at module load time.  Worker
processes inherit board/database pins from their dispatcher, so those variables
must be removed before a harness imports or resolves ``hermes_cli.kanban_db``.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_KANBAN_ENV_PREFIX = "HERMES_KANBAN_"


@dataclass(frozen=True)
class IsolatedKanbanTestEnvironment:
    """Resolved paths for a hermetic Kanban harness run."""

    home: Path
    db_path: Path


def _restore_kanban_environment(previous: dict[str, str]) -> None:
    for name in list(os.environ):
        if name == "HERMES_HOME" or name.startswith(_KANBAN_ENV_PREFIX):
            os.environ.pop(name, None)
    os.environ.update(previous)


@contextmanager
def isolated_kanban_test_context() -> Iterator[IsolatedKanbanTestEnvironment]:
    """Run an ad-hoc Kanban harness against a fresh temporary Hermes home.

    All inherited ``HERMES_KANBAN_*`` variables are cleared before importing
    the Kanban DB module.  The resolved default DB path is then required to be
    below the temporary ``HERMES_HOME``; otherwise the context fails closed
    before yielding to the harness.

    Import and use ``hermes_cli.kanban_db`` only inside this context.
    """
    previous = {
        name: value
        for name, value in os.environ.items()
        if name == "HERMES_HOME" or name.startswith(_KANBAN_ENV_PREFIX)
    }

    with tempfile.TemporaryDirectory(prefix="hermes-kanban-test-") as tmpdir:
        home = Path(tmpdir).resolve()
        for name in list(os.environ):
            if name.startswith(_KANBAN_ENV_PREFIX):
                os.environ.pop(name, None)
        os.environ["HERMES_HOME"] = str(home)

        try:
            from hermes_cli import kanban_db

            db_path = kanban_db.kanban_db_path().expanduser().resolve()
            try:
                db_path.relative_to(home)
            except ValueError as exc:
                raise RuntimeError(
                    "isolated Kanban test DB resolved outside its temporary "
                    f"HERMES_HOME: {db_path} (home: {home})"
                ) from exc

            yield IsolatedKanbanTestEnvironment(home=home, db_path=db_path)
        finally:
            _restore_kanban_environment(previous)
