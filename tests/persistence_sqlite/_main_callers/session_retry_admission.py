"""Main's ``gateway/session_retry_admission.py`` bodies (as of the pinned main), the legacy side of the backend parity checks."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable

from persistence.core.errors import AuthorityRefusal


class RetryRewriteRefused(RuntimeError):
    """The planned rewrite no longer applies (the transcript changed since the read, a turn lease is
    live, the session closed): the re-send was refused with it and nothing was written."""


@dataclass
class RetryRewrite:
    """One ``/retry`` rewrite, applied at most once inside the re-send's admission write."""
    db: Any
    session_id: str
    source: Any  # the re-send's own SessionSource object: only that admission takes the binding
    write: Callable[[Any], None]  # (conn) -> None, inside the admission transaction
    on_commit: Callable[[], None]  # on the loop, once the admission that applied it committed
    applied: bool = False
    taken: bool = field(default=False, repr=False)

    def guard(self, conn):
        self.applied = False  # the writer may retry the whole transaction
        try:
            self.write(conn)
        except (AuthorityRefusal, sqlite3.OperationalError):
            raise  # gate refusals and writer contention keep their own handling
        except Exception as exc:
            raise RetryRewriteRefused(str(exc)) from exc
        self.applied = True
