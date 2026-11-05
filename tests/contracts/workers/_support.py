"""Shared construction for the worker contract suites: a session, a live epoch, a registered execution and the receipt identity of its writes.

Every step goes through public verbs: the epoch through ``SessionRuntimeRepository.begin_epoch``, sessions through ``SessionRepository.create``, admissions through ``AdmissionRepository``, the execution through ``WorkerExecutionRepository.register``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import pytest

from persistence import AuthorityEpoch, ExecutionGeneration, SessionId
from persistence.conversation.sessions import NewSession
from persistence.core.errors import AuthorityRefusal
from persistence.workers.executions import WorkerExecutionId, WorkerExecutionRecord, WorkerKind
from tests.contracts.conftest import require_capability

SESSION = SessionId("sess-worker")
EXECUTION = WorkerExecutionId("exec-1")
SECRET = "adoption-secret-0123456789abcdef"


def require_workers(backend: Any) -> None:
    require_capability(backend, "receipted_writes")


def begin_epoch(backend: Any, scope: Any) -> AuthorityEpoch:
    return AuthorityEpoch(backend.open(scope).runtime.begin_epoch(instance_id="instance-contract"))


def seed(backend: Any, scope: Any, session_id: SessionId = SESSION, *, source: str = "contract", **fields: Any) -> AuthorityEpoch:
    """One session plus a live authority epoch."""
    require_workers(backend)
    backend.open(scope).sessions.create(NewSession(session_id=session_id, source=source, **fields))
    return begin_epoch(backend, scope)


def register(backend: Any, scope: Any, epoch: Any, *, generation: Any = 0, execution_id: Any = EXECUTION, session_id: Any = SESSION,
             kind: Any = WorkerKind.COMPUTE, adoption_secret: Any = SECRET, **kwargs: Any) -> WorkerExecutionRecord:
    return backend.open(scope).worker_executions.register(
        epoch=epoch, execution_id=execution_id, session_id=session_id, generation=generation, kind=kind, adoption_secret=adoption_secret, **kwargs)


def ident(epoch: Any, *, generation: Any = 0, sequence: Any = 1, execution_id: Any = EXECUTION, session_id: Any = SESSION) -> dict[str, Any]:
    """The receipt identity of one worker write."""
    return {"epoch": epoch, "execution_id": execution_id, "session_id": session_id, "generation": ExecutionGeneration(generation) if type(generation) is int else generation, "sequence": sequence}


def worker(backend: Any, scope: Any, *, source: str = "contract", **fields: Any) -> tuple[Any, AuthorityEpoch]:
    """A store with a registered execution on :data:`SESSION`, and its epoch."""
    epoch = seed(backend, scope, source=source, **fields)
    register(backend, scope, epoch)
    return backend.open(scope), epoch


def last_sequence(backend: Any, scope: Any, epoch: Any) -> int:
    """The execution's ``last_sequence`` through ``verify_claim``."""
    return backend.open(scope).worker_executions.verify_claim(execution_id=EXECUTION, session_id=SESSION, generation=ExecutionGeneration(0), adoption_secret=SECRET).last_sequence


def revision(store: Any, session_id: SessionId = SESSION) -> int:
    return store.sessions.get(session_id).runtime_revision


@contextmanager
def refused(reason: str) -> Iterator[Any]:
    """Expect an :class:`AuthorityRefusal` with ``reason`` (``FenceStale``, ``ProofRejected`` and ``UnknownExecution`` carry theirs)."""
    with pytest.raises(AuthorityRefusal) as caught:
        yield caught
    assert caught.value.reason == reason, caught.value.reason


def user(content: Any, **extra: Any) -> dict[str, Any]:
    return {"role": "user", "content": content, **extra}


def assistant(content: Any, **extra: Any) -> dict[str, Any]:
    return {"role": "assistant", "content": content, **extra}
