"""Presentation ownership follows the canonical cron agent inside the gateway.

External scheduler workers only submit and observe admissions. They never receive
presentation authority; neither persisted job metadata nor a locator grants it.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import math
import os
import threading
import time
import uuid
from pathlib import Path

from agent.memory_provider import spawn_context_thread
from cron.approval_relay import ApprovalLifetime, ApprovalRequest, ApprovalUnavailable, presenters_for
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
MAX_ACTIVE = 16
MAX_INVOCATIONS = 1024
_current = contextvars.ContextVar("cron_approval_execution", default=None)


class ExecutionPresentations:
    def __init__(self, owner, cancel_event):
        self.owner = owner
        self.home = str(Path(owner[0].db.db_path).resolve().parent)
        self.pid = os.getpid()
        self.cancel_event = cancel_event
        self.agent = None
        self._presenters = presenters_for(self.home)
        self._closed = threading.Event()
        self._lock = threading.RLock()
        self._seen = set()
        self._waits = set()

    def live(self):
        authority = self.owner[0]
        return (self.pid == os.getpid() and not self._closed.is_set()
                and not getattr(authority, "retired", False)
                and not getattr(authority.runner, "_draining", False)
                and not (self.cancel_event is not None and self.cancel_event.is_set()))

    def open(self, provider, *, session_id, external_session_id, call_id, deadline, check):
        if any(not isinstance(value, str) or not value or len(value) > 512
               for value in (session_id, external_session_id, call_id)):
            raise ApprovalUnavailable("Invalid cron approval identity")
        if (type(deadline) not in (int, float) or not math.isfinite(deadline)
                or deadline <= time.monotonic()):
            raise ApprovalUnavailable("Invalid cron approval deadline")
        invocation_context = contextvars.copy_context()
        with self._lock:
            presenter = self._presenters.get(provider)
            if (not self.live() or self.agent is None or self.agent.session_id != session_id
                    or presenter is None or presenter.closed.is_set()):
                raise ApprovalUnavailable("Cron approval execution or presenter unavailable")
            key = (provider, call_id)
            if key in self._seen or len(self._seen) >= MAX_INVOCATIONS or len(self._waits) >= MAX_ACTIVE:
                raise ApprovalUnavailable("Duplicate or excessive cron approval invocation")
            request = ApprovalRequest(
                self.home, self.owner[2], self.owner[3], session_id, external_session_id,
                call_id, uuid.uuid4().hex, min(deadline, time.monotonic() + 600),
            )

            def live():
                if not self.live() or presenter.closed.is_set() or self.agent.session_id != session_id:
                    return False
                try:
                    invocation_context.copy().run(check)
                except Exception:
                    logger.debug("Cron approval invocation ended", exc_info=True)
                    return False
                return True

            lifetime = ApprovalLifetime(request.deadline, live)
            # Fence before presenter I/O; an ambiguous send never gains an automatic retry.
            self._seen.add(key)
            self._waits.add(lifetime)
        try:
            spawn_context_thread(
                lambda: presenter.context.copy().run(self._present, presenter, request, lifetime),
                name="cron-approval-presenter",
            ).start()
        except BaseException:
            lifetime.close("failed")
            with self._lock:
                self._waits.discard(lifetime)
            raise
        return lifetime

    def _present(self, presenter, request, lifetime):
        try:
            lifetime.check()
            with presenter.callback(request, lifetime):
                while not lifetime.ended.wait(min(0.5, lifetime.remaining_seconds)):
                    lifetime.check()
        except ApprovalUnavailable:
            pass
        except Exception:
            logger.warning("Cron approval presenter failed", exc_info=True)
        finally:
            lifetime.close("failed")
            with self._lock:
                self._waits.discard(lifetime)

    def close(self):
        self._closed.set()
        with self._lock:
            waits = tuple(self._waits)
        for lifetime in waits:
            lifetime.close("execution ended")


@contextlib.contextmanager
def presentation_scope(cancel_event):
    from gateway.session_cron import current_execution

    owner = current_execution()
    scope = ExecutionPresentations(owner, cancel_event) if owner is not None else None
    if scope is not None and not scope._presenters:
        scope = None
    token = _current.set(scope)
    try:
        yield
    finally:
        if scope is not None:
            scope.close()
        _current.reset(token)


def current_scope():
    from cron.execution_identity import current_cron_execution
    from gateway.session_cron import current_execution

    scope, execution = _current.get(), current_cron_execution()
    if (scope is not None and execution is not None and scope.live()
            and current_execution() == scope.owner
            and (execution.job_id, execution.execution_id) == scope.owner[2:]
            and str(get_hermes_home().resolve()) == scope.home):
        return scope
    return None


def bind_agent(agent):
    """Only the original owner-created agent can bind; compression follows that object."""
    scope = current_scope()
    if scope is not None and scope.agent is None and agent.session_id == scope.owner[1]:
        scope.agent = agent
