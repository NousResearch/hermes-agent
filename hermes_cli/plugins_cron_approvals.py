"""Generic plugin access to canonical cron approval presentation (in-process only)."""

from __future__ import annotations

import contextlib
import logging
import threading
from functools import cached_property

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


class CronApprovalContextMixin:
    @cached_property
    def cron_approvals(self):
        return CronApprovalService(self)

    def current_cron_execution(self):
        """The current cron occurrence, or None outside its execution scope."""
        from cron.execution_identity import current_cron_execution

        return current_cron_execution()


class CronApprovalService:
    def __init__(self, ctx):
        self._ctx = ctx
        self.home = str(get_hermes_home().resolve())
        self._lock = threading.RLock()
        self._closed = False
        self._registrations = []
        # Acquire cleanup before publishing anything. An abandoned load may keep
        # running after the manager swept its ledger, so also check its latch.
        handle = ctx.on_unload(self.close)
        if handle is None or ctx._load_abandoned:
            self.close()
            if handle is not None:
                handle.dispose()

    @property
    def available(self) -> bool:
        from cron.approval_relay_scope import current_scope

        scope = current_scope()
        return (not self._closed and not self._ctx._load_abandoned
                and scope is not None and scope.home == self.home and scope.agent is not None)

    def register_presenter(self, callback):
        """Register callback(request, lifetime) -> context manager; unload retires it."""
        from cron.approval_relay import ApprovalUnavailable, PresenterRegistration

        with self._lock:
            if self._closed or self._ctx._load_abandoned:
                raise ApprovalUnavailable("Cron approval plugin generation retired")
            registration = PresenterRegistration(self.home, self._ctx.plugin_id, callback)
            self._registrations.append(registration)
            return registration

    def close(self):
        with self._lock:
            self._closed = True
            for registration in self._registrations:
                registration.dispose()
            self._registrations.clear()

    @contextlib.contextmanager
    def pending(self, *, session_id: str, external_session_id: str, call_id: str, deadline: float, check):
        """Best-effort presentation only. The caller continues its own bounded decision poll."""
        from cron.approval_relay_scope import current_scope

        check()
        scope, lifetime = current_scope(), None
        try:
            if scope is not None and self.available and self.home == scope.home:
                lifetime = scope.open(
                    self._ctx.plugin_id, session_id=session_id, external_session_id=external_session_id,
                    call_id=call_id, deadline=deadline, check=check,
                )
        except Exception:
            logger.warning("Cron approval presentation unavailable; retaining external approval wait", exc_info=True)
        reason = "completed"
        try:
            check()
            yield
            check()
        except BaseException:
            reason = "cancelled"
            raise
        finally:
            if lifetime is not None:
                lifetime.close(reason)
