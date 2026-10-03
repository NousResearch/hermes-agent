"""Authenticated runtime principal for scheduled cron executions.

Scheduled jobs have durable ownership in ``cron/executions.py``, but the values
visible to the model (``task_id``, ``session_id``, ``platform``, ledger rows)
are all reproducible strings.  This module mints a process-local opaque
:class:`CronExecutionGrant` only after the scheduler has established durable
``running`` ownership, and exposes :func:`verify_cron_execution` for trusted
plugin code to obtain a read-only :class:`VerifiedCronExecution` projection.

Security properties (see #130722):

* The nonce never leaves this process: it is not in prompts, model tool
  arguments, environment variables, session history, hook JSON, logs, CLI
  arguments, external-worker payloads, shell children, delegated agents, or
  nested Hermes processes.
* ``task_id`` / ``session_id`` / ``platform`` / ledger fields are consistency
  inputs, never the credential.  Verification requires the in-memory grant
  *and* a live ``running`` row owned by this process.
* Only ``source == "builtin"`` with a canonical non-null ``scheduled_instant``
  is eligible.  Manual ``hermes cron run``, ``source == "direct"``,
  off-schedule builtin runs with ``scheduled_instant IS NULL``, ordinary chat,
  and external providers never mint.
* The grant is a :class:`contextvars.ContextVar`: concurrent jobs/threads and
  profiles are isolated, terminal subprocesses never inherit it, and delegated
  subagent turns see ``None``.
* Losing ownership, reaching a terminal row, changing runtime generation, or
  changing profile scope invalidates verification immediately (checked on
  every call, not cached).
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import secrets
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

# Process-local runtime generation.  A new process gets a new value, so a grant
# can never survive a restart.  Tests may bump it to simulate a generation
# change; verification then fails for grants minted under the old value.
_RUNTIME_GENERATION: str = uuid.uuid4().hex


def get_runtime_generation() -> str:
    """Current process-local runtime generation."""
    return _RUNTIME_GENERATION


def bump_runtime_generation_for_tests() -> str:
    """Invalidate every outstanding grant (tests only)."""
    global _RUNTIME_GENERATION
    _RUNTIME_GENERATION = uuid.uuid4().hex
    return _RUNTIME_GENERATION


@dataclass
class CronExecutionGrant:
    """Process-local opaque authority for one actively owned execution.

    The ``nonce`` is the credential.  It is never exposed: no ``to_dict``,
    no logging, no serialization.  Use :func:`verify_cron_execution` to obtain
    the public :class:`VerifiedCronExecution` projection instead.
    """

    job_id: str
    execution_id: str
    source: str
    scheduled_instant: str
    profile_key: str
    profile_home: str
    pid: int
    process_started_at: Optional[int]
    process_id: str
    generation: str
    nonce: str

    def __repr__(self) -> str:  # pragma: no cover - redaction guard
        return (
            "CronExecutionGrant(job_id=%r, execution_id=%r, source=%r, "
            "scheduled_instant=%r, profile_key=%r, pid=%r, generation=%r, "
            "nonce=<redacted>)" % (
                self.job_id, self.execution_id, self.source,
                self.scheduled_instant, self.profile_key, self.pid,
                self.generation,
            )
        )


@dataclass(frozen=True)
class VerifiedCronExecution:
    """Read-only verified projection.  Never carries the opaque nonce."""

    job_id: str
    execution_id: str
    source: str
    scheduled_instant: str
    profile_home: str
    profile_key: str
    ownership_generation: str
    purpose_digest: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializable form for audit/logging.  Contains no secret material."""
        return {
            "job_id": self.job_id,
            "execution_id": self.execution_id,
            "source": self.source,
            "scheduled_instant": self.scheduled_instant,
            "profile_home": self.profile_home,
            "profile_key": self.profile_key,
            "ownership_generation": self.ownership_generation,
            "purpose_digest": self.purpose_digest,
        }


_GRANT_VAR: contextvars.ContextVar[Optional[CronExecutionGrant]] = contextvars.ContextVar(
    "hermes_cron_execution_grant", default=None
)


def _canonical_instant(value: Any) -> Optional[str]:
    try:
        from cron.occurrences import scheduled_instant as _canon
    except Exception:
        return None
    try:
        return _canon(value)
    except Exception:
        return None


def _current_profile_key() -> str:
    try:
        from hermes_constants import hermes_home_key
    except Exception:
        return ""
    try:
        return hermes_home_key()
    except Exception:
        return ""


def _current_profile_home() -> str:
    try:
        from hermes_constants import get_hermes_home

        return str(Path(get_hermes_home()).resolve())
    except Exception:
        return ""


def _current_process_started_at() -> Optional[int]:
    try:
        from gateway.status import get_process_start_time

        return get_process_start_time(os.getpid())
    except Exception:
        return None


def _is_delegated() -> bool:
    try:
        from agent.delegation_context import is_delegated_child_context

        if is_delegated_child_context():
            return True
    except Exception:
        pass
    return False


def has_grant() -> bool:
    """Whether the current context holds a grant (tests/diagnostics)."""
    return _GRANT_VAR.get() is not None


def try_mint_grant(execution_id: str) -> Optional[contextvars.Token]:
    """Mint a grant for *execution_id* after verifying durable ownership.

    Returns the ContextVar token to reset, or ``None`` when ineligible.
    Eligibility: not in a delegated child, ``source == "builtin"``, canonical
    non-null ``scheduled_instant``, row ``status == "running"``, row owned by
    this process (``process_id``/``pid`` match).  The row is the source of
    truth; caller-supplied job fields are never trusted.
    """
    if _is_delegated():
        return None
    eid = str(execution_id or "").strip()
    if not eid:
        return None
    try:
        from cron.executions import _PROCESS_ID, get_execution
    except Exception:
        return None
    try:
        row = get_execution(eid)
    except Exception:
        return None
    if not isinstance(row, dict):
        return None
    if row.get("status") != "running":
        return None
    source = str(row.get("source") or "")
    if source != "builtin":
        return None
    instant = _canonical_instant(row.get("scheduled_instant"))
    if instant is None:
        return None
    job_id = str(row.get("job_id") or "")
    if not job_id:
        return None
    # Ownership: the row must already name this process.
    try:
        if str(row.get("process_id") or "") != str(_PROCESS_ID):
            return None
        if int(row.get("pid")) != int(os.getpid()):
            return None
    except Exception:
        return None
    profile_key = _current_profile_key()
    if not profile_key:
        return None
    profile_home = _current_profile_home()
    grant = CronExecutionGrant(
        job_id=job_id,
        execution_id=eid,
        source=source,
        scheduled_instant=instant,
        profile_key=profile_key,
        profile_home=profile_home,
        pid=int(os.getpid()),
        process_started_at=_current_process_started_at(),
        process_id=str(_PROCESS_ID),
        generation=get_runtime_generation(),
        nonce=secrets.token_hex(32),
    )
    return _GRANT_VAR.set(grant)


@contextlib.contextmanager
def scoped_execution_grant(execution_id: str) -> Iterator[Optional[CronExecutionGrant]]:
    """Push a grant for *execution_id* (or ``None`` when ineligible) for the block.

    Always pushes: an ineligible inner execution hides an outer grant for the
    duration so a manual/direct child cannot observe its parent's authority.
    The previous value is restored on exit (stack discipline).
    """
    if _is_delegated():
        token: Optional[contextvars.Token] = _GRANT_VAR.set(None)
        try:
            yield None
        finally:
            try:
                _GRANT_VAR.reset(token)
            except Exception:
                pass
        return
    token = try_mint_grant(execution_id)
    if token is None:
        # Ineligible: hide any outer grant for the block, then restore.
        hide = _GRANT_VAR.set(None)
        try:
            yield None
        finally:
            try:
                _GRANT_VAR.reset(hide)
            except Exception:
                pass
        return
    try:
        yield _GRANT_VAR.get()
    finally:
        try:
            _GRANT_VAR.reset(token)
        except Exception:
            pass


def suspend_grant_for_child() -> Optional[contextvars.Token]:
    """Hide the current grant for a delegated child turn (returns reset token)."""
    try:
        current = _GRANT_VAR.get()
    except Exception:
        return None
    if current is None:
        return None
    try:
        return _GRANT_VAR.set(None)
    except Exception:
        return None


def restore_grant_after_child(token: Optional[contextvars.Token]) -> None:
    """Restore a grant hidden by :func:`suspend_grant_for_child`."""
    if token is None:
        return
    try:
        _GRANT_VAR.reset(token)
    except Exception:
        pass


def verify_cron_execution(
    expected_job_id: Optional[str] = None,
    purpose_digest: Optional[str] = None,
) -> Optional[VerifiedCronExecution]:
    """Verify the calling context runs inside the exact scheduled occurrence.

    Validates the in-memory grant together with the live durable row on every
    call.  Returns a read-only projection, or ``None`` when the caller is not
    the authorized top-level scheduled execution.  Never raises for an
    unauthorized caller.
    """
    grant = _GRANT_VAR.get()
    if grant is None:
        return None
    if _is_delegated():
        return None
    # Runtime generation + process binding.
    if grant.generation != get_runtime_generation():
        return None
    try:
        if int(grant.pid) != int(os.getpid()):
            return None
    except Exception:
        return None
    try:
        from cron.executions import _PROCESS_ID as _CUR_PROCESS_ID

        if str(grant.process_id) != str(_CUR_PROCESS_ID):
            return None
    except Exception:
        return None
    # Profile scope: no launch-profile/global state may authorize another
    # served profile.
    if grant.profile_key != _current_profile_key():
        return None
    # Purpose binding: a caller-supplied canonical digest rides the projection
    # so one verified call cannot be replayed for a different operation
    # without re-verifying for that purpose.
    norm_purpose: Optional[str] = None
    if purpose_digest is not None:
        if not isinstance(purpose_digest, str):
            return None
        norm_purpose = purpose_digest.strip()
        if not norm_purpose:
            return None
    if expected_job_id is not None and str(expected_job_id) != grant.job_id:
        return None
    # Durable row: must still be running and owned by this process.  Terminal
    # rows, lost ownership, and mismatched identities fail immediately.
    try:
        from cron.executions import get_execution

        row = get_execution(grant.execution_id)
    except Exception:
        return None
    if not isinstance(row, dict):
        return None
    if row.get("status") != "running":
        return None
    if str(row.get("job_id") or "") != grant.job_id:
        return None
    if str(row.get("source") or "") != grant.source:
        return None
    row_instant = _canonical_instant(row.get("scheduled_instant"))
    if row_instant is None or row_instant != grant.scheduled_instant:
        return None
    try:
        from cron.executions import _PROCESS_ID as _CUR_PID2
        from cron.executions import _owner_is_live

        if str(row.get("process_id") or "") != str(_CUR_PID2):
            return None
        if int(row.get("pid")) != int(os.getpid()):
            return None
        if str(row.get("process_id") or "") != str(grant.process_id):
            return None
        if int(row.get("pid")) != int(grant.pid):
            return None
        # Ownership is ``(pid, started_at)`` (cron/AGENTS.md): a recycled PID
        # reuses the numbers but is a different incarnation, so the row's
        # start-time fingerprint must match the live process.
        if not _owner_is_live(int(row.get("pid")), row.get("process_started_at")):
            return None
    except Exception:
        return None
    return VerifiedCronExecution(
        job_id=grant.job_id,
        execution_id=grant.execution_id,
        source=grant.source,
        scheduled_instant=grant.scheduled_instant,
        profile_home=grant.profile_home,
        profile_key=grant.profile_key,
        ownership_generation=grant.generation,
        purpose_digest=norm_purpose,
    )


class CronRuntimeAuthority:
    """Trusted runtime surface for plugins (``ctx.runtime``).

    ``verify_cron_execution`` is the only method: it reads the calling
    context's process-local grant, never model-visible values.
    """

    def __init__(self, plugin_id: str = "") -> None:
        self._plugin_id = str(plugin_id or "")

    def verify_cron_execution(
        self,
        expected_job_id: Optional[str] = None,
        purpose_digest: Optional[str] = None,
    ) -> Optional[VerifiedCronExecution]:
        """See :func:`verify_cron_execution`."""
        return verify_cron_execution(
            expected_job_id=expected_job_id, purpose_digest=purpose_digest
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "CronRuntimeAuthority()"
