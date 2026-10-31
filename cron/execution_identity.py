"""Which cron execution the current code runs inside, for plugins (``ctx.current_cron_execution()``).

The scheduler binds it once it has won the execution's ``claimed`` → ``running`` transition (or a
restart-safe worker adopted it) and resets it when the fire ends. It is a ContextVar, so it follows
the agent onto the threads the scheduler already runs it on via ``contextvars.copy_context()`` and
never leaks to another concurrently firing job or profile. Model output cannot reach it: tool
subprocesses do not share the interpreter. Delegated children read ``None`` — they are not the
scheduled occurrence.
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional


@dataclass(frozen=True)
class CronExecution:
    job_id: str
    job_name: str
    execution_id: str
    source: str  # "builtin" (ticker), "direct" (manual run / dashboard Run now), provider name...
    scheduled_instant: Optional[str]  # the occurrence fired; None for an off-schedule run
    started_at: Optional[str]
    profile: str


_CURRENT: ContextVar[Optional[CronExecution]] = ContextVar("hermes_cron_execution", default=None)


def enter_cron_execution(job: Mapping[str, Any], execution_id: str, record: Mapping[str, Any]) -> Token:
    """Bind this fire's identity from *record*, the ledger row of the running transition."""
    from hermes_cli.profiles import current_profile_name

    return _CURRENT.set(CronExecution(
        job_id=str(job["id"]),
        job_name=str(job.get("name") or job["id"]),
        execution_id=str(execution_id),
        source=str(record.get("source") or ""),
        scheduled_instant=record.get("scheduled_instant"),
        started_at=record.get("started_at"),
        profile=current_profile_name("default") or "default",
    ))


def exit_cron_execution(token: Optional[Token]) -> None:
    if token is not None:
        _CURRENT.reset(token)


def current_cron_execution() -> Optional[CronExecution]:
    from agent.delegation_context import is_delegated_child_context

    return None if is_delegated_child_context() else _CURRENT.get()


# --- The occurrence identity a fire EXPORTS to its children ----------------
#
# The contextvar above serves in-process plugins. A child process (a ``no_agent`` script,
# a pre-run gate script, a restart-safe worker) shares no interpreter with the scheduler
# and cannot re-derive the identity: ``create_execution`` records the SCHEDULER's pid, so a
# child cannot recognise its own ledger row, and picking a row by nearest window would put
# resolution logic in the consumer. So the fire exports the identity as env vars, once,
# from the job snapshot the scheduler dispatched — and the CONSUMER decides what an absent
# value means (a manual/off-schedule fire legitimately has no occurrence).


#: Every name the occurrence identity travels under — one list, so building the identity and
#: CLEARING it can never drift apart.
CRON_EXECUTION_ENV_NAMES = (
    "HERMES_CRON_JOB_ID",
    "HERMES_CRON_EXECUTION_ID",
    "HERMES_CRON_SCHEDULED_INSTANT",
    "HERMES_CRON_SOURCE",
)


def cron_execution_env(job: Mapping[str, Any]) -> dict[str, str]:
    """The occurrence identity of *job*, as the env a fire's child processes receive.

    ``_scheduled_instant`` is the occurrence fired, VERBATIM as the ledger holds it
    (``cron.occurrences.scheduled_instant`` stays its only normaliser) — never the wall
    clock. An absent instant is the single readable "off-schedule / manual fire" signal,
    so a key whose value is None or empty is OMITTED rather than exported empty: a
    present-but-empty variable is indistinguishable from a real one and would be a trap.
    Only ids, an instant and a source travel here — never a secret.

    Omission is only half the contract: install this with :func:`apply_cron_execution_env`,
    never with a plain ``env.update()``, or an omitted name survives in the child env.
    """
    values = (job.get("id"), job.get("execution_id"), job.get("_scheduled_instant"),
              job.get("source"))
    return {name: str(value) for name, value in zip(CRON_EXECUTION_ENV_NAMES, values)
            if value not in (None, "")}


def apply_cron_execution_env(env: MutableMapping[str, str],
                             identity: Optional[Mapping[str, str]]) -> MutableMapping[str, str]:
    """Make *identity* the ONLY occurrence identity *env* carries, and return *env*.

    A plain ``env.update(identity)`` is not enough, for a reason that is easy to get wrong:
    the child-env builder copies and scrubs the LAUNCHING process's environment, and the
    ``HERMES_CRON_*`` names are NOT among the names that scrub strips — they pass through.
    Hermes processes are themselves started from cron fires (a self-hosted scheduler, the
    DAG runners), so ``os.environ`` routinely holds an older fire's identity. Merging an
    identity that OMITS a name for "this fire has no occurrence" then leaves the launcher's
    value in place, and the child reports someone else's occurrence as its own.

    So an absent name is DELETED, never left to inherit: a child's identity is its own fire's
    or none at all. ``identity=None``/``{}`` therefore means "this spawn has no fire", which
    is what a caller with no occurrence must pass.
    """
    for name in CRON_EXECUTION_ENV_NAMES:
        env.pop(name, None)
    env.update(identity or {})
    return env
