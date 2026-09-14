"""The tenant boundary for the kanban execution kernel.

Before this module ``tasks.tenant`` was a *label*. ``kanban_db_graph`` says so in
as many words — "parent order breaks ties in this soft namespace". Nothing read it
back on a lookup, so ``get_task(conn, id)`` returned any tenant's task and
``archive_task(conn, id)`` archived it. Worse, the idempotency lookup in
``create_task`` matched on the key alone, so two tenants submitting the same key
collapsed onto ONE task: the second tenant's job never ran and it received a live
handle to the first tenant's card.

This module turns that label into a boundary, without rewriting the kernel:

* an **ambient tenant** (a contextvar) that every entry point binds once — the
  dispatcher before it spawns, the worker from ``HERMES_TENANT``, the control API
  from the authenticated principal;
* ``scope_clause()``, which the task accessors splice into their WHERE so a row
  belonging to another tenant simply is not there; and
* ``normalize()``, so ``"", None, "  acme  "`` cannot become three namespaces.

**The default is unscoped, deliberately.** A single-tenant workstation — the CLI,
the TUI, every existing test — binds nothing and behaves exactly as before. The
boundary engages when a tenant is bound, which is what a multi-tenant deployment
does at its edges. ``HERMES_TENANT_STRICT=1`` turns the unbound case into an
error for anyone who wants that guarantee enforced rather than trusted.

**Cross-tenant access reads as absent, not as forbidden.** A reader that answers
"forbidden" for a row that exists and "not found" for one that does not is an
existence oracle: it confirms another tenant's task ids. So the accessors return
their ordinary not-found value (``None``/``False``) and log. That also means no
caller's contract changes — a function that already had to handle "no such task"
still only has to handle that.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
from typing import Any, Iterator, Optional

__all__ = [
    "TenantViolation",
    "normalize",
    "current",
    "bind",
    "use",
    "from_env",
    "scope_clause",
    "row_tenant_matches",
    "strict",
]

_log = logging.getLogger("hermes.kanban.tenant")

#: The ambient tenant. ``None`` means unscoped: the historical single-tenant
#: behaviour, in which every row is visible.
_ACTIVE: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "hermes_kanban_tenant", default=None
)

#: Environment variable the dispatcher already exports to every worker
#: (``kanban_db_dispatch`` sets it from ``task.tenant``). Reusing it means a
#: worker inherits its boundary with no new plumbing.
ENV_VAR = "HERMES_TENANT"

#: When set truthy, operating on tasks with no tenant bound raises instead of
#: running unscoped. Opt-in: a deployment that has finished threading tenants
#: through its entry points can turn this on and have the gap fail loudly.
STRICT_ENV_VAR = "HERMES_TENANT_STRICT"

_TRUTHY = {"1", "true", "yes", "on"}


class TenantViolation(RuntimeError):
    """Raised on a tenant boundary breach that must not degrade to not-found.

    Used for *writes made under strict mode* and for internally inconsistent
    state (a row whose tenant changed underneath a claim). Ordinary cross-tenant
    reads do not raise — see this module's docstring on the existence oracle.
    """


def strict() -> bool:
    """Whether an unbound tenant is an error rather than "unscoped"."""
    return str(os.environ.get(STRICT_ENV_VAR, "")).strip().lower() in _TRUTHY


def normalize(value: Any) -> Optional[str]:
    """Canonical tenant id, or ``None`` for "no tenant".

    ``None``, ``""`` and whitespace all collapse to ``None``. Without this a
    tenant could be spelled three ways and occupy three namespaces, which would
    make the scope clause silently permissive — the failure mode that matters.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def current() -> Optional[str]:
    """The tenant bound for this context, or ``None`` when unscoped."""
    return _ACTIVE.get()


def bind(tenant: Any) -> contextvars.Token:
    """Bind the ambient tenant. Prefer :func:`use`; this is for entry points
    that own the whole process (a worker ``main``) and never need to unbind."""
    return _ACTIVE.set(normalize(tenant))


@contextlib.contextmanager
def use(tenant: Any) -> Iterator[Optional[str]]:
    """Bind ``tenant`` for the duration of the block, then restore.

    A contextvar rather than a global: the gateway dispatches many tenants from
    one process, and threads/tasks must not see each other's binding.
    """
    token = _ACTIVE.set(normalize(tenant))
    try:
        yield _ACTIVE.get()
    finally:
        _ACTIVE.reset(token)


def from_env() -> Optional[str]:
    """Bind from ``HERMES_TENANT`` and return it. Called by worker entry points,
    which the dispatcher already launches with that variable set."""
    tenant = normalize(os.environ.get(ENV_VAR))
    if tenant is not None:
        _ACTIVE.set(tenant)
    return tenant


def resolve(explicit: Any = None) -> Optional[str]:
    """The tenant an operation should run under: an explicit argument if given,
    otherwise the ambient one."""
    chosen = normalize(explicit)
    return chosen if chosen is not None else current()


def scope_clause(
    explicit: Any = None, *, alias: str = "", column: str = "tenant"
) -> tuple[str, tuple]:
    """SQL fragment and params restricting a query to the operating tenant.

    Returns ``("", ())`` when unscoped, so callers splice unconditionally::

        clause, params = scope_clause(tenant)
        conn.execute(f"SELECT * FROM tasks WHERE id = ?{clause}", (task_id, *params))

    ``IS`` rather than ``=`` so the comparison is still correct if a NULL ever
    reaches it; a scoped caller always passes a non-NULL tenant, and an
    unscoped one gets no clause at all.
    """
    tenant = resolve(explicit)
    if tenant is None:
        if strict():
            raise TenantViolation(
                "no tenant is bound and HERMES_TENANT_STRICT is set: this operation "
                "would run unscoped across every tenant on the board"
            )
        return "", ()
    prefix = f"{alias}." if alias else ""
    return f" AND {prefix}{column} IS ?", (tenant,)


def row_tenant_matches(row_tenant: Any, explicit: Any = None) -> bool:
    """Whether a row already in hand belongs to the operating tenant.

    For paths that fetched a row before the boundary existed and would be
    needlessly expensive to re-query.
    """
    tenant = resolve(explicit)
    return tenant is None or normalize(row_tenant) == tenant


def deny(operation: str, task_id: str, *, row_tenant: Any = None) -> None:
    """Record a refused cross-tenant access.

    Logged at WARNING because it is either a bug in a caller that forgot to
    bind, or an attempt — both worth seeing. The message deliberately does not
    echo the owning tenant to whoever asked.
    """
    _log.warning(
        "kanban tenant: refused %s on %s — it does not belong to tenant %r",
        operation, task_id, current(),
    )
    if strict():
        raise TenantViolation(
            f"{operation} on {task_id} crosses the tenant boundary"
        )
