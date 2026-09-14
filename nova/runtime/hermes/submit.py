"""Placing NOVA-planned work on the Hermes board.

The one place NOVA writes to a runtime-owned database, so the reasoning is worth stating
rather than leaving implicit.

**Which database.** ``kanban.db`` is the shared work board at the runtime home root, which
the runtime's own CLI, dashboard, dispatcher and in-agent tools all write to. It is not
``state.db``, ``hermes_state.db``, ``response_store.db``, ``memories`` or ``sessions`` —
those hold conversation and credential state, they are on ``materialize.NEVER_WRITE``, and
nothing here touches them.

**Through the runtime's own API, never raw SQL.** ``hermes_cli.kanban_db.create_task`` mints
ids, writes the task-event rows the dashboard and notifier read, honours the idempotency
key, and holds whatever locking the current schema requires. Hand-rolling an INSERT would
reimplement all of that against a schema upstream is free to change — exactly the coupling
the architecture exists to avoid. Calling it is a supported seam; copying it is a patch.

**Imported lazily, inside the function.** ``nova`` must load and test with only the standard
library and PyYAML present. Inside this function the runtime is, by definition, installed.
``tests/platform/test_boundaries.py`` permits that only in an adapter package and proves in
a subprocess that this module still imports without the runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from nova.errors import RuntimeAdapterError
from nova.runtime.base import SubmitResult, SubmittedItem, WorkItem

#: Recorded as the author of every task NOVA creates, so a board carrying both
#: human-created and NOVA-created work says which is which without a join.
CREATED_BY = "nova-supervisor"


def submit(
    home: Path,
    items: Sequence[WorkItem],
    *,
    dry_run: bool = False,
) -> SubmitResult:
    """Create each item in order, resolving dependency keys to runtime task ids.

    Items must already be ordered so every dependency precedes its dependants — the
    runtime resolves parents by id at creation time, and an id for a task that does not
    exist yet is not something a caller can supply.
    """
    if not items:
        return SubmitResult(dry_run=dry_run)

    if dry_run:
        # Reports what a real run would create without opening the store. The keys and the
        # ordering are the parts worth previewing; the ids do not exist yet and inventing
        # plausible-looking ones would make a preview indistinguishable from a result.
        return SubmitResult(
            items=tuple(
                SubmittedItem(
                    key=item.key,
                    task_id="",
                    created=True,
                    assignee=item.assignee,
                    state="pending",
                    detail="would be created",
                )
                for item in items
            ),
            dry_run=True,
        )

    kb, kbc = _runtime_modules()

    submitted: list[SubmittedItem] = []
    warnings: list[str] = []
    by_key: dict[str, str] = {}

    with kbc.connect_closing() as connection:
        for item in items:
            missing = [key for key in item.depends_on if key not in by_key]
            if missing:
                raise RuntimeAdapterError(
                    f"work item {item.key!r} depends on {', '.join(sorted(missing))}, which "
                    "has not been created yet — items must be submitted in dependency order"
                )
            parents = tuple(by_key[key] for key in item.depends_on)

            before = _existing_id(kb, connection, item.key, item.tenant_id or "")
            task_id = kb.create_task(
                connection,
                title=item.title,
                body=item.body,
                assignee=item.assignee,
                created_by=CREATED_BY,
                parents=parents,
                priority=item.priority,
                tenant=item.tenant_id or None,
                triage=bool(item.decompose),
                idempotency_key=item.key,
                max_runtime_seconds=item.max_runtime_seconds,
                max_retries=item.max_retries,
            )
            by_key[item.key] = task_id

            task = kb.get_task(connection, task_id)
            status = getattr(task, "status", "") if task is not None else ""
            created = before is None
            if not created and before != task_id:
                # The idempotency key matched something else entirely. Worth surfacing:
                # it means two different plans are using one key.
                warnings.append(
                    f"idempotency key {item.key!r} resolved to {task_id} but an earlier "
                    f"lookup found {before}"
                )
            submitted.append(
                SubmittedItem(
                    key=item.key,
                    task_id=task_id,
                    created=created,
                    assignee=getattr(task, "assignee", item.assignee) or item.assignee,
                    state=status,
                    detail="" if created else "already existed; left as it was",
                )
            )

    return SubmitResult(items=tuple(submitted), warnings=tuple(warnings))


def _existing_id(kb: Any, connection: Any, key: str, tenant: str = "") -> Optional[str]:
    """The task already carrying this idempotency key **for this tenant**, if any.

    Asked before creating so the result can distinguish "created" from "was already there".
    ``create_task`` returns the existing id either way, which is the behaviour that makes a
    re-submission safe but also makes it indistinguishable from a fresh one — and "did this
    run do anything?" is the first question an operator asks.

    Scoped by tenant for the same reason the runtime's own lookup now is: an
    idempotency key is a caller's natural name for a job, so two tenants both
    calling one "quarterly-refund-audit:pull-ledger" is ordinary. Unscoped, this
    probe reported another tenant's task as ours and the submission was labelled
    "already existed" when in fact nothing of ours had ever run.
    """
    finder = getattr(kb, "find_task_by_idempotency_key", None)
    if callable(finder):
        try:
            found = finder(connection, key)
        except Exception:  # noqa: BLE001 — a probe must never fail a submission
            return None
        if found is None:
            return None
        # Honour the boundary even when the runtime supplies its own finder.
        if tenant and (getattr(found, "tenant", None) or "") != tenant:
            return None
        return getattr(found, "id", None)
    try:
        row = connection.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' "
            "AND COALESCE(tenant, '') = COALESCE(?, '') LIMIT 1",
            (key, tenant or None),
        ).fetchone()
    except Exception:  # noqa: BLE001 — schema differences must not fail a submission
        return None
    return row[0] if row else None


def _runtime_modules() -> tuple[Any, Any]:
    """The runtime's work-store API. See this module's docstring for why it is imported here."""
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
    except ImportError as exc:  # pragma: no cover — requires the runtime to be absent
        raise RuntimeAdapterError(
            "the Hermes runtime is not importable from this process, so NOVA cannot place "
            f"work on its board ({exc}). Run `nova objective submit` where the runtime is "
            "installed, or use --dry-run to preview the plan"
        ) from exc
    return kb, kbc
