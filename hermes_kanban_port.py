"""The kanban storage port: boards, tasks, events, claims and notify subscriptions.

One port (:class:`KanbanStore`) at the level today's kanban call sites use — the operations of
``hermes_cli.kanban_db`` (boards, task rows and their lifecycle, the claim/dispatch transitions,
the event log) and ``hermes_cli.kanban_db_notify`` (notify subscriptions and their event
cursors) — never SQL and never a connection. The board a task belongs to is an argument, as it is
today (``board=None`` resolves through the caller's board context); a worker sees one board
because its scope and its board resolution are fenced, not because the port hides the argument.

A :class:`KanbanStorageProvider` supplies the store. This is the interface and its local backend
only: no production caller goes through it yet, and nothing in config.yaml selects a backend.
Today's callers still use ``hermes_cli.kanban_db`` directly; moving them behind the port, call
site by call site, is a follow-up, and until it lands a provider plugin stores nothing that
Hermes reads. :func:`resolve_kanban_storage` takes the backend's name explicitly: ``"local"``
(or empty) is ``hermes_kanban_port_local`` (today's ``kanban.db`` under today's kanban root,
``HERMES_KANBAN_HOME`` and all); any other name is a provider from
``plugins/kanban_storage/<name>/`` or ``$HERMES_HOME/plugins/<name>/``. A named provider that is
missing or unavailable is an error, never a fallback to the local store.

Providers load in-process only, for the same reason the session storage providers do: a host
process hands back opaque objects, and its calls would not carry the caller's scope.

Every operation reads its pin (tenant, agent, owner, actor) from ``hermes_scope.current_scope()``
when it runs, never from a path, ``HERMES_HOME`` or an argument; the port exposes no way to pin
anything else. Kanban records are agent level: the board belongs to one agent, its workers and
its subscribers are that agent's activities, and every row the store touches is that agent's. A
backend refuses a pin it cannot serve with :class:`KanbanPinRefused`, saying nothing about
whether the target exists. The local backend serves the local tenant (its one shared board root,
shared by every profile on the machine by design) and refuses any other tenant; it also serves
the ambient caller with no scope bound, as today's CLI does.

The local backend's shared root is the one declared exception to agent isolation
(:attr:`KanbanStore.LOCAL_SHARED_BOARD`): a single-user machine's dispatcher and its workers hand
off through that one root, so the local store serves any scope of its tenant, whatever the
profile. It is a property of the backend's own class, never a fixture option, and it is refused
wherever the process runs hosted (:func:`resolve_kanban_storage`): a shared deployment's tenants
each get their own agents' boards, behind a backend that pins the agent too.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Comment, Event, Run, Task

#: The ownership levels a store can hold (``hermes_state_port.LEVELS``): kanban records are
#: agent level only, so a backend that stores anything else should say so here and refuse it.
LEVELS = frozenset({"agent"})


class KanbanPinRefused(PermissionError):
    """The bound scope is not one this store serves: another tenant, another agent, or (for a
    backend that requires one) no scope at all. Says nothing about whether the target exists."""


class KanbanStore(ABC):
    """One agent's kanban board set: its boards, their tasks, the event log, the claim/dispatch
    transitions and the notify subscriptions, for the pinned agent.

    A foreign row (another tenant's, another agent's) is indistinguishable from a missing one:
    reads return nothing and a claim or a dispatch sweep moves nothing. Methods take the board
    slug where today's call sites pass one; ``board=None`` keeps today's resolution.
    """

    #: Whether this backend's one board root is shared by every profile of a single-user machine:
    #: the local exception, declared on the backend's own class and nowhere else. Such a store
    #: serves any scope of its tenant, whatever the profile, because that machine's dispatcher
    #: and its workers hand off through the one root; the conformance suite
    #: (``hermes_conformance.kanban``) asserts the inverse cross-agent matrix for a store that
    #: declares it. A backend that pins the agent — anything a shared deployment serves — never
    #: sets it, and a subclass cannot unset it: the declaration cannot be hidden behind a
    #: wrapper, a subclass or an un-declaring provider (:func:`resolve_kanban_storage` opens the
    #: store wherever the process runs hosted and refuses it).
    LOCAL_SHARED_BOARD: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        declared = cls.__dict__.get("LOCAL_SHARED_BOARD")
        if declared is not None and not declared and any(
                base.__dict__.get("LOCAL_SHARED_BOARD") for base in cls.__mro__[1:]):
            raise TypeError(
                f"{cls.__name__} cannot declare LOCAL_SHARED_BOARD false: a backend that keeps "
                "one board root shared across every profile cannot hide it behind a subclass")

    # --- Boards

    @abstractmethod
    def list_boards(self, *, include_archived: bool = True) -> list[dict[str, Any]]:
        """Metadata for every board of the pinned agent (``default`` first)."""

    @abstractmethod
    def create_board(self, slug: str, **fields: Any) -> dict[str, Any]:
        """Create a board (idempotent); returns its metadata, as ``kanban boards create`` does."""

    @abstractmethod
    def remove_board(self, slug: str, *, archive: bool = True) -> dict[str, Any]:
        """Archive (default) or delete a board; ``default`` cannot be removed."""

    # --- Tasks

    @abstractmethod
    def create_task(self, *, title: str, board: Optional[str] = None, **fields: Any) -> str:
        """Create a task (optionally under ``parents``); returns its id."""

    @abstractmethod
    def get_task(self, task_id: str, *, board: Optional[str] = None) -> Optional[Task]:
        """One task, or ``None`` (missing and foreign look the same)."""

    @abstractmethod
    def list_tasks(self, *, board: Optional[str] = None, **filters: Any) -> list[Task]:
        """The board's tasks (``assignee``, ``status``, ``include_archived``, ``limit`` and
        today's other filters)."""

    @abstractmethod
    def edit_task(self, task_id: str, *, board: Optional[str] = None, **fields: Any) -> bool:
        """Edit title/body/priority/result/summary/metadata; ``False`` when the task is missing."""

    @abstractmethod
    def assign_task(self, task_id: str, profile: Optional[str], *, board: Optional[str] = None) -> bool:
        """Point a task at the profile that runs it; ``False`` when the task is missing."""

    @abstractmethod
    def link_tasks(self, parent_id: str, child_id: str, *, board: Optional[str] = None,
                   expected_child_run_id: Optional[int] = None) -> bool:
        """Add a dependency edge; ``True`` when the link gated a ``ready`` child back to
        ``todo`` (see ``kanban_db.link_tasks``)."""

    @abstractmethod
    def unlink_tasks(self, parent_id: str, child_id: str, *, board: Optional[str] = None) -> bool:
        """Remove a dependency edge; ``False`` when there was no edge."""

    @abstractmethod
    def add_comment(self, task_id: str, author: str, body: str, *, board: Optional[str] = None) -> int:
        """Append a comment; returns its id."""

    @abstractmethod
    def list_comments(self, task_id: str, *, board: Optional[str] = None) -> list[Comment]:
        """The task's comments, oldest first; empty for a missing or foreign task."""

    @abstractmethod
    def list_events(self, task_id: str, *, board: Optional[str] = None) -> list[Event]:
        """The task's event log, oldest first; empty for a missing or foreign task."""

    @abstractmethod
    def list_runs(self, task_id: str, *, board: Optional[str] = None, include_active: bool = True,
                  **filters: Any) -> list[Run]:
        """The task's runs (``state_type``/``state_name`` filters as today)."""

    @abstractmethod
    def latest_run(self, task_id: str, *, board: Optional[str] = None) -> Optional[Run]:
        """The task's most recent run, or ``None``."""

    @abstractmethod
    def complete_task(self, task_id: str, *, board: Optional[str] = None, **fields: Any) -> bool:
        """``running|ready|blocked|review -> done`` (today's rules on live claims and empty
        evidence apply)."""

    @abstractmethod
    def block_task(self, task_id: str, *, board: Optional[str] = None, reason: Optional[str] = None,
                   kind: Optional[str] = None, expected_run_id: Optional[int] = None) -> bool:
        """``running``/``ready`` -> ``blocked`` (or ``todo``/``triage``, see ``kanban_db``)."""

    @abstractmethod
    def unblock_task(self, task_id: str, *, board: Optional[str] = None) -> bool:
        """``blocked``/``scheduled`` -> its resumable phase."""

    @abstractmethod
    def request_review(self, task_id: str, *, board: Optional[str] = None, **fields: Any) -> Any:
        """``running``/``ready`` -> ``review`` (a live claim needs ``expected_run_id`` or
        ``force``); the reviewer field reassigns, as today. ``with_reason`` returns
        ``(ok, reason)``, plain calls return the bool."""

    @abstractmethod
    def archive_task(self, task_id: str, *, board: Optional[str] = None) -> bool:
        """Archive a task; ``False`` when there was nothing to archive."""

    @abstractmethod
    def delete_task(self, task_id: str, *, board: Optional[str] = None) -> bool:
        """Delete a task and its relations; ``False`` when there was nothing to delete."""

    # --- Claim / dispatch

    @abstractmethod
    def claim_task(self, task_id: str, *, board: Optional[str] = None, ttl_seconds: Optional[int] = None,
                   claimer: Optional[str] = None) -> Optional[Task]:
        """Atomic ``ready -> running``; the claimed task, or ``None`` (already claimed, not
        ready, missing or foreign). At most one claimant wins, across replicas."""

    @abstractmethod
    def claim_review_task(self, task_id: str, *, board: Optional[str] = None,
                          ttl_seconds: Optional[int] = None,
                          claimer: Optional[str] = None) -> Optional[Task]:
        """Atomic ``review -> running``; ``None`` when lost, missing or foreign."""

    @abstractmethod
    def heartbeat_claim(self, task_id: str, *, board: Optional[str] = None,
                        ttl_seconds: Optional[int] = None, claimer: Optional[str] = None) -> bool:
        """Extend the live claim of the running task."""

    @abstractmethod
    def release_stale_claims(self, *, board: Optional[str] = None, signal_fn: Any = None,
                             failure_limit: Optional[int] = None) -> int:
        """The dispatch sweep: reclaim ``running`` tasks whose claim expired (a live worker's
        claim is extended instead); returns how many were reclaimed. Pinned to this agent: it
        never releases another agent's claim."""

    @abstractmethod
    def reclaim_task(self, task_id: str, *, board: Optional[str] = None, reason: Optional[str] = None,
                     signal_fn: Any = None) -> bool:
        """The operator reclaim: take a running task back to its retry status."""

    @abstractmethod
    def reassign_task(self, task_id: str, profile: Optional[str], *, board: Optional[str] = None,
                      reclaim_first: bool = False, reason: Optional[str] = None) -> bool:
        """Re-point a task at another profile, optionally reclaiming it first."""

    @abstractmethod
    def recompute_ready(self, *, board: Optional[str] = None, failure_limit: Optional[int] = None) -> int:
        """Promote ``todo`` tasks whose parents are all terminal; returns how many moved."""

    # --- Notify subscriptions

    @abstractmethod
    def add_notify_sub(self, *, task_id: str, platform: str, chat_id: str,
                       thread_id: Optional[str] = None, board: Optional[str] = None,
                       **fields: Any) -> None:
        """Register a chat that wants terminal-state notifications for the task (idempotent on
        ``(task, platform, chat, thread)``; today's field rules apply)."""

    @abstractmethod
    def list_notify_subs(self, task_id: Optional[str] = None, *, board: Optional[str] = None,
                         notifier_profiles: Any = None, include_unowned: bool = False) -> list[dict[str, Any]]:
        """The subscriptions (optionally one task's); another agent's targets are never here."""

    @abstractmethod
    def remove_notify_sub(self, *, task_id: str, platform: str, chat_id: str,
                          thread_id: Optional[str] = None, board: Optional[str] = None) -> bool:
        """Remove a subscription; ``False`` when there was none."""

    @abstractmethod
    def unseen_events_for_sub(self, *, task_id: str, platform: str, chat_id: str,
                              thread_id: Optional[str] = None, kinds: Any = None,
                              board: Optional[str] = None) -> tuple[int, list[Event]]:
        """``(new_cursor, events)`` past the subscription's cursor; the cursor is not advanced
        here (``advance_notify_cursor`` after delivery)."""

    @abstractmethod
    def claim_unseen_events_for_sub(self, *, task_id: str, platform: str, chat_id: str,
                                    thread_id: Optional[str] = None, kinds: Any = None,
                                    board: Optional[str] = None) -> tuple[int, int, list[Event]]:
        """Atomically claim the subscription's unseen events: ``(old_cursor, new_cursor,
        events)``, the cursor already advanced. Two watchers of the same subscription: one
        claims a given event range."""

    @abstractmethod
    def advance_notify_cursor(self, *, task_id: str, platform: str, chat_id: str,
                              thread_id: Optional[str] = None, new_cursor: int = 0,
                              board: Optional[str] = None) -> None:
        """Move the subscription's cursor after delivery."""

    @abstractmethod
    def rewind_notify_cursor(self, *, task_id: str, platform: str, chat_id: str,
                             thread_id: Optional[str] = None, claimed_cursor: int = 0,
                             old_cursor: int = 0, board: Optional[str] = None) -> bool:
        """Put a claimed-but-undelivered cursor back; ``False`` when the claim was superseded."""


@dataclass(frozen=True)
class KanbanStorage:
    """One opened store. ``close`` gives back whatever ``open`` took."""

    kanban: KanbanStore
    close: Callable[[], None] = lambda: None


class KanbanStorageProvider(ABC):
    """The plugin kind: supplies the kanban store. ``name``, ``open`` and ``assert_isolating``
    are required; every other method has a safe default, and growth stays additive."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``"local"``."""

    #: The ownership levels this backend stores (see :data:`LEVELS`).
    levels: frozenset[str] = frozenset({"agent"})

    #: Whether this provider's store keeps one board root shared across every profile of a
    #: single-user machine (``KanbanStore.LOCAL_SHARED_BOARD``, the local exception): refused
    #: wherever the process runs hosted (:func:`resolve_kanban_storage`), because a shared
    #: deployment's tenants each get their own agents' boards.
    LOCAL_SHARED_BOARD: bool = False

    def is_available(self) -> bool:
        """Whether this backend can run here. No network calls."""
        return True

    @abstractmethod
    def open(self) -> KanbanStorage:
        """The store for the bound scope's kanban root. The root is resolved once here, by the
        same rules as today (the local backend: ``HERMES_KANBAN_HOME`` else the default root),
        and every operation of the returned store stays inside it; a backend never re-resolves
        per call. Nothing calls this yet: the first caller that serves a scope from a resolved
        provider must call :meth:`assert_isolating`, and refuse the provider when it raises,
        before this method."""

    @abstractmethod
    def assert_isolating(self) -> None:
        """Raise unless this backend isolates tenants and agents from each other in storage
        itself (a shared role that could bypass tenant isolation fails). Required, with no
        default: a backend that has not checked must not pass."""


def _hosted_process(hosted: Optional[bool]) -> bool:
    """Whether this process serves hosted tenants: the caller's explicit answer, else what the
    hosted boot profile reports (``hermes_cli.hosted_mode.is_hosted``; a checkout without that
    module is not hosted — the first hosted-capable caller passes the answer explicitly)."""
    if hosted is not None:
        return hosted
    try:
        from hermes_cli.hosted_mode import is_hosted
    except ImportError:
        return False
    return bool(is_hosted())


def _serves_shared_board_root(provider: KanbanStorageProvider) -> bool:
    """Whether the store *provider* actually opens keeps the one shared board root — the
    declaration a wrapper or a subclass cannot hide: the store is opened and inspected (its
    own class, every class it inherits from, and the local store itself), then closed. Only
    :func:`resolve_kanban_storage` calls this, wherever the process runs hosted."""
    from hermes_kanban_port_local import LocalKanbanStore
    storage = provider.open()
    try:
        store = storage.kanban
        return (any(cls.__dict__.get("LOCAL_SHARED_BOARD") for cls in type(store).__mro__)
                or isinstance(store, LocalKanbanStore))
    finally:
        storage.close()


def resolve_kanban_storage(name: str, *, hosted: Optional[bool] = None) -> KanbanStorageProvider:
    """The provider named *name* (``"local"`` or empty: the local backend). Raises ``LookupError``
    with the reason for a named provider that is missing, refused or unavailable: storage never
    falls back to another store. A provider that keeps one board root shared across every profile
    of a single-user machine (``LOCAL_SHARED_BOARD``, the local exception) is refused wherever
    the process runs hosted: a shared deployment's tenants each get their own agents' boards,
    behind a backend that pins the agent too. The declaration is checked where it cannot be
    hidden — the provider's own class and, hosted, the store it actually opens."""
    name = name.strip()
    if name in ("", "local"):
        from hermes_kanban_port_local import LocalKanbanStorageProvider
        provider = LocalKanbanStorageProvider()
    else:
        from plugins.kanban_storage import KanbanStorageRefused, load_kanban_storage
        try:
            provider = load_kanban_storage(name)
        except KanbanStorageRefused as refused:
            raise LookupError(f"kanban storage {name!r}: {refused}") from refused
        if provider is None:
            raise LookupError(f"kanban storage {name!r}: no such kanban storage provider")
        if not provider.is_available():
            raise LookupError(f"kanban storage {name!r} is not available here")
    if _hosted_process(hosted) and (getattr(type(provider), "LOCAL_SHARED_BOARD", False)
                                     or _serves_shared_board_root(provider)):
        raise LookupError(
            f"kanban storage {name!r} keeps one board root shared across every profile of a "
            "single-user machine; it cannot serve hosted tenants")
    return provider
