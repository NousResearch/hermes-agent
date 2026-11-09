"""The local kanban storage backend: today's ``kanban.db`` boards behind the storage port.

:class:`LocalKanbanStore` is ``hermes_cli.kanban_db`` (and its boards, dispatch and notify
siblings): each operation is the same call the code makes today, after one check of the pin and
inside one pinned kanban root. The root is resolved once, when the store opens, by today's rule
(``HERMES_KANBAN_HOME`` else the default root), and every operation of the store runs inside it
(:func:`hermes_cli.kanban_db_home.scoped_kanban_home`): an operation never re-resolves the root
against a later environment, so one process can hold a store for another root (tests) and a
worker can never drift off the root it was dispatched into.

The pin: the local kanban root is shared across every profile on the machine **by design** (the
dispatcher and its workers hand off through it), so the store holds the local tenant's one agent
and its pin is the tenant. A bound scope of any other tenant is refused with
:class:`KanbanPinRefused` before anything is read or written; a local scope of any profile, and
the ambient caller with no scope bound (today's CLI), are served. A hosted backend instead pins
``(tenant, agent)`` and refuses another agent's scope outright.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from hermes_kanban_port import (
    KanbanPinRefused, KanbanStorage, KanbanStorageProvider, KanbanStore,
)


class LocalKanbanStore(KanbanStore):
    """Today's kanban operations, inside one frozen kanban root, for the local tenant."""

    def __init__(self) -> None:
        from hermes_cli import kanban_db as kb
        # Today's resolution, once: the root this store serves is fixed from here on.
        self.home = kb.kanban_home()

    def _pin(self) -> None:
        from hermes_scope import LOCAL_TENANT, current_scope
        scope = current_scope()
        if scope is not None and scope.tenant != LOCAL_TENANT:
            raise KanbanPinRefused(f"a local kanban store serves tenant {LOCAL_TENANT!r} only")

    @contextmanager
    def _root(self) -> Iterator[None]:
        """The store's frozen root around one operation (see ``kanban_db_home``)."""
        from hermes_cli.kanban_db_home import scoped_kanban_home
        with scoped_kanban_home(self.home):
            yield

    def _connect(self, board: Optional[str]):
        from hermes_cli.kanban_db_connect import connect_closing
        return connect_closing(board=board)

    # --- Boards

    def list_boards(self, *, include_archived: bool = True):
        self._pin()
        from hermes_cli.kanban_db_boards import list_boards
        with self._root():
            return list_boards(include_archived=include_archived)

    def create_board(self, slug: str, **fields: Any):
        self._pin()
        from hermes_cli.kanban_db_boards import create_board
        with self._root():
            return create_board(slug, **fields)

    def remove_board(self, slug: str, *, archive: bool = True):
        self._pin()
        from hermes_cli.kanban_db_boards import remove_board
        with self._root():
            return remove_board(slug, archive=archive)

    # --- Tasks

    def create_task(self, *, title: str, board: Optional[str] = None, **fields: Any) -> str:
        self._pin()
        from hermes_cli.kanban_db import create_task
        with self._root(), self._connect(board) as conn:
            return create_task(conn, title=title, board=board, **fields)

    def get_task(self, task_id: str, *, board: Optional[str] = None):
        self._pin()
        from hermes_cli.kanban_db import get_task
        with self._root(), self._connect(board) as conn:
            return get_task(conn, task_id)

    def list_tasks(self, *, board: Optional[str] = None, **filters: Any):
        self._pin()
        from hermes_cli.kanban_db import list_tasks
        with self._root(), self._connect(board) as conn:
            return list_tasks(conn, **filters)

    def edit_task(self, task_id: str, *, board: Optional[str] = None, **fields: Any) -> bool:
        self._pin()
        from hermes_cli.kanban_db import edit_task
        with self._root(), self._connect(board) as conn:
            return edit_task(conn, task_id, board=board, **fields)

    def assign_task(self, task_id: str, profile: Optional[str], *, board: Optional[str] = None) -> bool:
        self._pin()
        from hermes_cli.kanban_db import assign_task
        with self._root(), self._connect(board) as conn:
            return assign_task(conn, task_id, profile)

    def link_tasks(self, parent_id: str, child_id: str, *, board: Optional[str] = None,
                   expected_child_run_id: Optional[int] = None) -> bool:
        self._pin()
        from hermes_cli.kanban_db import link_tasks
        with self._root(), self._connect(board) as conn:
            return link_tasks(conn, parent_id, child_id, expected_child_run_id=expected_child_run_id)

    def unlink_tasks(self, parent_id: str, child_id: str, *, board: Optional[str] = None) -> bool:
        self._pin()
        from hermes_cli.kanban_db import unlink_tasks
        with self._root(), self._connect(board) as conn:
            return unlink_tasks(conn, parent_id, child_id)

    def add_comment(self, task_id: str, author: str, body: str, *, board: Optional[str] = None) -> int:
        self._pin()
        from hermes_cli.kanban_db import add_comment
        with self._root(), self._connect(board) as conn:
            return add_comment(conn, task_id, author, body)

    def list_comments(self, task_id: str, *, board: Optional[str] = None):
        self._pin()
        from hermes_cli.kanban_db import list_comments
        with self._root(), self._connect(board) as conn:
            return list_comments(conn, task_id)

    def list_events(self, task_id: str, *, board: Optional[str] = None):
        self._pin()
        from hermes_cli.kanban_db import list_events
        with self._root(), self._connect(board) as conn:
            return list_events(conn, task_id)

    def list_runs(self, task_id: str, *, board: Optional[str] = None, include_active: bool = True,
                  **filters: Any):
        self._pin()
        from hermes_cli.kanban_db import list_runs
        with self._root(), self._connect(board) as conn:
            return list_runs(conn, task_id, include_active=include_active, **filters)

    def latest_run(self, task_id: str, *, board: Optional[str] = None):
        self._pin()
        from hermes_cli.kanban_db import latest_run
        with self._root(), self._connect(board) as conn:
            return latest_run(conn, task_id)

    def complete_task(self, task_id: str, *, board: Optional[str] = None, **fields: Any) -> bool:
        self._pin()
        from hermes_cli.kanban_db import complete_task
        with self._root(), self._connect(board) as conn:
            return complete_task(conn, task_id, **fields)

    def block_task(self, task_id: str, *, board: Optional[str] = None, reason: Optional[str] = None,
                   kind: Optional[str] = None, expected_run_id: Optional[int] = None) -> bool:
        self._pin()
        from hermes_cli.kanban_db import block_task
        with self._root(), self._connect(board) as conn:
            return block_task(conn, task_id, reason=reason, kind=kind, expected_run_id=expected_run_id)

    def unblock_task(self, task_id: str, *, board: Optional[str] = None) -> bool:
        self._pin()
        from hermes_cli.kanban_db import unblock_task
        with self._root(), self._connect(board) as conn:
            return unblock_task(conn, task_id)

    def request_review(self, task_id: str, *, board: Optional[str] = None, **fields: Any):
        self._pin()
        from hermes_cli.kanban_db import request_review
        with self._root(), self._connect(board) as conn:
            return request_review(conn, task_id, **fields)

    def archive_task(self, task_id: str, *, board: Optional[str] = None) -> bool:
        self._pin()
        from hermes_cli.kanban_db import archive_task
        with self._root(), self._connect(board) as conn:
            return archive_task(conn, task_id)

    def delete_task(self, task_id: str, *, board: Optional[str] = None) -> bool:
        self._pin()
        from hermes_cli.kanban_db import delete_task
        with self._root(), self._connect(board) as conn:
            return delete_task(conn, task_id)

    # --- Claim / dispatch

    def claim_task(self, task_id: str, *, board: Optional[str] = None, ttl_seconds: Optional[int] = None,
                   claimer: Optional[str] = None):
        self._pin()
        from hermes_cli.kanban_db import claim_task
        with self._root(), self._connect(board) as conn:
            return claim_task(conn, task_id, ttl_seconds=ttl_seconds, claimer=claimer)

    def claim_review_task(self, task_id: str, *, board: Optional[str] = None,
                          ttl_seconds: Optional[int] = None, claimer: Optional[str] = None):
        self._pin()
        from hermes_cli.kanban_db import claim_review_task
        with self._root(), self._connect(board) as conn:
            return claim_review_task(conn, task_id, ttl_seconds=ttl_seconds, claimer=claimer)

    def heartbeat_claim(self, task_id: str, *, board: Optional[str] = None,
                        ttl_seconds: Optional[int] = None, claimer: Optional[str] = None) -> bool:
        self._pin()
        from hermes_cli.kanban_db import heartbeat_claim
        with self._root(), self._connect(board) as conn:
            return heartbeat_claim(conn, task_id, ttl_seconds=ttl_seconds, claimer=claimer)

    def release_stale_claims(self, *, board: Optional[str] = None, signal_fn: Any = None,
                             failure_limit: Optional[int] = None) -> int:
        self._pin()
        from hermes_cli.kanban_db import release_stale_claims
        with self._root(), self._connect(board) as conn:
            return release_stale_claims(conn, signal_fn=signal_fn, failure_limit=failure_limit)

    def reclaim_task(self, task_id: str, *, board: Optional[str] = None, reason: Optional[str] = None,
                     signal_fn: Any = None) -> bool:
        self._pin()
        from hermes_cli.kanban_db import reclaim_task
        with self._root(), self._connect(board) as conn:
            return reclaim_task(conn, task_id, reason=reason, signal_fn=signal_fn)

    def reassign_task(self, task_id: str, profile: Optional[str], *, board: Optional[str] = None,
                      reclaim_first: bool = False, reason: Optional[str] = None) -> bool:
        self._pin()
        from hermes_cli.kanban_db import reassign_task
        with self._root(), self._connect(board) as conn:
            return reassign_task(conn, task_id, profile, reclaim_first=reclaim_first, reason=reason)

    def recompute_ready(self, *, board: Optional[str] = None, failure_limit: Optional[int] = None) -> int:
        self._pin()
        from hermes_cli.kanban_db import recompute_ready
        with self._root(), self._connect(board) as conn:
            return recompute_ready(conn, failure_limit)

    # --- Notify subscriptions

    def add_notify_sub(self, *, task_id: str, platform: str, chat_id: str,
                       thread_id: Optional[str] = None, board: Optional[str] = None,
                       **fields: Any) -> None:
        self._pin()
        from hermes_cli.kanban_db_notify import add_notify_sub
        with self._root(), self._connect(board) as conn:
            add_notify_sub(conn, task_id=task_id, platform=platform, chat_id=chat_id,
                           thread_id=thread_id, **fields)

    def list_notify_subs(self, task_id: Optional[str] = None, *, board: Optional[str] = None,
                         notifier_profiles: Any = None, include_unowned: bool = False):
        self._pin()
        from hermes_cli.kanban_db_notify import list_notify_subs
        with self._root(), self._connect(board) as conn:
            return list_notify_subs(conn, task_id, notifier_profiles=notifier_profiles,
                                    include_unowned=include_unowned)

    def remove_notify_sub(self, *, task_id: str, platform: str, chat_id: str,
                          thread_id: Optional[str] = None, board: Optional[str] = None) -> bool:
        self._pin()
        from hermes_cli.kanban_db_notify import remove_notify_sub
        with self._root(), self._connect(board) as conn:
            return remove_notify_sub(conn, task_id=task_id, platform=platform, chat_id=chat_id,
                                     thread_id=thread_id)

    def unseen_events_for_sub(self, *, task_id: str, platform: str, chat_id: str,
                              thread_id: Optional[str] = None, kinds: Any = None,
                              board: Optional[str] = None):
        self._pin()
        from hermes_cli.kanban_db_notify import unseen_events_for_sub
        with self._root(), self._connect(board) as conn:
            return unseen_events_for_sub(conn, task_id=task_id, platform=platform, chat_id=chat_id,
                                         thread_id=thread_id, kinds=kinds)

    def claim_unseen_events_for_sub(self, *, task_id: str, platform: str, chat_id: str,
                                    thread_id: Optional[str] = None, kinds: Any = None,
                                    board: Optional[str] = None):
        self._pin()
        from hermes_cli.kanban_db_notify import claim_unseen_events_for_sub
        with self._root(), self._connect(board) as conn:
            return claim_unseen_events_for_sub(conn, task_id=task_id, platform=platform,
                                               chat_id=chat_id, thread_id=thread_id, kinds=kinds)

    def advance_notify_cursor(self, *, task_id: str, platform: str, chat_id: str,
                              thread_id: Optional[str] = None, new_cursor: int = 0,
                              board: Optional[str] = None) -> None:
        self._pin()
        from hermes_cli.kanban_db_notify import advance_notify_cursor
        with self._root(), self._connect(board) as conn:
            advance_notify_cursor(conn, task_id=task_id, platform=platform, chat_id=chat_id,
                                  thread_id=thread_id, new_cursor=new_cursor)

    def rewind_notify_cursor(self, *, task_id: str, platform: str, chat_id: str,
                             thread_id: Optional[str] = None, claimed_cursor: int = 0,
                             old_cursor: int = 0, board: Optional[str] = None) -> bool:
        self._pin()
        from hermes_cli.kanban_db_notify import rewind_notify_cursor
        with self._root(), self._connect(board) as conn:
            return rewind_notify_cursor(conn, task_id=task_id, platform=platform, chat_id=chat_id,
                                        thread_id=thread_id, claimed_cursor=claimed_cursor,
                                        old_cursor=old_cursor)


class LocalKanbanStorageProvider(KanbanStorageProvider):
    """Today's storage: one kanban root per machine (``HERMES_KANBAN_HOME`` else the default
    root), holding the local tenant's board set, shared by every local profile by design."""

    levels = frozenset({"agent"})

    @property
    def name(self) -> str:
        return "local"

    def open(self) -> KanbanStorage:
        """The store for the kanban root this process resolves today; the root freezes here and
        the pin is checked by every operation."""
        return KanbanStorage(LocalKanbanStore())

    def assert_isolating(self) -> None:
        """A local store is one root per machine and refuses every other tenant (``_pin``); its
        one shared file cannot bypass that."""
        return
