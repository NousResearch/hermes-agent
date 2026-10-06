"""Context-local state for delegate_task child execution.

A Hermes process may itself be a Kanban dispatcher worker with HERMES_KANBAN_* in
os.environ. In-process delegate_task children and cron jobs fired via
``cronjob(action="run")`` are NOT dispatcher-owned, so identity gates must fail
closed for them without mutating the process-global environment.
"""
from __future__ import annotations

import contextlib
import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Iterator, Mapping, MutableMapping, overload

_DELEGATED_CHILD_CONTEXT: ContextVar[bool] = ContextVar("hermes_delegated_child_context", default=False)
# Any in-process execution that is NOT the dispatcher-owned worker (cron jobs). Kept separate
# so delegate_task-specific behaviour (subprocess env scrubbing, its error strings) is unchanged.
_NON_DISPATCHER_OWNED_CONTEXT: ContextVar[bool] = ContextVar("hermes_non_dispatcher_owned_context", default=False)

DELEGATED_CHILD_ENV_MARKER = "HERMES_DELEGATED_CHILD_CONTEXT"

KANBAN_ENV_KEYS: tuple[str, ...] = (
    "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK",
    "HERMES_KANBAN_GOAL_MODE", "HERMES_KANBAN_GOAL_MAX_TURNS",
)


@contextmanager
def delegated_child_context(session_id: str | None = None) -> Iterator[None]:
    """Mark child execution and isolate its task-local session identity. Even a context
    entered without an id must restore the parent's session ContextVar (child
    construction calls ``set_current_session_id``)."""
    token = _DELEGATED_CHILD_CONTEXT.set(True)
    try:
        from gateway.session_context import scoped_current_session_id  # lazy: it calls is_delegated_child_context()

        with scoped_current_session_id(session_id):
            yield
    finally:
        _DELEGATED_CHILD_CONTEXT.reset(token)


def is_delegated_child_context() -> bool:
    """Return True while code is running for a delegate_task child."""
    return bool(_DELEGATED_CHILD_CONTEXT.get())


def enter_non_dispatcher_owned_context() -> Token[bool]:
    """Token form of :func:`non_dispatcher_owned_context` for long try/finally scopes."""
    return _NON_DISPATCHER_OWNED_CONTEXT.set(True)


def exit_non_dispatcher_owned_context(token: Token[bool]) -> None:
    """Restore the flag saved by :func:`enter_non_dispatcher_owned_context`."""
    _NON_DISPATCHER_OWNED_CONTEXT.reset(token)


@contextmanager
def non_dispatcher_owned_context() -> Iterator[None]:
    """Mark in-process execution that does NOT own the dispatcher's Kanban task; without it
    a cron agent run inside a worker is misread as that worker (kanban toolset force-added,
    ``kanban_complete`` defaulting to its task). ContextVar-scoped rather than clearing
    os.environ, which the worker's claim heartbeat and concurrent readers share."""
    token = enter_non_dispatcher_owned_context()
    try:
        yield
    finally:
        exit_non_dispatcher_owned_context(token)


def is_dispatcher_owned_worker_context() -> bool:
    """The single predicate every ``HERMES_KANBAN_*`` identity gate should use."""
    return not (is_delegated_child_process_context() or _NON_DISPATCHER_OWNED_CONTEXT.get())


def explicit_board_intent_is_pinned() -> bool:
    """Whether an explicit kanban ``board=`` argument must still resolve through
    the dispatcher-injected env pins (``HERMES_KANBAN_DB`` & friends) rather than
    its own board directory.

    True for in-process delegate children, descendants carrying
    :data:`DELEGATED_CHILD_ENV_MARKER`, and dispatched workers
    (``HERMES_KANBAN_TASK`` set). The pins are the "workers physically cannot
    see other boards" isolation, and :func:`kanban_path_is_fenced` checks the
    pinned path / fenced root — an explicit board that resolved elsewhere would
    also escape that fence. Outside these fences an explicit board is the
    caller's own intent and wins.
    """
    if _DELEGATED_CHILD_CONTEXT.get():
        return True
    if os.environ.get(DELEGATED_CHILD_ENV_MARKER):
        return True
    return bool((os.environ.get("HERMES_KANBAN_TASK") or "").strip())


def owned_kanban_task() -> str:
    """The board task this execution OWNS: ``HERMES_KANBAN_TASK`` for the dispatcher-owned
    worker, ``""`` otherwise. Tool access is not worker identity — a profile can expose the
    kanban toolset interactively, and children/cron runs inherit the env var — so every
    reader that turns the task id into worker behaviour (guidance, stop nudge, terminal
    outcomes) goes through this one helper."""
    if not is_dispatcher_owned_worker_context():
        return ""
    return (os.environ.get("HERMES_KANBAN_TASK") or "").strip()


def is_delegated_child_process_context() -> bool:
    """Return True in this process or a subprocess spawned by a child."""
    return bool(_DELEGATED_CHILD_CONTEXT.get()) or bool(os.environ.get(DELEGATED_CHILD_ENV_MARKER))


def _find_board_for_task(task_id: str) -> str | None:
    """Find the directory of the named board that contains task_id."""
    import sqlite3
    from hermes_cli.kanban_db import (
        DEFAULT_BOARD,
        board_dir,
        kanban_db_path,
        list_boards,
    )

    for entry in list_boards(include_archived=True):
        slug = entry.get("slug")
        if not slug or slug == DEFAULT_BOARD:
            continue
        b_db = kanban_db_path(slug)
        if not b_db.exists():
            continue
        with contextlib.suppress(sqlite3.Error, OSError):
            conn = sqlite3.connect(f"file:{b_db.resolve()}?mode=ro", uri=True)
            try:
                row = conn.execute(
                    "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
                ).fetchone()
                if row:
                    return str(board_dir(slug).resolve())
            finally:
                conn.close()
    return None


def _fenced_kanban_root(env: Mapping[str, str] | None = None) -> str:
    """The board root this process's Kanban lineage lives under; ``"1"`` when
    it cannot be resolved, which readers treat as "fence every board" (the
    pre-path marker)."""
    try:
        from hermes_cli.kanban_db import (
            DEFAULT_BOARD,
            _normalize_board_slug,
            board_dir,
            boards_root,
            kanban_db_path,
            kanban_home,
        )

        def _get(key: str) -> str:
            if env is not None and key in env:
                return str(env[key]).strip()
            return os.environ.get(key, "").strip()

        board_val = _get("HERMES_KANBAN_BOARD")
        if board_val:
            with contextlib.suppress(ValueError):
                slug = _normalize_board_slug(board_val)
                if slug and slug != DEFAULT_BOARD:
                    return str(board_dir(slug).resolve())
                if slug == DEFAULT_BOARD:
                    return str(kanban_home().resolve())

        db_val = _get("HERMES_KANBAN_DB")
        if db_val:
            db_path = Path(db_val).expanduser().resolve()
            b_root = boards_root().resolve()
            with contextlib.suppress(ValueError):
                rel = db_path.relative_to(b_root)
                if rel.parts:
                    slug = _normalize_board_slug(rel.parts[0])
                    if slug and slug != DEFAULT_BOARD:
                        return str(board_dir(slug).resolve())
            kh = kanban_home().resolve()
            with contextlib.suppress(OSError):
                if (
                    db_path == kanban_db_path(DEFAULT_BOARD).resolve()
                    or db_path.parent == kh
                ):
                    return str(kh)
            return str(db_path.parent)

        task_val = _get("HERMES_KANBAN_TASK")
        if task_val:
            found = _find_board_for_task(task_val)
            if found:
                return found

        return str(kanban_home().resolve())
    except (OSError, ValueError):
        return "1"
    except Exception:  # health: allow BLE001 -- fallback boundary
        return "1"


def scrub_kanban_env(
    env: Mapping[str, str] | MutableMapping[str, str],
) -> dict[str, str]:
    """Remove worker identity, retaining board/location and an inherited fence.

    TASK absence alone would promote a descendant to an orchestrator. The
    marker survives later execs, including scripts that remove TASK themselves.
    This is cooperative runtime scoping, not confinement of code with direct
    SQLite access.

    The marker's value is the fenced board ROOT, so the fence applies to the
    lineage's board and not to every Kanban DB the descendant touches: a child
    running a repro against a temp ``HERMES_HOME`` got a silently read-only
    board there. An inherited path-valued marker is kept (a grandchild that
    moved HERMES_HOME must not re-fence onto its scratch root and unfence the
    real one).
    """
    cleaned = {k: v for k, v in env.items() if k not in KANBAN_ENV_KEYS}
    inherited = str(env.get(DELEGATED_CHILD_ENV_MARKER) or "")
    cleaned[DELEGATED_CHILD_ENV_MARKER] = (
        inherited
        if inherited and inherited != "1"
        else _fenced_kanban_root(env)
    )
    return cleaned


def kanban_path_is_fenced(path: "os.PathLike[str] | str") -> bool:
    """Whether Kanban mutations at *path* (a board DB or board-metadata root)
    are denied for this process: always for an in-process delegate child (the
    parent's own board); for a spawned descendant only when *path* is the
    dispatcher-pinned ``HERMES_KANBAN_DB`` or lies under the fenced root the
    marker carries. A legacy ``"1"`` marker fences everything."""
    marker = os.environ.get(DELEGATED_CHILD_ENV_MARKER, "")
    if _DELEGATED_CHILD_CONTEXT.get():
        marker = marker or _fenced_kanban_root()
    if not marker:
        return False
    if marker == "1":
        return True

    target = Path(path).expanduser().resolve()
    pinned = os.environ.get("HERMES_KANBAN_DB", "").strip()
    if pinned and target == Path(pinned).expanduser().resolve():
        return True
    marker_path = Path(marker).expanduser().resolve()

    from hermes_cli.kanban_db import (
        DEFAULT_BOARD,
        board_dir,
        boards_root,
        kanban_home,
    )

    try:
        target.relative_to(marker_path)
    except ValueError:
        with contextlib.suppress(OSError, ValueError):
            if target == kanban_home().resolve():
                marker_path.relative_to(target)
                return True
        return False

    with contextlib.suppress(OSError, ValueError):
        if marker_path == kanban_home().resolve():
            b_root = boards_root().resolve()
            target.relative_to(b_root)
            default_dir = board_dir(DEFAULT_BOARD).resolve()
            try:
                target.relative_to(default_dir)
                return True
            except ValueError:
                return False
    return True


@overload
def delegated_child_subprocess_env(env: Mapping[str, str]) -> dict[str, str]: ...


@overload
def delegated_child_subprocess_env(env: None = None) -> dict[str, str] | None: ...


def delegated_child_subprocess_env(
    env: Mapping[str, str] | MutableMapping[str, str] | None = None,
) -> dict[str, str] | None:
    """Carry worker/delegate descendant denial across a real process spawn.

    Location and credentials are untouched; callers retain their existing secret policy.
    Dispatcher workers and supervised tool transports grant their own explicit scope.
    """
    if not (is_delegated_child_process_context() or os.environ.get("HERMES_KANBAN_TASK")
            or (env and (env.get("HERMES_KANBAN_TASK") or env.get(DELEGATED_CHILD_ENV_MARKER)))):
        return None if env is None else dict(env)
    return scrub_kanban_env(os.environ if env is None else env)
