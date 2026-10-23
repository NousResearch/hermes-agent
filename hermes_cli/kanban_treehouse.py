"""treehouse pool leases for kanban card workspaces.

A repo opts in to a capped, recycled worktree pool by committing
``treehouse.toml`` at its root. When it has one, the dispatcher acquires a
durable lease from that pool instead of materialising
``<repo>/.worktrees/<task-id>``; when it does not, the legacy
``git worktree add`` path is used and the reason is logged (never silent).

The pool is the whole point: ``max_trees`` is a hard cap, and a returned slot
is reusable, so a fleet of cards stops growing worktrees without bound.

A card holds **exactly one** lease at a time: :func:`resolve_pool_workspace`
reuses the lease the card already holds, returns any duplicate it finds (a
dispatcher that died between acquire and recording its workspace can leave
one), and only then asks for a free slot. Lease identity lives in treehouse's
own state file, keyed to the recorded slot path -- never rename a slot. The
card id travels in the branch name (``wt/<task-id>``) and in the lease holder
label (``kanban:<task-id>``), which is what lets
:func:`release_orphaned_leases` return a dead worker's lease.

CLI contract this module relies on (verified against treehouse 3.1.4):

* ``get --lease --json`` prints one JSON object on stdout (path, lease_id,
  lease_holder, ...); every banner goes to stderr. At the cap it exits 1 with
  ``all N worktrees are in use or dirty (max_trees = N)``.
* ``status --json`` prints one array of slots, each with ``name``, ``path``,
  ``status`` ('available' | 'leased' | ...), ``branch`` and ``lease_holder``.
* ``lease <name>`` is refused when another holder already owns the slot, so a
  card can never adopt a slot someone else is running in.
* ``return --force --if-lease-holder <label> <path>`` is idempotent, cleans and
  resets the slot, and works from any directory. Without ``--force`` a dirty
  slot exits 3 asking for confirmation, so the fleet always passes ``--force``;
  the holder fence makes a return a no-op for a lease that is not ours.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_log = logging.getLogger("hermes_cli.kanban_db")

CONFIG_FILENAME = "treehouse.toml"
POOL_DIRNAME = ".treehouse"
HOLDER_PREFIX = "kanban:"
_GET_TIMEOUT_SECONDS = 300
_STATUS_TIMEOUT_SECONDS = 60
_RETURN_TIMEOUT_SECONDS = 120

# Statuses in which a card is finished with its workspace. ``review`` is
# included: the reviewer re-leases a slot when it starts, so holding the lease
# while no worker runs only pins a pool slot for the whole review wait.
_TERMINAL_STATUSES = frozenset(
    {"done", "blocked", "review", "ready", "todo", "triage", "scheduled", "archived"}
)


class TreehouseError(RuntimeError):
    """treehouse is configured for this repo but the call failed."""


class TreehouseRequired(TreehouseError):
    """A repo lacks treehouse.toml; worktrees are acquired exclusively through treehouse."""


class PoolExhausted(TreehouseError):
    """Every slot is in use or dirty: the card must wait, not spawn outside."""


@dataclass(frozen=True)
class PoolConfig:
    """The parsed ``<repo>/treehouse.toml``, validated."""

    repo_root: Path
    config_path: Path
    max_trees: int
    base_branch: Optional[str]

    @property
    def pool_root(self) -> Path:
        return self.repo_root / POOL_DIRNAME


@dataclass(frozen=True)
class Lease:
    path: Path
    lease_id: str
    holder: str
    max_trees: int


def holder_label(task_id: str) -> str:
    return f"{HOLDER_PREFIX}{task_id}"


def treehouse_binary() -> Optional[str]:
    """Absolute path to the ``treehouse`` CLI, or ``None`` when absent."""
    found = shutil.which("treehouse")
    if found:
        return found
    for candidate in ("/opt/homebrew/bin/treehouse", "/usr/local/bin/treehouse"):
        if os.access(candidate, os.X_OK):
            return candidate
    return None


def read_pool_config(repo_root: Path) -> Optional[PoolConfig]:
    """Parse ``<repo_root>/treehouse.toml``; ``None`` means "not pool-enabled".

    A malformed or unusable config raises rather than returning ``None``: a
    repo that opted in must not silently degrade to the uncapped path.
    """
    repo_root = Path(repo_root).expanduser()
    config_path = repo_root / CONFIG_FILENAME
    if not config_path.is_file():
        return None
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise TreehouseError(f"cannot read {config_path}: {exc}") from exc
    max_trees = raw.get("max_trees")
    if not isinstance(max_trees, int) or isinstance(max_trees, bool) or max_trees < 1:
        raise TreehouseError(
            f"{config_path}: max_trees must be a positive integer (got {max_trees!r})"
        )
    base_branch = raw.get("base_branch")
    if base_branch is not None and not isinstance(base_branch, str):
        raise TreehouseError(f"{config_path}: base_branch must be a string")
    return PoolConfig(
        repo_root=repo_root,
        config_path=config_path,
        max_trees=max_trees,
        base_branch=base_branch or None,
    )


def pool_repo_root(path: Path | str | None) -> Optional[Path]:
    """Repo root owning *path* when it sits inside a ``.treehouse`` pool."""
    if not path:
        return None
    candidate = Path(path).expanduser()
    for parent in [candidate, *candidate.parents]:
        if parent.name == POOL_DIRNAME:
            return parent.parent
    return None


def _path_key(path: Path | str) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve(strict=False)))


def _run(args: list[str], *, cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _treehouse(config: PoolConfig, *args: str, timeout: int) -> subprocess.CompletedProcess:
    binary = treehouse_binary()
    if binary is None:
        raise TreehouseError(
            f"{config.config_path} enables a treehouse pool but the treehouse CLI "
            "is not on PATH (install it or remove the config)"
        )
    return _run([binary, *args], cwd=config.repo_root, timeout=timeout)


def _slot_status(config: PoolConfig) -> list[dict]:
    result = _treehouse(config, "status", "--json", timeout=_STATUS_TIMEOUT_SECONDS)
    if result.returncode != 0:
        raise TreehouseError(
            f"treehouse status failed in {config.repo_root}: "
            f"{(result.stderr or result.stdout or '').strip()}"
        )
    try:
        slots = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise TreehouseError(f"treehouse status returned non-JSON: {result.stdout!r}") from exc
    return slots if isinstance(slots, list) else []


def _check_slot_inside_repo(config: PoolConfig, slot: Path) -> None:
    """Invariant: a pool slot lives under ``<repo>/.treehouse``, never at the
    repo root, and never somewhere treehouse's config did not ask for."""
    pool_root = config.pool_root.resolve(strict=False)
    resolved = slot.resolve(strict=False)
    if resolved != pool_root and pool_root not in resolved.parents:
        raise TreehouseError(
            f"treehouse handed out {resolved}, which is outside the in-repo pool "
            f"{pool_root}; check root/worktree_path in {config.config_path}"
        )
    if resolved == config.repo_root.resolve(strict=False):
        raise TreehouseError(
            f"treehouse handed out the repo root ({resolved}) as a workspace slot"
        )


def _is_git_checkout(path: Path) -> bool:
    if not path.exists():
        return False
    result = _run(["git", "rev-parse", "--is-inside-work-tree"], cwd=path, timeout=30)
    return result.returncode == 0 and (result.stdout or "").strip() == "true"


def _switch_to_card_branch(slot: Path, branch_name: str) -> None:
    """Check the card branch out inside a freshly acquired/leased slot.

    treehouse hands a slot over in detached HEAD. The card id must be visible
    in the branch name, so recreate or reuse ``wt/<task-id>``. A reused branch
    may be checked out in another slot (a stale one), which git refuses; that
    is reported as a treehouse error rather than silently running on the wrong
    commit.
    """
    listed = _run(["git", "branch", "--list", branch_name], cwd=slot, timeout=30)
    exists = bool((listed.stdout or "").strip())
    args = ["git", "switch", branch_name] if exists else ["git", "switch", "-c", branch_name]
    result = _run(args, cwd=slot, timeout=60)
    if result.returncode != 0:
        raise TreehouseError(
            f"cannot check out {branch_name} in pool slot {slot}: "
            f"{(result.stderr or result.stdout or '').strip()}"
        )


def _lease_payload(config: PoolConfig, result: subprocess.CompletedProcess) -> dict:
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        if "max_trees" in stderr or "in use or dirty" in stderr:
            raise PoolExhausted(
                f"treehouse pool exhausted for {config.repo_root} "
                f"(max_trees = {config.max_trees}): {stderr}"
            )
        raise TreehouseError(f"treehouse get --lease failed in {config.repo_root}: {stderr}")
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise TreehouseError(
            f"treehouse get --lease returned non-JSON stdout: {result.stdout!r}"
        ) from exc
    if not payload.get("path"):
        raise TreehouseError(f"treehouse get --lease returned no path: {payload!r}")
    return payload


def acquire_lease(config: PoolConfig, task_id: str, branch_name: str) -> Lease:
    """Take a durable lease from the pool for *task_id*, at the cap raising
    :class:`PoolExhausted` instead of spawning outside the pool."""
    holder = holder_label(task_id)
    result = _treehouse(
        config,
        "get",
        "--lease",
        "--json",
        "--lease-holder",
        holder,
        timeout=_GET_TIMEOUT_SECONDS,
    )
    payload = _lease_payload(config, result)
    slot = Path(payload["path"]).expanduser()
    try:
        _check_slot_inside_repo(config, slot)
        _switch_to_card_branch(slot, branch_name)
    except (TreehouseError, OSError):
        # Never leave a lease behind for a slot this card cannot use.
        with contextlib.suppress(TreehouseError, OSError):
            release_lease(slot, task_id)
        raise
    lease = Lease(
        path=slot,
        lease_id=str(payload.get("lease_id") or ""),
        holder=holder,
        max_trees=config.max_trees,
    )
    _log.info(
        "kanban: leased treehouse slot %s (lease %s, max_trees %s) for %s",
        lease.path, lease.lease_id or "?", lease.max_trees, task_id,
    )
    return lease


def resolve_pool_workspace(
    config: PoolConfig,
    task_id: str,
    branch_name: str,
    *,
    preferred_path: Path | None = None,
) -> Lease:
    """Return the card's one pool lease, acquiring a slot when it has none.

    The card's own lease is reused when it still holds a usable slot (the
    retry and running-worker cases); every other lease the card holds is
    returned, because a dispatcher that died between acquire and recording its
    workspace can leave a duplicate behind that would otherwise pin a slot
    forever. With no usable lease, a free slot is acquired; at the cap
    :class:`PoolExhausted` propagates.
    """
    holder = holder_label(task_id)
    ours = [
        entry for entry in _slot_status(config)
        if entry.get("status") == "leased"
        and entry.get("lease_holder") == holder
        and entry.get("path")
    ]
    if preferred_path is not None and len(ours) > 1:
        preferred_key = _path_key(preferred_path)
        ours.sort(key=lambda entry: _path_key(Path(entry["path"])) != preferred_key)
    reused: Optional[tuple[Path, str]] = None
    for entry in ours:
        slot = Path(entry["path"]).expanduser()
        _check_slot_inside_repo(config, slot)
        if reused is None and _is_git_checkout(slot):
            try:
                _switch_to_card_branch(slot, branch_name)
            except TreehouseError as exc:
                _log.warning("kanban: not reusing pool slot %s for %s: %s", slot, task_id, exc)
            else:
                reused = (slot, str(entry.get("lease_id") or ""))
                continue
        release_lease(slot, task_id)
    if reused is not None:
        path, lease_id = reused
        _log.info(
            "kanban: reusing treehouse slot %s (lease %s) for %s",
            path, lease_id or "?", task_id,
        )
        return Lease(path=path, lease_id=lease_id, holder=holder, max_trees=config.max_trees)
    return acquire_lease(config, task_id, branch_name)


def cap_reason(config: PoolConfig) -> Optional[str]:
    """Quotable reason when the pool cannot hand out a slot, else ``None``.

    Read-only: the dispatcher runs this before claiming a card so a full pool
    parks the card without writing a failed run every tick. Only a pool that
    is at ``max_trees`` and has no available slot is reported; below the cap
    treehouse may still create a slot, so it is left to decide.
    """
    slots = _slot_status(config)
    if any(str(entry.get("status") or "") == "available" for entry in slots):
        return None
    if len(slots) < config.max_trees:
        return None
    leased = sum(1 for entry in slots if entry.get("status") == "leased")
    return (
        f"treehouse pool exhausted for {config.repo_root}: all {len(slots)} "
        f"worktrees are in use or dirty (max_trees = {config.max_trees}, "
        f"{leased} leased)"
    )


def release_lease(path: Path | str, task_id: str) -> bool:
    """Return a leased slot to the pool. Idempotent; never touches another
    card's lease (the holder precondition fences that)."""
    slot = Path(path).expanduser()
    repo_root = pool_repo_root(slot)
    if repo_root is None:
        return False
    config = read_pool_config(repo_root)
    if config is None:
        _log.warning("kanban: %s looks like a pool slot but %s no longer exists", slot, repo_root)
        return False
    dirty = _run(["git", "status", "--porcelain"], cwd=slot, timeout=60)
    if (dirty.stdout or "").strip():
        _log.warning(
            "kanban: pool slot %s has uncommitted changes; returning it with --force "
            "(committed work survives on branch wt/%s)",
            slot, task_id,
        )
    result = _treehouse(
        config, "return", "--force", "--if-lease-holder", holder_label(task_id), str(slot),
        timeout=_RETURN_TIMEOUT_SECONDS,
    )
    if result.returncode == 0:
        _log.info("kanban: returned treehouse slot %s for %s", slot, task_id)
        return True
    detail = (result.stderr or result.stdout or "").strip()
    if "lease holder does not match" in detail:
        _log.warning("kanban: not returning %s for %s: %s", slot, task_id, detail)
        return False
    raise TreehouseError(f"treehouse return failed for {slot}: {detail}")


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid or pid <= 0:
        return False
    from hermes_cli.kanban_db_dispatch import _pid_alive as dispatch_pid_alive

    return dispatch_pid_alive(int(pid))


def _task_lease_row(conn: sqlite3.Connection, task_id: str):
    return conn.execute(
        "SELECT status, claim_expires, worker_pid, workspace_path,"
        " (SELECT MAX(ended_at) FROM task_runs WHERE task_runs.task_id = tasks.id)"
        " AS last_ended FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()


def _row_is_live(row, now: int) -> bool:
    """True while the card still owns the slot recorded at its workspace path.

    ``running``: live until its recorded worker is gone (the killed-worker
    case) or, with no recorded worker, until the claim expires. Finished: live
    until the worker had the same grace ``reap_terminal_workers`` gives it, so
    a worker still finalising after ``kanban_complete`` is never reset
    underneath.
    """
    from hermes_cli.kanban_db_dispatch import TERMINAL_WORKER_REAP_GRACE_SECONDS

    status = row["status"]
    if status == "running":
        worker_pid = row["worker_pid"]
        if worker_pid:
            return _pid_alive(worker_pid)
        claim_expires = row["claim_expires"]
        return claim_expires is None or int(claim_expires) >= now
    if status not in _TERMINAL_STATUSES:
        return True
    last_ended = row["last_ended"]
    if last_ended is None:
        return False
    return (now - int(last_ended)) < TERMINAL_WORKER_REAP_GRACE_SECONDS


def _lease_verdict(conn: sqlite3.Connection, task_id: str, slot_path: str, now: int) -> str:
    """``"keep"``, ``"release"`` or ``"foreign"`` for one leased slot."""
    row = _task_lease_row(conn, task_id)
    if row is None:
        # Unknown here: another board's database may own this lease.
        return "foreign"
    recorded = str(row["workspace_path"] or "")
    if recorded and _path_key(recorded) != _path_key(slot_path):
        # A lease this card is not running in: a duplicate left by a
        # dispatcher that died between acquire and recording its workspace.
        return "release"
    return "keep" if _row_is_live(row, now) else "release"


def _pool_repo_roots(conn: sqlite3.Connection) -> set[Path]:
    """Every repo whose pool this DB may hold leases in.

    Card workspace paths are the primary trail; a board's ``default_workdir``
    covers a lease acquired before its path was recorded (the dispatcher died
    in between), which is otherwise invisible to the sweep.
    """
    roots: set[Path] = set()
    for row in conn.execute(
        "SELECT DISTINCT workspace_path FROM tasks WHERE workspace_path IS NOT NULL"
    ):
        repo_root = pool_repo_root(row["workspace_path"])
        if repo_root is not None:
            roots.add(repo_root)
    from hermes_cli import kanban_db as _kb
    from hermes_cli.kanban_db_workspace import _git_toplevel

    try:
        boards = _kb.list_boards()
    except (OSError, ValueError, sqlite3.Error) as exc:
        _log.warning("kanban: cannot enumerate boards for pool scan: %s", exc, exc_info=True)
        return roots
    for board in boards:
        default_workdir = str(board.get("default_workdir") or "").strip()
        if not default_workdir:
            continue
        path = Path(default_workdir).expanduser()
        if not path.exists():
            continue
        repo_root = _git_toplevel(path)
        if repo_root is not None:
            roots.add(repo_root)
    return roots


def release_orphaned_leases(conn: sqlite3.Connection) -> list[str]:
    """Return every lease whose card is no longer running a worker.

    This is the reconciling half of "release on every exit path": completion,
    block, failure, timeout, reap, a killed worker and a crashed dispatcher all
    end up here, because the test is board truth (task status + worker liveness)
    rather than an in-process callback that a SIGKILL skips. A lease the card
    is not running in -- a duplicate, or one acquired after a crash before the
    workspace was recorded -- is returned even while the card is alive.

    Leases whose holder task id this DB does not know are left alone: they
    belong to another board's database.
    """
    now = int(time.time())
    released: list[str] = []
    for repo_root in sorted(_pool_repo_roots(conn)):
        try:
            config = read_pool_config(repo_root)
            if config is None:
                continue
            slots = _slot_status(config)
        except (TreehouseError, OSError) as exc:
            _log.warning("kanban: treehouse pool scan failed for %s: %s", repo_root, exc)
            continue
        for entry in slots:
            holder = str(entry.get("lease_holder") or "")
            if not holder.startswith(HOLDER_PREFIX) or entry.get("status") != "leased":
                continue
            task_id = holder[len(HOLDER_PREFIX):]
            slot_path = str(entry.get("path") or "")
            if not task_id or not slot_path:
                continue
            if _lease_verdict(conn, task_id, slot_path, now) != "release":
                continue
            try:
                if release_lease(slot_path, task_id):
                    released.append(task_id)
            except (TreehouseError, OSError) as exc:
                _log.warning("kanban: could not return lease for %s: %s", task_id, exc)
    return released
