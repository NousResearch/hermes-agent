"""Durable, fail-closed Git worktrees for interactive conversation roots."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Iterator
import uuid

from agent.conversation_worktree_policy import ConversationWorktreePolicy
from hermes_cli.worktree_environment import bootstrap_worktree_environments
from hermes_cli._subprocess_compat import (
    IS_WINDOWS,
    kill_process_tree,
    noninteractive_git_env,
    windows_hide_flags,
)
from hermes_state import (
    ConversationWorktreeConflict,
    ConversationWorktreeRecord,
    SessionDB,
)


logger = logging.getLogger(__name__)

_REPOSITORY_THREAD_LOCKS: dict[str, threading.Lock] = {}
_REPOSITORY_THREAD_LOCKS_GUARD = threading.Lock()
_OWNER_MARKER = "hermes-conversation-owner-v1"
_COMMON_OWNER_CLAIMS_DIR = "hermes-conversation-owner-claims-v1"
_LEASE_MESSAGE_LIMIT = 300


class ConversationWorktreeError(RuntimeError):
    """A conversation worktree could not be safely created or reused."""

    def __init__(self, message: str, *, phase: str) -> None:
        super().__init__(message)
        self.phase = phase


@dataclass(frozen=True)
class ConversationWorktreeBinding:
    """A validated, task-owned working directory for one conversation root."""

    root_session_id: str
    path: Path
    branch: str
    base_commit: str
    repo_common_dir: Path


@dataclass(frozen=True)
class CleanupVerdict:
    """Complete fail-closed status for one explicit cleanup request."""

    allowed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CleanupResult:
    """Result of an explicit cleanup attempt and the verdict that governed it."""

    removed: bool
    verdict: CleanupVerdict
    failure_phase: str | None = None
    failure_message: str | None = None


@dataclass
class ConversationRootLease:
    """Mandatory cross-process ownership lease for one conversation root."""

    lease_id: str
    root_session_id: str
    state_path: Path
    lock_path: Path
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        try:
            with _lease_file_lock(self.lock_path, timeout=3.0):
                entries, valid = _read_root_leases(self.state_path)
                if not valid:
                    raise ConversationWorktreeError(
                        "conversation root lease registry is unavailable",
                        phase="lease",
                    )
                _write_root_leases(
                    self.state_path,
                    [e for e in entries if e.get("lease_id") != self.lease_id],
                )
        except ConversationWorktreeError:
            raise
        except Exception as exc:
            raise ConversationWorktreeError(
                "conversation root lease registry is unavailable", phase="lease"
            ) from exc
        self.released = True


def _root_lease_paths(repo_common_dir: Path, root_session_id: str) -> tuple[Path, Path]:
    digest = hashlib.sha256(root_session_id.encode("utf-8")).hexdigest()[:24]
    return (
        repo_common_dir / f"hermes-conversation-root-{digest}.leases.json",
        repo_common_dir / f"hermes-conversation-root-{digest}.leases.lock",
    )


@contextmanager
def _lease_file_lock(path: Path, *, timeout: float) -> Iterator[None]:
    deadline = time.monotonic() + timeout
    handle = None
    locked = False
    thread_lock = ConversationWorktreeManager._thread_lock(path)
    if not thread_lock.acquire(timeout=timeout):
        raise OSError("conversation root lease lock timed out")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise OSError("conversation root lease lock timed out")
                time.sleep(0.05)
        yield
    finally:
        if handle is not None:
            if locked:
                try:
                    if os.name == "nt":
                        import msvcrt

                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    logger.warning("conversation_worktree.lease_lock_release_failed")
            handle.close()
        thread_lock.release()


def _read_root_leases(path: Path) -> tuple[list[dict[str, object]], bool]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [], True
    except Exception:
        return [], False
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        return [], False
    return list(entries), True


def _write_root_leases(path: Path, entries: list[dict[str, object]]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump({"entries": entries}, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _process_liveness(entry: dict[str, object]) -> str:
    try:
        pid = int(entry.get("pid") or 0)
    except (TypeError, ValueError):
        return "unknown"
    if pid <= 0:
        return "unknown"
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return "dead"
    except Exception:
        return "unknown"
    expected = entry.get("process_start_time")
    if expected is None:
        return "active"
    try:
        from hermes_cli.active_sessions import _process_start_time

        current = _process_start_time(pid)
        if current is None:
            return "unknown"
        return "active" if abs(float(expected) - current) < 0.001 else "dead"
    except Exception:
        return "unknown"


def acquire_conversation_root_lease(
    *,
    root_session_id: str,
    worktree_path: Path,
    repo_common_dir: Path,
    surface: str,
) -> ConversationRootLease:
    """Acquire mandatory root liveness independent of concurrency limits."""
    state_path, lock_path = _root_lease_paths(repo_common_dir, root_session_id)
    lease_id = uuid.uuid4().hex
    try:
        with _lease_file_lock(lock_path, timeout=3.0):
            entries, valid = _read_root_leases(state_path)
            if not valid:
                raise ConversationWorktreeError(
                    "conversation root lease registry is unavailable", phase="lease"
                )
            kept = [e for e in entries if _process_liveness(e) != "dead"]
            kept.append(
                {
                    "lease_id": lease_id,
                    "root_session_id": root_session_id,
                    "worktree_path": str(worktree_path.resolve()),
                    "repo_common_dir": str(repo_common_dir.resolve()),
                    "surface": str(surface),
                    "pid": os.getpid(),
                    "process_start_time": __import__(
                        "hermes_cli.active_sessions", fromlist=["_process_start_time"]
                    )._process_start_time(os.getpid()),
                    "started_at": time.time(),
                }
            )
            _write_root_leases(state_path, kept)
    except ConversationWorktreeError:
        raise
    except Exception as exc:
        raise ConversationWorktreeError(
            "conversation root lease registry is unavailable", phase="lease"
        ) from exc
    return ConversationRootLease(
        lease_id=lease_id,
        root_session_id=root_session_id,
        state_path=state_path,
        lock_path=lock_path,
    )


def _common_owner_claim_path(repo_common_dir: Path, worktree_path: Path) -> Path:
    """Return the source-Git durable claim location for one exact worktree path."""
    digest = hashlib.sha256(str(worktree_path.resolve()).encode("utf-8")).hexdigest()
    return repo_common_dir / _COMMON_OWNER_CLAIMS_DIR / f"{digest}.json"


def _owner_claim_matches(
    data: object, *, worktree_path: Path, repo_common_dir: Path
) -> bool:
    return (
        isinstance(data, dict)
        and data.get("owner") == "conversation-worktree-manager"
        and data.get("worktree_path") == str(worktree_path.resolve())
        and data.get("repo_common_dir") == str(repo_common_dir.resolve())
        and isinstance(data.get("root_session_id"), str)
        and bool(data["root_session_id"])
    )


def _owner_claim_matches_record(
    data: object, *, record: ConversationWorktreeRecord
) -> bool:
    """Require ownership evidence for this exact durable conversation root."""
    return (
        _owner_claim_matches(
            data,
            worktree_path=Path(record.worktree_path),
            repo_common_dir=Path(record.repo_common_dir),
        )
        and data.get("root_session_id") == record.root_session_id
    )


def conversation_worktree_ownership_verdict(path: Path) -> bool | None:
    """Return manager ownership, absence, or an unverified ownership state.

    The common-repository claim is written before ``git worktree add``.  It
    closes the crash window where Git has created the worktree but the
    worktree-local marker cannot yet be persisted.  Any malformed or
    unreadable ownership evidence is deliberately ``None`` so every GC caller
    keeps the tree rather than guessing that it is safe to remove.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    repo_common_dir = Path(result.stdout.strip()).resolve()

    try:
        marker_result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--git-path", _OWNER_MARKER],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if marker_result.returncode != 0 or not marker_result.stdout.strip():
        return None
    marker = Path(marker_result.stdout.strip())
    if not marker.is_absolute():
        marker = path / marker
    if marker.exists():
        try:
            if _owner_claim_matches(
                json.loads(marker.read_text(encoding="utf-8")),
                worktree_path=path,
                repo_common_dir=repo_common_dir,
            ):
                return True
            return None
        except Exception:
            return None

    common_claim = _common_owner_claim_path(repo_common_dir, path)
    if not common_claim.exists():
        return False
    try:
        if _owner_claim_matches(
            json.loads(common_claim.read_text(encoding="utf-8")),
            worktree_path=path,
            repo_common_dir=repo_common_dir,
        ):
            return True
    except Exception:
        pass
    return None


def conversation_worktree_is_manager_owned(path: Path) -> bool | None:
    """Return ownership status; ``None`` means ownership cannot be verified."""
    return conversation_worktree_ownership_verdict(path)


def _write_owner_claim(path: Path, payload: dict[str, str]) -> None:
    """Atomically write ownership evidence without platform-specific renames."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        raise ConversationWorktreeError(
            "conversation worktree ownership claim is unavailable", phase="create"
        ) from exc


@contextmanager
def conversation_worktree_reclaim_guard(
    repo_root: Path, path: Path
) -> Iterator[bool | None]:
    """Hold the manager repository lock while rechecking durable ownership."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            yield None
            return
        common_dir = Path(result.stdout.strip()).resolve()
        with _lease_file_lock(
            common_dir / "hermes-conversation-worktree.lock", timeout=5.0
        ):
            yield conversation_worktree_is_manager_owned(path)
    except Exception:
        yield None


class ConversationWorktreeManager:
    """Own the Git lifecycle for interactive root-session worktrees only."""

    def __init__(self, policy: ConversationWorktreePolicy, db: SessionDB) -> None:
        self._policy = policy
        self._db = db

    def bind_new_root_session(
        self, root_session_id: str, *, conversation_kind: str
    ) -> ConversationWorktreeBinding | None:
        """Create or recover an interactive root binding without touching source work.

        Task/delegated callers deliberately bypass this manager: they retain their
        existing worktree ownership instead of being converted into conversation
        roots.
        """
        self._event("conversation_worktree.policy", root_session_id=root_session_id)
        if conversation_kind == "task":
            return None
        if conversation_kind != "interactive":
            raise ConversationWorktreeError(
                "conversation kind must be 'interactive' or 'task'", phase="policy"
            )
        if not self._policy.enabled:
            return None
        if not root_session_id:
            raise ConversationWorktreeError(
                "root session id must be non-empty", phase="identity"
            )

        existing = self._db.get_conversation_worktree(root_session_id)
        source, source_common_dir = self._source_repository_identity()
        path, branch = self._expected_identity(root_session_id)
        if self._is_within(path, source):
            raise ConversationWorktreeError(
                "worktree_root must not create conversation worktrees inside source_worktree",
                phase="policy",
            )
        record = existing
        try:
            with self._repository_lock(source_common_dir):
                record = self._db.get_conversation_worktree(root_session_id) or record
                if record is not None and record.state == "ready":
                    return self._resolve_ready_binding_locked(
                        source,
                        source_common_dir,
                        record,
                        path=path,
                        branch=branch,
                    )
                self._validate_worktree_root_ownership(
                    source,
                    source_common_dir,
                    expected_path=path,
                    existing=record,
                )
                if record is None:
                    base_commit = self._git_stdout(
                        source, ["rev-parse", "HEAD"], "identity"
                    )
                    try:
                        record = self._db.claim_conversation_worktree(
                            root_session_id=root_session_id,
                            worktree_path=str(path),
                            branch=branch,
                            base_commit=base_commit,
                            repo_common_dir=str(source_common_dir),
                        )
                    except ConversationWorktreeConflict as exc:
                        record = self._db.get_conversation_worktree(root_session_id)
                        if record is None:
                            raise ConversationWorktreeError(
                                "conversation worktree identity claim conflicted",
                                phase="identity",
                            ) from exc
                self._validate_record_identity(
                    record,
                    path=path,
                    branch=branch,
                    repo_common_dir=source_common_dir,
                )
                self._prepare_worktree(source, record)
                self._ensure_git_worktree_locked(source, record)

            # Bootstrap may be slow, network-bound, or intentionally interactive
            # at the project level. It must never monopolize the repository-wide
            # Git metadata lock: that lock protects only claim/create/validation.
            # A root-specific lock serializes bootstrap/readiness for retries of
            # this one root without blocking different conversation roots.
            with self._root_lock(source_common_dir, root_session_id):
                record = self._db.get_conversation_worktree(root_session_id) or record
                if record.state == "ready":
                    return self._validated_ready_binding(record)
                self._validate_record_identity(
                    record,
                    path=path,
                    branch=branch,
                    repo_common_dir=source_common_dir,
                )
                self._require_recoverable_record(record)
                self._validate_new_worktree(record)
                self._run_bootstrap(record)
                ready = self._db.mark_conversation_worktree_ready(root_session_id)
                binding = self._binding_from_record(ready)
                self._event("conversation_worktree.ready", root_session_id=root_session_id)
                return binding
        except ConversationWorktreeError as exc:
            self._record_failure(root_session_id, record, exc)
            self._event(
                "conversation_worktree.failure",
                root_session_id=root_session_id,
                phase=exc.phase,
            )
            raise

    def resolve_existing_session(
        self, root_session_id: str
    ) -> ConversationWorktreeBinding | None:
        """Resolve only an already-ready root binding; never create a new one."""
        record = self._db.get_conversation_worktree(root_session_id)
        if record is None:
            return None
        source, source_common_dir = self._source_repository_identity()
        path, branch = self._expected_identity(root_session_id)
        if self._is_within(path, source):
            raise ConversationWorktreeError(
                "worktree_root must not create conversation worktrees inside source_worktree",
                phase="policy",
            )
        with self._repository_lock(source_common_dir):
            record = self._db.get_conversation_worktree(root_session_id)
            if record is None:
                return None
            return self._resolve_ready_binding_locked(
                source,
                source_common_dir,
                record,
                path=path,
                branch=branch,
            )

    def _resolve_ready_binding_locked(
        self,
        source: Path,
        source_common_dir: Path,
        record: ConversationWorktreeRecord,
        *,
        path: Path,
        branch: str,
    ) -> ConversationWorktreeBinding:
        """Validate and retain one ready binding while holding its repository lock."""
        self._validate_worktree_root_ownership(
            source,
            source_common_dir,
            expected_path=path,
            existing=record,
        )
        binding = self._validated_ready_binding(record)
        self._ensure_git_worktree_locked(source, record)
        return binding

    def inspect_cleanup(
        self,
        root_session_id: str,
        *,
        active_session_bound: bool = False,
    ) -> CleanupVerdict:
        """Inspect one exact owned binding without changing Git or ledger state."""
        record = self._db.get_conversation_worktree(root_session_id)
        if record is None:
            return CleanupVerdict(False, ("unknown",))
        try:
            with self._root_lease_liveness(record) as root_liveness:
                return self._inspect_cleanup_record(
                    record,
                    active_session_bound=active_session_bound,
                    root_liveness=root_liveness,
                )
        except Exception:
            return CleanupVerdict(False, ("unknown",))

    def remove_after_explicit_request(
        self,
        root_session_id: str,
        *,
        active_session_bound: bool = False,
    ) -> CleanupResult:
        """Remove only a re-inspected safe binding after an explicit request."""
        record = self._db.get_conversation_worktree(root_session_id)
        if record is None:
            verdict = CleanupVerdict(False, ("unknown",))
            return CleanupResult(False, verdict)

        try:
            source, source_common_dir = self._source_repository_identity()
            with self._repository_lock(source_common_dir):
                with self._root_lock(source_common_dir, root_session_id):
                    current = self._db.get_conversation_worktree(root_session_id)
                    if current is None:
                        verdict = CleanupVerdict(False, ("unknown",))
                        return CleanupResult(False, verdict)
                    with self._root_lease_liveness(current) as root_liveness:
                        verdict = self._inspect_cleanup_record(
                            current,
                            active_session_bound=active_session_bound,
                            root_liveness=root_liveness,
                            source_identity=(source, source_common_dir),
                        )
                        if not verdict.allowed:
                            return CleanupResult(False, verdict)

                        path = Path(current.worktree_path).resolve()
                        unlocked = self._run_git(
                            source,
                            ["worktree", "unlock", str(path)],
                            self._policy.create_timeout,
                            "cleanup",
                        )
                        if unlocked.returncode != 0:
                            message = self._sanitize_remove_failure(unlocked.stderr)
                            logger.warning(
                                "conversation_worktree.remove_failed phase=unlock message=%s",
                                message,
                            )
                            return CleanupResult(
                                False,
                                CleanupVerdict(False, ("remove_failed",)),
                                failure_phase="unlock",
                                failure_message=message,
                            )
                        removed = self._run_git(
                            source,
                            ["worktree", "remove", str(path)],
                            self._policy.create_timeout,
                            "cleanup",
                        )
                        if removed.returncode != 0:
                            try:
                                self._ensure_git_worktree_locked(source, current)
                            except ConversationWorktreeError:
                                logger.warning(
                                    "conversation_worktree.relock_after_remove_failure_failed",
                                    exc_info=True,
                                )
                            message = self._sanitize_remove_failure(removed.stderr)
                            logger.warning(
                                "conversation_worktree.remove_failed phase=remove message=%s",
                                message,
                            )
                            return CleanupResult(
                                False,
                                CleanupVerdict(False, ("remove_failed",)),
                                failure_phase="remove",
                                failure_message=message,
                            )
                        if path.exists() or self._listed_worktree(source, path) is not None:
                            message = "worktree remained after git removal"
                            logger.warning(
                                "conversation_worktree.remove_failed phase=verify message=%s",
                                message,
                            )
                            return CleanupResult(
                                False,
                                CleanupVerdict(False, ("remove_failed",)),
                                failure_phase="verify",
                                failure_message=message,
                            )

                        self._db.mark_conversation_worktree_removed(root_session_id)
                        self._remove_common_owner_claim(current)
                        self._event(
                            "conversation_worktree.removed",
                            root_session_id=root_session_id,
                        )
                        return CleanupResult(True, verdict)
        except ConversationWorktreeError:
            return CleanupResult(False, CleanupVerdict(False, ("unknown",)))
        except Exception:
            logger.exception("conversation_worktree.cleanup_failed")
            return CleanupResult(False, CleanupVerdict(False, ("unknown",)))

    def _inspect_cleanup_record(
        self,
        record: ConversationWorktreeRecord,
        *,
        active_session_bound: bool,
        root_liveness: str = "inactive",
        source_identity: tuple[Path, Path] | None = None,
    ) -> CleanupVerdict:
        reasons: list[str] = []

        def block(reason: str) -> None:
            if reason not in reasons:
                reasons.append(reason)

        try:
            source, source_common_dir = source_identity or self._source_repository_identity()
            expected_path, expected_branch = self._expected_identity(record.root_session_id)
            if (
                record.state not in {"ready", "retained"}
                or Path(record.worktree_path).resolve() != expected_path.resolve()
                or record.branch != expected_branch
                or Path(record.repo_common_dir).resolve() != source_common_dir.resolve()
                or not expected_path.is_dir()
            ):
                return CleanupVerdict(False, ("mismatched identity",))

            listed = self._listed_worktree(source, expected_path)
            if listed != f"refs/heads/{record.branch}":
                return CleanupVerdict(False, ("mismatched identity",))

            actual_branch = self._git_stdout(
                expected_path, ["branch", "--show-current"], "cleanup"
            )
            actual_common = Path(
                self._git_stdout(
                    expected_path,
                    ["rev-parse", "--path-format=absolute", "--git-common-dir"],
                    "cleanup",
                )
            ).resolve()
            base_ancestor = self._run_git(
                expected_path,
                ["merge-base", "--is-ancestor", record.base_commit, "HEAD"],
                self._policy.create_timeout,
                "cleanup",
            )
            if (
                actual_branch != record.branch
                or actual_common != source_common_dir.resolve()
                or base_ancestor.returncode != 0
            ):
                return CleanupVerdict(False, ("mismatched identity",))

            if active_session_bound:
                block("active")
            if root_liveness == "active":
                block("active")
            elif root_liveness != "inactive":
                block("unknown")

            status = self._run_git(
                expected_path,
                ["status", "--porcelain", "--untracked-files=all"],
                self._policy.create_timeout,
                "cleanup",
            )
            if status.returncode != 0:
                block("unknown")
            elif status.stdout.strip():
                block("dirty")

            if self._git_operation_in_progress(expected_path):
                block("in-progress")

            head = self._git_stdout(expected_path, ["rev-parse", "HEAD"], "cleanup")
            integration_head = self._git_stdout(source, ["rev-parse", "HEAD"], "cleanup")
            integrated = self._run_git(
                source,
                ["merge-base", "--is-ancestor", head, integration_head],
                self._policy.create_timeout,
                "cleanup",
            )
            if integrated.returncode == 1:
                block("unintegrated")
            elif integrated.returncode != 0:
                block("unknown")

            remotes = self._run_git(
                expected_path,
                ["remote"],
                self._policy.create_timeout,
                "cleanup",
            )
            if remotes.returncode != 0:
                block("unknown")
            elif not remotes.stdout.strip():
                block("missing remote evidence")
            else:
                remote_refs = self._run_git(
                    expected_path,
                    [
                        "for-each-ref",
                        "--format=%(refname)",
                        "--contains",
                        head,
                        "refs/remotes",
                    ],
                    self._policy.create_timeout,
                    "cleanup",
                )
                if remote_refs.returncode != 0:
                    block("unknown")
                elif not remote_refs.stdout.strip():
                    block("unpushed")
        except Exception:
            block("unknown")

        return CleanupVerdict(not reasons, tuple(reasons))

    def _listed_worktree(self, source: Path, expected_path: Path) -> str | None:
        result = self._run_git(
            source,
            ["worktree", "list", "--porcelain"],
            self._policy.create_timeout,
            "cleanup",
        )
        if result.returncode != 0:
            raise ConversationWorktreeError(
                "git worktree inspection failed", phase="cleanup"
            )
        path: Path | None = None
        branch: str | None = None
        for line in [*result.stdout.splitlines(), ""]:
            if line.startswith("worktree "):
                path = Path(line.removeprefix("worktree ")).resolve()
                branch = None
            elif line.startswith("branch "):
                branch = line.removeprefix("branch ").strip()
            elif not line and path is not None:
                if path == expected_path.resolve():
                    return branch
                path = None
                branch = None
        return None

    def _git_operation_in_progress(self, path: Path) -> bool:
        markers = (
            "MERGE_HEAD",
            "CHERRY_PICK_HEAD",
            "REVERT_HEAD",
            "BISECT_LOG",
            "rebase-apply",
            "rebase-merge",
            "sequencer",
            "index.lock",
            "HEAD.lock",
            "packed-refs.lock",
        )
        branch = self._git_stdout(path, ["branch", "--show-current"], "cleanup")
        if branch:
            markers = (*markers, f"refs/heads/{branch}.lock")
        for marker in markers:
            marker_path = Path(
                self._git_stdout(path, ["rev-parse", "--git-path", marker], "cleanup")
            )
            if not marker_path.is_absolute():
                marker_path = path / marker_path
            if marker_path.exists():
                return True
        return False

    @contextmanager
    def _root_lease_liveness(
        self, record: ConversationWorktreeRecord
    ) -> Iterator[str]:
        state_path, lock_path = _root_lease_paths(
            Path(record.repo_common_dir), record.root_session_id
        )
        try:
            with _lease_file_lock(lock_path, timeout=self._policy.create_timeout):
                entries, valid = _read_root_leases(state_path)
                if not valid:
                    yield "unknown"
                    return
                relevant: list[dict[str, object]] = []
                uncertain = False
                for entry in entries:
                    if (
                        entry.get("root_session_id") != record.root_session_id
                        or Path(str(entry.get("worktree_path") or "")).resolve()
                        != Path(record.worktree_path).resolve()
                        or Path(str(entry.get("repo_common_dir") or "")).resolve()
                        != Path(record.repo_common_dir).resolve()
                    ):
                        uncertain = True
                        continue
                    state = _process_liveness(entry)
                    if state == "active":
                        relevant.append(entry)
                    elif state == "unknown":
                        uncertain = True
                if len(relevant) != len(entries):
                    live_or_unknown = [
                        entry
                        for entry in entries
                        if _process_liveness(entry) != "dead"
                    ]
                    if len(live_or_unknown) != len(entries):
                        _write_root_leases(state_path, live_or_unknown)
                yield "unknown" if uncertain else ("active" if relevant else "inactive")
        except Exception:
            yield "unknown"

    @staticmethod
    def _sanitize_remove_failure(stderr: str) -> str:
        text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(stderr or ""))
        text = re.sub(
            r"(?i)\b(token|password|secret|authorization)\s*=\s*\S+",
            r"\1=<redacted>",
            text,
        )
        text = " ".join(text.split()) or "git worktree remove failed"
        return text[:_LEASE_MESSAGE_LIMIT]

    def _source_repository_identity(self) -> tuple[Path, Path]:
        source = self._policy.source_worktree
        if source is None:
            raise ConversationWorktreeError(
                "enabled conversation worktree policy has no source_worktree", phase="policy"
            )
        source = source.resolve()
        if not source.is_dir():
            raise ConversationWorktreeError("source_worktree does not exist", phase="identity")
        common = self._git_stdout(
            source,
            ["rev-parse", "--path-format=absolute", "--git-common-dir"],
            "identity",
        )
        common_dir = Path(common).resolve()
        if not common_dir.is_dir():
            raise ConversationWorktreeError(
                "source_worktree did not resolve to a usable Git common directory",
                phase="identity",
            )
        return source, common_dir

    def _validate_worktree_root_ownership(
        self,
        source: Path,
        source_common_dir: Path,
        *,
        expected_path: Path,
        existing: ConversationWorktreeRecord | None,
    ) -> None:
        """Refuse a configured output root owned by a different repository.

        A worktree directory may sit under a parent repository even when the
        configured path itself has not been created yet. Resolve the nearest
        existing ancestor and let Git discover its common directory from there;
        only a same-common-dir owner is compatible with the configured source.
        """
        root = self._policy.worktree_root
        assert root is not None  # policy was validated by _expected_identity
        nearest = root.resolve()
        while not nearest.exists() and nearest != nearest.parent:
            nearest = nearest.parent
        if not nearest.exists():
            raise ConversationWorktreeError(
                "worktree_root has no existing ancestor", phase="policy"
            )
        result = self._run_git(
            nearest,
            ["rev-parse", "--path-format=absolute", "--git-common-dir"],
            self._policy.create_timeout,
            "policy",
        )
        if result.returncode == 0:
            owner_common_dir = Path(result.stdout.strip()).resolve()
            if owner_common_dir != source_common_dir.resolve():
                raise ConversationWorktreeError(
                    "worktree_root is inside an unrelated repository", phase="policy"
                )

        registered = self._git_stdout(
            source,
            ["worktree", "list", "--porcelain"],
            "policy",
        )
        registered_paths = [
            Path(line.removeprefix("worktree ")).resolve()
            for line in registered.splitlines()
            if line.startswith("worktree ")
        ]
        primary_path = registered_paths[0] if registered_paths else None
        for registered_path in registered_paths:
            # Git lists the primary checkout first. A conventional
            # <primary>/.worktrees root is safe after the common-dir check
            # above, while nesting under a linked sibling remains unsafe.
            if registered_path == primary_path:
                continue
            root_is_nested = self._is_within(root, registered_path)
            target_is_nested = self._is_within(expected_path, registered_path)
            existing_owns_target = (
                existing is not None
                and registered_path == expected_path.resolve()
                and Path(existing.worktree_path).resolve() == expected_path.resolve()
            )
            if root_is_nested or (target_is_nested and not existing_owns_target):
                raise ConversationWorktreeError(
                    "worktree_root or target is inside a registered worktree",
                    phase="policy",
                )

    def _expected_identity(self, root_session_id: str) -> tuple[Path, str]:
        worktree_root = self._policy.worktree_root
        if worktree_root is None:
            raise ConversationWorktreeError(
                "enabled conversation worktree policy has no worktree_root", phase="policy"
            )
        digest = hashlib.sha256(root_session_id.encode("utf-8")).hexdigest()[:24]
        name = f"conversation-{digest}"
        return worktree_root.resolve() / name, f"{self._policy.branch_prefix}/{name}"

    @staticmethod
    def _is_within(candidate: Path, parent: Path) -> bool:
        try:
            candidate.resolve().relative_to(parent.resolve())
        except ValueError:
            return False
        return True

    def _validate_record_identity(
        self,
        record: ConversationWorktreeRecord,
        *,
        path: Path,
        branch: str,
        repo_common_dir: Path,
    ) -> None:
        if (
            Path(record.worktree_path).resolve() != path.resolve()
            or record.branch != branch
            or Path(record.repo_common_dir).resolve() != repo_common_dir.resolve()
        ):
            raise ConversationWorktreeError(
                "conversation worktree identity conflicts with configured source or root",
                phase="identity",
            )

    @staticmethod
    def _require_recoverable_record(record: ConversationWorktreeRecord) -> None:
        if record.state == "removed":
            raise ConversationWorktreeError(
                "conversation worktree was explicitly removed", phase="recovery"
            )
        if record.state not in {"creating", "creation_failed"}:
            raise ConversationWorktreeError(
                f"conversation worktree is not recoverable from state {record.state!r}",
                phase="recovery",
            )

    def _prepare_worktree(self, source: Path, record: ConversationWorktreeRecord) -> None:
        """Create or identity-validate the tree while holding the Git lock."""
        self._require_recoverable_record(record)
        path = Path(record.worktree_path)
        if path.exists():
            try:
                self._validate_new_worktree(record)
            except ConversationWorktreeError as exc:
                if record.state == "creating":
                    raise ConversationWorktreeError(
                        "conversation worktree path already exists and is not a matching worktree",
                        phase="create",
                    ) from exc
                raise
            self._ensure_common_owner_claim(record)
            self._ensure_owner_marker(record)
            self._event("conversation_worktree.reuse", root_session_id=record.root_session_id)
        elif record.state == "creation_failed":
            raise ConversationWorktreeError(
                "failed conversation worktree is missing; retained state requires manual recovery",
                phase="recovery",
            )
        else:
            self._create_worktree(source, record)

    def _ensure_git_worktree_locked(
        self, source: Path, record: ConversationWorktreeRecord
    ) -> None:
        """Keep an active managed tree protected by Git's lifecycle lock."""
        path = Path(record.worktree_path).resolve()
        listing = self._git_stdout(
            source,
            ["worktree", "list", "--porcelain"],
            "create",
        )
        current_path: Path | None = None
        for line in listing.splitlines():
            if line.startswith("worktree "):
                current_path = Path(line.removeprefix("worktree ")).resolve()
            elif current_path == path and line.startswith("locked"):
                return

        # Preserve the lifecycle invariant introduced by #48699 / @JoaoMarcos44:
        # a Hermes-owned worktree stays Git-locked until its cleanup owner unlocks it.
        self._git_stdout(
            source,
            [
                "worktree",
                "lock",
                "--reason",
                f"Hermes conversation {record.root_session_id}",
                str(path),
            ],
            "create",
        )

    @contextmanager
    def _repository_lock(self, common_dir: Path) -> Iterator[None]:
        """Bound one repository's metadata mutation across Hermes processes."""
        with self._lock_path(common_dir / "hermes-conversation-worktree.lock"):
            yield

    @contextmanager
    def _root_lock(self, common_dir: Path, root_session_id: str) -> Iterator[None]:
        """Serialize bootstrap/readiness only for one durable conversation root."""
        digest = hashlib.sha256(root_session_id.encode("utf-8")).hexdigest()[:24]
        with self._lock_path(common_dir / f"hermes-conversation-root-{digest}.lock"):
            yield

    @contextmanager
    def _lock_path(self, lock_path: Path) -> Iterator[None]:
        """Take a bounded process + thread lock and normalize setup failures."""
        deadline = time.monotonic() + self._policy.create_timeout
        handle = None
        locked = False
        thread_lock = self._thread_lock(lock_path)
        if not thread_lock.acquire(timeout=self._policy.create_timeout):
            raise ConversationWorktreeError(
                "repository worktree lock timed out", phase="create"
            )
        try:
            try:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                handle = self._open_lock_file(lock_path)
                handle.seek(0)
                handle.write(b"0")
                handle.flush()
            except OSError as exc:
                raise ConversationWorktreeError(
                    "conversation worktree lock is unavailable", phase="create"
                ) from exc
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt

                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise ConversationWorktreeError(
                            "repository worktree lock timed out", phase="create"
                        )
                    time.sleep(0.05)
            yield
        finally:
            if handle is not None:
                if locked:
                    try:
                        if os.name == "nt":
                            import msvcrt

                            handle.seek(0)
                            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                        else:
                            import fcntl

                            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        logger.warning("conversation_worktree.lock_release_failed")
                try:
                    handle.close()
                except OSError:
                    logger.warning("conversation_worktree.lock_close_failed")
            thread_lock.release()

    @staticmethod
    def _open_lock_file(lock_path: Path):
        return lock_path.open("a+b")

    @staticmethod
    def _thread_lock(lock_path: Path) -> threading.Lock:
        """Return the in-process companion to the cross-process file lock."""
        key = str(lock_path.resolve())
        with _REPOSITORY_THREAD_LOCKS_GUARD:
            lock = _REPOSITORY_THREAD_LOCKS.get(key)
            if lock is None:
                lock = threading.Lock()
                _REPOSITORY_THREAD_LOCKS[key] = lock
            return lock

    def _create_worktree(self, source: Path, record: ConversationWorktreeRecord) -> None:
        path = Path(record.worktree_path)
        if path.exists():
            raise ConversationWorktreeError(
                "conversation worktree path already exists", phase="create"
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConversationWorktreeError(
                "conversation worktree parent could not be created", phase="create"
            ) from exc

        self._event("conversation_worktree.create", root_session_id=record.root_session_id)
        # This common-repository claim is intentionally written *before* Git
        # creates the worktree.  A crash or failed per-worktree marker write
        # after `git worktree add` must still be visible to every generic GC.
        self._ensure_common_owner_claim(record)
        self._git_stdout(
            source,
            [
                "worktree",
                "add",
                "--no-track",
                "-b",
                record.branch,
                str(path),
                record.base_commit,
            ],
            "create",
        )
        # Provision the source repository's own runtime before this new
        # conversation worktree can be bound to an agent.  This is additive:
        # an existing destination is never replaced, and a missing source
        # environment is left for the configured project bootstrap policy.
        bootstrap_worktree_environments(source, path, environment_names=(".venv",))
        self._validate_new_worktree(record)
        self._ensure_owner_marker(record)

    def _validate_new_worktree(self, record: ConversationWorktreeRecord) -> None:
        path = Path(record.worktree_path)
        if not path.is_dir():
            raise ConversationWorktreeError(
                "conversation worktree path is missing", phase="validate"
            )
        head = self._git_stdout(path, ["rev-parse", "HEAD"], "validate")
        branch = self._git_stdout(path, ["branch", "--show-current"], "validate")
        common = Path(
            self._git_stdout(
                path,
                ["rev-parse", "--path-format=absolute", "--git-common-dir"],
                "validate",
            )
        ).resolve()
        if (
            head != record.base_commit
            or branch != record.branch
            or common != Path(record.repo_common_dir).resolve()
        ):
            raise ConversationWorktreeError(
                "created conversation worktree failed identity validation", phase="validate"
            )

    def _validated_ready_binding(
        self, record: ConversationWorktreeRecord
    ) -> ConversationWorktreeBinding:
        if record.state != "ready":
            raise ConversationWorktreeError(
                f"conversation worktree is not ready (state {record.state!r})",
                phase="recovery",
            )
        _, source_common_dir = self._source_repository_identity()
        path, branch = self._expected_identity(record.root_session_id)
        self._validate_record_identity(
            record,
            path=path,
            branch=branch,
            repo_common_dir=source_common_dir,
        )
        if not path.is_dir():
            raise ConversationWorktreeError(
                "ready conversation worktree path is missing", phase="recovery"
            )
        actual_branch = self._git_stdout(path, ["branch", "--show-current"], "recovery")
        actual_common = Path(
            self._git_stdout(
                path,
                ["rev-parse", "--path-format=absolute", "--git-common-dir"],
                "recovery",
            )
        ).resolve()
        if actual_common != source_common_dir:
            raise ConversationWorktreeError(
                "ready conversation worktree failed identity validation",
                phase="recovery",
            )
        if actual_branch != record.branch:
            # A conversation can legitimately rename its branch while preparing
            # or merging a PR. Accept only that narrow drift: the checkout must
            # remain on a named branch and both independent owner claims must
            # still bind the exact root, path, and common repository.
            if not actual_branch or not self._exact_owner_claims_present(record):
                raise ConversationWorktreeError(
                    "ready conversation worktree failed identity validation",
                    phase="recovery",
                )
        ancestor = self._run_git(
            path,
            ["merge-base", "--is-ancestor", record.base_commit, "HEAD"],
            self._policy.create_timeout,
            "recovery",
        )
        if ancestor.returncode != 0:
            # Rebase/squash/reset workflows used while merging PRs can rewrite
            # every descendant without changing checkout ownership. Recover
            # only when this exact worktree's HEAD reflog proves the recorded
            # creation base was previously checked out here, and both durable
            # ownership claims still match. A replacement checkout with no
            # local continuity proof remains rejected.
            if not self._exact_owner_claims_present(
                record
            ) or not self._worktree_reflog_contains_base(record):
                raise ConversationWorktreeError(
                    "ready conversation worktree no longer descends from its base commit",
                    phase="recovery",
                )
        self._ensure_common_owner_claim(record)
        self._ensure_owner_marker(record)
        self._event(
            "conversation_worktree.reuse", root_session_id=record.root_session_id
        )
        binding = self._binding_from_record(record)
        if actual_branch != binding.branch:
            binding = ConversationWorktreeBinding(
                root_session_id=binding.root_session_id,
                path=binding.path,
                branch=actual_branch,
                base_commit=binding.base_commit,
                repo_common_dir=binding.repo_common_dir,
            )
        return binding

    def _exact_owner_claims_present(self, record: ConversationWorktreeRecord) -> bool:
        """Verify both durable ownership claims without repairing either one."""
        path = Path(record.worktree_path)
        try:
            marker_text = self._git_stdout(
                path, ["rev-parse", "--git-path", _OWNER_MARKER], "recovery"
            )
            marker = Path(marker_text)
            if not marker.is_absolute():
                marker = path / marker
            common_claim = _common_owner_claim_path(
                Path(record.repo_common_dir).resolve(), path
            )
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
            common_data = json.loads(common_claim.read_text(encoding="utf-8"))
        except (ConversationWorktreeError, OSError, ValueError, TypeError):
            return False
        return _owner_claim_matches_record(
            marker_data, record=record
        ) and _owner_claim_matches_record(common_data, record=record)

    def _worktree_reflog_contains_base(self, record: ConversationWorktreeRecord) -> bool:
        """Prove rewritten HEAD continuity using this worktree's own reflog."""
        path = Path(record.worktree_path)
        try:
            reflog_text = self._git_stdout(
                path, ["rev-parse", "--git-path", "logs/HEAD"], "recovery"
            )
            reflog_path = Path(reflog_text)
            if not reflog_path.is_absolute():
                reflog_path = path / reflog_path
            lines = reflog_path.read_text(encoding="utf-8").splitlines()
        except (ConversationWorktreeError, OSError):
            return False
        for line in lines:
            fields = line.split(None, 2)
            if record.base_commit in fields[:2]:
                return True
        return False

    def _ensure_owner_marker(self, record: ConversationWorktreeRecord) -> None:
        path = Path(record.worktree_path)
        marker_text = self._git_stdout(
            path, ["rev-parse", "--git-path", _OWNER_MARKER], "validate"
        )
        marker = Path(marker_text)
        if not marker.is_absolute():
            marker = path / marker
        _write_owner_claim(marker, self._owner_payload(record))

    @staticmethod
    def _owner_payload(record: ConversationWorktreeRecord) -> dict[str, str]:
        return {
            "owner": "conversation-worktree-manager",
            "root_session_id": record.root_session_id,
            "worktree_path": str(Path(record.worktree_path).resolve()),
            "repo_common_dir": str(Path(record.repo_common_dir).resolve()),
        }

    def _ensure_common_owner_claim(self, record: ConversationWorktreeRecord) -> None:
        _write_owner_claim(
            _common_owner_claim_path(
                Path(record.repo_common_dir).resolve(), Path(record.worktree_path)
            ),
            self._owner_payload(record),
        )

    @staticmethod
    def _remove_common_owner_claim(record: ConversationWorktreeRecord) -> None:
        """Drop only the exact durable claim after verified explicit removal."""
        claim = _common_owner_claim_path(
            Path(record.repo_common_dir).resolve(), Path(record.worktree_path)
        )
        try:
            claim.unlink(missing_ok=True)
        except OSError:
            # Git removal and the durable ledger transition have already
            # succeeded.  Retaining a stale claim is safe; deleting it later
            # is housekeeping, never a reason to misreport cleanup failure.
            logger.warning("conversation_worktree.common_claim_remove_failed")

    def _run_bootstrap(self, record: ConversationWorktreeRecord) -> None:
        if not self._policy.bootstrap:
            return
        self._event("conversation_worktree.bootstrap", root_session_id=record.root_session_id)
        popen_kwargs = (
            {"creationflags": windows_hide_flags()}
            if IS_WINDOWS
            else {"process_group": 0}
        )
        try:
            process = subprocess.Popen(
                list(self._policy.bootstrap_command),
                cwd=record.worktree_path,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                **popen_kwargs,
            )
        except OSError as exc:
            raise ConversationWorktreeError(
                "bootstrap command could not start", phase="bootstrap"
            ) from exc
        try:
            _stdout, _stderr = process.communicate(timeout=self._policy.bootstrap_timeout)
        except subprocess.TimeoutExpired as exc:
            # A bootstrap can spawn compilers/package managers and descendants.
            # Killing only its direct shell leaves those descendants running in
            # the conversation worktree after the failure was recorded. The
            # compatibility helper owns POSIX group termination and Windows
            # taskkill /T /F cleanup; the bounded drain avoids a hung pipe.
            kill_process_tree(process)
            try:
                process.communicate(timeout=1)
            except Exception:
                pass
            raise ConversationWorktreeError(
                f"bootstrap timed out after {self._policy.bootstrap_timeout:g} seconds",
                phase="bootstrap",
            ) from exc
        except Exception as exc:
            kill_process_tree(process)
            try:
                process.communicate(timeout=1)
            except Exception:
                pass
            raise ConversationWorktreeError(
                "bootstrap process communication failed", phase="bootstrap"
            ) from exc
        if process.returncode != 0:
            raise ConversationWorktreeError(
                f"bootstrap command exited with status {process.returncode}",
                phase="bootstrap",
            )

    def _git_stdout(self, cwd: Path, args: list[str], phase: str) -> str:
        result = self._run_git(cwd, args, self._timeout_for_phase(phase), phase)
        if result.returncode != 0:
            raise ConversationWorktreeError(
                f"git {args[0]} failed", phase=phase
            )
        return result.stdout.strip()

    @staticmethod
    def _run_git(
        cwd: Path, args: list[str], timeout: float, phase: str
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", "-C", str(cwd), *args],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                env=noninteractive_git_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise ConversationWorktreeError("git command timed out", phase=phase) from exc
        except OSError as exc:
            raise ConversationWorktreeError("git command could not start", phase=phase) from exc

    def _timeout_for_phase(self, phase: str) -> float:
        return self._policy.create_timeout if phase in {"create", "validate", "identity", "recovery"} else self._policy.create_timeout

    def _record_failure(
        self,
        root_session_id: str,
        record: ConversationWorktreeRecord | None,
        error: ConversationWorktreeError,
    ) -> None:
        if record is None or record.state not in {"creating", "creation_failed"}:
            return
        try:
            self._db.mark_conversation_worktree_failed(
                root_session_id,
                failure_phase=error.phase,
                failure_message=str(error)[:500],
            )
        except Exception:
            logger.exception("conversation_worktree.failure_record_failed")

    @staticmethod
    def _binding_from_record(record: ConversationWorktreeRecord) -> ConversationWorktreeBinding:
        return ConversationWorktreeBinding(
            root_session_id=record.root_session_id,
            path=Path(record.worktree_path),
            branch=record.branch,
            base_commit=record.base_commit,
            repo_common_dir=Path(record.repo_common_dir),
        )

    @staticmethod
    def _event(event: str, **fields: str) -> None:
        logger.info(event, extra={"conversation_worktree": fields})
