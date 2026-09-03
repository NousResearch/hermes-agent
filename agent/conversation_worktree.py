"""Durable, fail-closed Git worktrees for interactive conversation roots."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Iterator

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
        for line in registered.splitlines():
            if not line.startswith("worktree "):
                continue
            registered_path = Path(line.removeprefix("worktree ")).resolve()
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
        if actual_branch != record.branch or actual_common != source_common_dir:
            raise ConversationWorktreeError(
                "ready conversation worktree failed identity validation", phase="recovery"
            )
        ancestor = self._run_git(
            path,
            ["merge-base", "--is-ancestor", record.base_commit, "HEAD"],
            self._policy.create_timeout,
            "recovery",
        )
        if ancestor.returncode != 0:
            raise ConversationWorktreeError(
                "ready conversation worktree no longer descends from its base commit",
                phase="recovery",
            )
        self._event("conversation_worktree.reuse", root_session_id=record.root_session_id)
        return self._binding_from_record(record)

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
