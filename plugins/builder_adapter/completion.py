"""Race-resistant completion verification and hook-free adapter-owned commits."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from pathlib import Path
from pathlib import PurePosixPath

from .errors import AdapterError
from .gitops import GitVerifier, _run_git


class CompletionAttestor:
    def __init__(self, git: GitVerifier, validation, schemas, effective_profile):
        self.git = git
        self.validation = validation
        self.schemas = schemas
        self.effective_profile = effective_profile

    @staticmethod
    def _blob(repo: Path, commit: str, path: str) -> str | None:
        result = _run_git(repo, "rev-parse", f"{commit}:{path}", check=False)
        return result.stdout.decode().strip() if result.returncode == 0 else None

    @staticmethod
    def _fingerprint(root: Path, paths: list[str]) -> dict[str, tuple]:
        value = {}
        for raw in paths:
            path = root / raw
            if not path.exists():
                value[raw] = ("missing",)
                continue
            stat = path.lstat()
            if path.is_symlink() or not path.is_file() or stat.st_nlink != 1:
                raise AdapterError("MANIFEST_MISMATCH", f"unsafe file type: {raw}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            value[raw] = (
                stat.st_dev,
                stat.st_ino,
                stat.st_mode,
                stat.st_size,
                stat.st_mtime_ns,
                digest,
            )
        return value

    @staticmethod
    def _safe_blob(root: Path, raw: str) -> tuple[bytes, str]:
        """Read one file through no-follow directory descriptors."""
        parts = PurePosixPath(raw).parts
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for component in parts[:-1]:
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = child
            file_fd = os.open(
                parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            try:
                info = os.fstat(file_fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise AdapterError("MANIFEST_MISMATCH", f"unsafe file type: {raw}")
                chunks = []
                while True:
                    chunk = os.read(file_fd, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                mode = "100755" if info.st_mode & stat.S_IXUSR else "100644"
                return b"".join(chunks), mode
            finally:
                os.close(file_fd)
        except OSError as exc:
            raise AdapterError("MANIFEST_MISMATCH", f"unsafe file path: {raw}") from exc
        finally:
            os.close(descriptor)

    @staticmethod
    def _git_identity_env(index: Path | None = None) -> dict[str, str]:
        env = {
            "GIT_AUTHOR_NAME": "Hermes Builder Adapter",
            "GIT_AUTHOR_EMAIL": "builder-adapter@localhost.invalid",
            "GIT_COMMITTER_NAME": "Hermes Builder Adapter",
            "GIT_COMMITTER_EMAIL": "builder-adapter@localhost.invalid",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
        }
        if index is not None:
            env["GIT_INDEX_FILE"] = str(index)
        return env

    @staticmethod
    def _rollback_ref(root: Path, branch_ref: str, base: str, resulting: str) -> None:
        rollback = _run_git(
            root, "update-ref", branch_ref, base, resulting, check=False
        )
        if rollback.returncode != 0:
            raise AdapterError(
                "DISPATCH_STATE_UNKNOWN", "result ref rollback could not be proven"
            )
        _run_git(root, "read-tree", base)

    def _build_evidence(
        self,
        request,
        snapshot,
        principal: str,
        request_hash: str,
        base: str,
        resulting: str,
        committed: list[str],
        validation: dict,
    ) -> dict:
        changed = []
        for path in committed:
            before_blob = self._blob(Path(request.worktree_path), base, path)
            after_blob = self._blob(Path(request.worktree_path), resulting, path)
            status = "A" if before_blob is None else ("D" if after_blob is None else "M")
            changed.append(
                {
                    "path": path,
                    "status": status,
                    "before_blob": before_blob,
                    "after_blob": after_blob,
                }
            )
        evidence = {
            "schema_version": "1.0.0",
            "dispatch_id": str(request.dispatch_id),
            "cycle_id": request.cycle_id,
            "request_sha256": request_hash,
            "caller_principal": principal,
            "routing": self.effective_profile.evidence(),
            "kanban": {"task_id": snapshot.task_id, "run_ids": snapshot.run_ids},
            "git": {
                "repository_id": request.repository.repository_id,
                "canonical_remote": request.repository.canonical_remote,
                "worktree_path": request.worktree_path,
                "branch": request.branch,
                "starting_sha": base,
                "resulting_sha": resulting,
                "final_dirty_state": "CLEAN",
            },
            "changed_files": changed,
            "validation": validation,
            "terminal_execution": "SUCCEEDED",
            "audit_event_refs": ["pending-audit-correlation"],
            "live_execution_affected": False,
        }
        self.schemas.validate("completion_evidence", evidence)
        return evidence

    def _reconcile_existing(
        self,
        request,
        snapshot,
        principal: str,
        request_hash: str,
        manifest,
        resulting: str,
    ) -> dict:
        root = Path(request.worktree_path)
        base = request.expected_head_sha
        parent = _run_git(root, "rev-parse", f"{resulting}^").stdout.decode().strip()
        commit_object = _run_git(
            root, "cat-file", "commit", resulting
        ).stdout.decode()
        headers, _, message = commit_object.partition("\n\n")
        author = next(
            (line[7:] for line in headers.splitlines() if line.startswith("author ")),
            "",
        )
        committer = next(
            (line[10:] for line in headers.splitlines() if line.startswith("committer ")),
            "",
        )
        identity = [
            author.rsplit(" ", 2)[0],
            committer.rsplit(" ", 2)[0],
        ]
        if (
            parent != base
            or identity
            != [
                "Hermes Builder Adapter <builder-adapter@localhost.invalid>",
                "Hermes Builder Adapter <builder-adapter@localhost.invalid>",
            ]
            or f"dispatch_id={request.dispatch_id}" not in message
            or f"request_sha256={request_hash}" not in message
            or "live_execution_affected=false" not in message
        ):
            raise AdapterError(
                "DISPATCH_STATE_UNKNOWN", "existing result commit is not adapter-owned"
            )
        if not self.git.is_clean(root):
            raise AdapterError("WORKTREE_MISMATCH", "result worktree is not clean")
        committed = self.git.changed_paths(root, base, resulting)
        self.git.verify_paths(root, committed, manifest)
        self.git.verify_file_types(root, committed)
        validation = self.validation.run(
            request.validation_profile,
            root,
            resulting,
            scope_id=str(request.dispatch_id),
        )
        if validation["overall_status"] != "PASSED":
            raise AdapterError("VALIDATION_FAILED", "reconciled validation failed")
        return self._build_evidence(
            request,
            snapshot,
            principal,
            request_hash,
            base,
            resulting,
            committed,
            validation,
        )

    def complete(self, request, snapshot, principal: str, request_hash: str, manifest):
        root = Path(request.worktree_path)
        base = request.expected_head_sha
        branch_ref = f"refs/heads/{request.branch}"
        head = _run_git(root, "rev-parse", "HEAD").stdout.decode().strip()
        branch_head = _run_git(root, "rev-parse", branch_ref).stdout.decode().strip()
        if head != branch_head:
            raise AdapterError("HEAD_MISMATCH", "branch moved before attestation")
        if head != base:
            return self._reconcile_existing(
                request,
                snapshot,
                principal,
                request_hash,
                manifest,
                head,
            )

        tracked = _run_git(root, "diff", "--name-only", "-z").stdout
        untracked = _run_git(
            root, "ls-files", "--others", "--exclude-standard", "-z"
        ).stdout
        paths = sorted(
            {item.decode() for item in (tracked + untracked).split(b"\0") if item}
        )
        if not paths:
            raise AdapterError("VALIDATION_FAILED", "worker produced no changes")
        self.git.verify_paths(root, paths, manifest)
        self.git.verify_file_types(root, paths)
        before = self._fingerprint(root, paths)

        with tempfile.TemporaryDirectory(prefix="hermes-builder-index-") as tmp:
            index = Path(tmp) / "index"
            env = self._git_identity_env(index)
            _run_git(root, "read-tree", base, env=env)
            for path in paths:
                try:
                    data, mode = self._safe_blob(root, path)
                except AdapterError:
                    if (root / path).exists() or (root / path).is_symlink():
                        raise
                    _run_git(root, "update-index", "--remove", "--", path, env=env)
                    continue
                blob = _run_git(
                    root, "hash-object", "-w", "--stdin", "--no-filters", input=data, env=env
                ).stdout.decode().strip()
                _run_git(
                    root,
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    mode,
                    blob,
                    path,
                    env=env,
                )
            tree = _run_git(root, "write-tree", env=env).stdout.decode().strip()
            if before != self._fingerprint(root, paths):
                raise AdapterError(
                    "WORKTREE_RACE", "worktree changed while snapshotting"
                )

            snapshot_commit = _run_git(
                root,
                "commit-tree",
                tree,
                "-p",
                base,
                env=env,
            ).stdout.decode().strip()
            validation_root = Path(tmp) / "validation"
            self.git.materialize_tree(root, snapshot_commit, validation_root)
            validation = self.validation.run(
                request.validation_profile,
                validation_root,
                snapshot_commit,
                materialized_sha=snapshot_commit,
                scope_id=str(request.dispatch_id),
            )
            if validation["overall_status"] != "PASSED":
                raise AdapterError("VALIDATION_FAILED", "registered validation failed")

            if before != self._fingerprint(root, paths):
                raise AdapterError(
                    "WORKTREE_RACE", "worktree changed after validation"
                )
            message = (
                f"feat(builder-adapter): complete {request.cycle_id}\n\n"
                f"dispatch_id={request.dispatch_id}\n"
                f"request_sha256={request_hash}\n"
                "live_execution_affected=false\n"
            )
            result = _run_git(
                root,
                "commit-tree",
                tree,
                "-p",
                base,
                "-m",
                message,
                env=self._git_identity_env(),
            )
            resulting = result.stdout.decode().strip()
            cas = _run_git(
                root,
                "update-ref",
                branch_ref,
                resulting,
                base,
                check=False,
            )
            if cas.returncode != 0:
                raise AdapterError("WORKTREE_RACE", "branch compare-and-swap failed")
            # The branch ref now names the verified tree, but the linked
            # worktree index still describes the old base. Update only the
            # index from the newly created commit; worker bytes remain
            # untouched and are checked again below.
            _run_git(root, "read-tree", resulting)

        if before != self._fingerprint(root, paths):
            self._rollback_ref(root, branch_ref, base, resulting)
            raise AdapterError("WORKTREE_RACE", "worktree changed during commit")
        if not self.git.is_clean(root):
            self._rollback_ref(root, branch_ref, base, resulting)
            raise AdapterError("WORKTREE_MISMATCH", "final worktree is not clean")
        committed = self.git.changed_paths(root, base, resulting)
        self.git.verify_paths(root, committed, manifest)
        if committed != paths:
            raise AdapterError("MANIFEST_MISMATCH", "committed path set changed")

        return self._build_evidence(
            request,
            snapshot,
            principal,
            request_hash,
            base,
            resulting,
            committed,
            validation,
        )
