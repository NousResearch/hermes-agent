"""Argument-safe Git and governed-path verification."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath

from .errors import AdapterError


def _run_git(
    repo: Path,
    *args: str,
    check: bool = True,
    env: dict[str, str] | None = None,
    input: bytes | None = None,
) -> subprocess.CompletedProcess:
    safe_env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_COUNT": "12",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": "/dev/null",
        "GIT_CONFIG_KEY_1": "commit.gpgSign",
        "GIT_CONFIG_VALUE_1": "false",
        "GIT_CONFIG_KEY_2": "tag.gpgSign",
        "GIT_CONFIG_VALUE_2": "false",
        "GIT_CONFIG_KEY_3": "credential.helper",
        "GIT_CONFIG_VALUE_3": "",
        "GIT_CONFIG_KEY_4": "diff.external",
        "GIT_CONFIG_VALUE_4": "",
        "GIT_CONFIG_KEY_5": "core.attributesFile",
        "GIT_CONFIG_VALUE_5": "/dev/null",
        "GIT_CONFIG_KEY_6": "protocol.file.allow",
        "GIT_CONFIG_VALUE_6": "never",
        "GIT_CONFIG_KEY_7": "interactive.diffFilter",
        "GIT_CONFIG_VALUE_7": "",
        "GIT_CONFIG_KEY_8": "core.fsmonitor",
        "GIT_CONFIG_VALUE_8": "false",
        "GIT_CONFIG_KEY_9": "gpg.program",
        "GIT_CONFIG_VALUE_9": "/bin/false",
        "GIT_CONFIG_KEY_10": "gpg.ssh.program",
        "GIT_CONFIG_VALUE_10": "/bin/false",
        "GIT_CONFIG_KEY_11": "core.sshCommand",
        "GIT_CONFIG_VALUE_11": "/bin/false",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/false",
        "SSH_ASKPASS": "/bin/false",
    }
    if env:
        safe_env.update(
            {
                key: value
                for key, value in env.items()
                if not key.startswith("GIT_")
                or key
                in {
                    "GIT_INDEX_FILE",
                    "GIT_AUTHOR_NAME",
                    "GIT_AUTHOR_EMAIL",
                    "GIT_COMMITTER_NAME",
                    "GIT_COMMITTER_EMAIL",
                    "GIT_AUTHOR_DATE",
                    "GIT_COMMITTER_DATE",
                }
            }
        )
    return subprocess.run(
        ["/usr/bin/git", "--no-pager", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=False,
        timeout=30,
        env=safe_env,
        input=input,
    )


def safe_relative_path(raw: str) -> str:
    if not raw or raw.startswith(("/", "\\")) or "\\" in raw or "\x00" in raw:
        raise AdapterError("MANIFEST_MISMATCH", "unsafe governed path")
    path = PurePosixPath(raw)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise AdapterError("MANIFEST_MISMATCH", "unsafe governed path")
    return path.as_posix()


class AllowedPathManifest:
    def __init__(self, value: dict):
        if value.get("default_access") != "forbidden":
            raise AdapterError("MANIFEST_MISMATCH", "manifest must deny by default")
        if value.get("symlinks") != "reject" or value.get("submodules") != "reject":
            raise AdapterError(
                "MANIFEST_MISMATCH", "manifest must reject symlinks and submodules"
            )
        self.base_sha = value["base_sha"]
        read_policy = value.get("read_policy")
        if (
            not isinstance(read_policy, dict)
            or read_policy.get("source") != "git_tracked_regular_files"
            or read_policy.get("snapshot") != "base_sha"
            or not isinstance(read_policy.get("deny_patterns"), list)
            or not read_policy["deny_patterns"]
        ):
            raise AdapterError("MANIFEST_MISMATCH", "tracked read policy is required")
        self.read_denied_patterns = tuple(read_policy["deny_patterns"])
        self.patterns = [
            rule["pattern"]
            for rule in value["rules"]
            if rule.get("access") == "read_write"
        ]

    def permits(self, raw: str) -> bool:
        path = safe_relative_path(raw)
        return any(fnmatch.fnmatchcase(path, pattern) for pattern in self.patterns)

    def permits_read(self, raw: str) -> bool:
        path = safe_relative_path(raw)
        return not any(
            fnmatch.fnmatchcase(path, pattern)
            for pattern in self.read_denied_patterns
        )


class GitVerifier:
    def __init__(self, repository_allowlist: dict[str, str]):
        self._allowlist = repository_allowlist

    def artifact_bytes(self, repo: Path, commit: str, path: str) -> bytes:
        safe_relative_path(path)
        try:
            return _run_git(repo, "cat-file", "blob", f"{commit}:{path}").stdout
        except subprocess.CalledProcessError as exc:
            raise AdapterError(
                "CONTRACT_MISMATCH", "governed artifact is unavailable"
            ) from exc

    def verify_artifact(self, repo: Path, ref, code: str) -> bytes:
        raw = self.artifact_bytes(repo, ref.commit, ref.path)
        if hashlib.sha256(raw).hexdigest() != ref.sha256:
            raise AdapterError(code, "governed artifact hash mismatch")
        return raw

    def verify_worktree(self, request, *, require_clean: bool = True) -> Path:
        expected_remote = self._allowlist.get(request.repository.repository_id)
        if expected_remote != request.repository.canonical_remote:
            raise AdapterError("REPOSITORY_MISMATCH", "repository is not allowlisted")
        root = Path(request.worktree_path)
        try:
            real = root.resolve(strict=True)
        except OSError as exc:
            raise AdapterError("WORKTREE_MISMATCH", "worktree does not exist") from exc
        if real != root or root.is_symlink():
            raise AdapterError("WORKTREE_MISMATCH", "worktree path is not canonical")
        try:
            inside = _run_git(root, "rev-parse", "--is-inside-work-tree").stdout.strip()
            common_raw = _run_git(root, "rev-parse", "--git-common-dir").stdout
            git_dir_raw = _run_git(root, "rev-parse", "--git-dir").stdout
        except subprocess.CalledProcessError as exc:
            raise AdapterError("WORKTREE_MISMATCH", "path is not a linked worktree") from exc
        common = (root / common_raw.decode().strip()).resolve()
        git_dir = (root / git_dir_raw.decode().strip()).resolve()
        dot_git = root / ".git"
        if (
            inside != b"true"
            or not common_raw.strip()
            or not dot_git.is_file()
            or git_dir == common
            or git_dir.parent.name != "worktrees"
            or git_dir.parent.parent != common
        ):
            raise AdapterError("WORKTREE_MISMATCH", "path is not a linked worktree")
        remote = _run_git(
            root, "config", "--local", "--get", "remote.origin.url"
        ).stdout.decode().strip()
        if remote != expected_remote:
            raise AdapterError("REPOSITORY_MISMATCH", "canonical remote mismatch")
        branch_ref = _run_git(root, "symbolic-ref", "-q", "HEAD").stdout.decode().strip()
        branch = branch_ref.removeprefix("refs/heads/")
        if branch != request.branch:
            raise AdapterError("BRANCH_MISMATCH", "worktree branch mismatch")
        head = _run_git(root, "rev-parse", "HEAD").stdout.decode().strip()
        if head != request.expected_head_sha:
            raise AdapterError("HEAD_MISMATCH", "worktree HEAD mismatch")
        if require_clean and not self.is_clean(root):
            raise AdapterError("WORKTREE_MISMATCH", "worktree is not clean")
        self.verify_repository_types(root)
        return root

    @staticmethod
    def is_clean(root: Path) -> bool:
        _run_git(root, "update-index", "-q", "--refresh", check=False)
        tracked = _run_git(root, "diff-index", "--quiet", "HEAD", "--", check=False)
        untracked = _run_git(
            root, "ls-files", "--others", "--exclude-standard", "-z"
        ).stdout
        return tracked.returncode == 0 and not untracked

    def verify_repository_types(self, worktree: Path) -> None:
        entries = _run_git(worktree, "ls-files", "--stage", "-z").stdout
        for entry in entries.split(b"\0"):
            if not entry:
                continue
            mode = entry.split(b" ", 1)[0]
            if mode == b"120000":
                raise AdapterError("MANIFEST_MISMATCH", "tracked symlinks are forbidden")
            if mode == b"160000":
                raise AdapterError("MANIFEST_MISMATCH", "submodules are forbidden")

    def tracked_readable_paths(
        self,
        worktree: Path,
        snapshot: str,
        manifest: AllowedPathManifest,
    ) -> frozenset[str]:
        """Resolve readable regular blobs from the immutable Git snapshot."""
        if snapshot != manifest.base_sha:
            raise AdapterError("MANIFEST_MISMATCH", "read snapshot mismatch")
        entries = _run_git(
            worktree, "ls-tree", "-rz", "--full-tree", "-r", snapshot
        ).stdout
        readable: set[str] = set()
        for entry in entries.split(b"\0"):
            if not entry:
                continue
            metadata, raw_path = entry.split(b"\t", 1)
            mode, kind, _object_id = metadata.split(b" ", 2)
            if kind != b"blob" or mode not in {b"100644", b"100755"}:
                raise AdapterError(
                    "MANIFEST_MISMATCH", "unsafe object in readable snapshot"
                )
            try:
                path = safe_relative_path(raw_path.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise AdapterError(
                    "MANIFEST_MISMATCH", "non-UTF-8 tracked path is forbidden"
                ) from exc
            if manifest.permits_read(path):
                readable.add(path)
        return frozenset(readable)

    def changed_paths(self, worktree: Path, base_sha: str, head: str = "HEAD") -> list[str]:
        output = _run_git(
            worktree,
            "diff",
            "--name-only",
            "-z",
            "--diff-filter=ACDMRTUXB",
            base_sha,
            head,
        ).stdout
        paths = [item.decode("utf-8") for item in output.split(b"\0") if item]
        return [safe_relative_path(path) for path in paths]

    def verify_paths(
        self, worktree: Path, paths: list[str], manifest: AllowedPathManifest
    ) -> None:
        for raw in paths:
            path = safe_relative_path(raw)
            if not manifest.permits(path):
                raise AdapterError("MANIFEST_MISMATCH", f"changed path forbidden: {path}")
            current = worktree
            for component in PurePosixPath(path).parts:
                current = current / component
                if current.is_symlink():
                    raise AdapterError("MANIFEST_MISMATCH", f"symlink forbidden: {path}")
        submodules = _run_git(
            worktree, "ls-files", "--stage", check=False
        ).stdout.decode().splitlines()
        if any(line.startswith("160000 ") for line in submodules):
            raise AdapterError("MANIFEST_MISMATCH", "submodules are forbidden")

    def verify_file_types(
        self, worktree: Path, paths: list[str], *, allow_missing: bool = True
    ) -> None:
        for raw in paths:
            path = worktree / safe_relative_path(raw)
            if not path.exists():
                if allow_missing:
                    continue
                raise AdapterError("MANIFEST_MISMATCH", f"missing path: {raw}")
            stat = path.lstat()
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise AdapterError("MANIFEST_MISMATCH", f"unsafe file type: {raw}")
            if path.is_file() and stat.st_nlink != 1:
                raise AdapterError("MANIFEST_MISMATCH", f"hard link forbidden: {raw}")

    def manifest_from_artifact(self, raw: bytes) -> AllowedPathManifest:
        return AllowedPathManifest(json.loads(raw))

    def materialize_tree(self, repo: Path, commit: str, destination: Path) -> None:
        """Materialize blobs with plumbing only; never invoke checkout machinery."""
        destination.mkdir(mode=0o700)
        entries = _run_git(
            repo, "ls-tree", "-rz", "--full-tree", "-r", commit
        ).stdout
        for entry in entries.split(b"\0"):
            if not entry:
                continue
            metadata, raw_path = entry.split(b"\t", 1)
            mode, kind, object_id = metadata.split(b" ", 2)
            if kind != b"blob" or mode not in {b"100644", b"100755"}:
                raise AdapterError(
                    "MANIFEST_MISMATCH", "unsafe object in validation tree"
                )
            path = safe_relative_path(raw_path.decode("utf-8"))
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            data = _run_git(repo, "cat-file", "blob", object_id.decode()).stdout
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o700 if mode == b"100755" else 0o600,
            )
            try:
                os.write(descriptor, data)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
