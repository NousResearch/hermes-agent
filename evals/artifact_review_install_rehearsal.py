#!/usr/bin/env python3
"""Offline install/rollback rehearsal for an exact reviewed Hermes revision.

The rehearsal never writes to a Hermes home or live checkout.  It extracts the
Git object into a temporary target, verifies selected component bytes, swaps it
over a sentinel install, rolls back, and emits one deterministic receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
from typing import Sequence


SCHEMA = "hermes.isolated-install-rollback-rehearsal/v1"
_GIT = shutil.which("git", path=os.defpath)


def _git_command(repo: Path, *args: str) -> list[str]:
    if not _GIT:
        raise RuntimeError("git is unavailable on the system command path")
    return [
        _GIT, "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
        "-c", "core.untrackedCache=false", "-C", str(repo), *args,
    ]


def _git_env() -> dict[str, str]:
    # Do not inherit credential helpers, alternate object stores, replacement
    # refs, process hooks, or caller-selected Git configuration.
    return {
        "PATH": os.defpath,
        "HOME": "/nonexistent-hermes-rehearsal-home",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _run(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        _git_command(repo, *args), text=True, stdin=subprocess.DEVNULL, env=_git_env(),
    ).strip()


def _component_path(repo: Path, raw: str) -> tuple[str, Path]:
    text = str(raw).strip()
    posix = PurePosixPath(text)
    if (
        not text or posix.is_absolute() or text in {".", ".."}
        or any(part in {"", ".", ".."} for part in posix.parts)
        or "\\" in text
    ):
        raise ValueError(f"unsafe reviewed component path: {raw!r}")
    relative = posix.as_posix()
    candidate = (repo / relative).resolve(strict=True)
    if not candidate.is_relative_to(repo):
        raise ValueError(f"reviewed component escapes the repository: {raw!r}")
    return relative, candidate


def _tracked_regular_blob(repo: Path, relative: str) -> None:
    raw = subprocess.check_output(
        _git_command(repo, "ls-tree", "-z", "HEAD", "--", relative),
        stdin=subprocess.DEVNULL, env=_git_env(),
    )
    records = [item for item in raw.split(b"\0") if item]
    expected_suffix = b"\t" + relative.encode("utf-8")
    if len(records) != 1 or not records[0].endswith(expected_suffix):
        raise ValueError(f"reviewed component is not one exact tracked Git entry: {relative}")
    header = records[0].split(b"\t", 1)[0].split()
    if len(header) != 3 or header[0] not in {b"100644", b"100755"} or header[1] != b"blob":
        raise ValueError(f"reviewed component is not a tracked regular Git file: {relative}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rehearse(repo: Path, expected_commit: str, components: Sequence[str]) -> dict:
    repo = repo.resolve(strict=True)
    commit = _run(repo, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise ValueError(f"checkout is {commit}, expected {expected_commit}")
    if _run(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("checkout is dirty; rehearse an immutable reviewed revision")
    tree = _run(repo, "rev-parse", "HEAD^{tree}")
    if not components:
        raise ValueError("at least one reviewed component path is required")
    component_paths = dict(_component_path(repo, item) for item in components)
    normalized = sorted(component_paths)
    for relative in normalized:
        _tracked_regular_blob(repo, relative)

    with tempfile.TemporaryDirectory(prefix="hermes-install-rehearsal-") as raw:
        root = Path(raw)
        archive = root / "candidate.tar"
        subprocess.run(
            _git_command(repo, "archive", "--format=tar", "--output", str(archive), commit),
            check=True, stdin=subprocess.DEVNULL, env=_git_env(),
        )
        staged = root / "staged"
        staged.mkdir()
        with tarfile.open(archive, "r") as bundle:
            bundle.extractall(staged, filter="data")
        staged_root = staged.resolve(strict=True)

        files = []
        for relative in normalized:
            source = component_paths[relative]
            installed = (staged / relative).resolve(strict=True)
            if not installed.is_relative_to(staged_root):
                raise ValueError(f"archived component escapes the temporary install: {relative}")
            if source.is_symlink() or not source.is_file() or installed.is_symlink() or not installed.is_file():
                raise ValueError(f"reviewed component is not a regular tracked file: {relative}")
            source_sha = _file_sha256(source)
            if _file_sha256(installed) != source_sha:
                raise ValueError(f"archive bytes differ from reviewed checkout: {relative}")
            files.append({"path": relative, "sha256": source_sha, "size": source.stat().st_size})

        target = root / "install-target"
        target.mkdir()
        active = target / "active"
        rollback = target / "rollback"
        active.mkdir()
        sentinel = active / "prior-install.json"
        sentinel.write_text('{"revision":"prior"}\n', encoding="utf-8")
        prior_sha = _file_sha256(sentinel)
        active.rename(rollback)
        staged.rename(active)
        for item in files:
            if _file_sha256(active / item["path"]) != item["sha256"]:
                raise ValueError(f"installed component verification failed: {item['path']}")
        shutil.rmtree(active)
        rollback.rename(active)
        if _file_sha256(active / "prior-install.json") != prior_sha:
            raise ValueError("rollback did not restore the prior install bytes")

    receipt = {
        "schema": SCHEMA,
        "source_commit": commit,
        "source_tree": tree,
        "components": files,
        "checks": {
            "exact_commit": True,
            "clean_checkout": True,
            "caller_git_environment_scrubbed": True,
            "system_and_global_git_config_disabled": True,
            "hooks_fsmonitor_replace_objects_and_optional_locks_disabled": True,
            "git_archive_bytes_match": True,
            "isolated_swap_succeeded": True,
            "rollback_restored_prior_bytes": True,
            "live_install_touched": False,
            "credentials_used": False,
            "worker_or_model_launched": False,
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return receipt


def _write_receipt_exclusive(output: Path, receipt: dict) -> None:
    """Create one new receipt without following or replacing an existing path."""
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("receipt output parent is not a directory")
    target = parent / output.name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(target, flags, 0o600)
    try:
        data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--component", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = rehearse(args.repo, args.expected_commit, args.component)
    _write_receipt_exclusive(args.output, receipt)
    print(json.dumps({"receipt": str(args.output), "receipt_sha256": receipt["receipt_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

