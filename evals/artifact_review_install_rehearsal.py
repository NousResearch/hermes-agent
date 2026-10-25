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
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Sequence


SCHEMA = "hermes.isolated-install-rollback-rehearsal/v1"


def _run(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stdin=subprocess.DEVNULL,
    ).strip()


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
    normalized = sorted(dict.fromkeys(str(item).strip() for item in components if str(item).strip()))
    if not normalized:
        raise ValueError("at least one reviewed component path is required")

    with tempfile.TemporaryDirectory(prefix="hermes-install-rehearsal-") as raw:
        root = Path(raw)
        archive = root / "candidate.tar"
        subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", "--output", str(archive), commit],
            check=True, stdin=subprocess.DEVNULL,
        )
        staged = root / "staged"
        staged.mkdir()
        with tarfile.open(archive, "r") as bundle:
            bundle.extractall(staged, filter="data")

        files = []
        for relative in normalized:
            source = repo / relative
            installed = staged / relative
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--component", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = rehearse(args.repo, args.expected_commit, args.component)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(args.output), "receipt_sha256": receipt["receipt_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

