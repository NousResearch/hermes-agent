#!/usr/bin/env python3
"""Independent baseline-plus-hunk verifier for Context Governor candidates.

The candidate builder may *declare* which changes it needs, but this verifier
does not accept its classification as proof.  It obtains the current patch
from Git again, checks every normalized hunk against the declaration, replays
the required patch onto the declared base, and compares the reconstructed
tree to the sealed staging tree.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any


class BaselineMismatch(RuntimeError):
    pass


class MissingRequiredHunk(RuntimeError):
    pass


class UndeclaredHunk(RuntimeError):
    pass


class PatchReplayMismatch(RuntimeError):
    pass


class CandidateTreeMismatch(RuntimeError):
    pass


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def command(root: Path, *args: str, allowed: tuple[int, ...] = (0,)) -> bytes:
    completed = subprocess.run(
        ["git", *args], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if completed.returncode not in allowed:
        raise PatchReplayMismatch(completed.stderr.decode(errors="replace"))
    return completed.stdout


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def normalized_hunks(patch: bytes) -> dict[str, bytes]:
    """Derive stable IDs from zero-context diff hunks, independently."""
    chunks = patch.split(b"\ndiff --git ")
    output: dict[str, bytes] = {}
    for chunk in chunks:
        if not chunk:
            continue
        value = (
            b"diff --git " if not chunk.startswith(b"diff --git ") else b""
        ) + chunk
        headers = value.split(b"\n@@ ")
        if len(headers) == 1:
            value = value.rstrip(b"\n")
            output[sha(value)] = value
            continue
        prefix = headers[0]
        for hunk in headers[1:]:
            bytes_ = (prefix + b"\n@@ " + hunk).rstrip(b"\n")
            output[sha(bytes_)] = bytes_
    return output


def _untracked_files(root: Path) -> list[Path]:
    names = command(root, "ls-files", "--others", "--exclude-standard", "-z").split(
        b"\0"
    )
    return [root / name.decode() for name in names if name]


def complete_patch(
    root: Path, paths: list[str] | None = None, *, unified: int = 0
) -> bytes:
    """Return tracked and untracked changes without trusting a status parser."""
    path_args = ["--", *paths] if paths else ["--"]
    patch = command(
        root,
        "diff",
        "--binary",
        "--full-index",
        f"--unified={unified}",
        "HEAD",
        *path_args,
    )
    wanted = {Path(path) for path in paths} if paths else None
    for file in _untracked_files(root):
        rel = file.relative_to(root)
        if wanted is not None and not any(
            rel == selected or selected in rel.parents for selected in wanted
        ):
            continue
        # Older Git releases lack --label for diff --no-index. Normalize its
        # absolute destination back to a repository-relative b/ path before
        # replaying it.
        piece = command(
            root,
            "diff",
            "--no-index",
            "--binary",
            "--full-index",
            f"--unified={unified}",
            "/dev/null",
            str(file),
            allowed=(0, 1),
        )
        absolute = str(file).encode()
        relative = rel.as_posix().encode()
        piece = piece.replace(b"a" + absolute, b"a/" + relative)
        piece = piece.replace(b"b" + absolute, b"b/" + relative)
        patch += piece
    prefix = command(root, "rev-parse", "--show-prefix").decode().strip().rstrip("/")
    if prefix:
        token = prefix.encode() + b"/"
        patch = patch.replace(b"a/" + token, b"a/").replace(b"b/" + token, b"b/")
    return patch


def tree_entries(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "mode": path.stat().st_mode & 0o777,
            "size": path.stat().st_size,
            "sha256": sha(path.read_bytes()),
        }
        for path in sorted(
            item for item in root.rglob("*") if item.is_file() and not item.is_symlink()
        )
    ]


def tree_digest(root: Path) -> str:
    return sha(canonical(tree_entries(root)))


def selected_tree_entries(root: Path, paths: list[str]) -> list[dict[str, object]]:
    wanted = [Path(path) for path in paths]
    return [
        entry
        for entry in tree_entries(root)
        if any(
            Path(str(entry["path"])) == item or item in Path(str(entry["path"])).parents
            for item in wanted
        )
    ]


def selected_tree_digest(root: Path, paths: list[str] | None) -> str:
    return (
        sha(canonical(selected_tree_entries(root, paths)))
        if paths is not None
        else tree_digest(root)
    )


def declared_tree_digest(candidate_root: Path, declared: list[dict[str, Any]]) -> str:
    """Digest just the reconstructed source trees, excluding generated ledger files."""
    return sha(
        canonical([
            {
                "subtree": entry["candidate_subtree"],
                "entries": tree_entries(candidate_root / entry["candidate_subtree"]),
            }
            for entry in declared
        ])
    )


def _base_blob_ids(root: Path, baseline: str) -> dict[str, str]:
    values = command(root, "ls-tree", "-r", "-z", baseline).split(b"\0")
    result: dict[str, str] = {}
    for value in values:
        if not value:
            continue
        left, path = value.split(b"\t", 1)
        _mode, _kind, blob = left.split()
        result[path.decode()] = blob.decode()
    return result


def _extract_baseline(root: Path, baseline: str, destination: Path) -> None:
    raw = command(root, "archive", "--format=tar", baseline)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(destination, filter="data")


def _replay(root: Path, baseline: str, required_paths: list[str]) -> Path:
    replay_root = Path(tempfile.mkdtemp(prefix="ares-cg-scope-replay-"))
    _extract_baseline(root, baseline, replay_root)
    patch = complete_patch(root, required_paths, unified=3)
    if patch:
        result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=replay_root,
            input=patch,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode:
            shutil.rmtree(replay_root, ignore_errors=True)
            raise PatchReplayMismatch(result.stderr.decode(errors="replace"))
    return replay_root


def verify(manifest: dict[str, Any], candidate_root: Path) -> dict[str, Any]:
    declared = manifest.get("repositories")
    if not isinstance(declared, list) or not declared:
        raise BaselineMismatch("missing repositories")
    verified: list[dict[str, object]] = []
    replay_roots: list[Path] = []
    try:
        for entry in declared:
            root = Path(entry["root"])
            baseline = entry["baseline_commit"]
            if command(root, "rev-parse", baseline).decode().strip() != baseline:
                raise BaselineMismatch(entry["name"])
            if _base_blob_ids(root, baseline) != entry.get("base_blob_ids"):
                raise BaselineMismatch(f"base blob identity: {entry['name']}")
            actual = normalized_hunks(complete_patch(root))
            required = set(entry.get("required_hunk_ids", []))
            unrelated = set(entry.get("unrelated_hunk_ids", []))
            if not required <= actual.keys():
                raise MissingRequiredHunk(sorted(required - actual.keys())[0])
            undeclared = actual.keys() - required - unrelated
            if undeclared:
                raise UndeclaredHunk(sorted(undeclared)[0])
            if (required | unrelated) - actual.keys():
                raise MissingRequiredHunk("declared hunk no longer exists")
            required_paths = entry.get("required_paths")
            if not isinstance(required_paths, list):
                raise PatchReplayMismatch("missing required paths")
            replay = _replay(root, baseline, required_paths)
            replay_roots.append(replay)
            expected = candidate_root / entry["candidate_subtree"]
            candidate_paths = entry.get("candidate_paths")
            if candidate_paths is not None and not isinstance(candidate_paths, list):
                raise PatchReplayMismatch("invalid candidate paths")
            if selected_tree_digest(replay, candidate_paths) != tree_digest(expected):
                raise CandidateTreeMismatch(entry["name"])
            verified.append({
                "name": entry["name"],
                "required_hunk_ids": sorted(required),
                "unrelated_hunk_ids": sorted(unrelated),
                "reconstructed_tree_sha256": selected_tree_digest(
                    replay, candidate_paths
                ),
            })
        candidate_tree = (
            declared_tree_digest(candidate_root, declared)
            if manifest.get("candidate_tree_scope") == "declared_subtrees_v1"
            else tree_digest(candidate_root)
        )
        if candidate_tree != manifest.get("candidate_tree_sha256"):
            raise CandidateTreeMismatch(candidate_tree)
        return {
            "schema": "AresContextGovernorScopeProofV2",
            "pass": True,
            "repositories": verified,
            "candidate_tree_sha256": candidate_tree,
        }
    finally:
        for root in replay_roots:
            shutil.rmtree(root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(json.loads(args.manifest.read_text(encoding="utf-8")), args.candidate_root),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
