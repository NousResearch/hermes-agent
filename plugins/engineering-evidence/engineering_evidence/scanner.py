"""Read-only exact-HEAD codebase snapshot generation."""

from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path
from typing import Any

from .receipts import validate_receipt


_MAX_FILE_BYTES = 512 * 1024
_MAX_RELEVANT_FILES = 200
_MAX_DIAGNOSTIC_SAMPLES = 48


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _repository_name(repo: Path) -> str:
    try:
        remote = _git(repo, "config", "--get", "remote.origin.url").strip()
    except subprocess.CalledProcessError:
        remote = ""
    if remote:
        remote = remote.removesuffix(".git")
        return remote.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    return repo.name


def _python_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8")[:_MAX_FILE_BYTES], filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return sorted(imports)


def build_codebase_snapshot(repo: Path, *, query: str = "") -> dict[str, Any]:
    """Inspect tracked files without changing the repository or index."""

    repo = repo.expanduser().resolve()
    if not repo.is_dir():
        raise ValueError(f"repository is not a directory: {repo}")
    head_sha = _git(repo, "rev-parse", "HEAD").strip()
    tracked_files = sorted(item for item in _git(repo, "ls-files", "-z").split("\0") if item)
    normalized_query = query.strip().lower()
    relevant: list[str] = []
    import_map: dict[str, list[str]] = {}
    unreadable_count = 0
    unreadable_samples: list[str] = []
    for relative in tracked_files:
        path = repo / relative
        if normalized_query and normalized_query not in relative.lower():
            try:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    continue
                if normalized_query not in path.read_text(encoding="utf-8", errors="replace").lower():
                    continue
            except OSError:
                unreadable_count += 1
                if len(unreadable_samples) < _MAX_DIAGNOSTIC_SAMPLES:
                    unreadable_samples.append(relative)
                continue
        relevant.append(relative)
        if len(relevant) >= _MAX_RELEVANT_FILES:
            break
        if path.suffix == ".py":
            import_map[relative] = _python_imports(path)
    digest = hashlib.sha256("\0".join(tracked_files).encode()).hexdigest()
    incomplete = ["relevant_file_cap_reached"] if len(relevant) >= _MAX_RELEVANT_FILES else []
    if unreadable_count:
        incomplete.append(f"unreadable_file_count:{unreadable_count}")
        incomplete.extend(f"unreadable:{item}" for item in unreadable_samples)
    return validate_receipt(
        {
            "schema_name": "engineering_evidence_v1",
            "receipt_type": "codebase_snapshot",
            "receipt_id": f"snapshot-{head_sha[:12]}-{digest[:12]}",
            "repository": _repository_name(repo),
            "repository_path": str(repo),
            "head_sha": head_sha,
            "authority": "diagnostic-only",
            "query": query,
            "tracked_files": tracked_files,
            "relevant_files": sorted(relevant),
            "python_imports": import_map,
            "incomplete_reasons": incomplete,
        }
    )
