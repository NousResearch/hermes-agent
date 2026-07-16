"""Lint fork-features.json path and test coverage entries.

GitHub Actions wiring snippet for the follow-up CI PR:

```yaml
name: fork manifest lint
on:
  pull_request:
    paths:
      - docs/sync/fork-features.json
      - '**/*.py'
jobs:
  lint-manifest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python3.11 -m scripts.hermes_parity lint-manifest
```
"""

from __future__ import annotations

import glob
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import forkdelta, gitops, state


@dataclass(frozen=True)
class ManifestLintResult:
    ok: bool
    errors: tuple[str, ...]


def _matches(repo: Path, pattern: str) -> list[str]:
    return sorted(
        str(path.relative_to(repo))
        for path in repo.glob(pattern)
        if path.exists()
    ) or sorted(glob.glob(pattern, root_dir=repo, recursive=True))


def lint_paths(repo: Path, manifest_path: Path) -> list[str]:
    errors: list[str] = []
    for feature in forkdelta.load_manifest(manifest_path):
        for pattern in feature.paths:
            if not _matches(repo, pattern):
                errors.append(f"path matches zero files: {pattern}")
    return errors


def lint_nodeids(repo: Path, nodeids: Sequence[str]) -> list[str]:
    if not nodeids:
        return []
    python = repo / ".venv" / "bin" / "python"
    if not python.exists():
        python = repo / "venv" / "bin" / "python"
    if not python.exists():
        python = Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "python"
    # Last-resort fallback to the running interpreter — the fleet dev venvs
    # above are absent on a CI shard's clean checkout, and a nonexistent path
    # makes subprocess.run raise FileNotFoundError (the test-isolation red).
    python_exe = str(python) if python.exists() else sys.executable

    def collects(batch: Sequence[str]) -> bool:
        proc = subprocess.run(
            [python_exe, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=", "-p", "no:randomly", *batch],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return proc.returncode == 0 and bool(proc.stdout.strip())

    # Fast path: one invocation when everything collects. On failure, retry
    # per-nodeid so the error names ONLY the offender(s), not the whole set
    # (a single rotted nodeid must not smear every healthy one).
    if collects(nodeids):
        return []
    return [
        f"test nodeid does not collect: {node}"
        for node in nodeids
        if not collects([node])
    ]


def vacuous_coverage_errors(
    repo: Path,
    *,
    report: forkdelta.ForkDeltaReport,
    manifest_path: Path,
    touched_paths: set[str],
    vacuous_ok: bool = False,
) -> list[str]:
    if vacuous_ok or state.has_vacuous_ack(repo):
        return []
    if not forkdelta.load_manifest(manifest_path):
        return []
    if report.covered_paths:
        return []
    if not touched_paths:
        return []
    fork_delta_touched = set(report.changed_paths) & touched_paths
    if fork_delta_touched:
        return ["vacuous forkdelta coverage: 0 covered paths while fork-delta files were touched"]
    return []


def lint_manifest(
    repo: Path,
    *,
    manifest_path: Path | None = None,
    base: str | None = None,
    fork_ref: str = "fork/main",
    touched_paths: set[str] | None = None,
    vacuous_ok: bool = False,
) -> ManifestLintResult:
    manifest = repo / (manifest_path or forkdelta.DEFAULT_MANIFEST)
    if not manifest.exists():
        return ManifestLintResult(False, (f"manifest missing: {manifest.relative_to(repo)}",))
    errors = lint_paths(repo, manifest)
    nodeids = forkdelta.manifest_nodeids(manifest)
    errors.extend(lint_nodeids(repo, nodeids))
    if base is not None and touched_paths is not None:
        report = forkdelta.compute_fork_delta(repo, base=base, fork_ref=fork_ref, touched_paths=touched_paths)
        errors.extend(
            vacuous_coverage_errors(
                repo,
                report=report,
                manifest_path=manifest,
                touched_paths=touched_paths,
                vacuous_ok=vacuous_ok,
            )
        )
    return ManifestLintResult(not errors, tuple(errors))
