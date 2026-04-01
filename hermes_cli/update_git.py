from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional


def _git_stdout(repo_dir: Path, args: List[str], *, timeout: Optional[int] = None) -> tuple[int, str]:
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return result.returncode, (result.stdout or "").strip()


def resolve_update_remote(repo_dir: Path) -> str:
    """Resolve which git remote Hermes should use for update checks.

    Prefer the configured tracking remote for ``main`` when available. This
    avoids false "Already up to date" results in contributor checkouts where
    ``origin`` points to a personal fork and ``main`` tracks ``upstream/main``.

    Fallback order:
    1. ``branch.main.remote`` when it exists and the remote is configured
    2. ``origin``
    3. ``upstream``
    4. first configured remote
    5. literal ``origin`` as a final default
    """

    rc, configured_remote = _git_stdout(repo_dir, ["config", "--get", "branch.main.remote"], timeout=5)
    rc_remotes, remotes_out = _git_stdout(repo_dir, ["remote"], timeout=5)
    remotes = [line.strip() for line in remotes_out.splitlines() if line.strip()] if rc_remotes == 0 else []

    if rc == 0 and configured_remote and configured_remote in remotes:
        return configured_remote
    if "origin" in remotes:
        return "origin"
    if "upstream" in remotes:
        return "upstream"
    if remotes:
        return remotes[0]
    return "origin"
