#!/usr/bin/env python3
"""Deterministic smoke checks for the autonomy foundation layer."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from tools.autonomy_guard import (
    command_mutates_filesystem,
    enforce_write_policy,
    evaluate_terminal_command,
    load_autonomy_policy,
    run_bootstrap_preflight,
)


def _init_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True, text=True)
    (repo / "tracked.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-c", "user.name=Smoke", "-c", "user.email=smoke@example.com", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return repo


def main() -> int:
    policy = load_autonomy_policy()
    assert policy["version"] == 1
    assert policy["readiness"]["script"] == "scripts/run_readiness.py"
    assert command_mutates_filesystem("git add tracked.txt") is True

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = _init_repo(Path(tmpdir))
        blocked = enforce_write_policy("write_file", str(repo / "tracked.txt"))
        assert blocked["allowed"] is False

        approval = evaluate_terminal_command("git push origin HEAD", workdir=str(repo))
        assert approval["status"] == "approval_required"

    preflight = run_bootstrap_preflight(
        explicit_api_key="smoke-key",
        explicit_base_url="https://example.com/v1",
        requested_provider="openrouter",
    )
    assert preflight["ok"] is True

    print("autonomy smoke: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"autonomy smoke: FAIL - {exc}", file=sys.stderr)
        raise SystemExit(1)
