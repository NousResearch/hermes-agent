"""Smoke the real Docker-filtered source context without building the full image."""
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {
    "__init__.py", "desktop_identity.py", "git_subprocess.py", "processes.py",
    "process_identity.py", "resource_limits.py", "sqlite_runtime.py",
    "stdio.py", "subprocess_compat.py",
}


def test_docker_context_contains_runtime_source(tmp_path: Path) -> None:
    """BuildKit's scratch export applies the real .dockerignore before COPY."""
    output = tmp_path / "context"
    dockerfile = "FROM scratch\nCOPY runtime/ /runtime/\n"
    result = subprocess.run(
        ["docker", "build", "--file", "-", "--output", f"type=local,dest={output}", "."],
        input=dockerfile, cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    missing = sorted(name for name in REQUIRED if not (output / "runtime" / name).is_file())
    assert not missing, f"Filtered Docker context omitted runtime files: {missing}"
