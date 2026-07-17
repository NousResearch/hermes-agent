"""Behavioral bootstrap coverage for the standalone A2A entry point."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_standalone_entry_hardens_import_path_before_gateway_imports(tmp_path):
    shadow = tmp_path / "utils"
    shadow.mkdir()
    (shadow / "__init__.py").write_text(
        'raise RuntimeError("project-local utils was imported")\n',
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(repo_root),
        env.get("PYTHONPATH", ""),
    ]).rstrip(os.pathsep)

    result = subprocess.run(
        [sys.executable, "-m", "plugins.platforms.a2a", "--version"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
