"""Prepare native Windows build inputs without changing the caller's process."""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def prepare_windows_environment(*, source: Path, state: Path, env: Mapping[str, str]) -> dict[str, str]:
    """The distribution adapter decides whether this target needs ARM64 tools."""
    shell = shutil.which("powershell", path=env.get("PATH")) or shutil.which("pwsh", path=env.get("PATH"))
    if shell is None:
        raise FileNotFoundError("PowerShell is required to prepare Windows ARM64 build dependencies")
    state.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="environment-", dir=state) as scratch:
        output = Path(scratch) / "environment.json"
        subprocess.run(
            [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
             str(source / "scripts/build/windows-deps.ps1"), "-StateRoot", str(state),
             "-EnvironmentFile", str(output)],
            cwd=source, env=dict(env), check=True,
        )
        prepared = json.loads(output.read_text(encoding="utf-8-sig"))
    if not isinstance(prepared, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in prepared.items()):
        raise ValueError("Windows build dependency provider returned an invalid environment")
    return prepared
