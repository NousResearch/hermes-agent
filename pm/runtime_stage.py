"""One locked dependency builder for PM workers and packaged runtimes."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from pm.package import InstallError


def stage_runtime(uv: Path, python: Path, destination: Path, *,
                  project: Path | None = None, offline: bool = False,
                  wheelhouse: Path | None = None) -> Path:
    """Build at the final path; the caller owns publication and its marker.

    The scratch project prevents uv from discovering the application's workspace.
    No project install, application extra, or application lock enters this graph.
    """
    from pm.packages import uv_cache_dir
    from pm.runtime import runtime_environment

    project = project or Path(__file__).resolve().parent
    destination = destination.absolute()
    executable = destination / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    env = runtime_environment()
    env["UV_CACHE_DIR"] = str(uv_cache_dir())
    env["UV_PROJECT_ENVIRONMENT"] = str(destination)
    env["UV_PYTHON"] = str(python)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pm-project-", dir=destination.parent) as temp:
        snapshot = Path(temp)
        for name in ("pyproject.toml", "uv.lock"):
            shutil.copyfile(project / name, snapshot / name)
        commands = [
            [str(uv), "venv", "--relocatable", "--python", str(python), str(destination)],
            [str(uv), "sync", "--project", str(snapshot), "--locked",
             "--no-default-groups", "--no-install-project", "--python", str(python)],
        ]
        if wheelhouse is not None:
            # Locally rebuilt wheels are verified by the caller's wheelhouse
            # manifest, not the upstream wheel hashes in the PM lock.
            offline = True
            requirements = snapshot / "requirements.txt"
            commands[1:] = [
                [str(uv), "export", "--project", str(snapshot), "--frozen",
                 "--python", str(python), "--no-default-groups", "--no-emit-project",
                 "--no-hashes", "--output-file", str(requirements)],
                [str(uv), "pip", "install", "--python", str(executable),
                 "--no-index", "--only-binary", ":all:",
                 "--find-links", str(wheelhouse.absolute()), "-r", str(requirements)],
                [str(uv), "pip", "check", "--python", str(executable)],
            ]
        for command in commands:
            # Neither the caller's uv.toml nor user-wide settings may choose
            # PM's indexes, required uv version, or environment policy.
            command.append("--no-config")
            if offline:
                command.append("--offline")
            result = subprocess.run(command, cwd=snapshot, env=env, stdout=sys.stderr, stderr=sys.stderr, timeout=600)
            if result.returncode:
                raise InstallError("pm-runtime", f"{command[1]} exited {result.returncode}")
    checked = subprocess.run(
        [str(executable), "-I", "-B", "-c", "import packaging, tomli_w; from ruamel.yaml import YAML"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    if checked.returncode:
        raise InstallError("pm-runtime", f"dependency validation failed: {checked.stderr.strip()}")
    return executable
