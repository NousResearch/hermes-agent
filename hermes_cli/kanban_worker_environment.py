"""Bind worker commands to the control plane and assigned project separately."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def validate_profile_config(profile_home: str) -> None:
    import yaml

    config = Path(profile_home) / "config.yaml"
    if not config.is_file():
        return
    try:
        parsed = yaml.safe_load(config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"invalid profile config: {config}: {error}") from error
    if parsed is not None and not isinstance(parsed, dict):
        raise ValueError(f"invalid profile config: {config} must contain a mapping")


def bind_worker_environment(env: dict[str, str], workspace: str) -> None:
    from hermes_constants import get_default_hermes_root

    # These tokens are emitted by protected Kanban payloads. They must exist
    # in the terminal environment, not just in the model's sanitized text.
    env["HERMES_CONTROL_HOME"] = str(get_default_hermes_root())
    env["HERMES_KANBAN_HERMES_PYTHON"] = os.path.abspath(sys.executable)
    env.pop("HERMES_KANBAN_WORKTREE_PYTHON", None)
    env.pop("VIRTUAL_ENV", None)
    if not workspace or not os.path.isabs(workspace) or not os.path.isdir(workspace):
        return
    env["TERMINAL_CWD"] = workspace
    env["HERMES_WRITE_SAFE_ROOT"] = workspace
    from hermes_constants import venv_bin_dir
    python_name = "python.exe" if os.name == "nt" else "python"
    for name in (".venv", "venv"):
        venv = Path(workspace) / name
        executable = venv_bin_dir(venv) / python_name
        if not executable.is_file() or not os.access(executable, os.X_OK):
            continue
        # Preserve the venv spelling: resolving a Python symlink can turn it
        # into the base interpreter and discard the project's site-packages.
        env["HERMES_KANBAN_WORKTREE_PYTHON"] = str(executable)
        env["VIRTUAL_ENV"] = str(venv.resolve())
        env["PATH"] = os.pathsep.join(filter(None, (str(executable.parent), env.get("PATH"))))
        return
    if any((Path(workspace) / marker).is_file() for marker in (
        "pyproject.toml", "requirements.txt", "setup.py", "setup.cfg",
    )):
        raise RuntimeError(f"Python task workspace has no executable .venv/bin/python: {workspace}")
