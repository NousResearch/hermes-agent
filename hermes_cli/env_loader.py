"""Helpers for loading Hermes .env files consistently across entrypoints."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _load_dotenv_with_fallback(path: Path, *, override: bool) -> None:
    try:
        load_dotenv(dotenv_path=path, override=override, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(dotenv_path=path, override=override, encoding="latin-1")


def load_hermes_dotenv(
    *,
    hermes_home: str | os.PathLike | None = None,
    project_env: str | os.PathLike | None = None,
    cwd_env: bool = False,
) -> list[Path]:
    """Load Hermes environment files with user config taking precedence.

    Priority (highest to lowest):
    1. ``~/.hermes/.env`` — user's global config, overrides stale shell exports.
    2. CWD ``.env`` — project-specific keys (when *cwd_env* is True).
       Only fills missing values when the user env exists.
    3. *project_env* — developer fallback (hermes source-tree ``.env``).
       Only fills missing values when the user env exists.

    When ``~/.hermes/.env`` does not exist, the CWD and project files are
    allowed to override shell-exported values so a fresh install with only a
    local ``.env`` works out of the box.
    """
    loaded: list[Path] = []

    home_path = Path(hermes_home or os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    user_env = home_path / ".env"
    project_env_path = Path(project_env) if project_env else None

    if user_env.exists():
        _load_dotenv_with_fallback(user_env, override=True)
        loaded.append(user_env)

    # CWD .env — project-specific keys for the directory hermes is run from.
    if cwd_env:
        cwd_env_path = Path.cwd() / ".env"
        # Skip if it's the same file as the user env or the project env.
        if (
            cwd_env_path.exists()
            and cwd_env_path.resolve() != user_env.resolve()
            and (project_env_path is None or cwd_env_path.resolve() != project_env_path.resolve())
        ):
            _load_dotenv_with_fallback(cwd_env_path, override=not loaded)
            loaded.append(cwd_env_path)

    if project_env_path and project_env_path.exists():
        _load_dotenv_with_fallback(project_env_path, override=not loaded)
        loaded.append(project_env_path)

    return loaded
