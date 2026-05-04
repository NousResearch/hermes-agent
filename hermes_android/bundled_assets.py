from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def bundled_skills_dir() -> Path:
    return repo_root() / "skills"


def optional_skills_dir() -> Path:
    return repo_root() / "optional-skills"


def configure_skill_env() -> dict[str, str]:
    bundled = bundled_skills_dir()
    optional = optional_skills_dir()
    os.environ["HERMES_BUNDLED_SKILLS"] = str(bundled)
    os.environ["HERMES_OPTIONAL_SKILLS"] = str(optional)
    return {
        "HERMES_BUNDLED_SKILLS": str(bundled),
        "HERMES_OPTIONAL_SKILLS": str(optional),
    }


def sync_bundled_skills(*, quiet: bool = True) -> dict[str, Any]:
    configure_skill_env()
    from hermes_android.python_path import prefer_hermes_package_root

    prefer_hermes_package_root()
    import tools.skills_sync as skills_sync

    skills_sync = importlib.reload(skills_sync)
    return skills_sync.sync_skills(quiet=quiet)
