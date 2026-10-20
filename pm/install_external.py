"""Read-only health checks for tools supplied by an external package builder."""
from __future__ import annotations

import json
import os
from pathlib import Path

from pm import paths


def external_packages() -> dict[str, str]:
    """The install owner declares exact executables, not PM download identities.

    A developer checkout still owes PM's pins. Packaged Nix code instead
    resolves its stamp through HERMES_INSTALL_ROOT, outside the wheel.
    """
    root = paths.repo_root()
    if (root / ".git").exists():
        return {}
    try:
        stamp = json.loads(paths.install_stamp_path(root).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    if not isinstance(stamp, dict) or stamp.get("updateMechanism") != "external":
        return {}
    packages = stamp.get("externalPackages", {})
    return packages if isinstance(packages, dict) else {}


def external_drift(packages: dict[str, str]) -> dict[str, str]:
    """Never run a probe or accept a shell's replacement for a missing build input."""
    problems = {}
    for name, executable in packages.items():
        if (not isinstance(executable, str) or not Path(executable).is_absolute()
                or not Path(executable).is_file() or not os.access(executable, os.X_OK)):
            problems[name] = "external executable missing or not executable"
    return problems
