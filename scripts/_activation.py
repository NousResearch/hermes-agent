"""Activation guard for repository scripts.

Every script in this checkout assumes it runs under the PM-activated
environment (``source ./activate`` on POSIX, ``. .\\activate.ps1`` on Windows),
which is what puts the pinned interpreter and its dependency tree on
``PATH``/``PYTHONPATH``. A script run with a bare system Python instead fails
much later with a confusing ``ModuleNotFoundError``. Call
:func:`require_activation` first and it fails immediately, naming the exact
command for the caller's shell.

Import it by its bare name: ``scripts/`` is the running script's own directory,
so it is on ``sys.path`` whether or not the shell is activated::

    from _activation import require_activation

    require_activation()

This module must stay stdlib-only and side-effect free at import — it has to
load before the environment it checks for exists.

Activating from the shebang
---------------------------

A POSIX script can run itself in the environment, so ``./scripts/foo.py`` works
from any cwd with no manual ``source``: its shebang hands the file to
``scripts/run-in-hermes-env`` (the line, staleness rules and portability limits
are documented there). Windows keeps using ``python scripts\\foo.py`` with the
guard.
"""

from __future__ import annotations

import os
import sys

ACTIVATION_ENV_VAR = "__HERMES_ACTIVATED"
POSIX_COMMAND = "source ./activate"
WINDOWS_COMMAND = ". .\\activate.ps1"


def activation_command() -> str:
    """The exact command that activates this checkout in the caller's shell."""
    if os.name != "nt":
        return POSIX_COMMAND
    # A bash-family shell on Windows (Git Bash, MSYS, WSL interop) sources the
    # POSIX script; only a native host gets the PowerShell one.
    if os.environ.get("MSYSTEM") or os.path.basename(os.environ.get("SHELL", "")) in {"bash", "sh", "zsh"}:
        return POSIX_COMMAND
    return WINDOWS_COMMAND


def require_activation() -> None:
    """Exit the process unless the launching shell sourced the activate script.

    Only tests the sentinel for non-emptiness: its value is the installed-state
    path the environment was built from (and earlier revisions wrote a bare
    ``1``), so any value means "an ancestor shell activated".
    """
    if os.environ.get(ACTIVATION_ENV_VAR):
        return
    script = os.path.basename(sys.argv[0]) or "this script"
    print(
        f"{script}: the Hermes environment is not activated.\n"
        "From the repository root, run:\n"
        "\n"
        f"    {activation_command()}\n"
        "\n"
        "then re-run this script.",
        file=sys.stderr,
    )
    raise SystemExit(1)
