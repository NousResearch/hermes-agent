"""Activation guard for repository scripts.

Every script in this checkout assumes it runs under the PM-activated
environment (``source ./activate`` on bash/zsh, ``source ./activate.fish`` on
fish, ``. .\\activate.ps1`` on Windows), which is what puts the pinned
interpreter and its dependency tree on ``PATH``/``PYTHONPATH``. A script run
with a bare system Python instead fails much later with a confusing
``ModuleNotFoundError``. Call :func:`require_activation` first and it fails
immediately, naming the exact command for the caller's shell.

Import it by its bare name: ``scripts/`` is the running script's own directory,
so it is on ``sys.path`` whether or not the shell is activated::

    from _activation import require_activation

    require_activation()

This module must stay stdlib-only and side-effect free at import — it has to
load before the environment it checks for exists.

Activating from the shebang
---------------------------

A POSIX script can activate for itself, so ``./scripts/foo.py`` works from any
cwd with no manual ``source``. The shebang points at ``scripts/_hermes-python``,
a real repo script that sources ``activate`` and execs the interpreter on the
same file::

    #!/usr/bin/env -S bash -c 'exec "$BASH" "$(dirname "$0")/_hermes-python" "$0" "$@"'

The kernel appends the invoking script as the last argument, and ``bash -c``
binds it to ``$0`` — so the prologue finds the target relative to itself and
the shebang stays independent of the cwd. ``$BASH`` for the second hop keeps it
free of both the exec bit and a ``PATH`` lookup.

Activation is not re-run when the inherited environment is still current. pm
records one stamp per dependency input (``uv.lock``, ``pyproject.toml``,
``pm/lock.json``) beside the installed-state file that ``__HERMES_ACTIVATED``
names, each carrying the exact mtime that input had when the install was last
verified against it (``pm.environments.record_activation_inputs``). The
prologue re-activates when any input's mtime *differs* from its stamp: newer
or older, since switching branches can move it either way. ``[ -nt ]`` and
``[ -ot ]`` are bash builtins, so the check costs no process spawn. Every
successful activation records again, including no-op syncs, so a checkout that
touches an input without changing it costs one re-activation and then settles.

Two constraints worth knowing: the line is 83 bytes (a shebang must stay under
127), and ``/usr/bin/env -S`` is GNU and newer-BSD — verify it if an older
macOS or BSD is a target. macOS's stock bash 3.2 compares whole seconds, so an
input rewritten within the same second as its recorded mtime goes unnoticed
there. Windows keeps using ``python scripts\\foo.py`` with the guard.
"""

from __future__ import annotations

import os
import sys

ACTIVATION_ENV_VAR = "__HERMES_ACTIVATED"
POSIX_COMMAND = "source ./activate"
FISH_COMMAND = "source ./activate.fish"
WINDOWS_COMMAND = ". .\\activate.ps1"


def activation_command() -> str:
    """The exact command that activates this checkout in the caller's shell."""
    shell = os.path.basename(os.environ.get("SHELL", ""))
    if os.name != "nt":
        if shell == "fish":
            return FISH_COMMAND
        return POSIX_COMMAND
    # A bash-family shell on Windows (Git Bash, MSYS, WSL interop) sources the
    # POSIX script; fish gets activate.fish; only a native host gets PowerShell.
    if os.environ.get("MSYSTEM") or shell in {"bash", "sh", "zsh"}:
        return POSIX_COMMAND
    if shell == "fish":
        return FISH_COMMAND
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
