"""Runnable ``sudo`` recipes and system-scope remediation for ``hermes gateway``.

Every recipe printed here names an absolute binary or the service manager, never a bare ``sudo hermes``.
``sudo`` resets PATH to ``secure_path`` (``/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin``),
which holds neither the ``~/.local/bin/hermes`` shim of a per-user install nor a pipx/venv launcher, so
``sudo hermes gateway install --system`` — the advice the entry point used to print — dies with
``sudo: hermes: command not found``, and the user has no next step.

A sibling rather than part of the ``hermes_cli/gateway.py`` facade: the facade is over its line cap on
``main`` and may only shrink. Facade names are imported late (``from hermes_cli.gateway import …`` inside
the function) so the module-level import in the facade stays acyclic.
"""

from __future__ import annotations

import os
import shutil
import sys

_SYSTEMD_SERVICE_VERBS = frozenset({"start", "stop", "restart"})


def _hermes_binary_for_sudo() -> str:
    """Absolute path to this CLI, for a printed ``sudo …`` recipe.

    ``shutil.which`` runs against the caller's own PATH, which does include the shim; the ``sys.argv[0]``
    fallback covers an interpreter invoked directly on a launcher that PATH cannot see.
    """
    resolved = shutil.which("hermes")
    if resolved:
        return resolved
    argv0 = (sys.argv[0] if sys.argv else "") or ""
    if argv0 and os.path.isfile(argv0) and os.access(argv0, os.X_OK):
        return os.path.abspath(argv0)
    return "hermes"


def _sudo_hermes_prefix() -> str:
    """``sudo <absolute hermes> `` — the only sudo form of this CLI that runs as printed."""
    return f"sudo {_hermes_binary_for_sudo()} "


def _systemd_cli_bits(system: bool) -> tuple[str, str, str]:
    """``(sudo_prefix, scope_flag, user_flag)`` for printed hints: ``("sudo <abs hermes> ", " --system", "")``
    in system scope, ``("", "", "--user ")`` in user scope.

    The prefix spells the binary out: the CLI hints that append ``hermes …`` to it are exactly the ones a
    user pastes into a root shell.
    """
    return (_sudo_hermes_prefix(), " --system", "") if system else ("", "", "--user ")


def _print_system_scope_remediation(action: str, cli_args: str | None = None) -> None:
    """Print remediation for a system-scope action attempted without root.

    ``cli_args`` is the subcommand tail to re-run (the CLI entry point passes what the user typed); without
    it the recipe is ``gateway <action> --system``. The supervisor form is offered only for real systemd
    verbs — ``systemctl install`` does not exist.
    """
    from hermes_cli.gateway import get_service_name, print_info, print_warning

    print_warning(f"Gateway system service — {action} requires root.")
    print_info("  Options:")
    print_info("    1. Re-run it as root:")
    if action in _SYSTEMD_SERVICE_VERBS:
        print_info(f"         sudo systemctl {action} {get_service_name()}")
    recipe = cli_args or f"gateway {action} --system"
    print_info(f"         {_sudo_hermes_prefix()}{recipe}")
    print_info("    2. Switch to a per-user service (recommended for personal use):")
    print_info(f"         {_sudo_hermes_prefix()}gateway uninstall --system")
    print_info("         hermes gateway install")
    print_info("         hermes gateway start")


def _typed_gateway_args(argv: list[str]) -> str | None:
    """The ``gateway …`` tail of this process's argv, for a copy-pasteable sudo recipe.

    Global flags may precede the subcommand (``hermes --yolo gateway restart --system``), so the tail starts
    at ``gateway``; anything before it is not part of the command that needs root.
    """
    parts = list(argv[1:])
    if "gateway" in parts:
        return " ".join(parts[parts.index("gateway"):])
    return None


def _with_runnable_sudo(lines: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve the sudo recipe in module-level hint constants at print time.

    The status hints are constants, so the resolved binary cannot be baked in at import; every
    ``sudo hermes`` they carry would be unrunnable as printed on a per-user install.
    """
    return tuple(line.replace("sudo hermes ", _sudo_hermes_prefix()) for line in lines)
