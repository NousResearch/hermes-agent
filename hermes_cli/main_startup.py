"""The pre-parse startup prelude ``main.main`` runs before argparse.

Every ``hermes`` invocation used to pay this before dispatch — including
``hermes --help`` and read-only commands like ``hermes kanban list``, which
cannot be affected by any step below. It lives here, with the allowlist of
argv shapes that do not need it, so the skip and the steps are read together.

The skip list is an ALLOWLIST of read-only shapes, never a denylist keyed on
the word ``update``: ``hermes update`` must keep recovering the gateways a
killed update left paused (``_BOOT_BOOTSTRAP_OWNED_SHAPES`` defers exactly one
step for it, after the swap), and an unrecognised shape keeps the full prelude.

The steps late-import their owners through ``hermes_cli.main`` on purpose: main
re-exports them, so a patch on the main facade still intercepts the call (the
sibling rule in the root AGENTS.md).
"""

from __future__ import annotations

import sys

from hermes_cli._parser import command_argv

#: ``argparse`` answers these itself, at any depth, and exits — no prelude step
#: can change what gets printed.
_HELP_FLAGS = frozenset({"-h", "--help"})
#: Normally answered pre-import by ``_startup_fast.try_fast_version``; listed for
#: the paths that reach ``main()`` anyway (container mode defers the fast path).
_VERSION_FLAGS = frozenset({"-V", "--version"})

#: Read-only shapes: each one reads state and prints it, and no prelude step can
#: change what it prints. Add a shape only when that is true of the whole command.
_READ_ONLY_SHAPES: dict[tuple[str, ...], str] = {
    ("config",): "prints the effective config",
    ("config", "get"): "prints one resolved value",
    ("config", "path"): "prints a path",
    ("config", "env-path"): "prints a path",
    ("config", "show"): "prints the effective config",
    ("kanban", "boards"): "lists boards",
    ("kanban", "list"): "lists tasks",
    ("kanban", "ls"): "lists tasks",
    ("kanban", "show"): "shows one task",
}

#: Shapes that run every step except the boot bootstrap: they own that pass
#: themselves, after they have swapped in the checkout the prelude would bootstrap.
_BOOT_BOOTSTRAP_OWNED_SHAPES = frozenset({("update",)})


def shape(argv: list[str]) -> tuple[str, ...]:
    """``(subcommand, sub-subcommand)`` for *argv*, ignoring top-level flags."""
    return tuple(command_argv([str(token) for token in argv])[:2])


def skips_startup_prelude(argv: list[str] | None = None) -> bool:
    """True when *argv* dispatches to something the prelude cannot affect."""
    argv = list(sys.argv[1:] if argv is None else argv)
    for token in argv:
        if token == "--":  # everything after it is a value or a prompt, not a flag
            break
        if token in _HELP_FLAGS or token in _VERSION_FLAGS:
            return True
    return shape(argv) in _READ_ONLY_SHAPES


def run_startup_prelude(argv: list[str] | None = None) -> None:
    """Run the pre-dispatch startup steps, unless *argv* does not need them."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if skips_startup_prelude(argv):
        return
    _sweep_quarantine_and_stale_bytecode()
    _recover_update_debts()
    if shape(argv) not in _BOOT_BOOTSTRAP_OWNED_SHAPES:
        _run_boot_bootstrap()
    _warn_about_pm_runtime_problem()


def _sweep_quarantine_and_stale_bytecode() -> None:
    """Windows ``hermes.exe.old.*`` cleanup, then the once-per-checkout bytecode sweep."""
    from hermes_cli.main import (
        _cleanup_quarantined_exes,
        _sweep_stale_bytecode_if_checkout_changed,
    )

    try:
        _cleanup_quarantined_exes()
    except Exception:  # a Windows shim sweep never blocks the launch it cleans up after
        pass
    _sweep_stale_bytecode_if_checkout_changed()


def _recover_update_debts() -> None:
    """Owed fleet restarts and gateways a killed ``hermes update`` left paused."""
    from hermes_cli.main import _recover_update_debts_on_startup

    _recover_update_debts_on_startup()


def _run_boot_bootstrap() -> None:
    from hermes_cli.boot_bootstrap import maybe_run_boot_bootstrap
    from pm.paths import install_root

    maybe_run_boot_bootstrap(install_root())


def _warn_about_pm_runtime_problem() -> None:
    """One passive PM verdict per dispatch (the venv is out of sync with the lockfile)."""
    try:
        from hermes_cli.venv_sync import check_runtime
        from pm.paths import install_root

        problem = check_runtime(install_root())
        if problem:
            print(f"⚠ {problem}", file=sys.stderr)
    except Exception:
        import logging

        logging.getLogger(__name__).debug("pm startup check failed", exc_info=True)
