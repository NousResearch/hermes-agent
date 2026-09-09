"""``hermes doctor --upgrade-node[=MAJOR]``: opt-in Node major upgrade for the Hermes-managed
runtime under ``~/.hermes/node``. See issue #106456.

Deliberately separate from the silent, ``engines.node``-minimum-targeted automatic heal
(``hermes_constants.heal_hermes_managed_node``): this always requires the user to have typed
``--upgrade-node`` themselves.
"""

from __future__ import annotations

import sys

from hermes_cli.colors import Colors, color
from hermes_constants import (
    engines_node_allows_major, engines_node_default_upgrade_major, managed_node_major)


def _current_managed_major() -> int | None:
    """Major version of the currently installed Hermes-managed Node, or ``None`` if absent/broken."""
    return managed_node_major()


def _install_target_major(target_major: int) -> bool:
    """Run the staged install/heal machinery for *target_major*, POSIX or Windows."""
    if sys.platform == "win32":
        from hermes_constants import _heal_managed_node_windows
        return bool(_heal_managed_node_windows(target_major=target_major))
    from hermes_constants import _run_node_bootstrap
    return _run_node_bootstrap(
        "_nb_install_bundled_node", timeout=600, HERMES_NODE_TARGET_MAJOR=str(target_major))


def _update_node_dependencies(force: bool = False) -> list[str]:
    """Thin wrapper so tests can patch this module's own reference independently of
    ``hermes_cli.update_cmd_deps`` (imported lazily to avoid update_cmd's import-time side effects
    inside `hermes doctor`, which must stay fast)."""
    from hermes_cli.update_cmd_deps import _update_node_dependencies as _impl
    return _impl(force=force)


def upgrade_node(major_arg: str) -> None:
    """Entry point for ``hermes doctor --upgrade-node[=MAJOR]``. ``major_arg`` is the sentinel
    ``"<newest>"`` for a bare flag, or the literal digits the user passed after ``=``.
    """
    if major_arg == "<newest>":
        target_major = engines_node_default_upgrade_major()
    else:
        try:
            target_major = int(major_arg)
        except ValueError:
            print(color(f"--upgrade-node: {major_arg!r} is not a Node major version number", Colors.RED))
            sys.exit(2)
        if not engines_node_allows_major(target_major):
            print(color(
                f"--upgrade-node={target_major}: this checkout's package.json engines.node does "
                "not allow that major.", Colors.RED))
            print(
                "  engines.node in this checkout's package.json is the source of truth for "
                "--upgrade-node; add this major there to allow it. (HERMES_NODE_TARGET_MAJOR "
                "overrides the automatic silent heal only — it has no effect on --upgrade-node.)")
            sys.exit(2)

    current = _current_managed_major()
    if current == target_major:
        print(color(f"  ✓ already on Node {target_major} — nothing to do.", Colors.GREEN))
        return

    print(f"→ Upgrading Hermes-managed Node.js to major {target_major}"
          f"{f' (from {current})' if current else ''}...")
    if not _install_target_major(target_major):
        print(color(f"  ✗ Node {target_major} install failed — the previous install is unchanged.", Colors.RED))
        sys.exit(1)
    print(color(f"  ✓ Node {target_major} installed.", Colors.GREEN))

    print("→ Rebuilding node_modules/web against the new Node ABI...")
    failures = _update_node_dependencies(force=True)
    if failures:
        print(color(f"  ✗ Dependency rebuild failed for: {', '.join(failures)}", Colors.RED))
        sys.exit(1)
    print(color("  ✓ Dependencies rebuilt.", Colors.GREEN))
