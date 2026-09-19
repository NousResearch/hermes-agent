"""The tail every source install and update shares.

A fresh install (``scripts/install.sh`` / ``scripts/install.ps1``) and a finished
update must land in the same state: launchers published, product builds current,
and the post-build maintenance (skills sync, config migration) applied. One
implementation, two callers -- an install and an update cannot drift apart.

``main()`` is the installer entry. It is handed the bootstrap interpreter the
installer happens to have (uv, PM and no application dependencies), so it
re-executes itself in PM's selected interpreter before touching any product.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# The installers run this file directly under an isolated interpreter, which
# leaves the checkout off sys.path. Same idiom as hermes_cli/_launchers.py.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Second-phase marker: the re-executed interpreter, not the bootstrap one.
_PREPARED = "--prepared"


def complete_source_checkout(
    root: Path,
    *,
    desktop: bool,
    assume_yes: bool,
    gateway_mode: bool = False,
    pre_update_snapshot_id: str | None = None,
    pre_update_version: str | None = None,
    completion_message: str | None = None,
    announce: str | None = None,
) -> bool:
    """Publish commands, build the products, then run post-build maintenance.

    Returns the SQLite runtime verdict: a positive unsafe-runtime probe withholds
    success here exactly as it does at the end of an update.
    """
    from hermes_cli.source_build import build_update_products
    from hermes_cli.update_cmd_maint import _run_post_update_maintenance
    from hermes_cli.venv_sync import publish_launchers

    root = Path(root)
    publish_launchers(root)
    build_update_products(root, desktop=desktop)
    if announce:
        print(announce)
    return _run_post_update_maintenance(
        assume_yes=assume_yes,
        gateway_mode=gateway_mode,
        pre_update_snapshot_id=pre_update_snapshot_id,
        had_desktop_app_before_update=desktop,
        pre_update_version=pre_update_version,
        completion_message=completion_message,
    )


def _bootstrap_command(root: Path, argv: list[str]) -> list[str]:
    """Re-enter the checkout on PM's selected interpreter with its environment."""
    from pm.environments import project_python

    return [str(project_python(root)), "-I", "-B", "-u",
            str(Path(__file__).resolve()), "--source", str(root), *argv, _PREPARED]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    prepared = _PREPARED in argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--desktop", action="store_true",
                        help="Also build the packaged desktop app.")
    parser.add_argument("--interactive", action="store_true",
                        help="Allow prompts; installers pass no flag and stay unattended.")
    parser.add_argument("--finish-update", action="store_true",
                        help="Complete a source UPDATE tail instead of an install: the "
                             "same launchers/products/maintenance with update wording.")
    args = parser.parse_args([argument for argument in argv if argument != _PREPARED])
    root = args.source.resolve()
    if not (root / "hermes_cli/source_completion.py").is_file():
        print(f"✗ {root} is not a Hermes source checkout", file=sys.stderr)
        return 1

    if prepared:
        # Dependencies are selected before any application import: this is the
        # same ordering the update completion guarantees.
        from pm.environments import activate_dependencies

        activate_dependencies(root)
        if args.finish_update:
            # A source update that never reached its own completion -- a
            # pre-handoff release cannot flip during `hermes update`, so its
            # update ends with the tree at HEAD and nothing built -- lands here
            # on the next ordinary startup. Same tail as an install, so the two
            # states cannot drift apart.
            ok = complete_source_checkout(
                root, desktop=args.desktop, assume_yes=True,
                completion_message=None, announce="\n✓ Code updated!",
            )
        else:
            ok = complete_source_checkout(
                root, desktop=args.desktop, assume_yes=not args.interactive,
                completion_message="✓ Install complete!",
            )
        return 0 if ok else 1

    from pm.environments import activation_environment

    # This interpreter is a bootstrap one: the work happens in the re-exec, so
    # every flag that decides WHICH tail runs has to survive into it. Losing
    # --finish-update here silently reports an update as an install.
    passthrough = (["--desktop"] if args.desktop else []) + \
                  (["--finish-update"] if args.finish_update else [])
    command = _bootstrap_command(root, passthrough)
    return subprocess.call(command, cwd=root, env=activation_environment(root))


if __name__ == "__main__":
    raise SystemExit(main())