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
    followups: list[tuple[str, str]] | None = None,
    before_build=None,
) -> bool:
    """Publish commands, build the products, then run post-build maintenance.

    ``before_build`` (an update's paused-gateway restart) runs once the launchers are published:
    dependencies are already synced, the long product builds have not started.

    Every step runs even when an earlier one failed; each failure is printed as ``⚠``,
    recorded on the open update receipt and appended to ``followups`` as ``(step, reason)``.
    Returns True only when every step finished and the SQLite runtime is not unsafe.
    """
    from hermes_cli.update_lock import UpdateLock, describe_holder

    root = Path(root)
    # This tail is the last mutating step of an install or update, and its product
    # builds run for minutes. A gateway restarted while it runs (launchd KeepAlive,
    # a service manager, a second `hermes` launch) reaches the same tail through
    # venv_sync, and `hermes update` runs its own: two completions then build the
    # same output directories concurrently and race on install-stamp.json (#123376).
    # Claim the shared update lock so stacked completions serialize. A tail whose
    # orchestrator already holds the lock (venv_sync's interrupted-update finish,
    # the updater's completion child) runs under its parent's claim, exactly as
    # `hermes update` does under the desktop handoff pid. The CHECKOUT lock is acquired, not
    # sampled (R2): a completion from another home must not build a checkout an update owns;
    # one inside the update's tree joins the lock it inherited.
    lock = UpdateLock(install_root=root)
    if not lock.acquire():
        raise RuntimeError(
            f"an update is still running ({describe_holder(lock.holder)}); "
            "wait for it to exit, then relaunch Hermes"
        )
    try:
        return _complete_locked(
            root, desktop=desktop, assume_yes=assume_yes, gateway_mode=gateway_mode,
            pre_update_snapshot_id=pre_update_snapshot_id,
            pre_update_version=pre_update_version,
            completion_message=completion_message, announce=announce, followups=followups,
            before_build=before_build,
        )
    finally:
        lock.release()


def _complete_locked(
    root: Path,
    *,
    desktop: bool,
    assume_yes: bool,
    gateway_mode: bool,
    pre_update_snapshot_id: str | None,
    pre_update_version: str | None,
    completion_message: str | None,
    announce: str | None,
    followups: list[tuple[str, str]] | None = None,
    before_build=None,
) -> bool:
    """The completion body; callers hold the update lock already."""
    from hermes_cli.source_build import build_update_products
    from hermes_cli.update_cmd_maint import _run_post_update_maintenance
    from hermes_cli.venv_sync import publish_launchers

    try:
        from hermes_cli._subprocess_compat import expose_pm_git

        # The builds, the release-history refresh and the install stamp all run
        # git; a fresh Windows machine has only PM's.
        expose_pm_git(root)
    except Exception as exc:  # noqa: BLE001 — git-less steps below still complete
        print(f"⚠ Could not provide git for the source completion: {exc}", file=sys.stderr)
    owed: list[tuple[str, str]] = []

    def step(name: str, run) -> None:
        # Each step is independent: a failed launcher publish or web build must not skip the
        # config migration, the maintenance, or (in an update) the gateway restart after it.
        try:
            run()
        except (Exception, SystemExit) as exc:  # health: allow BLE001 -- reported as an owed follow-up
            from hermes_cli.update_receipt import record_followup

            reason = str(exc) or type(exc).__name__
            record_followup(name, reason)
            owed.append((name, reason))

    step("launchers", lambda: publish_launchers(root))
    if before_build is not None:
        before_build()  # never raises: a failed restart stays owed and is retried after the build
    step("build", lambda: build_update_products(root, desktop=desktop))
    if announce:
        print(announce)
    verdict: list[bool] = []
    step("maintenance", lambda: verdict.append(_run_post_update_maintenance(
        assume_yes=assume_yes,
        gateway_mode=gateway_mode,
        pre_update_snapshot_id=pre_update_snapshot_id,
        had_desktop_app_before_update=desktop,
        pre_update_version=pre_update_version,
        completion_message=completion_message,
        followups=owed,
    )))
    tail_done = not owed
    if tail_done:
        from hermes_cli.source_stamp import write_source_stamp

        try:
            write_source_stamp(root)
        except (OSError, ValueError) as exc:
            print(f"⚠ Source update completed, but the install stamp could not be written: {exc}",
                  file=sys.stderr)
    if followups is not None:
        followups.extend(owed)
    # The SQLite verdict is not tail work (re-running the tail cannot fix the interpreter); it
    # is reported by the maintenance step itself and only withholds an install's success.
    return tail_done and all(verdict)


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
            # states cannot drift apart. Only owed TAIL work keeps it pending: an
            # unsafe SQLite runtime is reported, but re-running the tail cannot fix it.
            owed: list[tuple[str, str]] = []
            complete_source_checkout(
                root, desktop=args.desktop, assume_yes=True,
                completion_message=None, announce="\n✓ Code updated!", followups=owed,
            )
            ok = not owed
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
    from hermes_cli.update_lock import checkout_lock_fds

    # Called inside an update tree (venv_sync's interrupted-update finish), the prepared child
    # keeps the checkout lock this bootstrap inherited.
    fds = checkout_lock_fds(root)
    return subprocess.call(command, cwd=root, env=activation_environment(root),
                           **({"pass_fds": fds} if fds else {}))


if __name__ == "__main__":
    raise SystemExit(main())