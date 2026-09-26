"""Pre-venv entry point for PM's dependency transaction.

Stdlib-only at import: installers call this before dependencies exist.
All checkout roots use PM's selected generation and facts; sealed payloads
remain build-owned. ``--check`` is passive and never provisions tools.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from hermes_cli.steward import UPDATE_MECHANISMS


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_sealed(project_root: Path) -> bool:
    """A sealed tree ships its interpreter; only checkouts own a venv.

    The stamp file is the authority (shared with hermes_cli.steward).
    A tree with BOTH a stamp and .git is a dev tree — treat as checkout.

    A stamp without a valid ``updateMechanism`` is a build-lane bug and
    must not be silently read as "not sealed" (that is exactly the
    misclassification that made sealed trees look updatable) — same
    guard as hermes_cli.version_info._stamp_version_info.
    """
    if (project_root / ".git").exists():
        return False
    from hermes_cli.steward import read_install_stamp
    from pm.paths import install_stamp_path
    stamp_path = install_stamp_path(project_root)
    data = read_install_stamp(project_root)
    if not data:
        return False
    if data.get("updateMechanism") not in UPDATE_MECHANISMS:
        raise RuntimeError(
            f"install-stamp.json at {stamp_path.parent} is missing a valid "
            f"'updateMechanism' (one of {', '.join(UPDATE_MECHANISMS)}). The "
            "build lane that wrote this stamp must pass --update-mechanism to "
            "scripts/write_install_stamp.py."
        )
    return True


def check_runtime(project_root: Path) -> str | None:
    """One passive startup verdict; callers only choose stderr or logging."""
    import pm
    from hermes_cli.steward import read_install_stamp, sealed_steward

    if (Path(project_root) / ".git").exists() and read_install_stamp(project_root).get("updateMechanism") != "self":
        return None  # A developer's checkout does not owe managed products.

    problems = pm.activate()
    if not problems:
        return None
    steward = sealed_steward(Path(project_root))
    remedy = (f"this {steward}-managed install must rebuild the artifact to fix"
              if steward else "run `hermes pm install`")
    return f"install out of sync ({'; '.join(problems)}) — {remedy}"


def publish_launchers(project_root: Path, *, create: bool = True) -> None:
    """Refresh durable commands; bootstrap repairs only existing PATH exposure."""
    import logging

    from hermes_cli._launchers import ENTRY_POINTS, ensure_install_launchers, expose_cli, resolve_store_python
    from hermes_cli.steward import read_install_stamp

    root = Path(project_root)
    log = logging.getLogger(__name__)
    if _is_sealed(root):
        log.info("launchers: sealed tree at %s keeps its own", root)
        return  # Sealed and external/Nix interpreters retain their own launchers.
    if read_install_stamp(root).get("updateMechanism") == "external":
        log.info("launchers: external runtime at %s keeps its own", root)
        return
    if resolve_store_python(root) is None:
        # A PM tree promises its launchers (tests/install/e2e-assets/
        # source-driver.sh refuses to let --version paper over the gap), so
        # this skip is a half-finished update, never a quiet no-op.
        log.warning("launchers: no managed interpreter under %s; %s not published",
                    root, root / ".hermes" / "bin")
        return
    written = ensure_install_launchers(root, root / ".hermes" / "bin")
    if len(written) != len(ENTRY_POINTS):
        from pm.package import InstallError

        raise InstallError("launchers", "source launcher publication failed", "retry the source update")
    result = expose_cli(root, create=create)
    if not result["ok"]:
        import logging

        logging.getLogger(__name__).warning("CLI exposure failed: %s", result["error"])


def sync(project_root: Path | None = None, *, check: bool = False) -> dict:
    """Report or sync dependencies. A malformed install stamp is a build error."""
    from hermes_cli.update_stage import publish_stage

    root = Path(project_root) if project_root is not None else _project_root()
    if _is_sealed(root):
        return {"state": "sealed", "ok": True}
    if not (root / "pyproject.toml").is_file():
        return {"state": "failed", "ok": False, "detail": f"no pyproject.toml under {root}"}
    try:
        import pm

        if pm.venv_is_current(project_root=root):
            if not check:
                publish_launchers(root)
            return {"state": "current", "ok": True}
        if check:
            return {"state": "would-sync", "ok": True}
        publish_stage("Updating Python dependencies")
        refuse_foreign_owned_venv(root)
        pm.sync_venv(explicit=True, project_root=root, evict_incompatible_plugins=True)
        collect_superseded_generations(root)
        publish_launchers(root)
        return {"state": "synced", "ok": True}
    except Exception as exc:
        return {"state": "failed", "ok": False, "detail": str(exc)}


def collect_superseded_generations(project_root: Path) -> None:
    """Collect what a publish just superseded, as the Docker boot already does.

    Without this only a manual `hermes pm gc` reclaimed old environments. Safe
    right after a sync: the collectors skip leased, selected and day-young
    generations and yield to any in-flight install instead of waiting.
    """
    import logging

    from hermes_cli.runtime_state import collect_generations
    from pm.environments import install_state_dir
    from pm.runtime import collect_runtime_generations

    try:
        removed = collect_generations(project_root) + collect_runtime_generations(
            install_state_dir(project_root) / "pm-runtime")
    except (OSError, ValueError) as exc:
        # Reclaiming space must never turn a committed update into a failure.
        logging.getLogger(__name__).warning("dependency generation cleanup skipped: %s", exc)
        return
    if removed:
        logging.getLogger(__name__).info("collected %d unused dependency generations", len(removed))


#: Answered from the tree alone; a metadata query must never wait on (or fail with)
#: a network-bound source-update completion.
_METADATA_FLAGS = frozenset({"-h", "--help", "-V", "--version"})


def completion_pending_path(project_root: Path) -> Path:
    """Marker for a source update whose dependency sync committed but whose tail
    (launchers, products, maintenance) has not finished.

    Lives beside PM's facts, not in the checkout: it is per-install state, and a
    root-level file would trip the ZIP updater's dirty-tree check.
    """
    from pm.environments import install_state_dir

    return install_state_dir(project_root) / "source-completion-pending"


def arm_completion(project_root: Path) -> Path:
    """Persist the tail obligation before selecting a new dependency generation."""
    pending = completion_pending_path(project_root)
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text("source update tail not finished\n", encoding="utf-8")
    return pending


def clear_completion(project_root: Path) -> None:
    completion_pending_path(project_root).unlink(missing_ok=True)


def refuse_foreign_owned_venv(project_root: Path) -> None:
    """Refuse cross-user mutation before PM changes the selected environment (#83529)."""
    if not hasattr(os, "geteuid"):
        return
    uid = os.geteuid()  # windows-footgun: ok — guarded POSIX ownership check
    root = Path(project_root)
    # A root-run update on a user's checkout is not safe even if a fresh
    # generation would be allocated: it publishes root-owned state for them.
    from pm.environments import selected_venv
    candidates = [root, root / "venv", root / ".venv", root / ".hermes", selected_venv(root)]
    for venv in (root / "venv", root / ".venv", candidates[-1]):
        for directory in (venv / ("Scripts" if os.name == "nt" else "bin"),
                          *venv.glob("lib/python*/site-packages")):
            if directory.is_dir():
                candidates.append(directory)
                for entry in list(directory.iterdir())[:2000]:
                    candidates.append(entry)
                    if entry.name.endswith(".dist-info") and entry.is_dir():
                        candidates.extend(list(entry.iterdir())[:100])
    for path in candidates:
        try:
            owner = path.lstat().st_uid
        except FileNotFoundError:
            continue
        if owner != uid:
            raise RuntimeError(
                f"refusing to update {root}: {path} is owned by uid {owner}, "
                f"not the current uid {uid}; repair ownership before retrying"
            )


def prepare_launch(project_root: Path, argv: list[str]) -> Path | None:
    """Finish a self-managed source update before importing app dependencies.

    PM's successful input stamp signals a finished dependency sync; the
    ``source-completion-pending`` marker signals the tail still owed after it,
    so a tail that failed is retried on the next launch WITHOUT rebuilding
    dependencies that are already current. Old updaters need not write a
    marker (and cannot accidentally clear this obligation).
    Return the store interpreter when this process must restart cleanly.
    """
    import os
    import sys
    from hermes_cli._parser import command_argv
    from hermes_cli.steward import read_install_stamp

    root = Path(project_root).resolve()
    if (command_argv(argv)[:1] == ["pm"]
            or _METADATA_FLAGS & set(argv)
            or os.environ.get("HERMES_DISABLE_LAZY_INSTALLS", "").lower() in ("1", "true", "yes")
            or not (root / ".git").exists()
            or not (root / "pyproject.toml").is_file()):
        return None
    stamp = read_install_stamp(root)
    if not stamp:
        from hermes_cli.post_update import step_adopt_blessed_checkout

        step_adopt_blessed_checkout(root)
        stamp = read_install_stamp(root)
    if stamp.get("updateMechanism") != "self":
        return None  # Developer checkouts and packaged runtimes retain their owner.

    import pm
    from hermes_cli._launchers import resolve_store_python
    from hermes_cli.update_lock import UpdateLock, read_live_update

    current = pm.venv_is_current(project_root=root)
    pending = completion_pending_path(root)
    if not current or pending.is_file():
        lock = UpdateLock()
        if not lock.acquire():
            raise RuntimeError("an update is still running; wait for it to exit, then relaunch Hermes")
        try:
            # The tail imports the application, whose entry point runs this very function:
            # under the launching process's own claim (its pid is our ancestor) we ARE that
            # tail and owe nothing — without this, a pending marker recurses forever.
            if not lock.acquired and read_live_update() is not None:
                if current:
                    return None
                # A process the update spawns before its dependencies are current (a restarted
                # gateway) would boot on a tree built for another interpreter. Sync — never the
                # tail, which is the updater's — then relaunch below into a current install.
                _sync_source_dependencies(root, arm=False)
                if not pm.venv_is_current(project_root=root):
                    # Relaunching would land back here and sync again, forever.
                    raise RuntimeError("dependency sync left this install out of date")
            else:
                _finish_source_update(root, current=current, pending=pending)
        finally:
            lock.release()
    python = resolve_store_python(root)
    if python is None:
        raise RuntimeError("source update has no managed Python; run `hermes pm install`")
    if not current or python.absolute() != Path(sys.executable).absolute():
        publish_launchers(root)
        return python
    return None


def _finish_source_update(root: Path, *, current: bool, pending: Path) -> None:
    """Sync dependencies when they are stale, then run the tail the marker still owes."""
    import sys
    from hermes_cli._early_recovery import _marker_owner_is_live
    from pm.environments import activation_environment

    if not current:
        # Existing markers guard liveness, never create the completion obligation.
        # Current post-sync verification children can boot under a live updater.
        legacy_markers = (root / ".update-incomplete", root / ".lazy-refresh-incomplete")
        if any(_marker_owner_is_live(marker) for marker in legacy_markers):
            raise RuntimeError("an update is still running; wait for it to exit, then relaunch Hermes")
        print("hermes: completing source-update dependencies...", file=sys.stderr, flush=True)
        _sync_source_dependencies(root, arm=True)
    else:
        print("hermes: finishing an interrupted source update...", file=sys.stderr, flush=True)
    # Sync commits the dependency generation, but a source update also owes
    # the product builds and the post-build maintenance -- the tail every
    # install and finished update shares (hermes_cli/source_completion.py).
    # Those builds need PM's selected interpreter with its dependencies
    # activated, so hand that file THIS interpreter and let it re-exec
    # itself, exactly as the installers do.
    desktop_app = root / "apps/desktop"
    desktop = ((desktop_app / "dist/index.html").is_file()
               or any((desktop_app / "release").glob("*")))
    # The tail's progress lines go to stderr: this is an automatic repair in
    # front of whatever command the user ran, and that command may be
    # emitting machine-readable stdout (a JSON probe, a piped query).
    code = subprocess.call(
        [sys.executable, "-I", "-B", "-u",
         str(root / "hermes_cli/source_completion.py"),
         "--source", str(root), "--finish-update",
         *(("--desktop",) if desktop else ())],
        cwd=root, env=activation_environment(root), stdout=sys.__stderr__,
    )
    if code != 0:
        raise RuntimeError(
            "source update completion failed; run `hermes update` to finish it"
        )
    clear_completion(root)


def _sync_source_dependencies(root: Path, *, arm: bool) -> None:
    """Commit the tree's dependency generation; *arm* also owes the tail afterwards."""
    import sys
    import pm
    from pm.client import ensure_tools_for_sync
    from pm.environments import runtime_facts_path
    from pm.extras import legacy_selection

    if not arm:
        print("hermes: preparing dependencies for this update...", file=sys.stderr, flush=True)
    refuse_foreign_owned_venv(root)
    if arm:
        # Owed from before the sync commits: a crash between the commit and the
        # tail must leave the tail, not a "current" install with nothing built.
        arm_completion(root)
    # Main-era installs have no PM ledger; carry what their venv held.
    # Established PM installs retain their recorded extras and plugin union instead.
    extras = legacy_selection(root) if not runtime_facts_path(root).is_file() else None
    # Same order as `hermes update`: an interrupted update or a hand-run
    # `git pull` leaves this tree's lockfile ahead of the installed tools.
    ensure_tools_for_sync()
    pm.sync_venv(extras, explicit=True, project_root=root, evict_incompatible_plugins=True)
    collect_superseded_generations(root)
    # These can predate the swap. Once PM commits the replacement they
    # must not make early recovery immediately rebuild it a second time.
    for name in (".update-incomplete", ".lazy-refresh-incomplete"):
        (root / name).unlink(missing_ok=True)


def _is_python_launcher(target: Path) -> bool:
    """Shebang-sniff a launcher path: True unless it is demonstrably not Python.

    Historical takeover shims are shell wrappers (see _write_shell in
    hermes_cli/_launchers.py); feeding one to runpy.run_path() parses shell
    as Python and raises SyntaxError. Missing, unreadable, or shebang-less
    paths (plain scripts, distlib exe zips) keep the legacy run_path shape.
    """
    try:
        with open(target, "rb") as handle:
            head = handle.read(512)
    except OSError:
        return True
    if head.startswith(b"#!"):
        first_line = head.split(b"\n", 1)[0]
        return b"python" in first_line.lower()
    return True


def relaunch_command(
    python: Path, root: Path, argv: list[str], original: list[str], module: str | None,
) -> list[str]:
    """Re-enter the same script/module/launcher with the managed interpreter.

    An old venv may use a different Python ABI. Do not add the new generation
    to that interpreter, and do not depend on its obsolete editable finder.
    """
    # Preserve interpreter options, not application flags with the same names.
    options: list[str] = []
    index = 1
    while index < len(original):
        option = original[index]
        if option in ("-c", "-m", "--", "-") or not option.startswith("-"):
            break
        options.append(option)
        index += 1
        if option in ("-W", "-X") and index < len(original):
            options.append(original[index])
            index += 1
    prefix = f"import sys, runpy; sys.path.insert(0, {str(root)!r}); sys.argv = {argv!r}; "
    if argv[0] == "-c":
        body = f"exec({original[index + 1]!r})"
    elif module and module != "__main__":
        body = f"runpy.run_module({module!r}, run_name='__main__', alter_sys=True)"
    else:
        # distlib .exe launchers are executable zip files with __main__, not
        # importable modules named '__main__'. run_path handles both shapes.
        target = Path(argv[0]).absolute()
        if not _is_python_launcher(target):
            # A historical takeover shell shim already re-execs its own
            # interpreter, so run it directly instead of runpy.run_path(),
            # which would parse shell as Python and die with SyntaxError
            # before the gateway restarts. Interpreter options cannot cross
            # the shim boundary, so they are dropped here.
            return [str(target), *argv[1:]]
        body = f"runpy.run_path({str(target)!r}, run_name='__main__')"
    return [str(python), *options, "-I", "-c", prefix + body]


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes_cli.venv_sync")
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--check", action="store_true", help="report; change nothing"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = sync(
        Path(args.project_root) if args.project_root else None, check=args.check
    )

    if args.json:
        print(json.dumps(result))
    else:
        detail = f" ({result['detail']})" if result.get("detail") else ""
        print(f"venv sync: {result['state']}{detail}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
