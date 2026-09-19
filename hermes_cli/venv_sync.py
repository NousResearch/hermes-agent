"""Pre-venv entry point for PM's dependency transaction.

Stdlib-only at import: installers call this before dependencies exist.
All checkout roots use PM's selected generation and facts; sealed payloads
remain build-owned. ``--check`` is passive and never provisions tools.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from hermes_cli.steward import UPDATE_MECHANISMS


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_sealed(project_root: Path) -> bool:
    """A sealed tree ships its interpreter; only checkouts own a venv.

    The stamp file is the authority (hermes_cli.steward reads the same
    file; restated here to keep the bare import stdlib-and-local). A
    tree with BOTH a stamp and .git is a dev tree — treat as checkout.

    A stamp without a valid ``updateMechanism`` is a build-lane bug and
    must not be silently read as "not sealed" (that is exactly the
    misclassification that made sealed trees look updatable) — same
    guard as hermes_cli.version_info._stamp_version_info.
    """
    if (project_root / ".git").exists():
        return False
    try:
        data = json.loads(
            (project_root / "install-stamp.json").read_text(encoding="utf-8-sig")
        )
    except (OSError, ValueError):
        return False
    if not (isinstance(data, dict) and bool(data)):
        return False
    if data.get("updateMechanism") not in UPDATE_MECHANISMS:
        raise RuntimeError(
            f"install-stamp.json at {project_root} is missing a valid "
            f"'updateMechanism' (one of {', '.join(UPDATE_MECHANISMS)}). The "
            "build lane that wrote this stamp must pass --update-mechanism to "
            "scripts/write_install_stamp.py."
        )
    return True


def check_runtime(project_root: Path) -> str | None:
    """One passive startup verdict; callers only choose stderr or logging."""
    import pm
    from hermes_cli.steward import sealed_steward

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
        pm.sync_venv(explicit=True, project_root=root)
        publish_launchers(root)
        return {"state": "synced", "ok": True}
    except Exception as exc:
        return {"state": "failed", "ok": False, "detail": str(exc)}


def prepare_launch(project_root: Path, argv: list[str]) -> Path | None:
    """Finish a self-managed source update before importing app dependencies.

    PM's successful input stamp is the only completion signal. Old updaters
    need not write a marker (and cannot accidentally clear this obligation).
    Return the store interpreter when this process must restart cleanly.
    """
    import os
    import sys
    from hermes_cli._parser import command_argv
    from hermes_cli.steward import read_install_stamp

    root = Path(project_root).resolve()
    if (command_argv(argv)[:1] == ["pm"]
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

    from hermes_cli._early_recovery import _marker_owner_is_live
    from hermes_cli.update_lock import read_live_update

    import pm
    from hermes_cli._launchers import resolve_store_python
    from pm.environments import activation_environment, runtime_facts_path

    current = pm.venv_is_current(project_root=root)
    if not current:
        # Existing markers guard liveness, never create the completion obligation.
        # Current post-sync verification children can boot under a live updater.
        legacy_markers = (root / ".update-incomplete", root / ".lazy-refresh-incomplete")
        if any(_marker_owner_is_live(marker) for marker in legacy_markers) or read_live_update():
            raise RuntimeError("an update is still running; wait for it to exit, then relaunch Hermes")
        print("hermes: completing source-update dependencies...", file=sys.stderr, flush=True)
        # Main-era installers selected [all] but had no PM ledger. Established
        # PM installs retain their recorded extras and plugin union instead.
        extras = ["all"] if not runtime_facts_path(root).is_file() else None
        pm.sync_venv(extras, explicit=True, project_root=root)
        # These can predate the swap. Once PM commits the replacement they
        # must not make early recovery immediately rebuild it a second time.
        for name in (".update-incomplete", ".lazy-refresh-incomplete"):
            (root / name).unlink(missing_ok=True)
        # Sync commits the dependency generation, but a source update also owes
        # the product builds and the post-build maintenance -- the tail every
        # install and finished update shares (hermes_cli/source_completion.py).
        # Those builds need PM's selected interpreter with its dependencies
        # activated, so hand that file THIS interpreter and let it re-exec
        # itself, exactly as the installers do.
        desktop_app = root / "apps/desktop"
        desktop = ((desktop_app / "dist/index.html").is_file()
                   or any((desktop_app / "release").glob("*")))
        code = subprocess.call(
            [sys.executable, "-I", "-B", "-u",
             str(root / "hermes_cli/source_completion.py"),
             "--source", str(root), "--finish-update",
             *(("--desktop",) if desktop else ())],
            cwd=root, env=activation_environment(root),
        )
        if code != 0:
            raise RuntimeError(
                "source update completion failed; run `hermes update` to finish it"
            )
    python = resolve_store_python(root)
    if python is None:
        raise RuntimeError("source update has no managed Python; run `hermes pm install`")
    if not current or python.absolute() != Path(sys.executable).absolute():
        publish_launchers(root)
        return python
    return None


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
        body = f"runpy.run_path({str(Path(argv[0]).absolute())!r}, run_name='__main__')"
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
