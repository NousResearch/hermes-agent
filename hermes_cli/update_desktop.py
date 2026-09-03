"""Desktop-presence, launcher-refresh, and rebuild operations.

Extracted mechanically from :mod:`hermes_cli.update_cmd`.  Runtime
references to the historical module surface resolve through the
compatibility facade so imports and monkeypatches remain effective.
"""

from pathlib import Path


def _desktop_app_present(desktop_dir: Path) -> bool:
    """Return whether a packaged or source Desktop build exists."""
    return (
        _m()._desktop_packaged_executable(desktop_dir) is not None
        or _m()._desktop_dist_exists(desktop_dir)
    )


def _rebuild_desktop_after_update(
    desktop_dir: Path, *, had_desktop_app_before_update: bool
) -> bool:
    """Rebuild an installed Desktop app when its source or artifact changed.

    Returns ``False`` only when a rebuild was attempted and failed, so the
    caller can withhold ``✓ Update complete!`` and (in gateway mode) write
    a failing ``.update_exit_code`` (#88251). Every other outcome — nothing
    to rebuild, up to date, build succeeded, Desktop never installed —
    returns ``True``.
    """
    # The release tree is ignored by git and can disappear during an update.
    # Its pre-update presence is enough to restore it; do not make people who
    # have never used Desktop pay for an Electron build.
    has_desktop_app = had_desktop_app_before_update or _desktop_app_present(desktop_dir)
    if not (
        (desktop_dir / "package.json").exists()
        and _m()._resolve_node_runtime_npm()
        and has_desktop_app
    ):
        return True

    print("→ Checking if desktop app needs rebuilding...")
    # Consult the content-hash stamp IN-PROCESS first. The spawned
    # `hermes desktop --build-only` subprocess re-imports the whole CLI stack
    # (~1-3 s) just to reach the same _m()._desktop_build_needed check; when
    # the stamp already says "up to date" we can skip the spawn entirely. The
    # update path never passes --source, so the subprocess would run with
    # source_mode=False — mirror that here. Any error in the pre-check falls
    # through to the subprocess.
    skip_desktop_build = False
    try:
        skip_desktop_build = not _m()._desktop_build_needed(
            desktop_dir, _m().PROJECT_ROOT, source_mode=False
        )
    except Exception:
        skip_desktop_build = False
    if skip_desktop_build:
        print("  ✓ Desktop app up to date")
        return True

    desktop_build_cmd = [sys.executable, "-m", "hermes_cli.main", "desktop", "--build-only"]
    # Capture the (very loud) Electron/vite build output into update.log
    # instead of streaming it to the terminal. On the rare nonzero exit,
    # retry once after waiting again for the venv — this covers a
    # still-settling rebuild window the first wait didn't fully catch — then
    # surface the captured tail so the failure is debuggable.
    #
    # Start the build subprocess with the Hermes-managed Node on PATH: when
    # `hermes update` runs inside the desktop updater chain (Desktop →
    # hermes-setup → hermes update), the shell PATH customizations are lost,
    # so a bare-PATH child would fail with `node: not found` before cmd_gui can
    # self-heal.
    from hermes_constants import with_hermes_node_path

    build_env = with_hermes_node_path()
    build_result = _m()._run_logged_subprocess(
        desktop_build_cmd, cwd=_m().PROJECT_ROOT, env=build_env
    )
    if build_result.returncode != 0:
        build_result = _m()._run_logged_subprocess(
            desktop_build_cmd, cwd=_m().PROJECT_ROOT, env=build_env
        )
    if build_result.returncode != 0:
        print("  ⚠ Desktop build failed (run `hermes desktop` to retry)")
        tail = "\n".join((build_result.stdout or "").strip().splitlines()[-15:])
        if tail:
            print(tail)
        from hermes_constants import display_hermes_home as _dhh

        print(f"  Full build log: {_dhh()}/logs/update.log")
        return False
    print("  ✓ Desktop app up to date")
    return True
