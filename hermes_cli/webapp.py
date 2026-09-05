"""Build and launch plumbing for the browser-hosted Desktop workspace.

The browser surface shares Hermes' hardened dashboard server. This module owns
only the separate renderer artifact and the one-way handoff into that server;
it never creates a second HTTP/WebSocket stack.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil
import time

from hermes_cli.main_desktop import _compute_desktop_content_hash
from hermes_cli.main_install_repair import _resolve_node_runtime_npm
from hermes_cli.main_tui_launch import _npm_lifecycle_env
from hermes_cli.main_web_build import _run_npm_install_deterministic, _run_with_idle_timeout


_DIST_NAME = "dist-webapp"
_STAMP_NAME = "desktop-webapp-build-stamp.json"
# Dashboard and Webapp both run root npm installs against the same workspace.
# One shared lock is therefore the authority for every browser-UI build.
_LOCK_NAME = ".web_ui_build.lock"
_LOCK_WAIT_SECONDS = 30 * 60


class WebappBuildError(RuntimeError):
    """The browser-hosted Desktop renderer could not be prepared."""


def webapp_dist_dir(project_root: Path) -> Path:
    return project_root / "apps" / "desktop" / _DIST_NAME


def _stamp_path() -> Path:
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / _STAMP_NAME


def _build_needed(project_root: Path, *, force: bool = False) -> bool:
    dist = webapp_dist_dir(project_root)
    if force or not (dist / "index.html").is_file():
        return True

    stamp = _stamp_path()
    try:
        payload = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True

    saved_hash = str(payload.get("contentHash") or "")
    return not saved_hash or saved_hash != _compute_desktop_content_hash(
        project_root
    )


def _write_stamp(project_root: Path) -> None:
    stamp = _stamp_path()
    payload = {
        "builtAt": datetime.now(timezone.utc).isoformat(),
        "contentHash": _compute_desktop_content_hash(project_root),
        "surface": "desktop-webapp",
    }
    stamp.parent.mkdir(parents=True, exist_ok=True)
    pending = stamp.with_suffix(".tmp")
    pending.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(pending, stamp)


def _workspace_install_args() -> tuple[str, ...]:
    """Install the existing JS closure without running native lifecycle scripts.

    ``npm ci`` prunes anything outside the selected workspace closure. Select
    the complete declared workspace graph plus the root so launching Webapp
    cannot silently remove another package's links before a later check/build.
    """
    return (
        "--workspaces",
        "--include-workspace-root",
        "--ignore-scripts",
        "--no-save",
        "--prefer-offline",
    )


def _try_file_lock(handle) -> bool:
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            getattr(msvcrt, "locking")(
                handle.fileno(), getattr(msvcrt, "LK_NBLCK"), 1
            )
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def _unlock_file(handle) -> None:
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            getattr(msvcrt, "locking")(
                handle.fileno(), getattr(msvcrt, "LK_UNLCK"), 1
            )
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


@contextmanager
def _exclusive_build_lock(path: Path):
    """Cross-platform exclusive lock for one renderer generation."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
    except OSError as exc:
        raise WebappBuildError(f"Could not open Webapp build lock {path}: {exc}") from exc

    windows = os.name == "nt"
    deadline = time.monotonic() + _LOCK_WAIT_SECONDS
    announced = False
    try:
        if windows:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()

        while True:
            if _try_file_lock(handle):
                break
            if time.monotonic() >= deadline:
                raise WebappBuildError(
                    f"Timed out waiting for another Webapp build ({path})"
                )
            if not announced:
                print("→ Another Hermes Webapp build is running; waiting for it...")
                announced = True
            time.sleep(0.1)

        yield
    finally:
        _unlock_file(handle)
        handle.close()


def _publish_dist(staging: Path, dist: Path) -> None:
    """Publish one complete renderer generation, restoring the old one on error."""
    backup = dist.with_name(f".dist-webapp-backup-{os.getpid()}-{secrets.token_hex(4)}")
    had_previous = dist.exists()
    preserve_backup = False
    try:
        if had_previous:
            os.replace(dist, backup)
        os.replace(staging, dist)
    except OSError as exc:
        if had_previous and backup.exists() and not dist.exists():
            try:
                os.replace(backup, dist)
            except OSError as restore_exc:
                preserve_backup = True
                raise WebappBuildError(
                    f"Could not publish Webapp renderer: {exc}. "
                    f"Restoring the prior renderer also failed; backup preserved at "
                    f"{backup}: {restore_exc}"
                ) from exc
        raise WebappBuildError(f"Could not publish Webapp renderer: {exc}") from exc
    finally:
        if backup.exists() and not preserve_backup:
            try:
                shutil.rmtree(backup)
            except OSError:
                pass


def _do_build(project_root: Path, *, force: bool) -> Path:
    dist = webapp_dist_dir(project_root)
    if not _build_needed(project_root, force=force):
        print(f"✓ Hermes Webapp renderer is up to date: {dist}")
        return dist

    desktop_dir = project_root / "apps" / "desktop"
    if not (desktop_dir / "package.json").is_file():
        raise WebappBuildError(f"Desktop workspace not found at {desktop_dir}")

    npm = _resolve_node_runtime_npm()
    if not npm:
        raise WebappBuildError(
            "Hermes Webapp needs Node.js/npm to build its Desktop renderer"
        )

    from hermes_constants import with_hermes_node_path

    install_env = _npm_lifecycle_env(with_hermes_node_path())
    install_env["ELECTRON_SKIP_BINARY_DOWNLOAD"] = "1"
    install_env["npm_config_ignore_scripts"] = "true"
    print("→ Installing the locked browser-renderer dependency closure...")
    installed = _run_npm_install_deterministic(
        npm,
        project_root,
        extra_args=_workspace_install_args(),
        capture_output=False,
        env=install_env,
    )
    if installed.returncode != 0:
        raise WebappBuildError(
            f"Browser-renderer dependency install failed (exit {installed.returncode})"
        )

    build_env = dict(install_env)
    build_env.pop("npm_config_ignore_scripts", None)
    print("→ Building the Hermes Desktop renderer for the browser...")
    staging = desktop_dir / f".dist-webapp-build-{os.getpid()}-{secrets.token_hex(4)}"
    try:
        built = _run_with_idle_timeout(
            [
                npm,
                "run",
                "--workspace",
                "apps/desktop",
                "build:webapp",
                "--",
                "--outDir",
                str(staging),
            ],
            cwd=project_root,
            env=build_env,
        )
        if built.returncode != 0 or not (staging / "index.html").is_file():
            raise WebappBuildError(
                f"Browser-hosted Desktop build failed (exit {built.returncode})"
            )
        _publish_dist(staging, dist)
    finally:
        if staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError:
                pass

    try:
        _write_stamp(project_root)
    except OSError as exc:
        # The artifact is authoritative; a read-only/contended cache directory
        # only means the next launch recomputes its content hash.
        print(f"⚠ Webapp renderer built, but its build stamp could not be saved: {exc}")
    print(f"✓ Hermes Webapp renderer built: {dist}")
    return dist


def prepare_webapp_renderer(
    project_root: Path,
    *,
    force: bool = False,
    skip_build: bool = False,
) -> Path:
    """Return a verified browser renderer, serializing concurrent builds."""
    project_root = project_root.resolve()
    dist = webapp_dist_dir(project_root)
    lock_path = project_root / _LOCK_NAME
    with _exclusive_build_lock(lock_path):
        if skip_build:
            if not (dist / "index.html").is_file():
                raise WebappBuildError(
                    f"--skip-build was passed but no Webapp renderer exists at {dist}"
                )
            print(f"→ Reusing Hermes Webapp renderer at {dist} (--skip-build)")
            return dist
        return _do_build(project_root, force=force)


def activate_webapp_dist(dist: Path) -> None:
    """Select the caller-managed Desktop bundle for the shared web server."""
    os.environ["HERMES_WEB_DIST"] = str(dist.resolve())
    os.environ.pop("HERMES_SERVE_HEADLESS", None)
