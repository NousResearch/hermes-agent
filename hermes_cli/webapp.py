"""Build and launch plumbing for the browser-hosted Desktop workspace.

The browser surface shares Hermes' hardened dashboard server. This module owns
only the separate renderer artifact and the one-way handoff into that server;
it never creates a second HTTP/WebSocket stack. The renderer is compiled by the
shared product builders (``apps/desktop/scripts/build-webapp.mjs``), whose
receipt inside ``dist-webapp`` decides freshness.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess

# Serialize renderer publication with Dashboard builds in this checkout.
_LOCK_NAME = ".web_ui_build.lock"
_LOCK_WAIT_SECONDS = 30 * 60


class WebappBuildError(RuntimeError):
    """The browser-hosted Desktop renderer could not be prepared."""


def webapp_dist_dir(project_root: Path) -> Path:
    return project_root / "apps" / "desktop" / "dist-webapp"


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
    from pm.filesystem import lock_fd

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
    except OSError as exc:
        raise WebappBuildError(f"Could not open Webapp build lock {path}: {exc}") from exc

    try:
        try:
            if not lock_fd(handle.fileno(), wait=False):
                print("→ Another Hermes Webapp build is running; waiting for it...")
                if not lock_fd(handle.fileno(), wait=True, timeout=_LOCK_WAIT_SECONDS):
                    raise WebappBuildError(f"Timed out waiting for another Webapp build ({path})")
        except OSError as exc:
            raise WebappBuildError(f"Could not lock Webapp build {path}: {exc}") from exc
        yield
    finally:
        _unlock_file(handle)
        handle.close()


def prepare_webapp_renderer(
    project_root: Path,
    *,
    force: bool = False,
    skip_build: bool = False,
    explicit: bool = False,
) -> Path:
    """Return a verified browser renderer, serializing concurrent builds.

    ``explicit`` (a requested build) may install dependencies even when lazy
    installs are disabled, like ``hermes desktop --build-only``.
    """
    from hermes_cli.source_build import build_source_webapp, source_build_env, source_product_current

    project_root = project_root.resolve()
    dist = webapp_dist_dir(project_root)
    with _exclusive_build_lock(project_root / _LOCK_NAME):
        if skip_build:
            if not (dist / "index.html").is_file():
                raise WebappBuildError(
                    f"--skip-build was passed but no Webapp renderer exists at {dist}"
                )
            print(f"→ Reusing Hermes Webapp renderer at {dist} (--skip-build)")
            return dist
        if not force and source_product_current(project_root, "webapp", dist):
            print(f"✓ Hermes Webapp renderer is up to date: {dist}")
            return dist
        try:
            build_source_webapp(project_root, env=source_build_env(explicit=explicit), explicit=explicit)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            raise WebappBuildError(f"Browser-hosted Desktop build failed: {exc}") from exc
        print(f"✓ Hermes Webapp renderer built: {dist}")
        return dist
