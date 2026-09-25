"""Where the running BUNDLED desktop app lives, and how to start it.

A bundled artifact runs its agent out of ``<app>/resources/agent-payload``,
so a Python process inside that payload sits at a known offset from the app
the user double-clicks. This module turns that offset into the launcher path
and starts it — the two things every in-bundle caller needs and each one
used to work out for itself.

The shape authority is the build stamp (``payload: bundled``, read through
:func:`hermes_cli.steward.is_bundled_payload`), never a filesystem probe: a
probe answers "is this artifact intact?", not "which shape am I?". Callers
gate on the stamp, then ask here WHERE the app is. A stamp that says bundled
over a tree that is not one is damage, and :func:`resolve_bundle_layout`
says so instead of degrading to a checkout.

Two consumers today: ``hermes desktop`` (start the app this CLI ships
inside) and the sealed self-updater (stop the app, then relaunch it).
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BundleLayout",
    "NotBundledApp",
    "PAYLOAD_DIR_NAME",
    "launch_detached",
    "resolve_bundle_layout",
]

#: The resources-resident payload directory every bundled artifact ships.
PAYLOAD_DIR_NAME = "agent-payload"

#: Executables that sit beside a Linux Electron launcher without being one.
#: AppRun is the AppImage entry stub, which re-execs the launcher below.
_LINUX_NON_LAUNCHERS = frozenset({"AppRun", "chrome-sandbox", "chrome_crashpad_handler"})


class NotBundledApp(Exception):
    """Raised when a tree is not a bundled desktop payload."""


@dataclass(frozen=True)
class BundleLayout:
    """The three directories and the launcher of a bundled desktop app.

    ``launcher`` is None when the app directory holds no single obvious
    executable. That is a real state on a damaged or unusual install, and
    both callers would rather report it than act on a guess.
    """

    app_root: Path
    resources: Path
    payload: Path
    launcher: Path | None


def resolve_bundle_layout(
    project_root: Path | str, *, platform: str | None = None
) -> BundleLayout:
    """Locate the desktop app that contains the payload at *project_root*.

    *project_root* is the payload's ``repo/`` tree — the install root a
    bundled backend runs from. The app sits two directories above the
    payload on Windows and Linux, and four above it on macOS, where the
    payload lands in ``Hermes.app/Contents/Resources``.

    The macOS offset is detected from the directory names themselves
    rather than from *platform*, so a layout can be resolved for any host
    from any host. *platform* (default :data:`sys.platform`) only decides
    the launcher-naming rule.

    Raises :class:`NotBundledApp` when the tree is not a bundled payload —
    a source checkout, or the docker/nix sealed shape, which has no app.
    """
    repo = Path(project_root).resolve()
    payload = repo.parent
    if payload.name != PAYLOAD_DIR_NAME:
        raise NotBundledApp(
            f"{repo} is not inside a {PAYLOAD_DIR_NAME}/ payload — no desktop app contains it"
        )

    resources = payload.parent
    app_root = resources.parent
    # macOS: .../Hermes.app/Contents/Resources/agent-payload/repo
    if (
        resources.name == "Resources"
        and app_root.name == "Contents"
        and app_root.parent.suffix == ".app"
    ):
        app_root = app_root.parent

    if not app_root.is_dir():
        raise NotBundledApp(f"no app directory above the payload at {payload}")

    return BundleLayout(
        app_root=app_root,
        resources=resources,
        payload=payload,
        launcher=_resolve_launcher(app_root, platform or sys.platform),
    )


def _resolve_launcher(app_root: Path, platform: str) -> Path | None:
    """The executable that starts the app, or None when it is ambiguous.

    One rule per platform, each one "the single launcher among files that
    are not launchers":

    * macOS — the one file in ``Contents/MacOS``.
    * Windows — the one top-level ``.exe`` that is not the uninstaller.
    * Linux — the one top-level executable that is not an Electron helper
      or a shared library.
    """
    if platform == "darwin":
        return _only(_files(app_root / "Contents" / "MacOS"))

    if platform == "win32":
        return _only(
            [
                p
                for p in _files(app_root)
                if p.suffix.lower() == ".exe"
                and not p.name.lower().startswith("uninstall")
            ]
        )

    return _only(
        [
            p
            for p in _files(app_root)
            if p.name not in _LINUX_NON_LAUNCHERS
            and ".so" not in p.name
            and os.access(p, os.X_OK)
        ]
    )


def _files(directory: Path) -> list[Path]:
    """Regular files directly in *directory*; empty when it is unreadable."""
    try:
        return sorted(p for p in directory.iterdir() if p.is_file())
    except OSError:
        return []


def _only(candidates: list[Path]) -> Path | None:
    return candidates[0] if len(candidates) == 1 else None


def launch_detached(
    argv: list[str], *, env: dict[str, str] | None = None, cwd: Path | str | None = None
) -> int:
    """Start *argv* as its own process and return its pid.

    The child outlives this process: it leads a new session on POSIX, and
    on Windows it gets the detach creation flags (new process group, own
    hidden console, breakaway from any job object). Its stdio goes to the
    null device, because the terminal that ran ``hermes desktop`` is free
    to close the moment this call returns.
    """
    from hermes_cli._subprocess_compat import windows_detach_popen_kwargs

    child = subprocess.Popen(
        argv,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        **windows_detach_popen_kwargs(),
    )
    return child.pid
