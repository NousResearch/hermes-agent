"""Pre-suffix Windows gateway launchers: the objects the current per-profile names cannot reach.

Before task/launcher names carried the profile suffix, an install wrote the Scheduled Task
``Hermes_Gateway``, the Startup entry ``Hermes_Gateway.vbs``/``.cmd`` and the launcher pair
``gateway-service\\Hermes_Gateway.{vbs,cmd}``. Every operation in ``windows_service`` is keyed on
``get_task_name()`` → ``Hermes_Gateway_<suffix>``, so those strays are never queried, rewritten,
reported or removed: they keep launching a second gateway at logon, never pick up launcher
fixes, and ``hermes gateway status`` prints ✓ while they do (#116157). This module enumerates
them once; ``status`` warns, ``uninstall`` and ``install --force`` remove.
"""

from __future__ import annotations

from pathlib import Path


def _w():
    import gateway.windows_service as windows_service

    return windows_service


def _targets_this_home(text: str) -> bool:
    """A bare-named launcher is OURS only when its action points into THIS home's ``gateway-service``
    dir. The bare name is also the live identity of the default ``~/.hermes`` profile, so a secondary
    profile that classified every ``Hermes_Gateway`` object as its own stray would delete the default
    profile's gateway autostart on ``uninstall`` / ``install --force``."""
    w = _w()
    marker = w._normalize_windows_path(str(w._hermes_home() / "gateway-service"))
    return marker in w._normalize_windows_path(text)


def legacy_launcher_artifacts() -> list[tuple[str, str, Path | str]]:
    """``(kind, label, target)`` for every pre-suffix launcher of THIS home still present; ``kind`` is
    ``"file"`` or ``"task"``. Empty when this home owns the bare name (its current objects ARE the bare
    ones). Bare-named objects belonging to another home (a sibling install) are left alone."""
    w = _w()
    bare = w._TASK_NAME_DEFAULT
    if w.get_task_name() == bare:
        return []
    found: list[tuple[str, str, Path | str]] = []
    startup = w._startup_dir()
    service_dir = w._hermes_home() / "gateway-service"
    for path, label in (
        (startup / f"{bare}.vbs", "legacy pre-suffix Windows login item"),
        (startup / f"{bare}.cmd", "legacy pre-suffix Windows login item"),
    ):
        try:
            if _targets_this_home(path.read_text(encoding="utf-8", errors="replace")):
                found.append(("file", label, path))
        except OSError:
            continue
    for path, label in (
        (service_dir / f"{bare}.vbs", "legacy pre-suffix task launcher"),
        (service_dir / f"{bare}.cmd", "legacy pre-suffix task script"),
    ):
        if path.exists():  # inside this home by construction
            found.append(("file", label, path))
    code, out, _err = w._exec_schtasks(["/Query", "/TN", bare, "/XML"])
    if code == 0 and _targets_this_home(out):
        found.append(("task", "legacy pre-suffix Scheduled Task", bare))
    return found


def warn_legacy_launchers() -> bool:
    """``hermes gateway status``: name every stray so nobody has to do file-mtime archaeology."""
    artifacts = legacy_launcher_artifacts()
    for _kind, label, target in artifacts:
        print(f"⚠ {label} still installed: {target}")
    if artifacts:
        print("  These predate per-profile launcher names and also start the gateway at logon.")
        print("  Remove them with: hermes gateway uninstall   (then: hermes gateway install)")
    return bool(artifacts)


def remove_legacy_launchers() -> None:
    """``uninstall`` / ``install --force``: delete the strays; a task that needs elevation is named
    with the exact elevated command instead of being silently left behind."""
    w = _w()
    for kind, label, target in legacy_launcher_artifacts():
        if kind == "file":
            try:
                Path(target).unlink()
                print(f"✓ Removed {label}: {target}")
            except OSError as exc:
                print(f"⚠ Could not remove {label} {target}: {exc}")
            continue
        code, _out, err = w._exec_schtasks(["/Delete", "/F", "/TN", str(target)])
        if code == 0:
            print(f"✓ Removed {label} {str(target)!r}")
        elif w._is_access_denied(err.strip()):
            print(f"⚠ {label} {str(target)!r} needs an elevated shell to remove; run as administrator:")
            print(f"    schtasks /Delete /F /TN {target}")
        else:
            print(f"⚠ schtasks /Delete {str(target)!r} returned code {code}: {err.strip()}")
