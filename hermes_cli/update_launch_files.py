"""Rewrite the launch files Hermes owns before ``hermes update`` starts any gateway.

A service manager starts a gateway from the definition it holds, so a unit, plist or task script
left from the previous release starts the new code under the old command line and environment.
The update therefore regenerates every profile's launch files once the inputs the generators read
are final (Python environment, published launchers, ``node_modules/.bin``) and before the first
gateway start or restart.

Each file is independent: a failure is reported as the ``launch_files`` follow-up and the rest
are still rewritten. A failure never keeps the source-update tail pending: every
``hermes gateway start``/``restart`` and every gateway boot regenerates its own file again.
"""

from __future__ import annotations

import os
import sys

FOLLOWUP_STEP = "launch_files"
_RETRY = "the next `hermes update` or `hermes gateway restart` rewrites it"


def refresh_launch_files() -> None:
    """Regenerate every profile's launch file that differs from what this checkout generates."""
    from hermes_cli.config_backend import ConfigBackendUnavailable
    from hermes_cli.update_receipt import _profile_homes, record_followup
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    try:
        refreshers = _refreshers()
        homes = _profile_homes() if refreshers else []
    except Exception as exc:  # health: allow BLE001 -- reported as a follow-up, never fails the update
        record_followup(FOLLOWUP_STEP, f"{type(exc).__name__}: {exc}", retry=_RETRY)
        return
    for profile, home in homes:
        for kind, refresh in refreshers:
            token = set_hermes_home_override(home)
            try:
                refresh()
            except (Exception, ConfigBackendUnavailable) as exc:  # health: allow BLE001 -- one file never blocks the others
                record_followup(FOLLOWUP_STEP, f"{kind} for profile {profile}: {exc}", retry=_RETRY)
            finally:
                reset_hermes_home_override(token)
    if any(kind == "systemd user unit" for kind, _ in refreshers) and os.geteuid() == 0:  # windows-footgun: ok — systemd hosts only
        try:
            _refresh_systemd_system_unit()
        except (Exception, ConfigBackendUnavailable) as exc:  # health: allow BLE001 -- reported as a follow-up, never fails the update
            record_followup(FOLLOWUP_STEP, f"systemd system unit: {exc}", retry=_RETRY)


def _refresh_systemd_system_unit() -> None:
    """The invoking home's system unit; the refresh adopts the unit's ``HERMES_HOME`` into the
    environment, which the rest of the update must not inherit."""
    from hermes_cli.gateway import refresh_systemd_unit_if_needed

    launch_home = os.environ.get("HERMES_HOME")
    try:
        refresh_systemd_unit_if_needed(system=True)
    finally:
        if launch_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = launch_home


def _refreshers():
    if sys.platform == "win32":
        return [("Windows gateway launcher", _refresh_windows_launcher_scripts)]
    if sys.platform == "darwin":
        return [("launchd plist", _refresh_unloaded_launchd_plist)]
    from hermes_cli.gateway import supports_systemd_services

    if supports_systemd_services():
        return [("systemd user unit", _refresh_systemd_user_unit)]
    return []


def _refresh_systemd_user_unit() -> None:
    from hermes_cli.gateway import refresh_systemd_unit_if_needed

    refresh_systemd_unit_if_needed(system=False)  # rewrites only a stale unit, then daemon-reloads


def _refresh_unloaded_launchd_plist() -> None:
    from hermes_cli.gateway_launchd import refresh_unloaded_launchd_plist

    refresh_unloaded_launchd_plist()


def _refresh_windows_launcher_scripts() -> None:
    from hermes_cli import gateway_windows

    if gateway_windows.is_installed() and gateway_windows.launcher_script_drift():
        script = gateway_windows._write_task_script()
        print(f"  ↻ Updated the {script.stem} gateway launcher scripts")
