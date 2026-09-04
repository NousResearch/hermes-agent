#!/usr/bin/env python3
"""Check for an App Installer update for the running MSIX package.

The out-of-store desktop installs are App Installer owned: the OS registered
the .appinstaller URI as the package's update source at install, so the
OS can tell us whether a newer package version is available. The desktop
shows its own prompt + tears down before the OS applies the swap.

Run with the BUNDLED payload python (the winrt package ships there):

    <payload>/tools/<python-entry>/python.exe scripts/check-appinstaller-update.py

Exit codes:
  0  no update available (or no App Installer source — nothing to do)
  2  update available (the caller decides whether to prompt/teardown)
  1  error (update availability unknown; the caller reports it honestly)
"""

import json
import sys


def main() -> int:
    try:
        from winrt.windows.applicationmodel import Package, PackageUpdateAvailability
        from winrt.windows.management.deployment import PackageManager
    except ImportError as exc:
        # The winrt module is absent (older payload without the dep): the
        # update check cannot run. Report unknown, not "no update".
        print(json.dumps({"available": None, "error": f"winrt import failed: {exc}"}))
        return 1

    try:
        package = Package.current()
    except Exception as exc:  # noqa: BLE001 — the API raises on non-packaged runs
        # Not running as a packaged app (dev run) — nothing to check.
        print(json.dumps({"available": False, "reason": "not-packaged"}))
        return 0

    package_family_name = package.id.family_name
    manager = PackageManager()
    try:
        result = manager.check_package_update_availability_async(package_family_name).get()
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"available": None, "error": f"check failed: {exc}"}))
        return 1

    availability = PackageUpdateAvailability(result.availability)
    # Available / Required both mean "a newer package exists on the source".
    available = availability in (
        PackageUpdateAvailability.AVAILABLE,
        PackageUpdateAvailability.REQUIRED,
    )
    print(json.dumps({"available": available, "availability": str(availability)}))
    return 2 if available else 0


if __name__ == "__main__":
    sys.exit(main())
