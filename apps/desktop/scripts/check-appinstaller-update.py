#!/usr/bin/env python3
"""Check for an App Installer update for the running MSIX package.

The out-of-store desktop installs are App Installer owned: the OS registered
the .appinstaller URI as the package's update source at install, so the
OS can tell us whether a newer package version is available. The desktop
shows its own prompt + tears down before the OS applies the swap.

Run with the BUNDLED payload python (the winrt package ships there):

    <payload>/tools/<python-entry>/python.exe scripts/check-appinstaller-update.py

Exit codes:
  0  no update available, or this process has no package identity
  2  update available (the caller decides whether to prompt/teardown)
  1  error, including a missing registered App Installer update source

API shape (pywinrt projection, verified against the installed winrt 3.2.1):
``Package.current`` is a static PROPERTY; the update check is the instance
method ``Package.check_update_availability_async`` — ``PackageManager`` has
no such method. ``PackageUpdateAvailability`` members (verified live in the
installed projection): UNKNOWN=0, NO_UPDATES=1, AVAILABLE=2, REQUIRED=3,
ERROR=4. Only the specific "no package identity" HRESULT (0x80073D54) means
not-packaged; every other failure is an unknown, never a "no update".
"""

import json
import sys

# HRESULT 0x80073D54 — "The process has no package identity." The one
# failure that positively identifies a non-packaged (dev) run.
_NO_PACKAGE_IDENTITY_HRESULT = -2147009196


def _load_projection():
    from winrt.windows.applicationmodel import Package, PackageUpdateAvailability

    return Package, PackageUpdateAvailability


def main() -> int:
    try:
        Package, PackageUpdateAvailability = _load_projection()
    except ImportError as exc:
        # The winrt module is absent (older payload without the dep): the
        # update check cannot run. Report unknown, not "no update".
        print(json.dumps({"available": None, "error": f"winrt import failed: {exc}"}))
        return 1

    # Property access, not a call: a packaged process gets the Package
    # instance; a dev run raises the package-identity HRESULT.
    try:
        package = Package.current
    except OSError as exc:
        if getattr(exc, "winerror", None) == _NO_PACKAGE_IDENTITY_HRESULT:
            print(json.dumps({"available": False, "reason": "not-packaged"}))
            return 0
        print(json.dumps({"available": None, "error": f"package identity failed: {exc}"}))
        return 1
    except Exception as exc:  # noqa: BLE001 — projection shape surprises are unknowns
        print(json.dumps({"available": None, "error": f"package identity failed: {exc}"}))
        return 1

    try:
        source = package.get_app_installer_info()
        if source is None:
            print(json.dumps({
                "available": None,
                "reason": "no-app-installer-source",
                "error": "No App Installer update source is registered. Install Hermes through its .appinstaller file to enable updates.",
            }))
            return 1
        source_uri = source.uri.absolute_uri
        result = package.check_update_availability_async().get()
        availability = PackageUpdateAvailability(result.availability)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"available": None, "error": f"check failed: {exc}"}))
        return 1

    # Only AVAILABLE / REQUIRED mean "a newer package exists". NO_UPDATES is
    # the one honest "no". UNKNOWN and ERROR are NOT "no update" — reporting
    # them as such would suppress a real update prompt — so they surface as
    # an unknown with the availability name as the extended error.
    if availability == PackageUpdateAvailability.NO_UPDATES:
        print(json.dumps({"available": False, "availability": availability.name, "source_uri": source_uri}))
        return 0

    if availability in (PackageUpdateAvailability.AVAILABLE, PackageUpdateAvailability.REQUIRED):
        print(json.dumps({"available": True, "availability": availability.name, "source_uri": source_uri}))
        return 2

    print(json.dumps({"available": None, "error": f"availability unknown: {availability.name}"}))
    return 1


if __name__ == "__main__":
    sys.exit(main())
