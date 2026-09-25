"""Shims to stop the old updater doing work until relaunch, not Android support."""

from pathlib import Path
from typing import NoReturn

from hermes_cli._old_updater import stop_for_relaunch

# Frozen data for old imports, not a download performed by this module.
PSUTIL_URL = (
    "https://files.pythonhosted.org/packages/aa/c6/"
    "d1ddf4abb55e93cebc4f2ed8b5d6dbad109ecb8d63748dd2b20ab5e57ebe/"
    "psutil-7.2.2.tar.gz"
)


def prepare_patched_psutil_sdist(archive: Path, destination: Path) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch. Extract nothing.
    stop_for_relaunch()


# Shim to stop the old updater doing work until relaunch. Historical callers
# download PSUTIL_URL BEFORE calling prepare_patched_psutil_sdist, so waiting
# for that call is too late. Stop the import without offering a fake URL.
stop_for_relaunch()
