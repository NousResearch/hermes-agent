"""Shims to suppress old updater work until relaunch. Live identity uses version_info."""


def get_code_identity(refresh: bool = False) -> dict:
    # Shim to suppress old updater work until relaunch, not certify the new checkout.
    return {"sha": None, "short_sha": None, "version": None, "source": "unknown"}
