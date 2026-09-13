"""Detect electron-winstaller Unix install-script failures."""

import sys


def _is_electron_winstaller_unix_failure(result) -> bool:
    if sys.platform == "win32":
        return False
    if getattr(result, "returncode", 1) == 0:
        return False
    text = (result.stderr or "") + (result.stdout or "")
    if "select-7z-arch.js" in text:
        return True
    if "7z-x64.exe" in text:
        return True
    if "electron-winstaller" in text and "ENOENT" in text:
        return True
    return False
