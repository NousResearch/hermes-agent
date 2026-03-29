"""Display utilities — theme detection and terminal rendering helpers.

Provides OS-level theme detection for adapting color schemes to the user's
system preference (dark mode vs light mode).
"""

import os
import sys


def get_system_theme() -> str:
    """Detect the OS theme preference. Returns 'dark', 'light', or 'unknown'.

    - Windows: reads HKCU registry AppsUseLightTheme (requires Windows 10 1607+)
    - macOS: runs 'defaults read -g AppleInterfaceStyle'
    - Linux: checks GTK_THEME env var (best-effort)
    - Falls back to 'dark' on any error (dark is the safer default for terminals)
    """
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            )
            val, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            winreg.CloseKey(key)
            return "light" if val else "dark"

        if sys.platform == "darwin":
            import subprocess
            r = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True, timeout=2
            )
            return "dark" if "Dark" in r.stdout else "light"

        # Linux — check GTK_THEME
        gtk = os.environ.get("GTK_THEME", "")
        if "dark" in gtk.lower():
            return "dark"
        if "light" in gtk.lower():
            return "light"

    except Exception:
        pass

    return "dark"  # safe default for terminals
