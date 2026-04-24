"""
Platform singleton detecting OS.
"""

import os
import platform

class PlatformInfo:
    def __init__(self):
        self.os_name = platform.system()
        self.is_windows = self.os_name == "Windows"
        self.is_linux = self.os_name == "Linux"
        self.is_macos = self.os_name == "Darwin"

        # Termux detection via PREFIX environment variable
        self.is_termux = "com.termux" in os.environ.get("PREFIX", "")

        self.is_wsl = False
        if self.is_linux:
            try:
                with open('/proc/version', 'r') as f:
                    if 'microsoft' in f.read().lower():
                        self.is_wsl = True
            except IOError:
                pass

platform_info = PlatformInfo()