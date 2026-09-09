"""
Platform Adapter Factory and Detector.
"""

from typing import Optional
import os
import sys
import platform

from hermes_platform.common.adapter import PlatformAdapter, PlatformType
from hermes_platform.windows.adapter import WindowsAdapter
from hermes_platform.linux.adapter import LinuxAdapter
from hermes_platform.macos.adapter import MacOSAdapter
from hermes_platform.android.adapter import AndroidAdapter
from hermes_platform.termux.adapter import TermuxAdapter
from hermes_platform.wsl.adapter import WSLAdapter
from hermes_platform.remote.adapter import RemoteAdapter


def detect_platform_type() -> PlatformType:
    if os.environ.get("HERMES_PLATFORM_OVERRIDE"):
        try:
            return PlatformType(os.environ["HERMES_PLATFORM_OVERRIDE"].lower())
        except ValueError:
            pass

    if "TERMUX_VERSION" in os.environ or os.path.exists("/data/data/com.termux"):
        return PlatformType.TERMUX

    if "ANDROID_ROOT" in os.environ:
        return PlatformType.ANDROID

    if sys.platform == "win32":
        return PlatformType.WINDOWS

    if sys.platform == "darwin":
        return PlatformType.MACOS

    if sys.platform.startswith("linux"):
        rel = platform.release().lower() if hasattr(platform, "release") else ""
        if "microsoft-standard" in rel or "wsl" in rel:
            return PlatformType.WSL
        return PlatformType.LINUX

    return PlatformType.LINUX


def get_platform_adapter(platform_type: Optional[PlatformType] = None) -> PlatformAdapter:
    if platform_type is None:
        platform_type = detect_platform_type()

    if platform_type == PlatformType.WINDOWS:
        return WindowsAdapter()
    elif platform_type == PlatformType.MACOS:
        return MacOSAdapter()
    elif platform_type == PlatformType.TERMUX:
        return TermuxAdapter()
    elif platform_type == PlatformType.ANDROID:
        return AndroidAdapter()
    elif platform_type == PlatformType.WSL:
        return WSLAdapter()
    elif platform_type == PlatformType.REMOTE:
        return RemoteAdapter()
    else:
        return LinuxAdapter()
