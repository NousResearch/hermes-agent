"""The macOS Bonjour host name (System Settings → Sharing → Local hostname).

``gethostname()`` on a Mac with no ``HostName`` set follows the network (``Mac``,
``Mac.localdomain``, ``<name>.local``); ``LocalHostName`` only changes when the user renames the
machine. Read from SystemConfiguration through ctypes without environment-variable input or
subprocesses.
"""

from __future__ import annotations

import ctypes

_UTF8 = 0x08000100  # kCFStringEncodingUTF8
_BUFFER_BYTES = 256  # LocalHostName is a DNS label: at most 63 bytes
_SYSTEM_CONFIGURATION = "/System/Library/Frameworks/SystemConfiguration.framework/SystemConfiguration"
_CORE_FOUNDATION = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"


def local_hostname() -> str:
    """``LocalHostName`` without the ``.local`` suffix, or ``""`` when unset or unreadable."""
    try:
        sc = ctypes.CDLL(_SYSTEM_CONFIGURATION)
        cf = ctypes.CDLL(_CORE_FOUNDATION)
    except OSError:
        return ""
    sc.SCDynamicStoreCopyLocalHostName.restype = ctypes.c_void_p
    sc.SCDynamicStoreCopyLocalHostName.argtypes = [ctypes.c_void_p]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    ref = sc.SCDynamicStoreCopyLocalHostName(None)
    if not ref:
        return ""
    try:
        buffer = ctypes.create_string_buffer(_BUFFER_BYTES)
        if not cf.CFStringGetCString(ref, buffer, _BUFFER_BYTES, _UTF8):
            return ""
        return buffer.value.decode(errors="replace").strip()
    finally:
        cf.CFRelease(ref)
