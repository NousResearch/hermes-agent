"""PCI display controllers on macOS, read from the IOKit registry.

Intel Macs name their GPU vendors here; Apple silicon has no PCI display controller. Read through
ctypes without environment-variable input or subprocesses.
"""

from __future__ import annotations

import ctypes
import ctypes.util

_UTF8 = 0x08000100  # kCFStringEncodingUTF8
_DISPLAY_CLASS = 0x03  # PCI base class: display controller
_MAX_DEVICES = 256


def _data_int(cf, ref: int | None) -> int | None:
    """Little-endian integer from a CFData property, or ``None`` for a missing or non-data value."""
    if not ref:
        return None
    try:
        if cf.CFGetTypeID(ref) != cf.CFDataGetTypeID():
            return None
        size = cf.CFDataGetLength(ref)
        return int.from_bytes(ctypes.string_at(cf.CFDataGetBytePtr(ref), size), "little") if size else None
    finally:
        cf.CFRelease(ref)


def _bind(iokit, cf) -> None:
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    cf.CFGetTypeID.restype = cf.CFDataGetTypeID.restype = ctypes.c_ulong
    cf.CFGetTypeID.argtypes = [ctypes.c_void_p]
    cf.CFDataGetLength.restype = ctypes.c_long
    cf.CFDataGetLength.argtypes = [ctypes.c_void_p]
    cf.CFDataGetBytePtr.restype = ctypes.c_void_p
    cf.CFDataGetBytePtr.argtypes = [ctypes.c_void_p]
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    iokit.IOServiceMatching.restype = ctypes.c_void_p
    iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
    iokit.IOServiceGetMatchingServices.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    iokit.IOIteratorNext.restype = ctypes.c_uint32
    iokit.IOIteratorNext.argtypes = [ctypes.c_uint32]
    iokit.IORegistryEntryCreateCFProperty.restype = ctypes.c_void_p
    iokit.IORegistryEntryCreateCFProperty.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    iokit.IOObjectRelease.argtypes = [ctypes.c_uint32]


def display_vendor_ids() -> list[str] | None:
    """Hex PCI vendor ids of every display controller, or ``None`` when IOKit cannot be read."""
    iokit = ctypes.CDLL(ctypes.util.find_library("IOKit"))
    cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    _bind(iokit, cf)
    iterator = ctypes.c_uint32()
    # The matching dictionary is consumed by the call; port 0 is kIOMainPortDefault.
    if iokit.IOServiceGetMatchingServices(0, iokit.IOServiceMatching(b"IOPCIDevice"), ctypes.byref(iterator)) != 0:
        return None
    keys = {name: cf.CFStringCreateWithCString(None, name.encode(), _UTF8) for name in ("class-code", "vendor-id")}
    vendors: list[str] = []
    try:
        for _ in range(_MAX_DEVICES):
            entry = iokit.IOIteratorNext(iterator)
            if not entry:
                break
            try:
                read = {name: _data_int(cf, iokit.IORegistryEntryCreateCFProperty(entry, key, None, 0))
                        for name, key in keys.items()}
            finally:
                iokit.IOObjectRelease(entry)
            if read["class-code"] is not None and read["class-code"] >> 16 == _DISPLAY_CLASS:
                vendors.append("" if read["vendor-id"] is None else f"{read['vendor-id'] & 0xFFFF:04x}")
    finally:
        iokit.IOObjectRelease(iterator.value)
        for key in keys.values():
            cf.CFRelease(key)
    return vendors
