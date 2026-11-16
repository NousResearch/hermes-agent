"""Windows graphics adapters as DXGI reports them.

Task Manager's "GPU memory" for an adapter is DXGI's dedicated video memory plus shared system
memory, and that sum is what Windows lets the adapter allocate. An integrated GPU has almost no
dedicated memory, so its shared allowance is its real pool: an Intel Arc B390 on a 31.5 GB machine
gets 18 GB. Read through ctypes without environment-variable input or subprocesses.
"""

from __future__ import annotations

import ctypes
import functools
from contextlib import suppress
from ctypes import wintypes
from dataclasses import dataclass

_SOFTWARE_FLAG = 0x2  # DXGI_ADAPTER_FLAG_SOFTWARE (Microsoft Basic Render Driver)
_MAX_ADAPTERS = 16
# vtable slots: IUnknown::Release, IDXGIFactory1::EnumAdapters1, IDXGIAdapter1::GetDesc1.
_RELEASE, _ENUM_ADAPTERS1, _GET_DESC1 = 2, 12, 10


@dataclass(frozen=True)
class Adapter:
    description: str
    vendor_id: int
    dedicated_bytes: int
    shared_bytes: int
    software: bool

    @property
    def memory_bytes(self) -> int:
        """What Windows lets this adapter allocate (Task Manager's GPU memory)."""
        return self.dedicated_bytes + self.shared_bytes


class _Guid(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8)]


class _Luid(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _AdapterDesc1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", wintypes.UINT),
                ("DeviceId", wintypes.UINT), ("SubSysId", wintypes.UINT), ("Revision", wintypes.UINT),
                ("DedicatedVideoMemory", ctypes.c_size_t), ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t), ("AdapterLuid", _Luid), ("Flags", wintypes.UINT)]


_IID_DXGI_FACTORY1 = _Guid(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))


def _method(obj: ctypes.c_void_p, slot: int, *argtypes):
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[slot])


@functools.cache
def windows_gpu_adapters() -> tuple[Adapter, ...]:
    """Every adapter DXGI enumerates, in its order; empty off Windows or when DXGI can't answer."""
    with suppress(OSError, AttributeError):
        factory = ctypes.c_void_p()
        if ctypes.WinDLL("dxgi").CreateDXGIFactory1(ctypes.byref(_IID_DXGI_FACTORY1), ctypes.byref(factory)) != 0:
            return ()
        found: list[Adapter] = []
        try:
            enum = _method(factory, _ENUM_ADAPTERS1, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))
            for index in range(_MAX_ADAPTERS):
                adapter = ctypes.c_void_p()
                if enum(factory, index, ctypes.byref(adapter)) != 0:  # DXGI_ERROR_NOT_FOUND ends the list
                    break
                try:
                    desc = _AdapterDesc1()
                    if _method(adapter, _GET_DESC1, ctypes.POINTER(_AdapterDesc1))(adapter, ctypes.byref(desc)) == 0:
                        found.append(Adapter(desc.Description, desc.VendorId, desc.DedicatedVideoMemory,
                                             desc.SharedSystemMemory, bool(desc.Flags & _SOFTWARE_FLAG)))
                finally:
                    _method(adapter, _RELEASE)(adapter)
        finally:
            _method(factory, _RELEASE)(factory)
        return tuple(found)
    return ()
