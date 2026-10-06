"""Cached OS, architecture, CPU, memory and GPU facts for the current machine.

Native architecture remains accurate when the Python process is emulated.
"""

from __future__ import annotations

import functools
import os
import platform
import sys

_ARCH_NAMES = {
    "amd64": "amd64", "x86_64": "amd64", "x64": "amd64",
    "arm64": "arm64", "aarch64": "arm64",
    "x86": "x86", "i386": "x86", "i686": "x86",
}
IMAGE_FILE_MACHINE = {0x8664: "amd64", 0xAA64: "arm64", 0x014C: "x86"}


def normalize_arch(raw: str | None) -> str:
    """Return a canonical architecture name."""
    return _ARCH_NAMES.get((raw or "").strip().lower(), "unknown")


@functools.cache
def os_family() -> str:
    """Return the current platform identifier."""
    return sys.platform


@functools.cache
def process_arch() -> str:
    """Return the architecture of this Python process."""
    return normalize_arch(platform.machine())


def windows_native_arch(*, wow64_native: int | None, machine: str, env_arch: str | None) -> str:
    """Return the native architecture from Windows observations.

    Order matters: ``PROCESSOR_ARCHITECTURE`` reads ``AMD64`` inside an x64-emulated process
    on ARM64 hardware, so the environment is consulted only when both APIs are unavailable.
    """
    name = IMAGE_FILE_MACHINE.get(wow64_native or 0)
    if name:
        return name
    normalized = normalize_arch(machine)
    if normalized != "unknown":
        return normalized
    return normalize_arch(env_arch)


def _wow64_native_machine() -> int | None:
    """Return the IsWow64Process2 native machine, or ``None`` when unavailable."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    probe = getattr(kernel32, "IsWow64Process2", None)
    if probe is None:
        return None
    # ctypes otherwise truncates the HANDLE pseudo-handle to c_int.
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    probe.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.USHORT), ctypes.POINTER(wintypes.USHORT)]
    probe.restype = wintypes.BOOL
    process_machine = wintypes.USHORT(0)
    native_machine = wintypes.USHORT(0)
    if not probe(kernel32.GetCurrentProcess(), ctypes.byref(process_machine), ctypes.byref(native_machine)):
        return None
    return native_machine.value or None


def _darwin_translated() -> bool:
    """Return whether this process runs under Rosetta."""
    return _sysctl_int(b"sysctl.proc_translated") == 1


@functools.cache
def native_arch() -> str:
    """Return the machine architecture even when this process is emulated."""
    if sys.platform == "win32":
        try:
            wow64 = _wow64_native_machine()
        except (OSError, AttributeError, TypeError, ValueError):
            wow64 = None
        env_arch = os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE")
        return windows_native_arch(wow64_native=wow64, machine=platform.machine(), env_arch=env_arch)
    if sys.platform == "darwin" and process_arch() == "amd64" and _darwin_translated():
        return "arm64"
    return process_arch()


_CPU_KEY = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"


def parse_cpuinfo(text: str, *, device_tree_model: str = "") -> str:
    """Return the first available CPU or device-tree model."""
    for key in ("model name", "Hardware"):
        for line in text.splitlines():
            if line.lower().startswith(key.lower()) and ":" in line:
                return line.split(":", 1)[1].strip()
    return device_tree_model.replace("\0", "").replace("_", " ").strip()


def _read_text(path: str, limit: int = 65536) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)
    except OSError:
        return ""


def _winreg_str(subkey: str, name: str) -> str:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, subkey) as key:
            value = winreg.QueryValueEx(key, name)[0]
    except OSError:
        return ""
    return str(value).strip()


def _sysctl_int(name: bytes) -> int | None:
    import ctypes
    import ctypes.util

    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    value = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
    if libc.sysctlbyname(name, ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
        return None
    return value.value


def _sysctl_u64(name: bytes) -> int | None:
    import ctypes
    import ctypes.util

    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    value = ctypes.c_uint64(0)
    size = ctypes.c_size_t(ctypes.sizeof(ctypes.c_uint64))
    if libc.sysctlbyname(name, ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
        return None
    return value.value


def _sysctl_str(name: bytes) -> str:
    import ctypes
    import ctypes.util

    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    size = ctypes.c_size_t(0)
    if libc.sysctlbyname(name, None, ctypes.byref(size), None, 0) != 0:
        return ""
    buffer = ctypes.create_string_buffer(size.value)
    if libc.sysctlbyname(name, buffer, ctypes.byref(size), None, 0) != 0:
        return ""
    return buffer.value.decode(errors="replace").strip()


@functools.cache
def cpu_model() -> str:
    """Return the CPU model, or an empty string when unavailable."""
    if sys.platform == "win32":
        return _winreg_str(_CPU_KEY, "ProcessorNameString")
    if sys.platform == "darwin":
        return _sysctl_str(b"machdep.cpu.brand_string")
    return parse_cpuinfo(_read_text("/proc/cpuinfo"), device_tree_model=_read_text("/proc/device-tree/model", 512))


@functools.cache
def cpu_vendor() -> str:
    """Return the CPU vendor, or an empty string when unavailable."""
    if sys.platform == "win32":
        return _winreg_str(_CPU_KEY, "VendorIdentifier")
    if sys.platform.startswith("linux"):
        for line in _read_text("/proc/cpuinfo").splitlines():
            if line.lower().startswith("vendor_id") and ":" in line:
                return line.split(":", 1)[1].strip()
    return ""


def parse_meminfo_total(text: str) -> int | None:
    """Return ``MemTotal`` from ``/proc/meminfo`` text, in bytes."""
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                return int(fields[1]) * 1024
    return None


def _windows_ram_total() -> int | None:
    import ctypes
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys) or None


@functools.cache
def ram_total_bytes() -> int | None:
    """Return the physical memory visible to this OS, or ``None`` when unavailable."""
    try:
        if sys.platform == "win32":
            return _windows_ram_total()
        if sys.platform == "darwin":
            return _sysctl_u64(b"hw.memsize") or None
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return parse_meminfo_total(_read_text("/proc/meminfo", 4096))


# PCI vendor ids; a discrete vendor wins over an integrated one on hybrid machines.
_GPU_PCI_VENDORS = {"10de": "nvidia", "1002": "amd", "8086": "intel"}
_GPU_PRIORITY = ("nvidia", "amd", "intel")
_DISPLAY_CLASS_KEY = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
_GPU_SCAN_LIMIT = 16


def classify_gpu_vendors(vendor_ids: list[str]) -> str:
    """Return the highest-priority GPU class for PCI vendor ids, or ``none``."""
    found = {_GPU_PCI_VENDORS.get(v.strip().lower().removeprefix("0x")) for v in vendor_ids}
    return next((name for name in _GPU_PRIORITY if name in found), "none")


def _linux_gpu_vendor_ids() -> list[str] | None:
    """PCI vendor ids of DRM cards; ``None`` when sysfs exposes no DRM tree at all."""
    root = "/sys/class/drm"
    try:
        entries = sorted(os.listdir(root))[:_GPU_SCAN_LIMIT * 4]
    except OSError:
        return None
    cards = [e for e in entries if e.startswith("card") and "-" not in e][:_GPU_SCAN_LIMIT]
    return [_read_text(f"{root}/{card}/device/vendor", 16).strip() for card in cards]


def _windows_gpu_vendor_ids() -> list[str] | None:
    import winreg

    vendors = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS_KEY) as root:
            for index in range(_GPU_SCAN_LIMIT):
                try:
                    subkey = winreg.EnumKey(root, index)
                except OSError:
                    break
                device_id = _winreg_str(f"{_DISPLAY_CLASS_KEY}\\{subkey}", "MatchingDeviceId").upper()
                if "VEN_" in device_id:
                    vendors.append(device_id.split("VEN_", 1)[1][:4])
    except OSError:
        return None
    return vendors


@functools.cache
def gpu_class() -> str:
    """Return ``nvidia``/``amd``/``intel``/``apple_silicon``/``none``, or ``unknown``.

    Reads sysfs, the registry or sysctl only: never a subprocess or a driver library.
    """
    try:
        if sys.platform == "darwin":
            return "apple_silicon" if native_arch() == "arm64" else "unknown"
        if sys.platform == "win32":
            vendors = _windows_gpu_vendor_ids()
            return "unknown" if vendors is None else classify_gpu_vendors(vendors)
    except (AttributeError, OSError, TypeError, ValueError):
        return "unknown"
    if os.path.exists("/proc/driver/nvidia/version"):
        return "nvidia"
    vendors = _linux_gpu_vendor_ids()
    return "unknown" if vendors is None else classify_gpu_vendors(vendors)


def _windows_interactive_session() -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    process_id = kernel32.GetCurrentProcessId()
    session_id = wintypes.DWORD()
    if not kernel32.ProcessIdToSessionId(process_id, ctypes.byref(session_id)) or session_id.value == 0:
        return False
    active_session = kernel32.WTSGetActiveConsoleSessionId()
    if active_session != 0xFFFFFFFF and active_session == session_id.value:
        return True
    return bool(kernel32.GetProcessWindowStation())


def interactive_session() -> bool:
    """Return whether this process can reach an interactive user session."""
    if sys.platform == "win32":
        try:
            return _windows_interactive_session()
        except (AttributeError, OSError, TypeError, ValueError):
            return False
    if sys.platform.startswith("linux"):
        session_id = _read_text("/proc/self/sessionid", 64).strip()
        return bool(session_id and session_id != "4294967295" and os.path.isdir(f"/run/user/{os.getuid()}"))
    return True


def clear_caches() -> None:
    """Clear every cached host fact."""
    for fact in (os_family, process_arch, native_arch, cpu_model, cpu_vendor, ram_total_bytes, gpu_class):
        fact.cache_clear()
