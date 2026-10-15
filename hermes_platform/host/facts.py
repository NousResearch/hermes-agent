"""Cached OS, architecture, CPU, memory and GPU facts for the current machine.

Native architecture remains accurate when the Python process is emulated. Facts read the registry,
sysctl, IOKit, procfs or sysfs: never the environment, a subprocess or a GPU driver library.
"""

from __future__ import annotations

import functools
import os
import platform
import sys
import sysconfig
from typing import NamedTuple

from hermes_platform.host import darwin_pci

_ARCH_NAMES = {
    "amd64": "amd64", "x86_64": "amd64", "x64": "amd64",
    "arm64": "arm64", "aarch64": "arm64",
    "x86": "x86", "i386": "x86", "i686": "x86",
}
IMAGE_FILE_MACHINE = {0x8664: "amd64", 0xAA64: "arm64", 0x014C: "x86"}
# The interpreter's own build: on Windows, platform.machine() names the native machine even
# inside an x64 process emulated on arm64.
_WINDOWS_BUILD_ARCH = {"win-amd64": "amd64", "win-arm64": "arm64", "win32": "x86"}


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
    if sys.platform == "win32":
        return _WINDOWS_BUILD_ARCH.get(sysconfig.get_platform(), "unknown")
    return normalize_arch(platform.machine())


def windows_native_arch(*, wow64_native: int | None, machine: str) -> str:
    """Return the native architecture: ``IsWow64Process2``'s native machine, else ``platform.machine()``."""
    return IMAGE_FILE_MACHINE.get(wow64_native or 0) or normalize_arch(machine)


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
        return windows_native_arch(wow64_native=wow64, machine=platform.machine())
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
        with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
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


# ARM's MIDR implementer codes, as arm64 Linux prints them in ``CPU implementer``.
_ARM_IMPLEMENTERS = {
    0x41: "ARM", 0x42: "Broadcom", 0x43: "Cavium", 0x46: "Fujitsu", 0x48: "HiSilicon", 0x4E: "NVIDIA",
    0x50: "Ampere", 0x51: "Qualcomm", 0x61: "Apple", 0x69: "Intel", 0x6D: "Microsoft", 0xC0: "Ampere",
}


def parse_cpuinfo_vendor(text: str) -> str:
    """Return ``vendor_id`` (x86) or the named ``CPU implementer`` (arm64) from ``/proc/cpuinfo``."""
    pairs = (line.split(":", 1) for line in text.splitlines() if ":" in line)
    fields = {key.strip(): value.strip() for key, value in pairs}
    if fields.get("vendor_id"):
        return fields["vendor_id"]
    try:
        return _ARM_IMPLEMENTERS.get(int(fields.get("CPU implementer", ""), 16), "")
    except ValueError:
        return ""


@functools.cache
def cpu_vendor() -> str:
    """Return the CPU vendor, or an empty string when unavailable.

    Windows reports the SoC vendor (``NVIDIA`` on an N1X); arm64 Linux reports the core designer.
    """
    if sys.platform == "win32":
        return _winreg_str(_CPU_KEY, "VendorIdentifier")
    if sys.platform == "darwin":
        return "Apple" if native_arch() == "arm64" else _sysctl_str(b"machdep.cpu.vendor")
    return parse_cpuinfo_vendor(_read_text("/proc/cpuinfo"))


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
    """Return the highest-priority GPU class for PCI vendor ids, or ``none``.

    ``unknown`` when devices were seen but none of their vendor ids could be read.
    """
    if vendor_ids and not any(v.strip() for v in vendor_ids):
        return "unknown"
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


def _hex_after(text: str, tag: str) -> str:
    return text.split(tag, 1)[1][:4] if tag in text else ""


def parse_windows_display_adapters(rows: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """``(vendor id, device id, description)`` per adapter from ``(subkey, MatchingDeviceId, DriverDesc)`` rows.

    Virtual adapters (no ``VEN_`` id, such as the remote-display driver) are skipped; an
    adapter subkey without a readable id keeps empty ids so it counts as unreadable.
    """
    adapters = []
    for subkey, matching_id, description in rows:
        upper = matching_id.upper()
        if "VEN_" in upper:
            adapters.append((_hex_after(upper, "VEN_"), _hex_after(upper, "DEV_"), description))
        elif subkey.isdigit() and not matching_id:
            adapters.append(("", "", description))
    return adapters


@functools.cache
def _windows_display_adapters() -> tuple[tuple[str, str, str], ...] | None:
    import winreg

    rows = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS_KEY) as root:
            for index in range(_GPU_SCAN_LIMIT):
                try:
                    subkey = winreg.EnumKey(root, index)
                except OSError:
                    break
                path = f"{_DISPLAY_CLASS_KEY}\\{subkey}"
                rows.append((subkey, _winreg_str(path, "MatchingDeviceId"), _winreg_str(path, "DriverDesc")))
    except OSError:
        return None
    return tuple(parse_windows_display_adapters(rows))


# The WSL GPU bridge mounts the Windows NVIDIA driver's CUDA library here only when that driver is installed.
_WSL_NVIDIA_CUDA = "/usr/lib/wsl/lib/libcuda.so.1"


@functools.cache
def gpu_class() -> str:
    """Return ``nvidia``/``amd``/``intel``/``apple_silicon``/``none``, or ``unknown``.

    Apple silicon by architecture; Intel Macs by IOKit PCI display controllers; Windows by the
    display-class registry; Linux by the NVIDIA driver's procfs, the WSL GPU bridge, then sysfs DRM cards.
    """
    try:
        if sys.platform == "darwin":
            if native_arch() == "arm64":
                return "apple_silicon"
            vendors = darwin_pci.display_vendor_ids()
        elif sys.platform == "win32":
            adapters = _windows_display_adapters()
            vendors = None if adapters is None else [vendor for vendor, _, _ in adapters]
        elif os.path.exists("/proc/driver/nvidia/version") or os.path.exists(_WSL_NVIDIA_CUDA):
            return "nvidia"
        else:
            vendors = _linux_gpu_vendor_ids()
    except (AttributeError, OSError, TypeError, ValueError):
        return "unknown"
    return "unknown" if vendors is None else classify_gpu_vendors(vendors)


_NVIDIA_PROC_GPUS = "/proc/driver/nvidia/gpus"
_NVIDIA_PCI_VENDOR = 0x10DE
_SYS_PCI_DEVICES = "/sys/bus/pci/devices"


def parse_nvidia_gpu_model(information: str) -> str:
    """Return the ``Model:`` value of a ``/proc/driver/nvidia/gpus/<bus>/information`` file."""
    for line in information.splitlines():
        if line.startswith("Model:"):
            return line.split(":", 1)[1].strip()
    return ""


class NvidiaGpu(NamedTuple):
    name: str
    pci_id: int | None  # NVML packing: device id above the 16-bit vendor id


def _pci_id(device_hex: str) -> int | None:
    try:
        return int(device_hex, 16) << 16 | _NVIDIA_PCI_VENDOR
    except ValueError:
        return None


def nvidia_gpus_from_adapters(adapters: list[tuple[str, str, str]]) -> tuple[NvidiaGpu, ...]:
    """The NVIDIA entries of parsed Windows display adapters."""
    return tuple(NvidiaGpu(description, _pci_id(device)) for vendor, device, description in adapters
                 if vendor.lower() == "10de")


def _linux_nvidia_gpus() -> tuple[NvidiaGpu, ...]:
    try:
        buses = sorted(os.listdir(_NVIDIA_PROC_GPUS))[:_GPU_SCAN_LIMIT]
    except OSError:
        return ()
    gpus = (NvidiaGpu(parse_nvidia_gpu_model(_read_text(f"{_NVIDIA_PROC_GPUS}/{bus}/information", 4096)),
                      _pci_id(_read_text(f"{_SYS_PCI_DEVICES}/{bus}/device", 16).strip().removeprefix("0x")))
            for bus in buses)
    return tuple(gpu for gpu in gpus if gpu.name)


@functools.cache
def nvidia_gpus() -> tuple[NvidiaGpu, ...]:
    """Name and PCI id of each NVIDIA GPU: the Windows display-class registry, or Linux driver procfs
    with sysfs PCI ids; empty elsewhere or without the driver."""
    if sys.platform == "win32":
        try:
            return nvidia_gpus_from_adapters(list(_windows_display_adapters() or ()))
        except (AttributeError, OSError, TypeError, ValueError):
            return ()
    return _linux_nvidia_gpus() if sys.platform.startswith("linux") else ()


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
        return bool(session_id and session_id != "4294967295" and os.path.isdir(f"/run/user/{os.getuid()}"))  # windows-footgun: ok — Linux branch
    return True


def clear_caches() -> None:
    """Clear every cached host fact."""
    for fact in (os_family, process_arch, native_arch, cpu_model, cpu_vendor, ram_total_bytes, gpu_class,
                 nvidia_gpus, _windows_display_adapters):
        fact.cache_clear()
