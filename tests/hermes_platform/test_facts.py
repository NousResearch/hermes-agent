from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from hermes_platform.host import facts

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AMD64", "amd64"),
        ("x86_64", "amd64"),
        ("aarch64", "arm64"),
        ("ARM64", "arm64"),
        ("i386", "x86"),
        ("weird", "unknown"),
    ],
)
def test_normalize_arch(raw: str, expected: str) -> None:
    assert facts.normalize_arch(raw) == expected

@pytest.mark.parametrize(
    ("wow64_native", "machine", "expected"),
    [
        (0xAA64, "AMD64", "arm64"),
        (0xAA64, "ARM64", "arm64"),  # x64 Python on an N1X, measured: platform.machine() is native
        (None, "x86_64", "amd64"),
        (None, "unknown", "unknown"),
    ],
)
def test_windows_native_arch_precedence(wow64_native: int | None, machine: str, expected: str) -> None:
    assert facts.windows_native_arch(wow64_native=wow64_native, machine=machine) == expected


# Display-class registry rows measured on an NVIDIA N1X laptop.
_N1X_DISPLAY_ROWS = [
    ("0000", r"pci\ven_10de&dev_2e03&subsys_33091043", "NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU)"),
    ("0001", r"acpi\ven_nvda&dev_200a&rev_0034", "NVIDIA NPU"),
    ("0002", r"pci\ven_10de&dev_2e06&subsys_33091043", "NVIDIA RTX Spark N1X (5120-core Blackwell RTX GPU)"),
    ("0003", "RdpIdd_IndirectDisplay", "Microsoft Remote Display Adapter"),
    ("Configuration", "", ""),
]


@pytest.mark.parametrize(
    ("rows", "gpu_class"),
    [
        (_N1X_DISPLAY_ROWS, "nvidia"),
        ([("0000", "RdpIdd_IndirectDisplay", "Microsoft Remote Display Adapter")], "none"),
        ([("0000", "", "")], "unknown"),
    ],
)
def test_windows_display_adapters_classify_the_gpu(rows, gpu_class) -> None:
    adapters = facts.parse_windows_display_adapters(rows)
    assert facts.classify_gpu_vendors([vendor for vendor, _, _ in adapters]) == gpu_class


@pytest.mark.parametrize(
    ("cpuinfo", "vendor"),
    [
        ("processor : 0\nvendor_id : GenuineIntel\nmodel name : Intel Xeon\n", "GenuineIntel"),
        ("processor : 0\nCPU implementer : 0x41\nCPU part : 0xd87\n", "ARM"),
        ("processor : 0\nCPU implementer : 0x61\n", "Apple"),
        ("processor : 0\nCPU implementer : 0xee\n", ""),
        ("processor : 0\n", ""),
    ],
)
def test_cpu_vendor_from_cpuinfo(cpuinfo: str, vendor: str) -> None:
    assert facts.parse_cpuinfo_vendor(cpuinfo) == vendor

@pytest.mark.parametrize(
    ("text", "device_tree_model", "expected"),
    [
        (
            "Hardware : Fallback board\nmodel name : Primary CPU\n",
            "Ignored_Model\0",
            "Primary CPU",
        ),
        ("processor : 0\nHardware : ARM Board\n", "Ignored_Model\0", "ARM Board"),
        ("processor : 0\n", "Vendor_Board_Name\0", "Vendor Board Name"),
    ],
)
def test_parse_cpuinfo_fallbacks(
    text: str,
    device_tree_model: str,
    expected: str,
) -> None:
    assert facts.parse_cpuinfo(text, device_tree_model=device_tree_model) == expected

@pytest.fixture
def cleared_fact_caches():
    facts.clear_caches()
    yield
    facts.clear_caches()

@pytest.mark.platforms("windows")
def test_windows_native_arch_matches_registry_identifier(cleared_fact_caches) -> None:
    winreg = importlib.import_module("winreg")
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, facts._CPU_KEY) as key:
        identifier = str(winreg.QueryValueEx(key, "Identifier")[0]).strip()

    expected = "arm64" if identifier.upper().startswith("ARMV8") else "amd64"
    assert facts.native_arch() == expected

@pytest.mark.platforms("macos")
def test_macos_live_cpu_facts(cleared_fact_caches) -> None:
    assert facts.cpu_model()
    assert facts.cpu_vendor()
    assert facts.native_arch() in {"arm64", "amd64"}
    assert facts.gpu_class() != "unknown"

@pytest.mark.platforms("linux")
def test_linux_live_cpu_facts_match_cpuinfo(cleared_fact_caches) -> None:
    cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")

    assert facts.cpu_vendor() == facts.parse_cpuinfo_vendor(cpuinfo)
    assert facts.cpu_model() in cpuinfo
