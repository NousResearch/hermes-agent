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
    ("wow64_native", "machine", "env_arch", "expected"),
    [
        (0xAA64, "AMD64", "x86", "arm64"),
        (None, "x86_64", "ARM64", "amd64"),
        (None, "unknown", "ARM64", "arm64"),
    ],
)
def test_windows_native_arch_precedence(
    wow64_native: int | None,
    machine: str,
    env_arch: str | None,
    expected: str,
) -> None:
    assert (
        facts.windows_native_arch(
            wow64_native=wow64_native,
            machine=machine,
            env_arch=env_arch,
        )
        == expected
    )


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


@pytest.mark.windows_only
def test_windows_native_arch_matches_registry_identifier(cleared_fact_caches) -> None:
    winreg = importlib.import_module("winreg")
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, facts._CPU_KEY) as key:
        identifier = str(winreg.QueryValueEx(key, "Identifier")[0]).strip()

    expected = "arm64" if identifier.upper().startswith("ARMV8") else "amd64"
    assert facts.native_arch() == expected


@pytest.mark.macos_only
def test_macos_live_cpu_facts(cleared_fact_caches) -> None:
    assert facts.cpu_model()
    assert facts.native_arch() in {"arm64", "amd64"}


@pytest.mark.linux_only
def test_linux_live_cpu_facts_match_cpuinfo(cleared_fact_caches) -> None:
    cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")

    assert facts.cpu_vendor() in cpuinfo
    assert facts.cpu_model() in cpuinfo


