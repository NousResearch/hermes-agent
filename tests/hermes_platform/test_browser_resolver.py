"""Native-browser selection verifies PE architecture instead of trusting installation paths."""

import struct

from hermes_platform.resolver.browser import _first_arm64_browser


def _executable(path, machine):
    data = bytearray(128)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 64)
    data[64:68] = b"PE\0\0"
    struct.pack_into("<H", data, 68, machine)
    path.write_bytes(data)
    return str(path)


def test_native_arm64_browser_is_selected_over_emulated_x64_and_stale_entries(tmp_path):
    x64 = _executable(tmp_path / "edge.exe", 0x8664)
    arm64 = _executable(tmp_path / "chrome.exe", 0xAA64)
    malformed = tmp_path / "invalid.exe"
    malformed.write_bytes(b"MZ")

    assert _first_arm64_browser([str(tmp_path / "missing.exe"), str(malformed), x64, arm64]) == arm64
    assert _first_arm64_browser([x64, str(malformed)]) is None
