"""Linux RAM availability must include reclaimable page cache (#102252).

``getconf _AVPHYS_PAGES`` counts only free pages, so the Desktop statusbar
treats page cache as used. ``MemAvailable`` is the reclaimable-aware figure.
A reported ``0`` is a real available value, not a missing field.
"""

from __future__ import annotations

import ctypes

import hermes_cli.local_runtime.hardware as hw

GIB = 1 << 30


def _as_linux(monkeypatch) -> None:
    """Reach the Linux branch on every host without editing the Windows probe."""
    monkeypatch.delattr(ctypes, "windll", raising=False)
    monkeypatch.setattr(hw.sys, "platform", "linux")


def test_memavailable_is_used_when_present():
    text = (
        "MemTotal:       67108864 kB\n"
        "MemFree:        12582912 kB\n"
        "MemAvailable:   56623104 kB\n"
        "Cached:         44040192 kB\n"
    )

    assert hw._linux_ram_from_meminfo(text) == (64 * GIB, 54 * GIB)


def test_zero_memavailable_is_a_real_value():
    text = (
        "MemTotal:       8388608 kB\n"
        "MemFree:        2097152 kB\n"
        "MemAvailable:         0 kB\n"
    )

    assert hw._linux_ram_from_meminfo(text) == (8 * GIB, 0)


def test_missing_memavailable_falls_back_to_memfree():
    text = "MemTotal:       8388608 kB\nMemFree:        2097152 kB\n"

    assert hw._linux_ram_from_meminfo(text) == (8 * GIB, 2 * GIB)


def test_unusable_meminfo_is_not_a_reading():
    assert hw._linux_ram_from_meminfo("MemAvailable: 1048576 kB\n") is None
    assert hw._linux_ram_from_meminfo(
        "MemTotal: 1048576 kB\nMemAvailable: 2097152 kB\n"
    ) is None


def test_ram_bytes_uses_injected_meminfo_including_zero(monkeypatch):
    _as_linux(monkeypatch)
    monkeypatch.setattr(
        hw,
        "_linux_meminfo_text",
        lambda: (
            "MemTotal:       8388608 kB\n"
            "MemFree:        2097152 kB\n"
            "MemAvailable:         0 kB\n"
        ),
    )

    def _getconf_must_not_run(*_args, **_kwargs):
        raise AssertionError("getconf must not run when MemAvailable is present")

    monkeypatch.setattr(hw, "_stdout", _getconf_must_not_run)

    assert hw._ram_bytes() == (8 * GIB, 0)


def test_ram_bytes_falls_back_when_memavailable_is_absent(monkeypatch):
    _as_linux(monkeypatch)
    monkeypatch.setattr(
        hw,
        "_linux_meminfo_text",
        lambda: "MemTotal:       8388608 kB\nMemFree:        2097152 kB\n",
    )

    def _getconf_must_not_run(*_args, **_kwargs):
        raise AssertionError("MemFree is a valid fallback; getconf must not run")

    monkeypatch.setattr(hw, "_stdout", _getconf_must_not_run)

    assert hw._ram_bytes() == (8 * GIB, 2 * GIB)


def test_ram_bytes_falls_back_to_getconf_when_meminfo_unusable(monkeypatch):
    _as_linux(monkeypatch)
    monkeypatch.setattr(hw, "_linux_meminfo_text", lambda: None)
    values = {
        "PAGE_SIZE": "4096\n",
        "_PHYS_PAGES": "2097152\n",
        "_AVPHYS_PAGES": "524288\n",
    }

    def fake_stdout(*argv):
        return values[argv[-1]]

    monkeypatch.setattr(hw, "_stdout", fake_stdout)

    assert hw._ram_bytes() == (8 * GIB, 2 * GIB)
