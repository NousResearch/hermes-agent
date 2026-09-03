"""Linux RAM availability must include reclaimable page cache (#102252)."""

from pathlib import Path
from types import SimpleNamespace

import hermes_cli.local_runtime.hardware as hw


GIB = 1 << 30


def test_linux_ram_bytes_prefers_memavailable(tmp_path: Path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       67108864 kB\n"
        "MemFree:        12582912 kB\n"
        "MemAvailable:   56623104 kB\n"
        "Cached:         44040192 kB\n",
        encoding="utf-8",
    )

    assert hw._linux_ram_bytes(meminfo) == (64 * GIB, 54 * GIB)


def test_linux_ram_bytes_falls_back_to_memfree_for_old_kernels(tmp_path: Path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       8388608 kB\nMemFree:        2097152 kB\n",
        encoding="utf-8",
    )

    assert hw._linux_ram_bytes(meminfo) == (8 * GIB, 2 * GIB)


def test_linux_ram_bytes_keeps_zero_memavailable_instead_of_memfree(tmp_path: Path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       8388608 kB\n"
        "MemFree:        2097152 kB\n"
        "MemAvailable:         0 kB\n",
        encoding="utf-8",
    )

    assert hw._linux_ram_bytes(meminfo) == (8 * GIB, 0)


def test_linux_ram_bytes_returns_none_when_procfs_is_missing(tmp_path: Path):
    assert hw._linux_ram_bytes(tmp_path / "missing-meminfo") is None


def test_linux_ram_bytes_rejects_incomplete_or_invalid_values(tmp_path: Path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: 1048576 kB\n", encoding="utf-8")
    assert hw._linux_ram_bytes(meminfo) is None

    meminfo.write_text(
        "MemTotal: 1048576 kB\nMemAvailable: 2097152 kB\n",
        encoding="utf-8",
    )
    assert hw._linux_ram_bytes(meminfo) is None


def test_ram_bytes_uses_linux_memavailable_before_getconf(monkeypatch):
    expected = (64 * GIB, 54 * GIB)
    monkeypatch.setattr(hw.sys, "platform", "linux")
    monkeypatch.setattr(hw, "_linux_ram_bytes", lambda: expected)
    monkeypatch.setattr(
        hw.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("getconf must not run when /proc/meminfo is usable")
        ),
    )

    assert hw._ram_bytes() == expected


def test_ram_bytes_falls_back_to_getconf_when_meminfo_unavailable(monkeypatch):
    monkeypatch.setattr(hw.sys, "platform", "linux")
    monkeypatch.setattr(hw, "_linux_ram_bytes", lambda: None)
    values = {
        "PAGE_SIZE": "4096\n",
        "_PHYS_PAGES": "2097152\n",
        "_AVPHYS_PAGES": "524288\n",
    }

    def fake_run(command, **kwargs):
        return SimpleNamespace(stdout=values[command[1]])

    monkeypatch.setattr(hw.subprocess, "run", fake_run)

    assert hw._ram_bytes() == (8 * GIB, 2 * GIB)
