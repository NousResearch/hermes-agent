"""_ram_bytes() must report MemAvailable on Linux, not MemFree.

The desktop statusbar RAM meter renders ``ram_total - ram_available``, so
whatever this function calls "available" is what the user reads as free
memory. The old implementation used ``getconf _AVPHYS_PAGES``, which counts
only free pages: page cache and reclaimable slab were reported as consumed.
The desktop showed 7.5 GB / 7.8 GB (98%) on a machine with 5.1 GiB genuinely
available, while the web dashboard — which reads MemAvailable from the gateway
heartbeat — correctly showed 2.7 GB (34%). Same machine, same moment.

These tests pin the field, not a snapshot value.
"""

from __future__ import annotations

import sys

import pytest

from hermes_cli.local_runtime import hardware as hw

# Reported numbers from the box that surfaced the bug: cache plus reclaimable
# slab account for the whole 4.95 GiB gap between the two candidate answers.
_MEMINFO = """\
MemTotal:        8131784 kB
MemFree:          236448 kB
MemAvailable:    5357484 kB
Buffers:          465444 kB
Cached:          3611976 kB
SReclaimable:    1485000 kB
"""

_KB = 1024
_TOTAL = 8131784 * _KB
_AVAILABLE = 5357484 * _KB
_FREE = 236448 * _KB


@pytest.fixture
def fake_meminfo(tmp_path, monkeypatch):
    """Point the Linux branch at a controlled /proc/meminfo."""

    def _write(text: str):
        path = tmp_path / "meminfo"
        path.write_text(text, encoding="utf-8")
        real_open = open

        def _open(target, *args, **kwargs):
            if target == "/proc/meminfo":
                return real_open(path, *args, **kwargs)
            return real_open(target, *args, **kwargs)

        monkeypatch.setattr("builtins.open", _open)
        monkeypatch.setattr(sys, "platform", "linux")
        return path

    return _write


def _guard_getconf(monkeypatch):
    """Fail loudly if the Linux branch falls through to the getconf ladder."""

    def _boom(*args, **kwargs):  # pragma: no cover - only on regression
        raise AssertionError(
            "fell through to the getconf ladder instead of /proc/meminfo")

    monkeypatch.setattr(hw, "_stdout", _boom)


def test_reports_mem_available_not_mem_free(fake_meminfo, monkeypatch):
    fake_meminfo(_MEMINFO)
    _guard_getconf(monkeypatch)

    total, avail = hw._ram_bytes()

    assert total == _TOTAL
    assert avail == _AVAILABLE
    # The bug this test exists for: MemFree is 4.95 GiB lower.
    assert avail != _FREE


def test_statusbar_percent_matches_real_pressure(fake_meminfo, monkeypatch):
    """total - available is exactly what the RAM meter renders."""
    fake_meminfo(_MEMINFO)
    _guard_getconf(monkeypatch)

    total, avail = hw._ram_bytes()
    used_percent = round((total - avail) / total * 100)

    # ~34% real use. The MemFree reading rendered 97%.
    assert used_percent == 34
    assert round((total - _FREE) / total * 100) == 97


def test_missing_mem_available_falls_through(fake_meminfo, monkeypatch):
    """A trimmed /proc must not yield a bogus zero; the ladder continues."""
    fake_meminfo("MemTotal:        8131784 kB\nMemFree:          236448 kB\n")

    calls: list[tuple[str, ...]] = []

    def _fake_stdout(*argv: str) -> str:
        calls.append(argv)
        return {"PAGE_SIZE": "4096", "_PHYS_PAGES": "2032946",
                "_AVPHYS_PAGES": "39536"}.get(argv[-1], "0")

    monkeypatch.setattr(hw, "_stdout", _fake_stdout)

    total, avail = hw._ram_bytes()

    assert calls, "expected the getconf ladder to run"
    assert total == 2032946 * 4096
    assert avail == 39536 * 4096
