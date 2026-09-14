"""The RAM probe must not read a machine as memoryless because a child process misbehaved.

On unified-memory machines the whole budget derives from total RAM, so a probe that answers 0
does not degrade gracefully: every catalog row reads "too big for this machine" and a 128 GiB
Mac is told it has no memory (#102619). Total therefore comes from os.sysconf in-process; the
sysctl/getconf subprocesses remain only as the fallback and for the available figure."""

from __future__ import annotations

import ctypes
import subprocess

import hermes_cli.local_runtime.hardware as hw

GIB = 1 << 30
PAGE = 16384
MAC_128 = 128 * GIB
MAC_SYSCONF = {"SC_PHYS_PAGES": MAC_128 // PAGE, "SC_PAGE_SIZE": PAGE}

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                              123456.
Pages active:                           3000000.
Pages inactive:                         2000000.
Pages speculative:                       100000.
Pages throttled:                              0.
Pages wired down:                        500000.
Pages purgeable:                          50000.
"Translation faults":                 999999999.
Pages compressed:                        400000.
"""
VM_STAT_RECLAIMABLE = (123456 + 2000000 + 100000 + 50000) * PAGE


def _sysconf(values: dict):
    def fake(name):
        if name not in values:
            raise ValueError(f"unrecognized configuration name: {name}")
        return values[name]
    return fake


def _host(monkeypatch, *, platform, sysconf, stdout=None):
    """Pin the platform and stub every process boundary the probe can cross. Returns the argv of
    each subprocess the probe attempted; ``stdout=None`` makes any attempt a failure."""
    # Windows answers from GlobalMemoryStatusEx before any of this runs; take that path away so
    # the POSIX branches are exercised on every CI runner.
    monkeypatch.delattr(ctypes, "windll", raising=False)
    monkeypatch.setattr(hw.sys, "platform", platform)
    if sysconf is None:
        monkeypatch.delattr(hw.os, "sysconf", raising=False)
    else:
        monkeypatch.setattr(hw.os, "sysconf", _sysconf(sysconf), raising=False)
    calls: list[tuple[str, ...]] = []

    def fake_stdout(*argv):
        calls.append(argv)
        assert stdout is not None, f"unexpected subprocess {argv}"
        out = stdout(argv)
        if isinstance(out, BaseException):
            raise out
        return out

    monkeypatch.setattr(hw, "_stdout", fake_stdout)
    return calls


# ── macOS ────────────────────────────────────────────────────


def test_darwin_total_is_read_in_process(monkeypatch):
    """Only vm_stat (the available figure) is spawned; the total never waits on a child."""
    calls = _host(monkeypatch, platform="darwin", sysconf=MAC_SYSCONF,
                  stdout=lambda argv: VM_STAT if argv[0] == "/usr/bin/vm_stat" else "")
    assert hw._ram_bytes() == (MAC_128, VM_STAT_RECLAIMABLE)
    assert [c[0] for c in calls] == ["/usr/bin/vm_stat"]


def test_darwin_empty_subprocess_output_keeps_the_total(monkeypatch):
    """#102619's shape: a subprocess spawned by the packaged backend produced no output (the
    sibling report #102616 shows exactly that for --version). The old probe turned an empty
    sysctl pipe into total=0; now only the available figure degrades, to half."""
    _host(monkeypatch, platform="darwin", sysconf=MAC_SYSCONF, stdout=lambda argv: "")
    assert hw._ram_bytes() == (MAC_128, MAC_128 // 2)


def test_darwin_spawn_failure_keeps_the_total(monkeypatch):
    """fork/exec refused (EAGAIN under load, a stripped or sandboxed environment): the total
    stands, the available figure degrades to half."""
    _host(monkeypatch, platform="darwin", sysconf=MAC_SYSCONF,
          stdout=lambda argv: BlockingIOError(11, "Resource temporarily unavailable"))
    assert hw._ram_bytes() == (MAC_128, MAC_128 // 2)


def test_darwin_hung_vm_stat_keeps_the_total(monkeypatch):
    """vm_stat past its timeout must not take the reading (and the catalog endpoint) down."""
    _host(monkeypatch, platform="darwin", sysconf=MAC_SYSCONF,
          stdout=lambda argv: subprocess.TimeoutExpired(argv, 5))
    assert hw._ram_bytes() == (MAC_128, MAC_128 // 2)


def test_darwin_falls_back_to_sysctl_without_sysconf(monkeypatch):
    calls = _host(monkeypatch, platform="darwin", sysconf=None,
                  stdout=lambda argv: f"{MAC_128}\n" if argv[0] == "/usr/sbin/sysctl" else VM_STAT)
    assert hw._ram_bytes() == (MAC_128, VM_STAT_RECLAIMABLE)
    assert calls[0] == ("/usr/sbin/sysctl", "-n", "hw.memsize")


def test_darwin_sysconf_answering_zero_falls_back_to_sysctl(monkeypatch):
    """A libc that has the name but no answer (-1 / 0) is treated as absent, not as a 0-byte
    machine."""
    calls = _host(monkeypatch, platform="darwin",
                  sysconf={"SC_PHYS_PAGES": -1, "SC_PAGE_SIZE": PAGE},
                  stdout=lambda argv: f"{MAC_128}\n" if argv[0] == "/usr/sbin/sysctl" else VM_STAT)
    assert hw._ram_bytes()[0] == MAC_128
    assert calls[0][0] == "/usr/sbin/sysctl"


# ── Linux / other POSIX ──────────────────────────────────────


def test_linux_total_and_available_are_read_in_process(monkeypatch):
    """glibc carries both names: no getconf fork at all (stdout=None fails on any attempt)."""
    _host(monkeypatch, platform="linux", sysconf={
        "SC_PHYS_PAGES": 64 * GIB // 4096, "SC_AVPHYS_PAGES": 20 * GIB // 4096, "SC_PAGE_SIZE": 4096})
    assert hw._ram_bytes() == (64 * GIB, 20 * GIB)


def test_posix_without_avphys_halves_the_total(monkeypatch):
    """BSD-flavoured libc: SC_PHYS_PAGES only. Half of total stands in for available, as before."""
    _host(monkeypatch, platform="freebsd14", sysconf={"SC_PHYS_PAGES": 32 * GIB // 4096, "SC_PAGE_SIZE": 4096})
    assert hw._ram_bytes() == (32 * GIB, 16 * GIB)


def test_linux_falls_back_to_getconf_without_sysconf(monkeypatch):
    def getconf(argv):
        return {"PAGE_SIZE": "4096\n", "_PHYS_PAGES": f"{64 * GIB // 4096}\n"}[argv[1]]

    calls = _host(monkeypatch, platform="linux", sysconf=None, stdout=getconf)
    assert hw._ram_bytes() == (64 * GIB, 32 * GIB)
    assert [c[1] for c in calls] == ["PAGE_SIZE", "_PHYS_PAGES"]


def test_nothing_answers_reads_unavailable(monkeypatch):
    """Every source gone: (0, 0) remains the 'unknown machine' signal downstream code expects."""
    _host(monkeypatch, platform="darwin", sysconf=None,
          stdout=lambda argv: FileNotFoundError(2, "no such file"))
    assert hw._ram_bytes() == (0, 0)


def test_hung_fallback_reads_unavailable_instead_of_raising(monkeypatch):
    """No sysconf and a hung sysctl: unavailable, not an exception out of probe_budget."""
    _host(monkeypatch, platform="darwin", sysconf=None,
          stdout=lambda argv: subprocess.TimeoutExpired(argv, 5))
    assert hw._ram_bytes() == (0, 0)


# ── the symptom, end to end ──────────────────────────────────


def test_unified_mac_prices_the_catalog_from_the_in_process_total(monkeypatch):
    """A 128 GiB Mac whose subprocesses return nothing must still budget the pool minus headroom
    and price the 27B as resident — the row #102619 saw tagged 'Too big for this machine'."""
    from hermes_cli.local_runtime import catalog

    _host(monkeypatch, platform="darwin", sysconf=MAC_SYSCONF, stdout=lambda argv: "")
    monkeypatch.setattr(hw, "_nvidia_vram", lambda: None)
    monkeypatch.setattr(hw, "_device_pool_view", lambda: None)
    budget = hw.probe_budget(planning=True)
    assert budget.uma is True
    assert budget.total_device_bytes == MAC_128
    assert budget.usable_vram_bytes == int(MAC_128 * (1 - hw._UMA_HEADROOM_FRACTION))
    choice = catalog.select_variant(catalog.catalog_by_id()["qwen3.8-27b"], budget)
    assert choice is not None and choice.zero_spill
