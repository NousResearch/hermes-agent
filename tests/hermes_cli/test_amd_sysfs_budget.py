"""The AMD sysfs probe must never misclassify a small-carve APU.

On APUs with a large BIOS-dedicated carve (Strix Halo and friends),
amdgpu's mem_info_vram_total bar IS the real pool: the kernel removed
it from system RAM, the OS cannot reclaim it, and placement inside it
runs at full bandwidth. Budgeting those machines from OS RAM (the
UMA fallback) sees a fraction of the truth and prices every model
wrong.

The other direction is the regression this file guards: a small-carve
APU (0.5-4 GiB bar of a shared RAM pool) must keep today's
RAM-as-UMA budget — trusting its bar would budget 2 GiB against a
machine with 64 GiB. The RAM-sized-carve gate (_POOL_RAM_FRACTION,
shared with the NVIDIA quirk) separates the two shapes."""

from __future__ import annotations

import hermes_cli.local_runtime.hardware as hw

GIB = 1 << 30

# Representative Strix Halo shape: 96 GiB BIOS carve, 32 GiB OS RAM.
STRIX_CARVE = 96 * GIB
STRIX_CARVE_USED = 2 * GIB
STRIX_RAM = 32 * GIB

# Representative small-carve APU: 512 MiB bar, 64 GiB shared pool.
SMALL_CARVE = 512 << 20
SMALL_CARVE_USED = 128 << 20
SMALL_RAM = 64 * GIB


def _machine(monkeypatch, *, nvidia=None, amd=None, ram=(0, 0)):
    monkeypatch.setattr(hw, "_nvidia_vram", lambda: nvidia)
    monkeypatch.setattr(hw, "_amd_sysfs_vram", lambda: amd)
    monkeypatch.setattr(hw, "_ram_bytes", lambda: ram)


# ── the gate in probe_budget ─────────────────────────────────


def test_strix_halo_budgets_from_the_carve(monkeypatch):
    """Large BIOS carve passes the RAM-sized gate: the budget becomes the
    dedicated-pool shape (discrete math, RAM kept as spill room), not
    the 32 GiB RAM-as-UMA budget that started this."""
    _machine(monkeypatch, amd=(STRIX_CARVE, STRIX_CARVE - STRIX_CARVE_USED),
             ram=(STRIX_RAM, 16 * GIB))
    b = hw.probe_budget(planning=True)
    assert b.uma is False
    assert b.total_device_bytes == STRIX_CARVE
    margin = max(hw._MARGIN_FLOOR, int(STRIX_CARVE * hw._MARGIN_FRACTION))
    assert b.usable_vram_bytes == STRIX_CARVE - margin
    # Planning budgets report ram_total as the spill room (by design).
    assert b.ram_available_bytes == STRIX_RAM
    # The whole point: the budget must dwarf the OS-visible RAM.
    assert b.usable_vram_bytes > STRIX_RAM


def test_strix_halo_live_budget_uses_carve_free(monkeypatch):
    _machine(monkeypatch, amd=(STRIX_CARVE, STRIX_CARVE - STRIX_CARVE_USED),
             ram=(STRIX_RAM, 16 * GIB))
    b = hw.probe_budget(planning=False)
    assert b.uma is False
    assert b.usable_vram_bytes == (STRIX_CARVE - STRIX_CARVE_USED) - max(
        hw._MARGIN_FLOOR, int(STRIX_CARVE * hw._MARGIN_FRACTION))


def test_small_carve_apu_keeps_uma_budget(monkeypatch):
    """512 MiB bar in a 64 GiB box fails the gate — exactly today's
    RAM-as-UMA behavior, bit for bit."""
    _machine(monkeypatch, amd=(SMALL_CARVE, SMALL_CARVE - SMALL_CARVE_USED),
             ram=(SMALL_RAM, 48 * GIB))
    b = hw.probe_budget(planning=True)
    assert b.uma is True
    assert b.total_device_bytes == SMALL_RAM
    assert b.usable_vram_bytes == int(
        SMALL_RAM * (1 - hw._UMA_HEADROOM_FRACTION))
    assert b.ram_available_bytes == 0


def test_no_amd_device_keeps_uma_budget(monkeypatch):
    """Probe miss (non-AMD Linux box) -> exactly today's behavior."""
    _machine(monkeypatch, amd=None, ram=(STRIX_RAM, 16 * GIB))
    b = hw.probe_budget(planning=True)
    assert b.uma is True
    assert b.total_device_bytes == STRIX_RAM


def test_zero_ram_stays_uma_conservative(monkeypatch):
    """RAM unreadable: the gate has no denominator, so it must not fire —
    an unclassified carve claim alone never flips the verdict."""
    _machine(monkeypatch, amd=(STRIX_CARVE, STRIX_CARVE), ram=(0, 0))
    b = hw.probe_budget(planning=True)
    assert b.uma is True
    assert b.total_device_bytes == 0


def test_nvidia_device_takes_precedence(monkeypatch):
    """An NVIDIA device answering means the AMD probe never runs: the
    discrete-NVIDIA path is untouched."""
    seen = []
    monkeypatch.setattr(hw, "_amd_sysfs_vram",
                        lambda: seen.append(1) or (STRIX_CARVE, STRIX_CARVE))
    _machine(monkeypatch, nvidia=(24 * GIB, 23 * GIB),
             ram=(64 * GIB, 48 * GIB))
    b = hw.probe_budget(planning=True)
    assert b.uma is False
    assert b.total_device_bytes == 24 * GIB
    assert not seen


# ── _amd_sysfs_vram parsing ──────────────────────────────────


def test_probe_reads_amdgpu_sysfs(tmp_path, monkeypatch):
    """The probe reads vendor + mem_info_* from the amdgpu bind point and
    picks the largest card (multi-GPU boxes)."""
    drm = tmp_path / "drm"
    for name, vendor, total, used in (
        ("card0", "0x8086", 0, 0),               # Intel iGPU: no mem_info
        ("card1", "0x1002", STRIX_CARVE, STRIX_CARVE_USED),
        ("card2", "0x1002", 24 * GIB, 1 * GIB),
    ):
        dev = drm / name / "device"
        dev.mkdir(parents=True)
        (dev / "vendor").write_text(vendor + "\n")
        if total:
            (dev / "mem_info_vram_total").write_text(f"{total}\n")
            (dev / "mem_info_vram_used").write_text(f"{used}\n")
    monkeypatch.setattr(hw.Path, "glob",
                        lambda self, pat: (drm / "card1" / "device",
                                           drm / "card2" / "device"))
    monkeypatch.setattr(hw.sys, "platform", "linux")
    assert hw._amd_sysfs_vram() == (STRIX_CARVE, STRIX_CARVE - STRIX_CARVE_USED)


def test_probe_skips_unreadable_or_broken_entries(tmp_path, monkeypatch):
    drm = tmp_path / "drm"
    dev = drm / "card1" / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text("0x1002\n")
    (dev / "mem_info_vram_total").write_text("not a number\n")
    monkeypatch.setattr(hw.Path, "glob", lambda self, pat: (dev,))
    monkeypatch.setattr(hw.sys, "platform", "linux")
    assert hw._amd_sysfs_vram() is None


def test_probe_other_platforms_return_none(monkeypatch):
    for platform in ("darwin", "win32"):
        monkeypatch.setattr(hw.sys, "platform", platform)
        assert hw._amd_sysfs_vram() is None
