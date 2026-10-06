"""Multi-GPU NVIDIA hosts are priced against AGGREGATE VRAM, not the first card (#107960).

The managed runtime read one line from nvidia-smi, so a 24 GB + 32 GB pair budgeted as a
24 GB machine and models that fit the aggregate were marked "Won't fit". The fix prices the
sum (per-card margin subtracted per card) and exposes VRAM-proportional --tensor-split
ratios, because llama.cpp's default EVEN split starves the larger card on uneven pairs.

Single-card machines must stay byte-identical: every test here that feeds one card asserts
the legacy answer.
"""
from __future__ import annotations

import pytest

import hermes_cli.local_runtime.hardware as hw

GIB = 1 << 30
MIB = 1 << 20

# The reporter's pair from #107960, plus the uneven pair this fix was designed around.
CARD_A_TOTAL = 24 * GIB   # RTX 4090
CARD_B_TOTAL = 32 * GIB   # RTX PRO 4500 Blackwell


def _no_cache(monkeypatch):
    monkeypatch.setattr(hw, "_gpu_query_cache", None)
    monkeypatch.setattr(hw, "_pool_probe_cache", None)


def _cards_query(monkeypatch, cards):
    """cards: list of (total_mib, free_mib, name). Primary view = first row (smi order)."""
    rows = []
    for total_mib, free_mib, name in cards:
        rows.append({
            "gpu_name": name,
            "total_bytes": total_mib << 20,
            "free_bytes": free_mib << 20,
            "used_bytes": (total_mib - free_mib) << 20,
            "gpu_util_percent": 0,
            "gpu_pci_id": None,
        })
    primary = dict(rows[0])
    primary["cards"] = rows
    monkeypatch.setattr(hw, "_cached_nvidia_gpu_query", lambda ttl_s=4.0: primary)


def _card_row(monkeypatch, total_gib: float, free_gib: float, name: str = "NVIDIA GeForce RTX 4090"):
    """The legacy single-card shape (no 'cards' key) — what pre-fix builds return."""
    monkeypatch.setattr(hw, "_cached_nvidia_gpu_query", lambda ttl_s=4.0: {
        "gpu_name": name,
        "total_bytes": int(total_gib * GIB),
        "free_bytes": int(free_gib * GIB),
        "used_bytes": int((total_gib - free_gib) * GIB),
        "gpu_util_percent": 0,
        "gpu_pci_id": None,
    })


# ── _nvidia_cards: parsing and ordering ─────────────────────


def test_single_card_query_exposes_itself_as_one_card(monkeypatch):
    _no_cache(monkeypatch)
    _card_row(monkeypatch, 24, 22)
    cards = hw._nvidia_cards()
    assert len(cards) == 1 and cards[0]["total_bytes"] == int(24 * GIB)


def test_multi_card_query_parses_every_row(monkeypatch):
    _no_cache(monkeypatch)
    _cards_query(monkeypatch, [
        (24576, 22000, "NVIDIA GeForce RTX 4090"),
        (32768, 31000, "NVIDIA RTX PRO 4500 Blackwell"),
    ])
    cards = hw._nvidia_cards()
    assert len(cards) == 2
    # Sorted by total VRAM descending regardless of smi row order.
    assert cards[0]["total_bytes"] == 32 * GIB
    assert cards[1]["total_bytes"] == 24 * GIB


def test_no_query_yields_no_cards(monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setattr(hw, "_cached_nvidia_gpu_query", lambda ttl_s=4.0: None)
    assert hw._nvidia_cards() == []


# ── tensor_split_ratio: VRAM-proportional, never even-by-default ──


def test_split_ratio_is_vram_proportional(monkeypatch):
    _no_cache(monkeypatch)
    _cards_query(monkeypatch, [
        (24576, 22000, "NVIDIA GeForce RTX 4090"),
        (32768, 31000, "NVIDIA RTX PRO 4500 Blackwell"),
    ])
    # 32:24 = 4:3. Ratios follow the LARGER card first (device order = sorted order), on a
    # fine-grained scale: --tensor-split 4,3 and --tensor-split 114,86 split identically.
    assert hw.tensor_split_ratio() == [114, 86]


def test_split_ratio_equal_cards_is_even(monkeypatch):
    _no_cache(monkeypatch)
    _cards_query(monkeypatch, [
        (16303, 15000, "NVIDIA GeForce RTX 5070 Ti"),
        (16311, 15000, "NVIDIA GeForce RTX 5060 Ti"),
    ])
    assert hw.tensor_split_ratio() == [100, 100]


def test_split_ratio_single_card_is_none(monkeypatch):
    _no_cache(monkeypatch)
    _card_row(monkeypatch, 24, 22)
    assert hw.tensor_split_ratio() is None


def test_split_ratio_never_zero_on_a_tiny_second_card(monkeypatch):
    """A 4 GB card beside a 32 GB card still gets at least 1 share."""
    _no_cache(monkeypatch)
    _cards_query(monkeypatch, [
        (32768, 31000, "NVIDIA RTX PRO 4500 Blackwell"),
        (4096, 3800, "NVIDIA T400"),
    ])
    ratios = hw.tensor_split_ratio()
    assert ratios is not None and all(r >= 1 for r in ratios)
    assert ratios[0] > ratios[1]


# ── aggregate_nvidia_budget: per-card margin, summed ─────────


def test_aggregate_capacity_prices_the_sum_minus_per_card_margin(monkeypatch):
    _no_cache(monkeypatch)
    _cards_query(monkeypatch, [
        (24576, 22000, "NVIDIA GeForce RTX 4090"),
        (32768, 31000, "NVIDIA RTX PRO 4500 Blackwell"),
    ])
    budget = hw.aggregate_nvidia_budget(planning=True)
    assert budget is not None
    expected = sum(
        max(0, total - max(hw._MARGIN_FLOOR, int(total * hw._MARGIN_FRACTION)))
        for total in (CARD_A_TOTAL, CARD_B_TOTAL))
    assert budget.usable_vram_bytes == expected
    assert budget.total_device_bytes == CARD_A_TOTAL + CARD_B_TOTAL
    assert budget.uma is False
    # Both names surface so the UI can show per-GPU rows.
    assert "4090" in budget.gpu_name and "PRO 4500" in budget.gpu_name


def test_aggregate_live_prices_free_memory(monkeypatch):
    _no_cache(monkeypatch)
    _cards_query(monkeypatch, [
        (24576, 20000, "NVIDIA GeForce RTX 4090"),
        (32768, 28000, "NVIDIA RTX PRO 4500 Blackwell"),
    ])
    budget = hw.aggregate_nvidia_budget(planning=False)
    assert budget is not None
    expected = sum(
        max(0, free - max(hw._MARGIN_FLOOR, int(total * hw._MARGIN_FRACTION)))
        for total, free in ((CARD_A_TOTAL, 20000 << 20), (CARD_B_TOTAL, 28000 << 20)))
    assert budget.usable_vram_bytes == expected


def test_aggregate_is_none_for_single_card(monkeypatch):
    _no_cache(monkeypatch)
    _card_row(monkeypatch, 24, 22)
    assert hw.aggregate_nvidia_budget(planning=True) is None


# ── probe_budget: multi-GPU hosts get the aggregate, single-card unchanged ──


def test_probe_budget_planning_uses_aggregate_on_dual_host(monkeypatch):
    _no_cache(monkeypatch)
    _cards_query(monkeypatch, [
        (24576, 22000, "NVIDIA GeForce RTX 4090"),
        (32768, 31000, "NVIDIA RTX PRO 4500 Blackwell"),
    ])
    monkeypatch.setattr(hw, "_unified_pool_bytes", lambda smi_total, ram_total: None)
    budget = hw.probe_budget(planning=True)
    # The legacy answer would have been the 32 GB card alone; now it is the sum.
    single_card_capacity = CARD_B_TOTAL - max(hw._MARGIN_FLOOR, int(CARD_B_TOTAL * hw._MARGIN_FRACTION))
    assert budget.usable_vram_bytes > single_card_capacity
    assert budget.total_device_bytes == CARD_A_TOTAL + CARD_B_TOTAL


def test_probe_budget_single_card_unchanged(monkeypatch):
    _no_cache(monkeypatch)
    _card_row(monkeypatch, 24, 22)
    monkeypatch.setattr(hw, "_unified_pool_bytes", lambda smi_total, ram_total: None)
    budget = hw.probe_budget(planning=True)
    expected = CARD_A_TOTAL - max(hw._MARGIN_FLOOR, int(CARD_A_TOTAL * hw._MARGIN_FRACTION))
    assert budget.usable_vram_bytes == expected
    assert budget.total_device_bytes == CARD_A_TOTAL


def test_probe_budget_unified_pool_still_wins(monkeypatch):
    """A pooled (carve-out) primary card keeps the unified-memory path, not the aggregate."""
    _no_cache(monkeypatch)
    pool = 46464 << 20
    _cards_query(monkeypatch, [
        (16320, 15000, "NVIDIA Unified Device"),
        (8192, 7000, "NVIDIA Secondary"),
    ])
    monkeypatch.setattr(hw, "_unified_pool_bytes", lambda smi_total, ram_total: pool)
    budget = hw.probe_budget(planning=True)
    assert budget.uma is True
    assert budget.usable_vram_bytes == int(pool * (1 - hw._UMA_HEADROOM_FRACTION))
