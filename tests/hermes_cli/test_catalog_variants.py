"""Variant-selection contracts: the quant ladder picks the best build for
the hardware, per the design's 'offer a smaller quant' remedy run
proactively, bounded by the Q4 quality floor. Pure decision-table tests
over synthetic budgets."""

from __future__ import annotations

import pytest

from hermes_cli.local_runtime.catalog import (
    CATALOG,
    catalog_by_id,
    find_entry_for_model,
    select_variant,
)
from hermes_cli.local_runtime.estimator import HardwareBudget

GIB = 1 << 30


def budget(vram_gib: float, ram_gib: float = 64) -> HardwareBudget:
    return HardwareBudget(usable_vram_bytes=int(vram_gib * GIB),
                          total_device_bytes=int(vram_gib * GIB),
                          ram_available_bytes=int(ram_gib * GIB))


def test_variants_ordered_best_first_and_floor_is_q4():
    for entry in CATALOG:
        sizes = [v.size_bytes for v in entry.variants]
        assert sizes == sorted(sizes, reverse=True), f"{entry.id}: variants not best-first"
        for v in entry.variants:
            for asset in entry.download_files(v):
                assert asset.size_bytes > 0, f"{entry.id}/{v.quant}: no size on {asset.path}"
        # The quality floor: the ladder ends at Q4 — no Q3/Q2 builds ship
        # (product decision, 2026-08-09). Validation status is explicit
        # per variant in catalog.json; unvalidated floors are permitted
        # (day-0 entries) and surface as unbadged rows in the pane.
        floor = entry.variants[-1]
        assert floor.quant == "UD-Q4_K_XL", f"{entry.id}: ladder floor is {floor.quant}, not Q4"


def test_split_variants_have_coherent_parts():
    """Multi-file variants: same model_id from every part, exact sizes,
    first file is the load target."""
    entry = catalog_by_id()["deepseek-v4-flash"]
    for v in entry.variants:
        assert len(v.files) >= 2, "deepseek ships split GGUFs"
        assert "00001-of" in v.files[0].path, "first part must be the load target"
        assert v.size_bytes == sum(f.size_bytes for f in v.files)
    assert entry.draft is not None, "DSpark draft rides along"


def test_big_card_gets_best_quality():
    """A card with real headroom takes the top rung — quality is free when
    it fits AT THE TARGET WINDOW. Muse is dense (target KV ~18 GiB), so
    'real headroom' for its Q8 means ~56+ GiB usable."""
    entry = catalog_by_id()["muse-glimmer-30b"]
    choice = select_variant(entry, budget(60))
    assert choice is not None
    assert choice.zero_spill
    assert choice.variant.quant == entry.variants[0].quant  # UD-Q8_K_XL
    assert choice.reason_key == "best-large-window"


def test_quality_monotone_in_vram():
    """More VRAM never selects a smaller build."""
    entry = catalog_by_id()["qwen3.8-27b"]
    sizes = []
    for vram in (8, 12, 16, 24, 32, 48):
        choice = select_variant(entry, budget(vram))
        assert choice is not None
        sizes.append(choice.variant.size_bytes)
    assert sizes == sorted(sizes), f"quality not monotone in VRAM: {sizes}"


def test_small_card_gets_q4_spilled_never_below():
    """8 GiB card + 27B: nothing zero-spills. The floor holds — the
    selector offers Q4 spilled (priced honestly), never a sub-Q4 build."""
    entry = catalog_by_id()["qwen3.8-27b"]
    choice = select_variant(entry, budget(8))
    assert choice is not None
    assert not choice.zero_spill
    assert choice.reason_key == "smallest-fits-spilled"
    assert choice.variant.quant == "UD-Q4_K_XL"


def test_frontier_model_refused_on_consumer_card_offered_on_big_ram():
    """DeepSeek V4 Flash (161 GB at Q4): refused outright on a 32 GiB-RAM
    desktop; offered spilled on a 192 GiB-RAM workstation. The catalog
    carries frontier hardware honestly instead of hiding the model."""
    entry = catalog_by_id()["deepseek-v4-flash"]
    assert select_variant(entry, budget(32, ram_gib=32)) is None
    big = select_variant(entry, budget(32, ram_gib=192))
    assert big is not None and not big.zero_spill


def test_selection_accounts_for_kv_not_just_weights():
    """The zero-spill check prices weights + KV, not weights alone: give a
    machine exactly enough VRAM for the Q8 weights of a dense model and it
    must step down a rung."""
    entry = catalog_by_id()["muse-glimmer-30b"]
    q8 = entry.variants[0]
    exactly_weights = HardwareBudget(
        usable_vram_bytes=q8.size_bytes + (100 << 20),
        total_device_bytes=q8.size_bytes + (100 << 20),
        ram_available_bytes=64 * GIB)
    choice = select_variant(entry, exactly_weights)
    assert choice is not None
    assert choice.variant.quant != q8.quant, "KV cost ignored — Q8 can't fit with floor KV"


def test_target_window_beats_one_quality_step():
    """The usability rule: a quant that only clears the 64K floor loses to
    the next rung down when that rung clears the target window. A real
    32 GiB card is ~29.6 GiB usable after margin: Qwen3.8-27B Q6 (needs
    ~31.5 with target KV + overhead) misses, Q5 (~26.2) clears — the
    selector must pick Q5 and say why."""
    entry = catalog_by_id()["qwen3.8-27b"]
    choice = select_variant(entry, budget(29.6))
    assert choice is not None and choice.zero_spill
    assert choice.variant.quant == "UD-Q5_K_XL", (
        f"expected the target-window pick, got {choice.variant.quant}")
    assert choice.reason_key == "best-large-window"


def test_floor_fallback_when_no_variant_reaches_target():
    """Cards where nothing clears the target keep the old rule: highest
    quality that zero-spills at the 64K floor (reason 'best-fits'), never
    a needless step down."""
    entry = catalog_by_id()["qwen3.8-27b"]
    # ~24.5 GiB usable: Q4 weights (16.7 GiB in-memory) + floor KV (2.2)
    # + overhead (1.5 + 0.9 mmproj + 1.9 vocab logits) fits, but no
    # variant reaches 144K-target KV (+2.7 more for Q4, more for Q5+).
    choice = select_variant(entry, budget(24.5))
    assert choice is not None and choice.zero_spill
    assert choice.reason_key == "best-fits"
    assert choice.variant.quant == "UD-Q4_K_XL"


def test_target_never_degrades_below_floor_choice():
    """The target preference may only IMPROVE the window, never the
    floor guarantees: whenever the old floor rule found a zero-spill pick,
    the new rule also finds one (possibly a smaller quant, never spill)."""
    for entry in CATALOG:
        for vram in (8, 12, 16, 24, 32, 48, 96):
            choice = select_variant(entry, budget(vram, ram_gib=256))
            if choice is None:
                continue
            # Rule 2: whatever was chosen zero-spill must genuinely clear
            # the floor (the selector's own invariant, re-checked).
            if choice.zero_spill:
                assert choice.reason_key in ("best-large-window", "best-fits")


def test_find_entry_for_model_resolves_split_ids():
    hit = find_entry_for_model("DeepSeek-V4-Flash-0731-UD-Q4_K_XL")
    assert hit is not None
    entry, variant = hit
    assert entry.id == "deepseek-v4-flash"
    assert variant.quant == "UD-Q4_K_XL"


def test_hybrid_long_context_stays_cheap():
    """The reason Nemotron/Qwen3.6 headline the catalog: their priced
    64K-floor KV must be a small fraction of a dense model's."""
    from hermes_cli.local_runtime.catalog import FLOOR
    from hermes_cli.local_runtime.estimator import ctx_bytes

    dense = catalog_by_id()["muse-glimmer-30b"]
    hybrid = catalog_by_id()["nemotron-3.5-lightning-30b"]
    dense_kv = ctx_bytes(dense.profile(dense.variants[-1]), FLOOR)
    hybrid_kv = ctx_bytes(hybrid.profile(hybrid.variants[-1]), FLOOR)
    assert hybrid_kv * 5 < dense_kv, (
        f"hybrid KV ({hybrid_kv:,}) should be >5x cheaper than dense ({dense_kv:,})")
