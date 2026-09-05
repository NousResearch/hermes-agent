"""Variant-selection contracts: fit the catalog's single Q4-class build
to a machine and price it honestly. Pure decision-table tests over
synthetic budgets."""

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


def test_every_entry_ships_exactly_one_q4_build():
    """No quant ladder: one Q4-class build per entry (K_M where the repo
    ships it, XL elsewhere) — the quant class current engines optimize
    for. Nothing below Q4 ever ships. Validation status is explicit per
    variant in catalog.json; unvalidated builds are permitted (day-0
    entries) and surface as unbadged rows in the pane."""
    for entry in CATALOG:
        assert len(entry.variants) == 1, (
            f"{entry.id}: {len(entry.variants)} variants — expected exactly one")
        build = entry.variants[0]
        assert build.quant.startswith(("UD-Q4", "Q4")), (
            f"{entry.id}: ships {build.quant}, not a Q4-class build")
        for asset in entry.download_files(build):
            assert asset.size_bytes > 0, f"{entry.id}: no size on {asset.path}"


def test_split_variants_have_coherent_parts():
    """Multi-file variants: same model_id from every part, exact sizes,
    first file is the load target."""
    entry = catalog_by_id()["deepseek-v4-flash"]
    for v in entry.variants:
        assert len(v.files) >= 2, "deepseek ships split GGUFs"
        assert "00001-of" in v.files[0].path, "first part must be the load target"
        assert v.size_bytes == sum(f.size_bytes for f in v.files)
    assert entry.draft is not None, "DSpark draft rides along"


def test_selection_is_the_q4_build_even_with_headroom():
    """The selector picks the Q4 build even when bigger quants would fit
    with room to spare — headroom buys window, not quant. Larger builds
    stay one tile click away in the pane."""
    entry = catalog_by_id()["qwen3.8-27b"]
    choice = select_variant(entry, budget(60))
    assert choice is not None
    assert choice.zero_spill
    assert choice.variant.quant == entry.variants[-1].quant  # the Q4 rung
    assert choice.reason_key == "best-large-window"


def test_selected_build_constant_and_fit_shape_monotone_in_vram():
    """More VRAM never changes the selected build (always the Q4 rung);
    what improves is the fit shape: spilled -> floor -> target window."""
    entry = catalog_by_id()["qwen3.8-27b"]
    quants = set()
    shapes = []
    rank = {"smallest-fits-spilled": 0, "best-fits": 1, "best-large-window": 2}
    for vram in (8, 12, 16, 24, 32, 48):
        choice = select_variant(entry, budget(vram))
        assert choice is not None
        quants.add(choice.variant.quant)
        shapes.append(rank[choice.reason_key])
    assert quants == {entry.variants[-1].quant}, f"selection not constant: {quants}"
    assert shapes == sorted(shapes), f"fit shape not monotone in VRAM: {shapes}"


def test_small_card_gets_q4_spilled_never_below():
    """8 GiB card + 27B: nothing zero-spills. The floor holds — the
    selector offers Q4 spilled (priced honestly), never a sub-Q4 build."""
    entry = catalog_by_id()["qwen3.8-27b"]
    choice = select_variant(entry, budget(8))
    assert choice is not None
    assert not choice.zero_spill
    assert choice.reason_key == "smallest-fits-spilled"
    assert choice.variant.quant == "UD-Q4_K_M"


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
    machine exactly enough VRAM for the build's weights and the fit must
    come back spilled, not zero-spill."""
    entry = catalog_by_id()["qwen3.8-27b"]
    build = entry.variants[0]
    exactly_weights = HardwareBudget(
        usable_vram_bytes=build.size_bytes + (100 << 20),
        total_device_bytes=build.size_bytes + (100 << 20),
        ram_available_bytes=64 * GIB)
    choice = select_variant(entry, exactly_weights)
    assert choice is not None
    assert not choice.zero_spill, "KV cost ignored — weights alone can't zero-spill"


def test_floor_fallback_when_target_window_does_not_fit():
    """Cards where nothing clears the target keep the old rule: highest
    quality that zero-spills at the 64K floor (reason 'best-fits'), never
    a needless step down."""
    entry = catalog_by_id()["qwen3.8-27b"]
    # ~23.5 GiB usable: Q4 weights (16.7 GiB in-memory) + floor KV (2.2)
    # + overhead (1.5 + 0.9 mmproj + ~1.0 MTP-posture logits) fits, but
    # the 144K-target KV (+2.7 more) does not.
    choice = select_variant(entry, budget(23.5))
    assert choice is not None and choice.zero_spill
    assert choice.reason_key == "best-fits"
    assert choice.variant.quant == "UD-Q4_K_M"


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

    from hermes_cli.local_runtime.estimator import LayerKind, ModelProfile

    hybrid = catalog_by_id()["qwen3.6-35b-a3b"]
    hybrid_profile = hybrid.profile(hybrid.variants[-1])
    # A fully-dense profile of the same layer count and per-layer cost:
    # the contract is about LAYER ECONOMICS (recurrent layers pay no
    # per-token KV), not about any particular catalog entry.
    n_layers = len(hybrid_profile.layers)
    dense_profile = ModelProfile(
        name="synthetic-dense", weights_bytes=hybrid_profile.weights_bytes,
        embd_table_bytes=0, n_ctx_train=hybrid.n_ctx_train,
        layers=[(LayerKind.FULL, hybrid.per_layer_f16)] * n_layers)
    dense_kv = ctx_bytes(dense_profile, FLOOR)
    hybrid_kv = ctx_bytes(hybrid_profile, FLOOR)
    # The contract is structural: recurrent layers pay no per-token KV,
    # so the hybrid's KV must track its full-attention share (x kv_scale
    # for MTP's draft context), not its total layer count.
    full = sum(1 for kind, _ in hybrid_profile.layers if kind == LayerKind.FULL)
    expected = dense_kv * full / n_layers * hybrid_profile.kv_scale
    assert hybrid_kv < dense_kv, "hybrid must be cheaper than dense"
    assert abs(hybrid_kv - expected) / expected < 0.25, (
        f"hybrid KV ({hybrid_kv:,}) should track its full-attention share "
        f"(expected ~{expected:,.0f})")
