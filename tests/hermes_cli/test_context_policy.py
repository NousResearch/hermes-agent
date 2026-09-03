"""Context-policy decision-table tests (Rollout 3).

Per the design's verification plan: synthetic per-layer profiles ->
relationships, never exact numbers. Real-model spot checks pin the
estimator to constants measured on real GGUFs (those ARE relationships —
constants with tolerance bands, not change-detecting catalog snapshots).
"""

from __future__ import annotations

import pytest

from hermes_cli.local_runtime.context_policy import (
    FLOOR,
    SPEED_FLOOR_TOK_S,
    GrowthDecision,
    WindowDecision,
    growth_decision,
    initial_window,
    ladder,
    launch_args,
    spill_overrides,
    ub_logits_bytes,
)
from hermes_cli.local_runtime.estimator import (
    HardwareBudget,
    LayerKind,
    ModelProfile,
    PhysicsRefusal,
    ctx_bytes,
    kv_dtype_factor,
    physics_check,
)

GIB = 1 << 30
KIB = 1024


# ── synthetic profiles (per-layer tuples, per the verification plan) ──


def dense(name="dense-32b", layers=64, per_token_f16=4096, weights_gib=20,
          native=128 * 1024) -> ModelProfile:
    return ModelProfile(
        name=name, weights_bytes=weights_gib * GIB, embd_table_bytes=0,
        n_ctx_train=native,
        layers=[(LayerKind.FULL, per_token_f16)] * layers)


def hybrid(name="hybrid-30b", full_layers=16, recurrent_layers=48,
           per_token_f16=4096, weights_gib=22, native=1024 * 1024) -> ModelProfile:
    layers = ([(LayerKind.FULL, per_token_f16)] * full_layers
              + [(LayerKind.RECURRENT, 0)] * recurrent_layers)
    return ModelProfile(name=name, weights_bytes=weights_gib * GIB,
                        embd_table_bytes=0, n_ctx_train=native, layers=layers)


def moe(name="moe-30b", layers=48, per_token_f16=3072, weights_gib=17,
        native=256 * 1024) -> ModelProfile:
    return ModelProfile(name=name, weights_bytes=weights_gib * GIB,
                        embd_table_bytes=0, n_ctx_train=native,
                        layers=[(LayerKind.FULL, per_token_f16)] * layers,
                        moe=True)


def card(vram_gib, ram_gib=64, uma=False) -> HardwareBudget:
    return HardwareBudget(usable_vram_bytes=int(vram_gib * GIB),
                          total_device_bytes=int(vram_gib * GIB),
                          ram_available_bytes=int(ram_gib * GIB), uma=uma)


# ── estimator invariants ─────────────────────────────────────


def test_full_attention_linear_in_window():
    p = dense()
    b32, b64, b128 = (ctx_bytes(p, w * KIB) for w in (32, 64, 128))
    assert abs(b64 / b32 - 2) < 0.01
    assert abs(b128 / b64 - 2) < 0.01


def test_recurrent_state_constant_in_window():
    p = hybrid()
    full_share_32 = ctx_bytes(p, 32 * KIB)
    full_share_1m = ctx_bytes(p, 1024 * KIB)
    # Grows only through the 16 full-attn layers — the recurrent share is
    # identical, so the ratio tracks the full-attn ratio exactly.
    full_only = ModelProfile(name="x", weights_bytes=0, embd_table_bytes=0,
                             n_ctx_train=p.n_ctx_train,
                             layers=[(LayerKind.FULL, 4096)] * 16)
    expected_delta = ctx_bytes(full_only, 1024 * KIB) - ctx_bytes(full_only, 32 * KIB)
    assert abs((full_share_1m - full_share_32) - expected_delta) <= 1


def test_swa_layers_capped_at_window():
    p = ModelProfile(
        name="swa", weights_bytes=0, embd_table_bytes=0, n_ctx_train=128 * KIB,
        layers=[(LayerKind.SWA, 4096)] * 5 + [(LayerKind.FULL, 4096)] * 1,
        swa_window=1024)
    small, big = ctx_bytes(p, 4 * KIB), ctx_bytes(p, 32 * KIB)
    # Full layer grew 8x; the 5 SWA layers stayed capped at 1024 — total
    # growth must land well under the all-full 8x (here ~4.1x).
    assert big / small < 0.6 * 8


def test_q8_factor_is_exactly_34_over_64():
    assert kv_dtype_factor(True) == pytest.approx(34 / 64)
    assert kv_dtype_factor(False) == 1.0


def test_non_fa_fallback_doubles_ctx_cost():
    p = dense()
    assert ctx_bytes(p, FLOOR, flash_attention=False) == pytest.approx(
        ctx_bytes(p, FLOOR, flash_attention=True) * 64 / 34, rel=0.001)


def test_hybrid_vs_dense_100x_class_spread():
    """The whole reason for the per-layer walk: equal-size models, ~100x
    per-token spread between classic dense and a mostly-recurrent hybrid."""
    d = dense(layers=64, per_token_f16=8192)          # 256 KiB/tok class
    h = hybrid(full_layers=4, recurrent_layers=60, per_token_f16=8192)
    window = 256 * KIB
    dense_cost = ctx_bytes(d, window)
    hybrid_cost = ctx_bytes(h, window)
    assert dense_cost / hybrid_cost > 10


# ── measured-constant spot checks (real models, tolerance bands) ──


def test_measured_dense_4b_per_token():
    """Qwen3-4B: 36 layers x 8 kv-heads x (128+128) x 2B = 144 KiB/tok f16."""
    p = ModelProfile(name="qwen3-4b", weights_bytes=0, embd_table_bytes=0,
                     n_ctx_train=262144,
                     layers=[(LayerKind.FULL, 8 * 256 * 2)] * 36)
    per_token_bytes = ctx_bytes(p, 32 * KIB, flash_attention=False) / (32 * KIB)
    assert per_token_bytes == pytest.approx(144 * KIB, rel=0.02)


def test_measured_gdn_27b_per_token_q8():
    """Qwen3.6-27B: 16 full-attn of 64; measured 34.0 KiB/tok @ q8 (B4).
    Per-layer f16 = 34 KiB * 64/34 / 16 = 4 KiB."""
    per_layer_f16 = 4 * KIB
    kv_only = ModelProfile(name="kv", weights_bytes=0, embd_table_bytes=0,
                           n_ctx_train=262144,
                           layers=[(LayerKind.FULL, per_layer_f16)] * 16)
    per_token_bytes = ctx_bytes(kv_only, 128 * KIB) / (128 * KIB)
    assert per_token_bytes == pytest.approx(34 * KIB, rel=0.02)


def test_measured_nemotron_1m_within_band():
    """1M @ q8 measured 3264 MiB KV (B3): ~3.19 KiB/token TOTAL across the
    16 full-attn layers -> per-layer f16 ~384 B. Estimator must land in the
    measured band, not the dense-formula 100x miss."""
    p = hybrid(full_layers=16, recurrent_layers=46, per_token_f16=384,
               native=1024 * KIB)
    total = ctx_bytes(p, 1024 * KIB)
    assert 2.5 * GIB < total < 4.0 * GIB


# ── physics check ────────────────────────────────────────────


def test_physics_refusal_only_past_vram_plus_ram():
    p = dense(weights_gib=60)
    ok = physics_check(p, card(24, ram_gib=64), FLOOR)
    assert ok is None  # 60 GiB weights fit in 24+64
    refused = physics_check(p, card(24, ram_gib=16), FLOOR)
    assert isinstance(refused, PhysicsRefusal)
    assert "smaller quant" in refused.message


def test_physics_check_prices_at_floor_not_native():
    """A 1M-native hybrid must not be refused for its native window —
    the check prices the floor only."""
    p = hybrid(weights_gib=22)
    assert physics_check(p, card(24, ram_gib=8), FLOOR) is None


# ── ladder + initial window ──────────────────────────────────


def test_ladder_shape():
    rungs = ladder(262144)
    assert rungs[0] == FLOOR
    assert rungs[-1] == 262144
    assert all(a < b for a, b in zip(rungs, rungs[1:]))
    # geometric-ish: each step grows, none more than 2x
    assert all(b / a <= 2.0 for a, b in zip(rungs, rungs[1:]))


def test_initial_window_never_below_floor_and_never_above_native():
    for profile in (dense(), hybrid(), moe(), dense(native=32 * KIB)):
        for vram in (8, 16, 24, 32):
            d = initial_window(profile, card(vram))
            if isinstance(d, WindowDecision):
                assert d.window >= min(FLOOR, profile.n_ctx_train)
                assert d.window <= profile.n_ctx_train


def test_initial_window_monotone_in_vram():
    p = dense()
    windows = []
    for vram in (8, 12, 16, 24, 32, 48):
        d = initial_window(p, card(vram))
        assert isinstance(d, WindowDecision)
        windows.append(d.window)
    assert all(a <= b for a, b in zip(windows, windows[1:]))


def test_flat_curve_reaches_native_where_dense_does_not():
    """Design invariant: equal-size hybrid rides to native spill-free where
    the dense model cannot. Hybrid KV priced at the B3 class (~3.2 KiB/tok
    total: per-layer f16 384 B x 16 layers)."""
    h = hybrid(weights_gib=18, native=1024 * KIB, per_token_f16=384)
    d = dense(weights_gib=18, per_token_f16=8192, native=1024 * KIB)
    vram = card(24)
    dh = initial_window(h, vram)
    dd = initial_window(d, vram)
    assert isinstance(dh, WindowDecision) and isinstance(dd, WindowDecision)
    assert dh.window == 1024 * KIB and not dh.spilled
    assert dd.window < 1024 * KIB


def test_dense_on_small_card_holds_floor_and_spills():
    """The deliberate price of the guarantee (design table: dense 32B on
    24 GB starts at the floor with a few GiB spilled)."""
    d = initial_window(dense(weights_gib=20), card(12))
    assert isinstance(d, WindowDecision)
    assert d.window == FLOOR
    assert d.spilled


def test_uma_budget_caps_the_window_through_physics():
    """Unified memory needs no special context rule: the budget already
    encodes the constraint (usable = RAM minus headroom, ram_available=0),
    so the ladder stops where weights + KV genuinely stop fitting."""
    p = hybrid(weights_gib=8, native=1024 * KIB)
    unified = card(38.4, ram_gib=0, uma=True)  # 48 GiB machine, 20% headroom
    d = initial_window(p, unified)
    assert isinstance(d, WindowDecision)
    assert not d.spilled, "UMA budget must produce a resident decision"
    need = 8 * GIB + ctx_bytes(p, d.window)
    assert need <= unified.usable_vram_bytes


# ── growth ───────────────────────────────────────────────────


def _grow(profile, budget, **kw):
    defaults = dict(current_window=FLOOR, session_tokens=int(FLOOR * 0.9),
                    measured_decode_tok_s=40.0, server_idle=True)
    defaults.update(kw)
    return growth_decision(profile, budget, **defaults)


def test_growth_holds_below_occupancy():
    d = _grow(dense(), card(24), session_tokens=int(FLOOR * 0.5))
    assert d.action == "hold"


def test_growth_requires_idle_server():
    d = _grow(dense(), card(24), server_idle=False)
    assert d.action == "hold"
    assert "idle" in d.reason


def test_growth_steps_one_rung_and_is_monotone():
    p = dense(native=262144)
    d = _grow(p, card(32))
    assert d.action == "grow"
    assert d.next_window > FLOOR
    rungs = ladder(262144)
    assert d.next_window == rungs[rungs.index(FLOOR) + 1]


def test_growth_stops_at_native():
    p = dense(native=128 * KIB)
    d = _grow(p, card(48), current_window=128 * KIB,
              session_tokens=int(128 * KIB * 0.9))
    assert d.action == "compress-default"


def test_speed_floor_flips_default_to_compression():
    d = _grow(dense(), card(24), measured_decode_tok_s=SPEED_FLOOR_TOK_S - 2)
    assert d.action == "compress-default"
    assert "explicit per-session choice" in d.reason


def test_growth_refits_against_live_budget():
    """V3C: a rung that no longer fits (external pressure ate the memory)
    is not granted."""
    p = dense(weights_gib=20)
    starved = card(2, ram_gib=1)
    d = _grow(p, starved)
    assert d.action == "compress-default"
    assert "physics" in d.reason


# ── spill placement + launch args ────────────────────────────


def test_spill_overrides_prefer_expert_and_recurrent_ffn():
    assert "exps" in " ".join(spill_overrides(moe()))
    assert "ffn" in " ".join(spill_overrides(hybrid()))
    assert spill_overrides(dense()) == []


def test_launch_args_contract():
    p = moe()
    spilled = WindowDecision(window=FLOOR, spill_bytes=4 * GIB, kv_on_gpu=True)
    resident = WindowDecision(window=131072, spill_bytes=0, kv_on_gpu=True)

    a = launch_args(p, spilled, mtp_capable=True)
    assert a[:2] == ["-c", str(FLOOR)]           # explicit window, always
    assert "q8_0" in a                            # q8 KV under flash attn
    assert "-ot" in a                             # spill placement
    assert "--spec-type" in a                     # MTP on spilled

    # MTP is not gated on spill: resident decode measured +16% at depth 2.
    b = launch_args(p, resident, mtp_capable=True, mtp_draft_depth=2)
    assert "-ot" not in b, "placement is spill-only"
    assert "--spec-type" in b, "MTP must run on resident configs too"
    assert b[b.index("--spec-draft-n-max") + 1] == "2"
    assert "--backend-sampling" in b
    assert "--spec-draft-backend-sampling" in b

    # Stacking MTP with the large microbatch is a FIT question, decided
    # by the caller (presets' posture ladder) and passed as mtp_prefill.
    # Default (no headroom proven): decode posture, small ubatch — the
    # stacked logits buffers once packed a 32 GiB card 3.9 GiB past a
    # fit that ignored them.
    assert "-ub" not in b, "default MTP posture stays at the small ubatch"

    # Headroom proven: the stacked posture carries the large microbatch
    # (measured best on both axes where it fits: 93.3 vs 89.5 tok/s
    # decode on Qwen3.8 Q4). ub_logits_bytes must price the same choice.
    s = launch_args(p, resident, mtp_capable=True, mtp_draft_depth=2,
                    mtp_prefill=True)
    assert "-ub" in s and s[s.index("-ub") + 1] == "2048"
    assert "--spec-type" in s
    v = 248320
    assert ub_logits_bytes(v, mtp_capable=True) == 512 * v * 4 * 2
    assert ub_logits_bytes(v, mtp_capable=True, mtp_prefill=True) == int(2048 * v * 4 * 1.5)

    c = launch_args(p, resident, mtp_capable=False)
    assert "-ub" in c and c[c.index("-ub") + 1] == "2048"  # prefill hint
    assert "--spec-type" not in c

    d = launch_args(p, spilled, flash_attention=False, mtp_capable=False)
    assert "q8_0" not in d                        # f16 on non-FA fallback


def test_launch_args_uma_never_pins_tensors():
    """On unified memory, -ot pinning is off even for spilled decisions:
    "CPU" and "GPU" are the same silicon, and forcing FFN weights down
    the host compute path measures far slower than letting the
    allocator place everything. The discrete ~1.75x -ot win does not
    transfer. Everything else about the launch shape is identical to
    discrete."""
    p = moe()
    spilled = WindowDecision(window=FLOOR, spill_bytes=4 * GIB, kv_on_gpu=True)

    u = launch_args(p, spilled, mtp_capable=False, uma=True)
    assert "-ot" not in u, "UMA must never pin tensors to the host path"
    assert u[:2] == ["-c", str(FLOOR)]            # window contract unchanged
    assert "q8_0" in u                            # KV policy unchanged

    # Same call on discrete keeps the pinning — the flag is the ONLY delta.
    disc = launch_args(p, spilled, mtp_capable=False, uma=False)
    assert "-ot" in disc
    assert [x for x in disc if x != "-ot" and not x.startswith("blk")] == \
        [x for x in u if x != "-ot" and not x.startswith("blk")]


def test_ub_logits_bytes_prices_the_flag_choice():
    """The logits-buffer price must match the microbatch launch_args
    chooses: 2048 x vocab x 4 for non-MTP, 512 x vocab x 4 x 2 for MTP
    (draft context doubles it). 248320-vocab receipts: ~1.9 GiB at
    ub2048, ~0.95 GiB under MTP."""
    v = 248320
    assert ub_logits_bytes(v, mtp_capable=False) == 2048 * v * 4
    assert ub_logits_bytes(v, mtp_capable=True) == 512 * v * 4 * 2
    assert ub_logits_bytes(0, mtp_capable=True) == 0   # unknown vocab: no charge


def test_no_refusal_branch_past_physics():
    """Design invariant: anything past the physics check is servable —
    initial_window never refuses on its own."""
    for vram in (4, 6, 8, 12):
        d = initial_window(dense(weights_gib=20), card(vram, ram_gib=64))
        assert isinstance(d, WindowDecision)


def test_kv_scale_prices_mtp_draft_context():
    """MTP profiles carry kv_scale > 1 (the draft context's KV share,
    calibrated from measured server RSS); ctx_bytes must scale with it so
    every consumer — launch fit, catalog rows, growth — prices what the
    server actually allocates. Four-point calibration held within
    +1.4 GiB conservative, never optimistic."""
    import dataclasses

    p = moe()
    base = ctx_bytes(p, 131072)
    scaled = ctx_bytes(dataclasses.replace(p, kv_scale=1.2), 131072)
    assert scaled == int(base * 1.2)

    # The safety direction: the estimate must never be BELOW measured.
    # (Calibration receipts: predicted-measured was +233..+1400 MiB.)
    assert scaled > base
