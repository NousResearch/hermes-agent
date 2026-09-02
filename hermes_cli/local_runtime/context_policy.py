"""Context policy — the window ladder for managed local models.

One contract: any model runs at any window up to its native max; hardware
and session depth only change tokens/s. Constants, not knobs — nothing in
this module reads config.

The policy encodes behavior measured on real hardware (llama.cpp,
discrete NVIDIA GPUs on Windows/WDDM, and unified-memory devices):

- Windows never over-allocates VRAM ahead of need. On WDDM, allocating
  past residency slows decode roughly 9x even at identical conversation
  depth — the driver silently demotes pages instead of failing. Every
  window grant therefore re-fits against live memory at grant time.
- Models launch at the largest window that fits entirely in GPU memory
  (zero-spill) and grow toward their native max as the session needs
  room, at request boundaries only.
- Growth re-prefills the conversation into the larger window. Measured
  cost is comparable to save/restore on discrete GPUs, and recurrent or
  hybrid-attention models cannot rewind mid-sequence anyway, so
  re-prefill is the only mechanism that works for every architecture.
- Every recommended model gets at least a 64K window. When weights alone
  exceed VRAM, the fit deliberately spills weights to host RAM to
  protect that floor (measured: an explicit context size makes the fit
  spill weights and hold the window rather than shrink it).
- Below ~6 tok/s decode, growth stops and compression becomes the
  default; deeper context is an explicit per-session choice. The deepest
  measured host-spilled configuration bottomed out near this rate.
- Spilled mixture-of-experts configs pin expert/FFN weights to host so
  attention and KV stay GPU-resident — measured ~1.75x faster than
  spilling layers naively at the same host byte count.
- Speculative decoding (MTP) defaults on only for spilled configs, where
  its speedup is largest (measured 1.43x spilled vs 1.35x resident).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hermes_cli.local_runtime.estimator import (
    HardwareBudget,
    ModelProfile,
    PhysicsRefusal,
    ctx_bytes,
    physics_check,
)

FLOOR = 64 * 1024                     # = target; one internal constant
_LADDER_GROWTH = 1.5
_GROW_AT_OCCUPANCY = 0.85             # of the current window, at turn boundary
SPEED_FLOOR_TOK_S = 6.0               # deepest measured spill bottomed near this
_EARLY_COST_CTX_FRACTION = 0.15       # bounded early cost when weights spill

# TARGET_WINDOW: the smallest ladder rung at which compression becomes the
# exception rather than the routine. Measured over 161 real agentic
# sessions: 66% complete uncompressed in 64K, 82% in 96K, 91% in 144K —
# and the marginal gain past 144K (+6 points for 216K) falls below the
# quality cost of stepping down another quant. Quant selection prefers
# the best build that reaches this; the FLOOR remains the guarantee.
TARGET_WINDOW = 144 * 1024

# What a load really costs beyond weights + KV: CUDA contexts and compute
# buffers at the DEFAULT microbatch (-ub 512, no MTP). Measured on a
# 32 GiB card: a model estimated at 29.3 GiB (weights+KV) loaded at
# ~31.2 GiB resident and the server's own fit still shaved a layer to
# CPU. Microbatch/MTP logits buffers are priced separately per model
# (ub_logits_bytes — they scale with the model's vocab and doubled once
# packed a card 3.9 GiB past this constant). Callers add mmproj bytes on
# top.
RUNTIME_OVERHEAD_BYTES = int(1.5 * (1 << 30))


def ladder(native: int) -> list[int]:
    """64K -> 96K -> 128K -> ... -> native (native always the last rung)."""
    rungs: list[int] = []
    step = float(FLOOR)
    while step < native:
        rungs.append(int(step))
        step *= _LADDER_GROWTH
    rungs.append(native)
    return rungs


@dataclass
class WindowDecision:
    window: int
    spill_bytes: int              # weights displaced to host at this window
    kv_on_gpu: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def spilled(self) -> bool:
        return self.spill_bytes > 0


def initial_window(profile: ModelProfile, budget: HardwareBudget,
                   *, flash_attention: bool = True,
                   overhead_bytes: int = 0) -> WindowDecision | PhysicsRefusal:
    """The launch decision: largest cheap rung, never below the floor.

    Zero-spill rung: weights + ctx + overhead fit usable VRAM entirely.
    Bounded-early-cost rung: weights already exceed VRAM; take the largest
    rung whose ctx stays <= ~15% of usable VRAM.
    Floor everywhere, capped at native.

    ``overhead_bytes``: runtime cost beyond weights+KV (RUNTIME_OVERHEAD
    plus the vision projector when one loads). Zero keeps this function
    pure physics for decision-table tests; production callers pass it.
    """
    refusal = physics_check(profile, budget, FLOOR, flash_attention=flash_attention)
    if refusal:
        return refusal

    native = profile.n_ctx_train or FLOOR
    rungs = ladder(native)

    reasons: list[str] = []
    best_zero_spill: int | None = None
    for rung in rungs:
        need = (profile.weights_bytes + overhead_bytes
                + ctx_bytes(profile, rung, flash_attention=flash_attention))
        if need <= budget.usable_vram_bytes:
            best_zero_spill = rung
        else:
            break

    if best_zero_spill is not None and best_zero_spill >= min(FLOOR, native):
        window = best_zero_spill
        reasons.append(f"largest zero-spill rung ({window // 1024}K)")
    else:
        # Weights spill from turn one (steep-curve model on a small card) —
        # hold the floor, bound the early ctx cost.
        cap = int(budget.usable_vram_bytes * _EARLY_COST_CTX_FRACTION)
        window = min(FLOOR, native)
        for rung in rungs:
            if rung < window:
                continue
            if ctx_bytes(profile, rung, flash_attention=flash_attention) <= cap:
                window = rung
            else:
                break
        reasons.append(f"floor held at {window // 1024}K; weights spill (deliberate price of the guarantee)")

    kv = ctx_bytes(profile, window, flash_attention=flash_attention)
    spill = max(0, profile.weights_bytes + kv - budget.usable_vram_bytes)
    return WindowDecision(window=window, spill_bytes=spill,
                          kv_on_gpu=kv <= budget.usable_vram_bytes,
                          reasons=reasons)


@dataclass
class GrowthDecision:
    action: str                   # "grow" | "hold" | "compress-default"
    next_window: int | None = None
    reason: str = ""


def growth_decision(profile: ModelProfile, budget: HardwareBudget, *,
                    current_window: int, session_tokens: int,
                    measured_decode_tok_s: float | None,
                    server_idle: bool,
                    flash_attention: bool = True,
                    occupancy_confirmed: bool = False) -> GrowthDecision:
    """One growth evaluation, END-OF-TURN ONLY (caller guarantees the turn
    boundary; recurrent state cannot rewind mid-sequence).

    Gate ordering:
    1. occupancy (~85%) — nothing to do before the edge;
    2. native cap — the contract tops out at trained context;
    3. idleness — growth re-grants only on an otherwise-idle
       server (concurrency design);
    4. speed floor — below it, compression becomes the default and deeper
       is an explicit user choice;
    5. re-fit against LIVE free memory (the rung must fit residency
       NOW, not at launch time — over-allocation is the slow path).

    ``occupancy_confirmed``: the caller has independently established that
    the session is at its window's edge (the agent's compression gate fired
    on its own threshold). Skips gate 1 so two separately-derived edge
    definitions can't deadlock into compress-before-grow.
    """
    if not occupancy_confirmed and session_tokens < current_window * _GROW_AT_OCCUPANCY:
        return GrowthDecision("hold", reason="session below growth occupancy")

    native = profile.n_ctx_train or current_window
    if current_window >= native:
        return GrowthDecision("compress-default",
                              reason="at native window; compression is the only move")

    if not server_idle:
        return GrowthDecision("hold", reason="server busy; re-grant deferred to idle")

    if measured_decode_tok_s is not None and measured_decode_tok_s < SPEED_FLOOR_TOK_S:
        return GrowthDecision(
            "compress-default",
            reason=(f"decode {measured_decode_tok_s:.1f} tok/s below the "
                    f"~{SPEED_FLOOR_TOK_S:.0f} tok/s floor; growth is now an "
                    "explicit per-session choice"))

    next_rung = next((r for r in ladder(native) if r > current_window), native)

    # Re-fit against live free memory: allocation beyond residency is the
    # slow path, so a rung that no longer fits doesn't get granted.
    kv = ctx_bytes(profile, next_rung, flash_attention=flash_attention)
    total_need = profile.weights_bytes + kv
    if total_need > budget.usable_vram_bytes + budget.ram_available_bytes:
        return GrowthDecision("compress-default",
                              reason="next rung exceeds physics; compression instead")

    return GrowthDecision("grow", next_window=next_rung,
                          reason=f"rung {current_window // 1024}K -> {next_rung // 1024}K")


def spill_overrides(profile: ModelProfile) -> list[str]:
    """-ot placement for spilled configs: expert/FFN weights to host so
    attention + KV stay GPU-resident. MoE gets the expert pattern;
    hybrids push recurrent-layer FFNs (their n_head_kv==0 layers carry no
    KV worth protecting)."""
    if profile.moe:
        return ["-ot", r"blk\.\d+\.ffn_.*_exps\.weight=CPU"]
    if profile.recurrent_layer_count:
        return ["-ot", r"blk\.\d+\.ffn_.*\.weight=CPU"]
    return []  # dense: fit's back-to-front layer cut is the only axis


def launch_args(profile: ModelProfile, decision: WindowDecision, *,
                flash_attention: bool = True,
                mtp_capable: bool = False,
                mtp_draft_depth: int = 3,
                uma: bool = False,
                mtp_prefill: bool = False) -> list[str]:
    """Per-model launch flags from a window decision. Explicit -c puts fit
    into spill-weights-and-hold-ctx; q8 KV cache wherever flash attention
    exists; -ot placement on spilled configs — DISCRETE cards only.

    ``uma``: on unified memory there is no bus to protect tensors from —
    "CPU" and "GPU" are the same silicon, and pinning FFN weights to the
    host path just forces CPU compute (measured well over 2x slower than
    letting the allocator place everything). The discrete
    ~1.75x win the -ot pattern encodes does not transfer; a spilled UMA
    config runs unpinned.

    MTP and the large prefill microbatch both win, and whether they may
    STACK is a fit question, not a rule: backend sampling keeps a
    ubatch x vocab x fp32 logits buffer on the GPU and MTP's draft
    context doubles it, so the stacked posture costs a few GiB extra at
    large vocab. Where it fits, it measures best on both axes (Qwen3.8
    Q4 on a 32 GiB card: 93.3 tok/s decode vs 89.5 at ub512, prefill
    slightly better too); where it doesn't, ub512 keeps the decode win
    without packing the card. ``mtp_prefill`` is that fit verdict —
    presets decide it against the priced margin, and ub_logits_bytes()
    prices the same choice so the flag and its cost travel together."""
    args = ["-c", str(decision.window)]
    if mtp_capable:
        args += ["--spec-type", "draft-mtp",
                 "--spec-draft-n-max", str(mtp_draft_depth),
                 "--backend-sampling", "--spec-draft-backend-sampling"]
        if mtp_prefill:
            args += ["-b", "4096", "-ub", "2048"]
    else:
        args += ["-b", "2048", "-ub", "2048"]
    if flash_attention:
        args += ["-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on"]
    if decision.spilled and not uma:
        args += spill_overrides(profile)
    return args


def ub_logits_bytes(n_vocab: int, *, mtp_capable: bool,
                    mtp_prefill: bool = False) -> int:
    """GPU logits/compute-buffer cost of the microbatch posture chosen by
    launch_args, priced from the model's own vocab and calibrated against
    measured server RSS (Qwen3.8 Q4, both postures, three windows):

      stacked (MTP + ub2048):  ubatch x vocab x fp32 x 1.5  (~2.9 GiB at
                               248K vocab; fitted 2.5, rounded up)
      decode  (MTP + ub512):   ubatch x vocab x fp32 x 2    (~1.0 GiB)
      plain   (ub2048):        ubatch x vocab x fp32        (~1.9 GiB)

    Callers add this to RUNTIME_OVERHEAD per model — the flag and its
    price travel together or the fit lies."""
    v = max(0, int(n_vocab))
    if mtp_capable and mtp_prefill:
        return int(2048 * v * 4 * 1.5)
    if mtp_capable:
        return 512 * v * 4 * 2
    return 2048 * v * 4
