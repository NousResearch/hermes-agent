"""Context policy — the window ladder for managed local models.

One contract: any model runs at any window up to its native max; hardware and session depth only
change tokens/s. Constants, not knobs — nothing in this module reads config. The policy encodes
behavior measured on real hardware (llama.cpp, discrete NVIDIA on Windows/WDDM, unified-memory).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hermes_cli.local_runtime.estimator import (
    HardwareBudget, ModelProfile, PhysicsRefusal, ctx_bytes, physics_check)

FLOOR = 64 * 1024                     # = target; one internal constant
_LADDER_GROWTH = 1.5
_GROW_AT_OCCUPANCY = 0.85             # of the current window, at turn boundary
SPEED_FLOOR_TOK_S = 6.0               # deepest measured spill bottomed near this
_EARLY_COST_CTX_FRACTION = 0.15       # bounded early cost when weights spill

# The smallest ladder rung at which compression becomes the exception rather than the routine.
# Measured over 161 real agentic sessions: 66% complete uncompressed in 64K, 82% in 96K, 91% in
# 144K — and the marginal gain past 144K (+6 points for 216K) falls below the quality cost of
# stepping down another quant. The FLOOR remains the guarantee.
TARGET_WINDOW = 144 * 1024

# What a load really costs beyond weights + KV: CUDA contexts and compute buffers at the DEFAULT
# microbatch (-ub 512, no MTP). Measured on a 32 GiB card: a model estimated at 29.3 GiB loaded at
# ~31.2 GiB resident and fit still shaved a layer to CPU. Microbatch/MTP logits buffers are priced
# separately per model (ub_logits_bytes — they scale with vocab and once packed a card 3.9 GiB
# past this constant). Callers add mmproj bytes on top.
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


def initial_window(profile: ModelProfile, budget: HardwareBudget, *, flash_attention: bool = True,
                   overhead_bytes: int = 0) -> WindowDecision | PhysicsRefusal:
    """The launch decision: largest cheap rung, never below the floor.

    Zero-spill rung: weights + ctx + overhead fit usable VRAM entirely. Bounded-early-cost rung
    (weights already exceed VRAM): largest rung whose ctx stays <= ~15% of usable VRAM. Floor
    everywhere, capped at native. ``overhead_bytes`` is runtime cost beyond weights+KV; zero keeps
    this pure physics for decision-table tests, production callers pass it.
    """
    refusal = physics_check(profile, budget, FLOOR, flash_attention=flash_attention)
    if refusal:
        return refusal

    native = profile.n_ctx_train or FLOOR
    rungs = ladder(native)

    def kv(rung: int) -> int:
        return ctx_bytes(profile, rung, flash_attention=flash_attention)

    best_zero_spill: int | None = None
    for rung in rungs:
        if profile.weights_bytes + overhead_bytes + kv(rung) > budget.usable_vram_bytes:
            break
        best_zero_spill = rung

    if best_zero_spill is not None and best_zero_spill >= min(FLOOR, native):
        window = best_zero_spill
        reason = f"largest zero-spill rung ({window // 1024}K)"
    else:
        # Weights spill from turn one (steep-curve model on a small card) — hold the floor, bound
        # the early ctx cost.
        cap = int(budget.usable_vram_bytes * _EARLY_COST_CTX_FRACTION)
        window = min(FLOOR, native)
        for rung in rungs:
            if rung < window:
                continue
            if kv(rung) > cap:
                break
            window = rung
        reason = f"floor held at {window // 1024}K; weights spill (deliberate price of the guarantee)"

    kv_bytes = kv(window)
    return WindowDecision(window=window, reasons=[reason],
                          spill_bytes=max(0, profile.weights_bytes + kv_bytes - budget.usable_vram_bytes),
                          kv_on_gpu=kv_bytes <= budget.usable_vram_bytes)


@dataclass
class GrowthDecision:
    action: str                   # "grow" | "hold" | "compress-default"
    next_window: int | None = None
    reason: str = ""


def growth_decision(profile: ModelProfile, budget: HardwareBudget, *,
                    current_window: int, session_tokens: int, measured_decode_tok_s: float | None,
                    server_idle: bool, flash_attention: bool = True,
                    occupancy_confirmed: bool = False) -> GrowthDecision:
    """One growth evaluation, END-OF-TURN ONLY (recurrent state cannot rewind mid-sequence).

    Gate order: occupancy (~85%) → native cap → idleness (growth only on an otherwise-idle
    server) → speed floor (below it compression is the default) → re-fit against LIVE free memory
    (the rung must fit NOW, not at launch). ``occupancy_confirmed`` skips gate 1 when the caller's
    own compression gate already fired, so two edge definitions can't deadlock into
    compress-before-grow.
    """
    if not occupancy_confirmed and session_tokens < current_window * _GROW_AT_OCCUPANCY:
        return GrowthDecision("hold", reason="session below growth occupancy")

    native = profile.n_ctx_train or current_window
    if current_window >= native:
        return GrowthDecision("compress-default", reason="at native window; compression is the only move")

    if not server_idle:
        return GrowthDecision("hold", reason="server busy; re-grant deferred to idle")

    if measured_decode_tok_s is not None and measured_decode_tok_s < SPEED_FLOOR_TOK_S:
        return GrowthDecision(
            "compress-default",
            reason=(f"decode {measured_decode_tok_s:.1f} tok/s below the "
                    f"~{SPEED_FLOOR_TOK_S:.0f} tok/s floor; growth is now an "
                    "explicit per-session choice"))

    next_rung = next((r for r in ladder(native) if r > current_window), native)

    # Re-fit against live free memory: allocation beyond residency is the slow path, so a rung
    # that no longer fits doesn't get granted.
    kv = ctx_bytes(profile, next_rung, flash_attention=flash_attention)
    if profile.weights_bytes + kv > budget.usable_vram_bytes + budget.ram_available_bytes:
        return GrowthDecision("compress-default", reason="next rung exceeds physics; compression instead")

    return GrowthDecision("grow", next_window=next_rung,
                          reason=f"rung {current_window // 1024}K -> {next_rung // 1024}K")


def spill_overrides(profile: ModelProfile) -> list[str]:
    """-ot placement for spilled configs: expert/FFN weights to host so attention + KV stay
    GPU-resident. MoE gets the expert pattern; hybrids push recurrent-layer FFNs (their
    n_head_kv==0 layers carry no KV worth protecting)."""
    if profile.moe:
        return ["-ot", r"blk\.\d+\.ffn_.*_exps\.weight=CPU"]
    if profile.recurrent_layer_count:
        return ["-ot", r"blk\.\d+\.ffn_.*\.weight=CPU"]
    return []  # dense: fit's back-to-front layer cut is the only axis


def launch_args(profile: ModelProfile, decision: WindowDecision, *, flash_attention: bool = True,
                mtp_capable: bool = False, mtp_draft_depth: int = 3, uma: bool = False,
                mtp_prefill: bool = False) -> list[str]:
    """Per-model launch flags from a window decision. Explicit -c puts fit into
    spill-weights-and-hold-ctx; q8 KV cache wherever flash attention exists; -ot placement on
    spilled configs — DISCRETE cards only."""
    args = ["-c", str(decision.window)]
    if mtp_capable:
        args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(mtp_draft_depth),
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


def ub_logits_bytes(n_vocab: int, *, mtp_capable: bool, mtp_prefill: bool = False) -> int:
    """GPU logits/compute-buffer cost of the microbatch posture chosen by launch_args, priced from
    the model's own vocab and calibrated against measured server RSS (Qwen3.8 Q4, both postures,
    three windows)."""
    v = max(0, int(n_vocab))
    if mtp_capable and mtp_prefill:
        return int(2048 * v * 4 * 1.5)
    if mtp_capable:
        return 512 * v * 4 * 2
    return 2048 * v * 4
