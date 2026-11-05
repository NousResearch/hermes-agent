"""Context policy — the window ladder for managed local models.

One contract: any model runs at any window up to its native max; hardware and session depth only
change tokens/s. Constants, not knobs — nothing in this module reads config. The policy encodes
behavior measured on real hardware (llama.cpp, discrete NVIDIA on Windows/WDDM, unified-memory).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from hermes_cli.local_runtime.estimator import (
    HardwareBudget, LayerKind, ModelProfile, PhysicsRefusal, ctx_bytes, footprint_bytes,
    physics_check)

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

# Partial MoE spill covers the spill budget with this multiple of expert weight. Covering it
# exactly leaves CUDA residency exactly full, and WDDM answers a bare overcommit by demoting
# whole allocations to shared memory — a paging loop that measured ~30x decode collapse on a
# 96 GB card (DeepSeek-V4-Flash, 21/43 expert blocks GPU, ~4 GiB over). The margin keeps
# partial-offload launches inside dedicated VRAM across fit-table rounding.
SPILL_COVER_MARGIN = 1.25

# llama.cpp's default microbatch, and the larger one launch_args passes for faster prefill.
DEFAULT_UBATCH = 512
PREFILL_UBATCH = 2048


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
    refusal = physics_check(profile, budget, FLOOR, flash_attention=flash_attention,
                            overhead_bytes=overhead_bytes)
    if refusal:
        return refusal

    native = profile.n_ctx_train or FLOOR
    rungs = ladder(native)

    def kv(rung: int) -> int:
        return ctx_bytes(profile, rung, flash_attention=flash_attention)

    def need(rung: int) -> int:
        return footprint_bytes(profile, rung, flash_attention=flash_attention,
                               overhead_bytes=overhead_bytes)

    best_zero_spill: int | None = None
    for rung in rungs:
        if need(rung) > budget.usable_vram_bytes:
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
            if (kv(rung) > cap
                    or need(rung) > budget.usable_vram_bytes + budget.ram_available_bytes):
                break
            window = rung
        reason = f"floor held at {window // 1024}K; weights spill (deliberate price of the guarantee)"

    kv_bytes = kv(window)
    return WindowDecision(window=window, reasons=[reason],
                          spill_bytes=max(0, need(window) - budget.usable_vram_bytes),
                          kv_on_gpu=kv_bytes + overhead_bytes <= budget.usable_vram_bytes)


@dataclass
class LaunchPlan:
    decision: WindowDecision | PhysicsRefusal
    mtp_prefill: bool
    overhead_bytes: int
    profile: ModelProfile | None = None   # as priced at the chosen posture (posture_profile)


def posture_profile(profile: ModelProfile, *, mtp_capable: bool, mtp_prefill: bool) -> ModelProfile:
    """``profile`` at one launch posture: window-scaled compute buffers grow with the microbatch
    launch_args passes, and MTP runs a second context beside the target."""
    ubatch = PREFILL_UBATCH if mtp_prefill or not mtp_capable else DEFAULT_UBATCH
    contexts = 2 if mtp_capable else 1
    return replace(profile, window_compute_per_token=profile.window_compute_bytes * ubatch * contexts)


def plan_launch(profile: ModelProfile, budget: HardwareBudget, *, mtp_capable: bool = False,
                fixed_overhead: int = RUNTIME_OVERHEAD_BYTES,
                requested_window: int | None = None) -> LaunchPlan:
    """Window first, then prefill; price both postures at the effective window.

    A restored window may fit only under lean MTP. Evaluate it before discarding it because
    stacked exceeds memory, and keep deliberate spill when neither posture is resident.
    """
    if mtp_capable and profile.kv_scale == 1.0:
        profile = replace(profile, kv_scale=1.2)

    initial: dict[bool, WindowDecision | PhysicsRefusal] = {}

    def candidate(stacked: bool) -> LaunchPlan:
        posture = posture_profile(profile, mtp_capable=mtp_capable, mtp_prefill=stacked)
        overhead = fixed_overhead + ub_logits_bytes(
            profile.n_vocab, mtp_capable=mtp_capable, mtp_prefill=stacked)
        decision = initial_window(posture, budget, overhead_bytes=overhead)
        initial[stacked] = decision
        if isinstance(decision, WindowDecision) and requested_window:
            target = min(requested_window, profile.n_ctx_train or requested_window)
            if target > decision.window and physics_check(
                    posture, budget, target, overhead_bytes=overhead) is None:
                need = footprint_bytes(posture, target, overhead_bytes=overhead)
                decision = WindowDecision(
                    window=target, spill_bytes=max(0, need - budget.usable_vram_bytes),
                    kv_on_gpu=ctx_bytes(posture, target) + overhead <= budget.usable_vram_bytes,
                    reasons=[f"grown window restored ({target // 1024}K)"])
        return LaunchPlan(decision, stacked, overhead, posture)

    lean = candidate(False)
    if not mtp_capable:
        return lean
    stacked = candidate(True)
    if isinstance(stacked.decision, PhysicsRefusal):
        return lean
    if isinstance(lean.decision, PhysicsRefusal):
        return stacked
    if stacked.decision.window < lean.decision.window:
        return lean
    if not stacked.decision.spilled:
        return stacked
    # A previously granted window keeps its spill policy unless lean can make it resident.
    stacked_initial, lean_initial = initial[True], initial[False]
    if (requested_window and isinstance(stacked_initial, WindowDecision)
            and isinstance(lean_initial, WindowDecision) and not stacked_initial.spilled
            and stacked_initial.window >= lean_initial.window
            and stacked.decision.window > stacked_initial.window and lean.decision.spilled):
        return stacked
    return lean


def fit_to_free_memory(plan: LaunchPlan, profile: ModelProfile, live: HardwareBudget, *,
                       mtp_capable: bool = False,
                       fixed_overhead: int = RUNTIME_OVERHEAD_BYTES) -> LaunchPlan:
    """Narrow a resident capacity plan to the window the card's free memory holds now.

    Only the window moves, and only down to the floor. A plan that already spills keeps its
    placement, and so does a card too busy to hold even the floor: moving weights to the CPU on a
    live reading is how a launch once pinned a fitting model to the CPU, because the reading still
    counted memory the outgoing server was about to free.
    """
    decision = plan.decision
    if not isinstance(decision, WindowDecision) or decision.spilled:
        return plan
    now = plan_launch(profile, live, mtp_capable=mtp_capable, fixed_overhead=fixed_overhead)
    if isinstance(now.decision, WindowDecision) and not now.decision.spilled:
        if now.decision.window >= decision.window:
            return plan
        return LaunchPlan(replace(now.decision, reasons=[
            f"{now.decision.window // 1024}K fits beside other programs' GPU memory "
            f"({decision.window // 1024}K would not)"]), now.mtp_prefill, now.overhead_bytes, now.profile)
    floor = min(FLOOR, profile.n_ctx_train or FLOOR)
    if decision.window <= floor:
        return plan
    return LaunchPlan(WindowDecision(window=floor, spill_bytes=0, kv_on_gpu=True, reasons=[
        f"floor held at {floor // 1024}K; other programs hold most of the GPU memory"]),
        plan.mtp_prefill, plan.overhead_bytes, plan.profile)


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


def recurrent_spill_blocks(profile: ModelProfile, spill_bytes: int | None = None) -> list[int]:
    """Recurrent (n_head_kv==0) block indices whose FFN weights cover ``spill_bytes``.

    Takes the fewest blocks, lowest index first (llama.cpp's -ngl also keeps the front blocks on
    the host). Every recurrent block when ``spill_bytes`` is None, when any recurrent block's FFN
    size is unknown, or when all of them together still fall short (fit spills the rest).
    """
    recurrent = [i for i, (kind, _) in enumerate(profile.layers) if kind == LayerKind.RECURRENT]
    if spill_bytes is None or any(i not in profile.ffn_block_bytes for i in recurrent):
        return recurrent
    chosen: list[int] = []
    covered = 0
    for i in recurrent:
        if covered >= spill_bytes:
            break
        chosen.append(i)
        covered += profile.ffn_block_bytes[i]
    return chosen


def moe_spill_blocks(profile: ModelProfile, spill_bytes: int) -> list[int]:
    """Fewest front blocks whose expert weights cover ``spill_bytes`` with margin.

    MoE mirror of ``recurrent_spill_blocks``: routed experts are the dominant, latency-tolerant
    weights, so displacing only enough of them to satisfy the spill budget keeps attention + KV
    GPU-resident and preserves the rest of the VRAM budget for compute buffers. Expert weights
    that stay GPU-resident keep their layer's decode off the host memory bus, so the fewer blocks
    displaced the faster the model runs.

    Every expert block when the model's experts cannot cover the budget — the all-to-host
    placement this falls back to is also the legacy behavior for unknown block tables.
    """
    blocks = [i for i in sorted(profile.ffn_block_bytes) if i < len(profile.layers)]
    covered_target = int(spill_bytes * SPILL_COVER_MARGIN)
    chosen: list[int] = []
    covered = 0
    for i in blocks:
        if covered >= covered_target:
            break
        chosen.append(i)
        covered += profile.ffn_block_bytes[i]
    return chosen


def spill_overrides(profile: ModelProfile, spill_bytes: int | None = None) -> list[str]:
    """-ot placement for spilled configs: expert/FFN weights to host so attention + KV stay
    GPU-resident. MoE spills just enough expert blocks to cover the budget (margin included) —
    all of them only when the experts cannot cover it; hybrids push the FFNs of just enough
    recurrent blocks to cover ``spill_bytes`` (their n_head_kv==0 layers carry no KV worth
    protecting, and full-attention FFNs stay on the GPU)."""
    if profile.moe:
        if spill_bytes:
            blocks = moe_spill_blocks(profile, spill_bytes)
            expert_blocks = [i for i in sorted(profile.ffn_block_bytes) if i < len(profile.layers)]
            if 0 < len(blocks) < len(expert_blocks):
                pattern = "|".join(map(str, sorted(blocks, reverse=True)))
                return ["-ot", r"blk\.(%s)\.ffn_.*_exps\.weight=CPU" % pattern]
        return ["-ot", r"blk\.\d+\.ffn_.*_exps\.weight=CPU"]
    blocks = recurrent_spill_blocks(profile, spill_bytes)
    if blocks:
        return ["-ot", r"blk\.({})\.ffn_.*\.weight=CPU".format("|".join(map(str, blocks)))]
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
            args += ["-b", "4096", "-ub", str(PREFILL_UBATCH)]
    else:
        args += ["-b", str(PREFILL_UBATCH), "-ub", str(PREFILL_UBATCH)]
    if flash_attention:
        args += ["-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on"]
    if decision.spilled and not uma:
        args += spill_overrides(profile, decision.spill_bytes)
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
