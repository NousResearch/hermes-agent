"""Per-layer context-memory estimator + physics check.

The estimator is ADVISORY: fit's allocation is authoritative at launch and the touch generation is
ground truth after it. Unknown shapes round UP (never underestimate memory).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from hermes_cli.local_runtime.gguf import GGUFHeader

# q8_0: 34-byte blocks of 32 f16-equivalent elements (exact).
_Q8_BYTES_PER_ELEM = 34 / 32
_F16_BYTES_PER_ELEM = 2.0

# Architectures with a known SWA layer pattern: arch -> fraction of layers that are
# sliding-window. Unknown SWA archs treat every layer as full attention (overestimate; safe).
_SWA_LAYER_FRACTION = {"gemma3": 5 / 6, "gemma2": 1 / 2}

# Per-recurrent-layer state allowance (bytes/seq). Deliberately generous: an entire measured
# hybrid slot state is ~99 MB including 8K tokens of full-attn KV, so tens of MiB total is the
# right order; unknown SSM shapes must never underestimate.
_RECURRENT_STATE_PER_LAYER = 4 << 20


class LayerKind(Enum):
    FULL = "full"
    SWA = "swa"
    RECURRENT = "recurrent"


@dataclass
class ModelProfile:
    """Everything the policy needs, decoupled from GGUF parsing so decision-table tests can
    construct profiles directly."""

    name: str
    weights_bytes: int
    embd_table_bytes: int
    n_ctx_train: int
    layers: list[tuple[LayerKind, int]]   # (kind, kv_bytes_per_token_f16); SWA capped, recurrent ignored
    swa_window: int = 0
    moe: bool = False
    architecture: str = ""
    n_vocab: int = 0            # prices logits buffers (ubatch x vocab)
    # Context-cost multiplier. MTP spec decode keeps a small draft context beside the main one;
    # calibrated against four measured server-RSS points on Qwen3.8 Q4 (128K/221K/256K, both
    # postures): the draft adds ~17% to per-token KV; 1.2 rounds up so the error stays on the safe
    # side (+250 MiB at 256K, never negative).
    kv_scale: float = 1.0

    @property
    def per_token_kv_f16(self) -> int:
        """Uncapped per-token KV cost (full + SWA share)."""
        return sum(b for kind, b in self.layers if kind != LayerKind.RECURRENT)

    @property
    def recurrent_layer_count(self) -> int:
        return sum(1 for kind, _ in self.layers if kind == LayerKind.RECURRENT)


@dataclass
class HardwareBudget:
    """Memory the physics check may budget against. Discrete cards may trust the device query;
    unified-memory devices must budget from OS free memory minus headroom (device queries observed
    off by 3x). Callers construct this accordingly; the estimator just consumes it."""

    usable_vram_bytes: int      # live free (discrete) / derived (UMA)
    total_device_bytes: int
    ram_available_bytes: int
    uma: bool = False


def profile_from_gguf(header: GGUFHeader) -> ModelProfile:
    kv_heads = header.head_counts_kv()
    dk, dv = header.head_dim_k, header.head_dim_v
    swa_fraction = _SWA_LAYER_FRACTION.get(header.architecture, 0.0)
    has_swa = header.sliding_window > 0 and swa_fraction > 0

    layers: list[tuple[LayerKind, int]] = []
    n_attn_seen = 0
    n_attn_total = sum(1 for h in kv_heads if h > 0)
    n_swa = round(n_attn_total * swa_fraction) if has_swa else 0
    for heads in kv_heads:
        if heads == 0:
            layers.append((LayerKind.RECURRENT, 0))
            continue
        per_token = round(heads * (dk + dv) * _F16_BYTES_PER_ELEM)
        # Distribute the SWA share across the first n_swa attention layers; only the full/SWA
        # SPLIT matters to the totals, not which indexes.
        kind = LayerKind.SWA if n_attn_seen < n_swa else LayerKind.FULL
        layers.append((kind, per_token))
        n_attn_seen += 1

    return ModelProfile(
        name=header.path, weights_bytes=header.tensor_bytes, embd_table_bytes=header.embd_table_bytes,
        n_ctx_train=header.n_ctx_train, layers=layers, swa_window=header.sliding_window,
        moe=header.expert_count > 0, architecture=header.architecture, n_vocab=header.n_vocab)


def kv_dtype_factor(flash_attention: bool) -> float:
    """q8_0 with FA (every backend we ship); f16 on exotic non-FA fallbacks — the 64K guarantee
    stands either way, the physics check just prices the doubled KV."""
    return (_Q8_BYTES_PER_ELEM / _F16_BYTES_PER_ELEM) if flash_attention else 1.0


def ctx_bytes(profile: ModelProfile, window: int, *, flash_attention: bool = True) -> int:
    """Context memory for one window: full layers linear in T, SWA layers capped at the sliding
    window, recurrent layers constant. Scaled by profile.kv_scale (MTP draft context)."""
    factor = kv_dtype_factor(flash_attention)
    total = 0.0
    for kind, per_token_f16 in profile.layers:
        if kind == LayerKind.RECURRENT:
            total += _RECURRENT_STATE_PER_LAYER
        elif kind == LayerKind.SWA:
            total += per_token_f16 * factor * min(window, profile.swa_window)
        else:
            total += per_token_f16 * factor * window
    return int(total * profile.kv_scale)


@dataclass
class PhysicsRefusal:
    """The only true refusal: weights + floor-KV + state exceed VRAM + RAM. The remedy is a
    smaller quant, never a smaller window."""

    needed_bytes: int
    available_bytes: int
    message: str


# OS free-memory floor for UNIFIED-memory machines. On a shared pool the model's
# working set comes out of the same physical RAM the OS lives in, so "it fits the
# budget" is not the same as "the machine still works". The reserve in
# hardware.py shapes the budget; this is the hard admission gate that refuses a
# load which would leave the OS too little to run.
#
# Shape is min(cap, max(floor, fraction)) — the same idiom the discrete path
# already uses. Justified per tier: a single constant is wrong (3 GiB of a 512 GiB
# Mac Pro is 0.6%), and an UNCAPPED max(3 GiB, 10%) is also wrong — it would newly
# refuse a 240 GiB model on a 256 GiB machine, a regression. The cap is
# load-bearing, not decoration.
_OS_FLOOR_FRACTION = 0.10
_OS_FLOOR_MIN = 3 << 30            # 3 GiB — binds up to 32 GiB of RAM
_OS_FLOOR_CAP = 16 << 30           # 16 GiB — binds from 160 GiB of RAM up


def os_free_floor_bytes(total_ram: int) -> int:
    """Bytes that must remain free for the OS after a model is resident."""
    return int(min(_OS_FLOOR_CAP, max(_OS_FLOOR_MIN, total_ram * _OS_FLOOR_FRACTION)))


def physics_check(profile: ModelProfile, budget: HardwareBudget,
                  floor: int, *, flash_attention: bool = True,
                  overhead_bytes: int = 0) -> PhysicsRefusal | None:
    """Refuse a model that cannot physically run.

    ``overhead_bytes`` is runtime cost beyond weights+KV. It defaults to 0 so the
    pure-physics decision-table tests are unchanged, but production callers pass
    it: without it every refusal decision was under-priced by that amount.
    """
    needed = (profile.weights_bytes
              + ctx_bytes(profile, min(floor, profile.n_ctx_train or floor),
                          flash_attention=flash_attention)
              + overhead_bytes)
    available = budget.usable_vram_bytes + budget.ram_available_bytes
    gib = 1 << 30
    if needed > available:
        return PhysicsRefusal(
            needed_bytes=needed, available_bytes=available,
            message=(f"{profile.name}: needs ~{needed / gib:.1f} GiB at the "
                     f"{floor // 1024}K floor but only ~{available / gib:.1f} GiB "
                     "of VRAM+RAM exist — try a smaller quant (UD-Q3/Q2)"))

    # Unified memory only: on a discrete card the weights live in VRAM and do not
    # consume host RAM, so the OS floor does not apply there.
    if budget.uma:
        total_ram = budget.total_device_bytes
        os_floor = os_free_floor_bytes(total_ram)
        left_for_os = total_ram - needed
        if left_for_os < os_floor:
            return PhysicsRefusal(
                needed_bytes=needed, available_bytes=available,
                message=(f"{profile.name}: needs ~{needed / gib:.1f} GiB of the "
                         f"~{total_ram / gib:.1f} GiB shared pool, leaving only "
                         f"~{left_for_os / gib:.1f} GiB for the OS (floor is "
                         f"~{os_floor / gib:.1f} GiB) — the machine would swap; "
                         "try a smaller quant (UD-Q3/Q2) or a shorter context"))
    return None
