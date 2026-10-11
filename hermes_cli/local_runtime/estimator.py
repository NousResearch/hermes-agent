"""Per-layer context-memory estimator + physics check.

The estimator is ADVISORY: fit's allocation is authoritative at launch and the touch generation is
ground truth after it. Unknown shapes round UP (never underestimate memory).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

from hermes_cli.local_runtime.gguf import GGUFHeader

# q8_0: 34-byte blocks of 32 f16-equivalent elements (exact).
_Q8_BYTES_PER_ELEM = 34 / 32
_F16_BYTES_PER_ELEM = 2.0

# Architectures with a known SWA layer pattern: arch -> fraction of layers that are
# sliding-window. Unknown SWA archs treat every layer as full attention (overestimate; safe).
_SWA_LAYER_FRACTION = {"gemma3": 5 / 6, "gemma2": 1 / 2}

# Architectures whose llama.cpp loader expands a scalar `sliding_window_pattern` period with
# dense_first=True (the full-attention layer opens each period-length cycle instead of closing
# it). Confirmed against llama.cpp's per-arch hparams.set_swa_pattern() calls; every other
# architecture — including the whole Gemma family — uses the dense_first=False default.
_SWA_DENSE_FIRST_ARCHS = {"cohere2moe", "modern-bert", "smallthinker", "laguna"}


def _expand_swa_period(period: int, n_layer: int, dense_first: bool) -> list[int]:
    """Per-layer SWA/full split from a scalar period, matching llama.cpp's
    `llama_hparams::set_swa_pattern()`: one full-attention layer per `period`-length cycle, the
    rest sliding-window."""
    if period <= 0:
        return []
    if dense_first:
        return [0 if i % period == 0 else 1 for i in range(n_layer)]
    return [1 if i % period < period - 1 else 0 for i in range(n_layer)]


# Per-recurrent-layer state allowance (bytes/seq). Deliberately generous: an entire measured
# hybrid slot state is ~99 MB including 8K tokens of full-attn KV, so tens of MiB total is the
# right order; unknown SSM shapes must never underestimate.
_RECURRENT_STATE_PER_LAYER = 4 << 20

# Compute-buffer bytes per microbatch token per window token, per llama.cpp context, for
# architectures whose attention scores the whole window each microbatch. qwen4exp's QSA indexer:
# measured on b11370 with the router's unified KV (every slot sees the full window), device plus
# host buffers, target and MTP head contexts alike (39.5 + 6.1 B; 5.9 GiB per context at 256K and
# -ub 512, ~23 GiB at -ub 2048).
_WINDOW_COMPUTE_BYTES = {"qwen4exp": 46}


class LayerKind(Enum):
    FULL = "full"
    SWA = "swa"
    RECURRENT = "recurrent"


@dataclass
class ModelProfile:
    """Everything the policy needs, decoupled from GGUF parsing so decision-table tests can
    construct profiles directly."""

    name: str
    # Weights the engine loads when it reads lazy_bytes from disk on demand; as_loaded() adds them
    # back on a machine where it reads them up front.
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
    # block index -> FFN weight bytes (from the tensor table); empty when unknown.
    ffn_block_bytes: dict[int, int] = field(default_factory=dict)
    # Bytes of architecture-marked tensors (gguf._LAZY_READ_TENSORS) the engine can read from disk
    # on demand instead of loading.
    lazy_bytes: int = 0
    # Compute-buffer bytes per window token at the launch posture (plan_launch sets it from
    # window_compute_bytes, the microbatch and the context count); zero prices none.
    window_compute_per_token: int = 0

    @property
    def window_compute_bytes(self) -> int:
        return _WINDOW_COMPUTE_BYTES.get(self.architecture, 0)

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
    gpu_name: str = ""          # display name; legacy fallback for performance estimates
    platform: str = ""          # sys.platform of the machine being priced
    gpu_pci_id: int | None = None  # nvidia-smi's packed PCI device/vendor ID
    # The engine runs on the CUDA backend here, so its launch carries a CUDA context. The policy
    # prices runtime overhead from this; Metal/Vulkan/HIP/CPU backends have no such context.
    cuda: bool = False
    # The engine reads lazy tensors from disk here. llama.cpp's own default does so everywhere
    # except integrated GPUs (b11370 #28160); Hermes passes --lazy-mode on to NVIDIA's, so only
    # AMD/Intel integrated GPUs load them up front.
    lazy_reads: bool = True


def as_loaded(profile: ModelProfile, budget: HardwareBudget) -> ModelProfile:
    """The profile as this machine loads it: on-demand tensors count as weights where the engine
    reads them up front."""
    if profile.lazy_bytes and not budget.lazy_reads:
        return replace(profile, weights_bytes=profile.weights_bytes + profile.lazy_bytes, lazy_bytes=0)
    return profile


def profile_from_gguf(header: GGUFHeader) -> ModelProfile:
    kv_heads = header.head_counts_kv()
    dk, dv = header.head_dim_k, header.head_dim_v
    dk_swa = header.key_length_swa or dk
    dv_swa = header.value_length_swa or dv

    # Priority ladder, highest first: (1) the file's own per-layer pattern, array or scalar-period
    # form — architecture-agnostic and exact; (2) a known-architecture fraction, for older files
    # that declare `sliding_window` but no per-layer pattern; (3) no signal at all -> every layer
    # priced as full attention (overestimate; safe).
    pattern = header.sliding_window_pattern
    if pattern is None and header.sliding_window_pattern_period > 0:
        pattern = _expand_swa_period(header.sliding_window_pattern_period, header.n_layer,
                                      header.architecture in _SWA_DENSE_FIRST_ARCHS)
    has_pattern = (pattern is not None and header.sliding_window > 0
                  and len(pattern) == len(kv_heads))
    swa_fraction = _SWA_LAYER_FRACTION.get(header.architecture, 0.0)
    has_fraction = not has_pattern and header.sliding_window > 0 and swa_fraction > 0
    n_attn_total = sum(1 for h in kv_heads if h > 0)
    n_swa = round(n_attn_total * swa_fraction) if has_fraction else 0

    layers: list[tuple[LayerKind, int]] = []
    n_attn_seen = 0
    for i, heads in enumerate(kv_heads):
        if heads == 0:
            layers.append((LayerKind.RECURRENT, 0))
            continue
        if has_pattern:
            is_swa = bool(pattern[i])
        else:
            # Distribute the SWA share across the first n_swa attention layers; only the
            # full/SWA split matters to the totals, not which indexes.
            is_swa = n_attn_seen < n_swa
        layer_dk, layer_dv = (dk_swa, dv_swa) if is_swa else (dk, dv)
        per_token = round(heads * (layer_dk + layer_dv) * _F16_BYTES_PER_ELEM)
        layers.append((LayerKind.SWA if is_swa else LayerKind.FULL, per_token))
        n_attn_seen += 1

    return ModelProfile(
        name=header.path, weights_bytes=header.tensor_bytes - header.lazy_bytes,
        embd_table_bytes=header.embd_table_bytes,
        n_ctx_train=header.n_ctx_train, layers=layers, swa_window=header.sliding_window,
        moe=header.expert_count > 0, architecture=header.architecture, n_vocab=header.n_vocab,
        ffn_block_bytes=dict(header.ffn_block_bytes), lazy_bytes=header.lazy_bytes)


def kv_dtype_factor(flash_attention: bool) -> float:
    """q8_0 with FA (every backend we ship); f16 on exotic non-FA fallbacks — the 64K guarantee
    stands either way, the physics check just prices the doubled KV."""
    return (_Q8_BYTES_PER_ELEM / _F16_BYTES_PER_ELEM) if flash_attention else 1.0


def ctx_bytes(profile: ModelProfile, window: int, *, flash_attention: bool = True) -> int:
    """Memory that grows with the window: full layers linear in T, SWA layers capped at the sliding
    window, recurrent layers constant, KV scaled by profile.kv_scale (MTP draft context), plus the
    posture's window-scaled compute buffers."""
    factor = kv_dtype_factor(flash_attention)
    total = 0.0
    for kind, per_token_f16 in profile.layers:
        if kind == LayerKind.RECURRENT:
            total += _RECURRENT_STATE_PER_LAYER
        elif kind == LayerKind.SWA:
            total += per_token_f16 * factor * min(window, profile.swa_window)
        else:
            total += per_token_f16 * factor * window
    return int(total * profile.kv_scale) + profile.window_compute_per_token * window


@dataclass
class PhysicsRefusal:
    """The only true refusal: weights + floor-KV + state exceed VRAM + RAM. The remedy is a
    smaller quant, never a smaller window."""

    needed_bytes: int
    available_bytes: int
    message: str


def footprint_bytes(profile: ModelProfile, window: int, *, flash_attention: bool = True,
                    overhead_bytes: int = 0) -> int:
    """Complete estimated footprint; the hardware budget already excludes its reserve."""
    return (profile.weights_bytes + ctx_bytes(profile, window, flash_attention=flash_attention)
            + max(0, overhead_bytes))


def physics_check(profile: ModelProfile, budget: HardwareBudget,
                  floor: int, *, flash_attention: bool = True,
                  overhead_bytes: int = 0) -> PhysicsRefusal | None:
    needed = footprint_bytes(profile, min(floor, profile.n_ctx_train or floor),
                             flash_attention=flash_attention, overhead_bytes=overhead_bytes)
    available = budget.usable_vram_bytes + budget.ram_available_bytes
    if needed <= available:
        return None
    gib = 1 << 30
    return PhysicsRefusal(
        needed_bytes=needed, available_bytes=available,
        message=(f"{profile.name}: needs ~{needed / gib:.1f} GiB at the "
                 f"{floor // 1024}K floor but only ~{available / gib:.1f} GiB "
                 "of VRAM+RAM are available — try a smaller model or a supported smaller quant"))
