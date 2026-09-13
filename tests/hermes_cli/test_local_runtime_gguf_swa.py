"""profile_from_gguf must trust the GGUF file's own per-layer SWA pattern over the hardcoded
architecture-name table (#109551): an architecture absent from `_SWA_LAYER_FRACTION` that still
declares `attention.sliding_window_pattern` should be priced from that pattern, not as fully
global attention."""

from __future__ import annotations

from hermes_cli.local_runtime.estimator import LayerKind, ctx_bytes, profile_from_gguf
from hermes_cli.local_runtime.gguf import GGUFHeader

_ARCH = "gemma4"  # deliberately absent from _SWA_LAYER_FRACTION


def _header(**extra_metadata) -> GGUFHeader:
    metadata = {
        "general.architecture": _ARCH,
        f"{_ARCH}.block_count": 6,
        f"{_ARCH}.context_length": 262144,
        f"{_ARCH}.attention.head_count_kv": 8,
        f"{_ARCH}.attention.sliding_window": 1024,
        f"{_ARCH}.attention.key_length": 512,
        f"{_ARCH}.attention.value_length": 512,
        **extra_metadata,
    }
    return GGUFHeader(path="test.gguf", version=3, metadata=metadata,
                      n_tensors=0, tensor_bytes=0, embd_table_bytes=0)


def test_unknown_arch_without_pattern_falls_back_to_full_attention():
    header = _header()
    profile = profile_from_gguf(header)
    assert all(kind == LayerKind.FULL for kind, _ in profile.layers)


def test_unknown_arch_with_pattern_is_priced_per_layer():
    header = _header(**{
        # 5 SWA layers followed by 1 global layer, exactly as the GGUF declares it.
        f"{_ARCH}.attention.sliding_window_pattern": [1, 1, 1, 1, 1, 0],
        f"{_ARCH}.attention.key_length_swa": 256,
        f"{_ARCH}.attention.value_length_swa": 256,
    })
    profile = profile_from_gguf(header)

    kinds = [kind for kind, _ in profile.layers]
    assert kinds == [LayerKind.SWA] * 5 + [LayerKind.FULL]

    swa_layer_bytes = profile.layers[0][1]
    full_layer_bytes = profile.layers[-1][1]
    # SWA layers use the file's *_swa key/value dims (256+256), not the global ones (512+512).
    assert swa_layer_bytes == round(8 * (256 + 256) * 2.0)
    assert full_layer_bytes == round(8 * (512 + 512) * 2.0)

    # At a window far beyond the 1024-token SWA cap, the per-layer pricing keeps the SWA share
    # capped while only the single global layer keeps growing — the false-positive refusal from
    # the issue came from every layer growing unbounded like this instead.
    small = ctx_bytes(profile, window=2048, flash_attention=False)
    large = ctx_bytes(profile, window=65536, flash_attention=False)
    full_layer_only_growth = full_layer_bytes * (65536 - 2048)
    assert large - small == full_layer_only_growth
