"""profile_from_gguf must trust the GGUF file's own per-layer SWA pattern over the hardcoded
architecture-name table (#109551): an architecture absent from `_SWA_LAYER_FRACTION` that still
declares `attention.sliding_window_pattern` should be priced from that pattern, not as fully
global attention.

llama.cpp permits `attention.sliding_window_pattern` as either a per-layer array or a scalar
period (writers for Gemma-family and others use the scalar form); the tests below round-trip both
through the real binary parser (`read_gguf_header`), not just `GGUFHeader` built by hand, and cover
malformed/mismatched metadata falling back safely.
"""

from __future__ import annotations

import struct
from pathlib import Path

from hermes_cli.local_runtime.estimator import LayerKind, ctx_bytes, profile_from_gguf
from hermes_cli.local_runtime.gguf import GGUFHeader, read_gguf_header

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


# ── binary round-trip: real GGUF bytes through read_gguf_header(), not GGUFHeader built by hand ──

_STRING, _ARRAY = 8, 9
_SCALAR_FMT = {0: "<B", 1: "<b", 4: "<I", 7: "<?"}


def _pack_value(vtype: int, value) -> bytes:
    if vtype == _STRING:
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw
    if vtype == _ARRAY:
        etype, items = value
        return struct.pack("<IQ", etype, len(items)) + b"".join(_pack_value(etype, i) for i in items)
    return struct.pack(_SCALAR_FMT[vtype], value)


def _write_gguf(path: Path, metadata: dict) -> None:
    """metadata: key -> (vtype, value); ARRAY values are (element_vtype, [items])."""
    body = b"".join(struct.pack("<Q", len(k.encode())) + k.encode() + struct.pack("<I", vt) +
                     _pack_value(vt, v) for k, (vt, v) in metadata.items())
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata)) + body)


def test_binary_array_pattern_uint8_elements_round_trips(tmp_path):
    """Some writers encode the per-layer array with uint8 (not bool) elements; the parser must
    not care which scalar element type carries the 0/1 flags."""
    path = tmp_path / "m.gguf"
    _write_gguf(path, {
        "general.architecture": (_STRING, _ARCH),
        f"{_ARCH}.block_count": (4, 6),
        f"{_ARCH}.context_length": (4, 262144),
        f"{_ARCH}.attention.head_count_kv": (4, 8),
        f"{_ARCH}.attention.sliding_window": (4, 1024),
        f"{_ARCH}.attention.key_length": (4, 512),
        f"{_ARCH}.attention.value_length": (4, 512),
        f"{_ARCH}.attention.sliding_window_pattern": (_ARRAY, (0, [1, 1, 1, 1, 1, 0])),
        f"{_ARCH}.attention.key_length_swa": (4, 256),
        f"{_ARCH}.attention.value_length_swa": (4, 256),
    })
    profile = profile_from_gguf(read_gguf_header(path))
    assert [k for k, _ in profile.layers] == [LayerKind.SWA] * 5 + [LayerKind.FULL]


def test_binary_scalar_period_defaults_to_dense_first_false(tmp_path):
    """Gemma-family writers declare `sliding_window_pattern` as a scalar period rather than a
    per-layer array; for architectures outside the small dense_first=True set this must expand
    with the full-attention layer closing each period (llama.cpp's dense_first=False default)."""
    path = tmp_path / "m.gguf"
    _write_gguf(path, {
        "general.architecture": (_STRING, _ARCH),
        f"{_ARCH}.block_count": (4, 7),
        f"{_ARCH}.context_length": (4, 262144),
        f"{_ARCH}.attention.head_count_kv": (4, 8),
        f"{_ARCH}.attention.sliding_window": (4, 1024),
        f"{_ARCH}.attention.key_length": (4, 512),
        f"{_ARCH}.attention.value_length": (4, 512),
        f"{_ARCH}.attention.sliding_window_pattern": (4, 4),  # scalar period, not an array
    })
    profile = profile_from_gguf(read_gguf_header(path))
    kinds = [k for k, _ in profile.layers]
    assert kinds == [LayerKind.SWA, LayerKind.SWA, LayerKind.SWA, LayerKind.FULL,
                      LayerKind.SWA, LayerKind.SWA, LayerKind.SWA]


def test_binary_scalar_period_honors_dense_first_true_architectures(tmp_path):
    """`laguna` is one of the few llama.cpp architectures whose loader expands the scalar period
    with dense_first=True (the full-attention layer opens each cycle instead of closing it) —
    with 7 layers and period 4 that puts FULL at indices 0 and 4, not just index 3."""
    arch = "laguna"
    path = tmp_path / "m.gguf"
    _write_gguf(path, {
        "general.architecture": (_STRING, arch),
        f"{arch}.block_count": (4, 7),
        f"{arch}.context_length": (4, 262144),
        f"{arch}.attention.head_count_kv": (4, 8),
        f"{arch}.attention.sliding_window": (4, 1024),
        f"{arch}.attention.key_length": (4, 512),
        f"{arch}.attention.value_length": (4, 512),
        f"{arch}.attention.sliding_window_pattern": (4, 4),
    })
    profile = profile_from_gguf(read_gguf_header(path))
    kinds = [k for k, _ in profile.layers]
    assert kinds == [LayerKind.FULL, LayerKind.SWA, LayerKind.SWA, LayerKind.SWA,
                      LayerKind.FULL, LayerKind.SWA, LayerKind.SWA]


def test_binary_length_mismatched_array_falls_back_to_full_attention(tmp_path):
    """A declared array whose length doesn't match block_count is malformed/unusable — must fall
    back to the safe all-global default rather than misaligning it against the layers."""
    path = tmp_path / "m.gguf"
    _write_gguf(path, {
        "general.architecture": (_STRING, _ARCH),
        f"{_ARCH}.block_count": (4, 6),
        f"{_ARCH}.context_length": (4, 262144),
        f"{_ARCH}.attention.head_count_kv": (4, 8),
        f"{_ARCH}.attention.sliding_window": (4, 1024),
        f"{_ARCH}.attention.key_length": (4, 512),
        f"{_ARCH}.attention.value_length": (4, 512),
        f"{_ARCH}.attention.sliding_window_pattern": (_ARRAY, (7, [1, 1, 1, 0])),  # 4 != 6 layers
    })
    profile = profile_from_gguf(read_gguf_header(path))
    assert all(k == LayerKind.FULL for k, _ in profile.layers)
