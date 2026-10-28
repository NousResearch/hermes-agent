"""Short-convolution hybrids retain state without Mamba's SSM metadata."""

import struct

from hermes_cli.local_runtime.estimator import LayerKind, ctx_bytes, profile_from_gguf
from hermes_cli.local_runtime.gguf import read_gguf_header


def _string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def _write_shortconv(path):
    full = {2, 6, 10, 14, 18, 22}
    swa = {4, 8, 12, 16, 20}
    recurrent = set(range(24)) - full - swa
    metadata = {
        "general.architecture": (8, _string("lfm3")),
        "lfm3.block_count": (4, struct.pack("<I", 24)),
        "lfm3.context_length": (4, struct.pack("<I", 262144)),
        "lfm3.embedding_length": (4, struct.pack("<I", 2048)),
        "lfm3.shortconv.l_cache": (4, struct.pack("<I", 7)),
        "lfm3.attention.key_length": (4, struct.pack("<I", 128)),
        "lfm3.attention.value_length": (4, struct.pack("<I", 128)),
        "lfm3.attention.sliding_window": (4, struct.pack("<I", 256)),
        "lfm3.attention.head_count_kv": (9, struct.pack("<IQ24I", 4, 24,
            *[0 if i in recurrent else 4 for i in range(24)])),
        "lfm3.attention.sliding_window_pattern": (9, struct.pack("<IQ24?", 7, 24,
            *[i in swa for i in range(24)])),
    }
    tensors = [f"blk.{i}.shortconv.conv.weight" for i in sorted(recurrent)]
    # Projections do not retain state: they must not create extra recurrent blocks.
    tensors += ["blk.2.shortconv.in_proj.weight", "blk.2.shortconv.out_proj.weight"]
    data = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, (kind, value) in metadata.items():
        data += _string(key) + struct.pack("<I", kind) + value
    for name in tensors:
        data += _string(name) + struct.pack("<IQIQ", 1, 7 * 2048, 0, 0)
    path.write_bytes(data)
    return full, swa, recurrent


def test_shortconv_state_and_three_way_attention_layout_roundtrip(tmp_path):
    path = tmp_path / "shortconv.gguf"
    full, swa, recurrent = _write_shortconv(path)

    header = read_gguf_header(path)
    profile = profile_from_gguf(header)

    assert header.recurrent_blocks == recurrent
    assert header.recurrent_state_bytes == 13 * 4 * 2048 * 6 == 638976
    assert profile.recurrent_state_bytes == header.recurrent_state_bytes
    assert {i for i, (kind, _) in enumerate(profile.layers) if kind == LayerKind.FULL} == full
    assert {i for i, (kind, _) in enumerate(profile.layers) if kind == LayerKind.SWA} == swa
    assert {i for i, (kind, _) in enumerate(profile.layers) if kind == LayerKind.RECURRENT} == recurrent
    assert all(nbytes == 2048 for kind, nbytes in profile.layers if kind != LayerKind.RECURRENT)
    assert ctx_bytes(profile, 65536, flash_attention=False) - ctx_bytes(profile, 32768, flash_attention=False) == 6 * 2048 * 32768
