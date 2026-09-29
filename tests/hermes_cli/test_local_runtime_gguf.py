"""Behavior contracts for the managed local runtime's GGUF reader."""

from __future__ import annotations

import re
import struct

from hermes_cli.local_runtime.context_policy import spill_overrides
from hermes_cli.local_runtime.estimator import profile_from_gguf
from hermes_cli.local_runtime.gguf import read_gguf_header


def _gguf_str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _write_gguf(path, metadata: dict, tensors: list[tuple[str, int]]) -> None:
    """Minimal GGUF v3: uint32/string metadata, 1-D f32 tensors of ``elems`` elements."""
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        out += _gguf_str(key)
        out += (struct.pack("<I", 8) + _gguf_str(value) if isinstance(value, str)
                else struct.pack("<II", 4, value))
    for name, elems in tensors:
        out += _gguf_str(name) + struct.pack("<IQIQ", 1, elems, 0, 0)
    path.write_bytes(out)


def test_hybrid_ffn_block_sizes_drive_per_block_spill(tmp_path):
    """#113329 end to end on synthetic metadata: a qwen35-style hybrid (every 4th of 12 blocks
    full attention) spills only the recurrent blocks whose FFN bytes cover the spill."""
    tensors = [("token_embd.weight", 64)]
    for i in range(12):
        tensors += [(f"blk.{i}.attn_norm.weight", 16), (f"blk.{i}.ffn_norm.weight", 16),
                    (f"blk.{i}.ffn_up.weight", 256), (f"blk.{i}.ffn_down.weight", 240)]
    gguf = tmp_path / "hybrid.gguf"
    _write_gguf(gguf, {
        "general.architecture": "qwen35", "qwen35.block_count": 12,
        "qwen35.context_length": 65536, "qwen35.full_attention_interval": 4,
        "qwen35.attention.head_count": 8, "qwen35.attention.head_count_kv": 2,
        "qwen35.attention.key_length": 128, "qwen35.attention.value_length": 128,
    }, tensors)

    header = read_gguf_header(gguf)
    per_block = (16 + 256 + 240) * 4   # f32; attn_norm is not an FFN tensor
    assert header.ffn_block_bytes == {i: per_block for i in range(12)}

    profile = profile_from_gguf(header)
    assert header.head_counts_kv() == [0, 0, 0, 2] * 3
    args = spill_overrides(profile, 4 * per_block + 1)
    pattern = args[1].removesuffix("=CPU")
    moved = [i for i in range(12) if re.search(pattern, f"blk.{i}.ffn_up.weight")]
    assert moved == [0, 1, 2, 4, 5]
    assert not re.search(pattern, "blk.1.attn_norm.weight")


def test_reader_sizes_mxfp4_tensor_blocks(tmp_path):
    """MXFP4 stores 32 elements in one 17-byte block."""
    name = b"token_embd.weight"
    gguf = tmp_path / "gpt-oss.gguf"
    gguf.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 1, 0)
        + struct.pack("<Q", len(name))
        + name
        + struct.pack("<IQIQ", 1, 64, 39, 0)
    )

    header = read_gguf_header(gguf)

    assert header.tensor_bytes == 34
    assert header.embd_table_bytes == header.tensor_bytes


def test_reader_sizes_upstream_ternary_tensor_blocks(tmp_path):
    """TQ1_0 / TQ2_0 (ggml-common.h) pack 256 elements into 54 / 66 bytes."""
    for ggml_type, block_bytes in ((34, 54), (35, 66)):
        gguf = tmp_path / f"bitnet-{ggml_type}.gguf"
        gguf.write_bytes(
            b"GGUF"
            + struct.pack("<IQQ", 3, 1, 0)
            + _gguf_str("token_embd.weight")
            + struct.pack("<IQIQ", 1, 512, ggml_type, 0)
        )

        assert read_gguf_header(gguf).tensor_bytes == 2 * block_bytes
