"""Behavior contracts for the managed local runtime's GGUF reader."""

from __future__ import annotations

import re
import struct

import pytest

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


@pytest.mark.parametrize(("ggml_type", "elems", "expected"), [
    (40, 128, 72),  # NVFP4: 64 elements per block, 4 E4M3 scales + 32 bytes of 4-bit values
    (41, 256, 36),  # Q1_0: 128 elements per block, an f16 scale + 16 bytes of bits
    (42, 128, 36),  # Q2_0: 64 elements per block, an f16 scale + 16 bytes of 2-bit values
])
def test_reader_sizes_ggml_types_added_after_mxfp4(tmp_path, ggml_type, elems, expected):
    """Nemotron 3.5 Bolt stores its SSM projections as NVFP4; an unknown type refuses the model."""
    name = b"blk.0.ssm_in.weight"
    gguf = tmp_path / "bolt.gguf"
    gguf.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 1, 0)
        + struct.pack("<Q", len(name))
        + name
        + struct.pack("<IQIQ", 1, elems, ggml_type, 0)
    )

    assert read_gguf_header(gguf).tensor_bytes == expected


def _ssm_model(path, arch: str, n_layer: int, recurrent: set[int], ssm: dict) -> None:
    tensors = [("token_embd.weight", 64)]
    for i in range(n_layer):
        tensors.append((f"blk.{i}.attn_norm.weight", 16))
        if i in recurrent:
            tensors.append((f"blk.{i}.ssm_conv1d.weight", 16))
    _write_gguf(path, {"general.architecture": arch, f"{arch}.block_count": n_layer,
                       f"{arch}.context_length": 262144,
                       **{f"{arch}.ssm.{k}": v for k, v in ssm.items()}}, tensors)


def test_reader_prices_one_recurrent_state_copy_from_the_ssm_shape(tmp_path):
    """llama.cpp keeps one f32 conv window plus SSM state per recurrent block, per copy. Qwen3.8
    27B's shape is 149.6 MiB: the 1,796 MiB it allocates for 4 slots x 3 MTP positions."""
    gguf = tmp_path / "qwen.gguf"
    _ssm_model(gguf, "qwen35", 64, {i for i in range(64) if (i + 1) % 4},
               {"inner_size": 6144, "state_size": 128, "conv_kernel": 4, "group_count": 16})

    header = read_gguf_header(gguf)

    assert header.recurrent_state_bytes == 48 * 4 * (6144 * 128 + 3 * (6144 + 2 * 16 * 128))
    assert round(header.recurrent_state_bytes / 2**20, 1) == 149.6
    assert profile_from_gguf(header).recurrent_state_bytes == header.recurrent_state_bytes


def test_reader_counts_only_blocks_that_hold_recurrent_state(tmp_path):
    """Nemotron-H's MoE blocks also have no KV heads, but hold no state. Bolt: 40 Mamba blocks of
    88 -> 164.7 MiB per copy, the 659 MiB llama.cpp allocates for 4 slots."""
    gguf = tmp_path / "bolt.gguf"
    _ssm_model(gguf, "nemotron_h_moe", 88, set(range(0, 80, 2)),
               {"inner_size": 8192, "state_size": 128, "conv_kernel": 4, "group_count": 8})

    assert round(read_gguf_header(gguf).recurrent_state_bytes / 2**20, 1) == 164.7


def test_split_model_counts_recurrent_blocks_across_parts(tmp_path):
    ssm = {"inner_size": 64, "state_size": 8, "conv_kernel": 4, "group_count": 1}
    first = tmp_path / "m-00001-of-00002.gguf"
    _ssm_model(first, "qwen35", 4, {0, 1}, ssm)
    _write_gguf(tmp_path / "m-00002-of-00002.gguf", {"split.count": 2},
                [("blk.2.ssm_conv1d.weight", 16), ("blk.3.ssm_conv1d.weight", 16)])

    assert read_gguf_header(first).recurrent_state_bytes == 4 * 4 * (64 * 8 + 3 * (64 + 2 * 8))


def test_no_ssm_shape_prices_no_recurrent_state(tmp_path):
    gguf = tmp_path / "dense.gguf"
    _ssm_model(gguf, "llama", 4, set(), {})

    assert read_gguf_header(gguf).recurrent_state_bytes == 0
