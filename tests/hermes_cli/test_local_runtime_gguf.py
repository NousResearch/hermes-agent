"""Behavior contracts for the managed local runtime's GGUF reader."""

from __future__ import annotations

import struct

from hermes_cli.local_runtime import presets
from hermes_cli.local_runtime.estimator import HardwareBudget
from hermes_cli.local_runtime.gguf import read_gguf_header


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


def test_reader_and_presets_accept_ternary_tensor_types(tmp_path):
    """PQ2_0 and PTQ1_0 headers must remain eligible for local presets."""
    def tensor(name, tensor_type):
        encoded = name.encode()
        return (struct.pack("<Q", len(encoded)) + encoded
                + struct.pack("<IQIQ", 1, 128, tensor_type, 0))

    gguf = tmp_path / "Ternary-Bonsai-PQ2_0.gguf"
    gguf.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 2, 0)
        + tensor("token_embd.weight", 142)
        + tensor("blk.0.weight", 143)
    )

    header = read_gguf_header(gguf)

    assert header.tensor_bytes == 34 + 28
    assert header.embd_table_bytes == 34
    generated = presets.generate_presets(
        tmp_path, HardwareBudget(2 << 30, 2 << 30, 8 << 30), tmp_path / "presets.ini")
    assert [entry.model_id for entry in generated] == ["Ternary-Bonsai-PQ2_0"]
