"""Behavior contracts for the managed local runtime's GGUF reader."""

from __future__ import annotations

import struct

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
