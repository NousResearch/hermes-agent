"""Behavior contracts for the managed local runtime's GGUF reader."""

from __future__ import annotations

import struct

import pytest

from hermes_cli.local_runtime.gguf import read_gguf_header


def _one_tensor_gguf(path, ggml_type: int, n_elems: int):
    name = b"token_embd.weight"
    path.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 1, 0)
        + struct.pack("<Q", len(name))
        + name
        + struct.pack("<IQIQ", 1, n_elems, ggml_type, 0)
    )
    return path


def test_reader_sizes_mxfp4_tensor_blocks(tmp_path):
    """MXFP4 stores 32 elements in one 17-byte block."""
    header = read_gguf_header(_one_tensor_gguf(tmp_path / "gpt-oss.gguf", 39, 64))

    assert header.tensor_bytes == 34
    assert header.embd_table_bytes == header.tensor_bytes


@pytest.mark.parametrize(("ggml_type", "block_bytes"), [(34, 54), (35, 66)])
def test_reader_sizes_upstream_ternary_tensor_blocks(tmp_path, ggml_type, block_bytes):
    """TQ1_0 / TQ2_0 (ggml-common.h) pack 256 elements into 54 / 66 bytes."""
    header = read_gguf_header(_one_tensor_gguf(tmp_path / "bitnet.gguf", ggml_type, 512))

    assert header.tensor_bytes == 2 * block_bytes
