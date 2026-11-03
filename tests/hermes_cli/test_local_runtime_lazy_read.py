"""Tensors llama.cpp reads from disk on demand are not priced as loaded weights.

The engine's model loader leaves a tensor its architecture marks TENSOR_READ_LAZY in the file and
reads that tensor's rows when a token needs them. Under the default ``--lazy-mode auto`` it does so
only for a marked tensor larger than 4 GiB, and never on an integrated GPU, where it loads the
table in full. Qwen3.8 Flash Next's 26.8 GiB per-layer embedding table is one: priced as loaded,
the IQ4_XS build needs ~87 GiB; read on demand, it holds ~60 GiB. On a DGX Spark (GB10) the table
took 27.2 GiB of resident memory under ``auto`` and 0.4 GiB under ``--lazy-mode on``, at the same
decode speed and within 2% of the prefill speed, so Hermes passes ``on`` wherever it prices the
table as left on disk. AMD/Intel integrated GPUs keep the engine's default (reading on demand
halved their prefill, llama.cpp #28160) and pay for the table.
"""

from __future__ import annotations

import struct

from hermes_cli.local_runtime import presets
from hermes_cli.local_runtime.catalog import AssetFile, QuantVariant
from hermes_cli.local_runtime.estimator import HardwareBudget, as_loaded, profile_from_gguf
from hermes_cli.local_runtime.gguf import read_gguf_header

GIB = 1 << 30
OVER_4_GIB = (4 * GIB) // 4 + 256   # f32 elements: 4 GiB + 1 KiB
EXACTLY_4_GIB = (4 * GIB) // 4


def _gguf_str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _write_gguf(path, metadata: dict, tensors: list[tuple[str, int]]) -> None:
    """Header-only GGUF v3: uint32/string metadata, 1-D f32 tensors of ``elems`` elements. The
    reader never touches tensor data, so a header prices exactly like the full file."""
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        out += _gguf_str(key)
        out += (struct.pack("<I", 8) + _gguf_str(value) if isinstance(value, str)
                else struct.pack("<II", 4, value))
    for name, elems in tensors:
        out += _gguf_str(name) + struct.pack("<IQIQ", 1, elems, 0, 0)
    path.write_bytes(out)


def _arch(name: str) -> dict:
    return {"general.architecture": name, f"{name}.block_count": 4,
            f"{name}.context_length": 65536, f"{name}.attention.head_count": 8,
            f"{name}.attention.head_count_kv": 2, f"{name}.attention.key_length": 128}


def test_a_marked_table_over_4_gib_stays_out_of_the_loaded_weights(tmp_path):
    gguf = tmp_path / "flash.gguf"
    _write_gguf(gguf, _arch("qwen4exp"), [("per_layer_token_embd.weight", OVER_4_GIB),
                                          ("blk.0.attn_q.weight", 16 << 20)])

    header = read_gguf_header(gguf)

    assert header.lazy_bytes == OVER_4_GIB * 4
    assert profile_from_gguf(header).weights_bytes == 64 << 20


def test_a_marked_table_of_4_gib_or_less_loads_like_any_weight(tmp_path):
    """The engine's auto mode skips a tensor of at most 4 GiB: reading it row by row costs more
    than it saves, so it loads."""
    gguf = tmp_path / "small.gguf"
    _write_gguf(gguf, _arch("qwen4exp"), [("per_layer_token_embd.weight", EXACTLY_4_GIB)])

    header = read_gguf_header(gguf)

    assert header.lazy_bytes == 0
    assert profile_from_gguf(header).weights_bytes == header.tensor_bytes


def test_an_architecture_that_marks_nothing_loads_the_same_table(tmp_path):
    """Only the architecture's own model code marks a tensor lazy; the name alone means nothing."""
    gguf = tmp_path / "other.gguf"
    _write_gguf(gguf, _arch("llama"), [("per_layer_token_embd.weight", OVER_4_GIB)])

    assert profile_from_gguf(read_gguf_header(gguf)).weights_bytes == OVER_4_GIB * 4


def test_the_table_is_found_in_a_later_split_part(tmp_path):
    """The architecture lives in part 1 and the table in a later part (Flash Next keeps it in
    part 2), so the decision has to be made for the model, not per file."""
    _write_gguf(tmp_path / "flash-00001-of-00002.gguf", _arch("qwen4exp"), [])
    _write_gguf(tmp_path / "flash-00002-of-00002.gguf",
                {"split.no": 1, "split.count": 2, "split.tensors.count": 0},
                [("per_layer_token_embd.weight", OVER_4_GIB), ("blk.0.attn_q.weight", 16 << 20)])

    header = read_gguf_header(tmp_path / "flash-00001-of-00002.gguf")

    assert header.lazy_bytes == OVER_4_GIB * 4
    assert profile_from_gguf(header).weights_bytes == 64 << 20


def test_a_catalog_build_is_priced_without_its_lazy_tables():
    """Before download the catalog stands in for the tensor table, so the build carries the bytes
    the engine leaves on disk."""
    build = QuantVariant(quant="UD-IQ4_XS", lazy_bytes=30 * GIB,
                         files=(AssetFile("m-00001-of-00002.gguf", 10 << 20),
                                AssetFile("m-00002-of-00002.gguf", 90 * GIB)))

    assert build.size_bytes == 90 * GIB + (10 << 20)
    assert build.weights_bytes == build.size_bytes - 30 * GIB


def _unified(*, lazy_reads: bool) -> HardwareBudget:
    return HardwareBudget(usable_vram_bytes=64 * GIB, total_device_bytes=80 * GIB,
                          ram_available_bytes=0, uma=True, lazy_reads=lazy_reads)


def test_an_engine_that_loads_the_table_up_front_pays_for_it(tmp_path):
    gguf = tmp_path / "flash.gguf"
    _write_gguf(gguf, _arch("qwen4exp"), [("per_layer_token_embd.weight", OVER_4_GIB),
                                          ("blk.0.attn_q.weight", 16 << 20)])
    profile = profile_from_gguf(read_gguf_header(gguf))

    assert as_loaded(profile, _unified(lazy_reads=True)).weights_bytes == 64 << 20
    assert as_loaded(profile, _unified(lazy_reads=False)).weights_bytes == (64 << 20) + OVER_4_GIB * 4


def test_the_preset_reads_the_table_on_demand_wherever_it_prices_it_that_way(tmp_path, monkeypatch):
    """The price and the flag travel together, so the engine's own per-device default never decides
    whether the table is resident."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    gguf = tmp_path / "flash.gguf"
    _write_gguf(gguf, _arch("qwen4exp"), [("per_layer_token_embd.weight", OVER_4_GIB),
                                          ("blk.0.attn_q.weight", 16 << 20)])

    lazy = presets.preset_for_model(gguf, _unified(lazy_reads=True), set())
    eager = presets.preset_for_model(gguf, _unified(lazy_reads=False), set())

    assert lazy.keys["lazy-mode"] == "on"
    assert "lazy-mode" not in eager.keys
    window = min(lazy.window, eager.window)
    assert (presets.resident_footprint(gguf, _unified(lazy_reads=False), window)
            - presets.resident_footprint(gguf, _unified(lazy_reads=True), window)) == OVER_4_GIB * 4
