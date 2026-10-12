"""A split GGUF is priced as the SET of its shards, never as its first file.

Publishers lay split shards out so the first part can hold little more than metadata while the
bulk of the weights sits in the later ones. A reader that stops at part 1 therefore prices the
model at whatever fraction of its weights that part happens to hold — cheap enough for the physics
check and the residency cap to admit giants the card cannot hold, which is how two such giants
came to co-reside and take the whole router down with OOM.

These are the synthetic equivalents of that report: a first part with an empty tensor table, an
embedding table that lives outside part 1, and per-block FFN weights spread across shards (whose
truncation also silently disabled spill placement, the lever that would have relieved the
pressure).
"""

from __future__ import annotations

import re
import struct

import pytest

from hermes_cli.local_runtime.context_policy import spill_overrides
from hermes_cli.local_runtime.estimator import profile_from_gguf
from hermes_cli.local_runtime.gguf import read_gguf_header

GIB = 1 << 30


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


ARCH = {
    "general.architecture": "llama",
    "llama.block_count": 12,
    "llama.context_length": 65536,
    "llama.attention.head_count": 8,
    "llama.attention.head_count_kv": 2,
    "llama.attention.key_length": 128,
    "llama.attention.value_length": 128,
}


def _split_keys(no: int, count: int) -> dict:
    """What gguf-split writes into every part after the first: its split keys and nothing else
    (checked against a published Flash Next split: part 1 carries 67 keys, parts 2-3 carry 3)."""
    return {"split.no": no, "split.count": count, "split.tensors.count": 0}


def _split(tmp_path, stem: str, parts: list[dict], total: int | None = None):
    """A split on disk laid out as gguf-split writes it: ``parts[i]`` is the tensor table of shard
    i+1, and only shard 1 carries the model's metadata unless a spec overrides it."""
    total = total or len(parts)
    written = []
    for i, spec in enumerate(parts, start=1):
        path = tmp_path / f"{stem}-{i:05d}-of-{total:05d}.gguf"
        default = ARCH if i == 1 else _split_keys(i - 1, total)
        _write_gguf(path, spec.get("metadata", default), spec.get("tensors", []))
        written.append(path)
    return written


def test_split_weights_are_the_sum_across_shards(tmp_path):
    """The reported bug exactly: part 1 is a metadata stub (``n_tensors = 0``), the weights are in
    the later parts, so a reader that stops at part 1 prices the model at nothing."""
    parts = _split(tmp_path, "qwen-flash", [
        {"tensors": []},                                             # metadata stub, 0 tensors
        {"tensors": [("blk.0.attn_q.weight", 32 << 20)]},             # 128 MiB
        {"tensors": [("blk.1.attn_q.weight", 64 << 20)]},             # 256 MiB
    ])

    header = read_gguf_header(parts[0])

    assert header.n_tensors == 2
    assert header.tensor_bytes == ((32 + 64) * 4) << 20
    # Architecture still comes from the first part, so KV/layer pricing is unaffected.
    assert header.n_layer == 12 and header.n_ctx_train == 65536
    assert profile_from_gguf(header).weights_bytes == header.tensor_bytes


def test_single_shard_model_is_unchanged(tmp_path):
    """The fan-out must not disturb an unsplit file."""
    gguf = tmp_path / "single.gguf"
    _write_gguf(gguf, ARCH, [("blk.0.attn_q.weight", 16 << 20)])
    assert read_gguf_header(gguf).tensor_bytes == 64 << 20


def test_naming_alone_is_not_a_split(tmp_path):
    """A split whose siblings are not on disk prices as the one file that is, rather than raising."""
    only = tmp_path / "half-00001-of-00003.gguf"
    _write_gguf(only, ARCH, [("blk.0.attn_q.weight", 16 << 20)])
    assert read_gguf_header(only).tensor_bytes == 64 << 20


def test_embedding_table_outside_part_one_is_still_priced(tmp_path):
    """``embd_table_bytes`` prices the host-side duplicate of a fully offloaded embedding table.
    When that tensor sits outside part 1 the duplicate was priced at 0, understating the load by
    however large the table is."""
    parts = _split(tmp_path, "embd-split", [
        {"tensors": []},
        {"tensors": [("token_embd.weight", 8 << 20)]},
    ])

    header = read_gguf_header(parts[0])

    assert header.embd_table_bytes == 32 << 20
    assert header.embd_table_bytes <= header.tensor_bytes


def test_ffn_blocks_spread_across_shards_still_place_spill(tmp_path):
    """Truncating the per-block FFN map also disabled spill placement: ``recurrent_spill_blocks``
    spills every recurrent block as soon as one is missing from the map, so a split whose FFNs sit
    in later parts lost the ability to spill just enough."""
    hybrid = dict(ARCH, **{"llama.full_attention_interval": 4})
    tensors = [("token_embd.weight", 64)]
    for i in range(12):
        tensors += [(f"blk.{i}.attn_norm.weight", 16), (f"blk.{i}.ffn_norm.weight", 16),
                    (f"blk.{i}.ffn_up.weight", 256), (f"blk.{i}.ffn_down.weight", 240)]
    parts = _split(tmp_path, "hybrid-split", [
        {"metadata": hybrid, "tensors": tensors[:1]},
        {"tensors": tensors[1:37]},   # blocks 0-8 (4 tensors each)
        {"tensors": tensors[37:]},    # blocks 9-11
    ], total=3)

    profile = profile_from_gguf(read_gguf_header(parts[0]))
    per_block = (16 + 256 + 240) * 4   # f32; attn_norm is not an FFN tensor
    assert profile.ffn_block_bytes == {i: per_block for i in range(12)}

    # Enough spill for two recurrent blocks' FFNs, so the placement is a choice, not "everything".
    args = spill_overrides(profile, 2 * per_block)
    pattern = args[1].removesuffix("=CPU")
    moved = [i for i in range(12) if re.search(pattern, f"blk.{i}.ffn_up.weight")]
    assert moved == [0, 1]


def test_unreadable_shard_does_not_price_the_split_away(tmp_path):
    """A part that is corrupt or truncated is skipped, not fatal: a half-arrived split prices at
    what is on disk. Refusing it to serve is ``staged_in(require_complete=True)``'s job."""
    parts = _split(tmp_path, "half-downloaded", [
        {"tensors": [("blk.0.attn_q.weight", 16 << 20)]},
        {"tensors": [("blk.1.attn_q.weight", 16 << 20)]},
        {"tensors": [("blk.2.attn_q.weight", 16 << 20)]},
    ])
    parts[1].write_bytes(b"GGUF\x03\x00")   # header cut mid-stream

    header = read_gguf_header(parts[0])

    assert header.tensor_bytes == 2 * (64 << 20)   # parts 1 and 3; the truncated one dropped


def test_residency_cap_prices_a_split_against_the_card(tmp_path, monkeypatch):
    """The consequence, not just the reader: a fleet whose largest model is a split must not be
    admitted ``models_max`` times over. Two giants co-resided because the probe saw a 175 GB model
    as a few GiB and booted the router with room for three."""
    from hermes_cli.local_runtime import presets
    from hermes_cli.local_runtime.estimator import HardwareBudget

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    mdir = tmp_path / "models"
    mdir.mkdir()
    # A 48 GiB split laid out the reported way: part 1 a metadata stub, the weights in part 2.
    _write_gguf(mdir / "giant-00001-of-00002.gguf", ARCH, [])
    _write_gguf(mdir / "giant-00002-of-00002.gguf", _split_keys(1, 2),
                [("blk.0.attn_q.weight", 48 * GIB // 4)])
    # A 10 GB single-shard neighbour: a small second resident must stay possible.
    _write_gguf(mdir / "utility.gguf", ARCH, [("blk.0.attn_q.weight", 10 * GIB // 4)])

    budget = HardwareBudget(usable_vram_bytes=64 * GIB, total_device_bytes=64 * GIB,
                            ram_available_bytes=64 * GIB)

    assert presets.admitted_residency_count(mdir, budget, 4) == 1


def test_a_truncated_header_skips_the_model_instead_of_raising(tmp_path, monkeypatch):
    """Every caller skips a model whose header raises ValueError or OSError. A header cut short
    used to raise struct.error, which none of them catch, so one bad file took down the whole
    preset pass instead of dropping out of it."""
    from hermes_cli.local_runtime import presets
    from hermes_cli.local_runtime.estimator import HardwareBudget

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    parts = _split(tmp_path, "cut-head", [
        {"tensors": []},
        {"tensors": [("blk.0.attn_q.weight", 16 << 20)]},
    ])
    parts[0].write_bytes(b"GGUF\x03\x00")
    budget = HardwareBudget(usable_vram_bytes=64 * GIB, total_device_bytes=64 * GIB,
                            ram_available_bytes=64 * GIB)

    with pytest.raises(ValueError):
        read_gguf_header(parts[0])
    assert presets.resident_footprint(parts[0], budget, 8192) is None