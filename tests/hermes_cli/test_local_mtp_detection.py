"""MTP (multi-token prediction) heads on staged GGUFs the catalog has never seen.

The catalog declares ``mtp: true`` for the files it lists; a drop-in file only has its tensor
table. llama.cpp names an MTP head ``blk.<N>.nextn.*`` and serves it with ``--spec-type
draft-mtp`` — without that flag the same weights decode at plain speed (measured 22.6 vs 30.8
tok/s on a Radeon 780M). These tests pin the detection and the preset keys it must produce.
"""

from __future__ import annotations

import struct

import pytest

from hermes_cli.local_runtime.gguf import GGUFHeader, read_gguf_header

_STRING, _UINT32 = 8, 4


def _gguf_bytes(tensor_names: list[str]) -> bytes:
    """A minimal, readable GGUF v3 header: one metadata key plus the given tensor table."""
    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    out = bytearray(b"GGUF")
    out += struct.pack("<IQQ", 3, len(tensor_names), 1)
    out += s("general.architecture") + struct.pack("<I", _STRING) + s("qwen3moe")
    for i, name in enumerate(tensor_names):
        out += s(name)
        out += struct.pack("<I", 1) + struct.pack("<Q", 256)   # one dim: 256 elements
        out += struct.pack("<I", 0)                            # ggml type F32
        out += struct.pack("<Q", i * 1024)                     # offset (unused by the reader)
    return bytes(out)


def test_mtp_head_detected_from_tensor_table(tmp_path):
    path = tmp_path / "Qwen3.6-35B-A3B-MTP.gguf"
    path.write_bytes(_gguf_bytes([
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.40.attn_norm.weight",
        "blk.40.nextn.eh_proj.weight",
        "blk.40.nextn.shared_head_norm.weight",
    ]))
    header = read_gguf_header(path)
    assert header.mtp_blocks == frozenset({40})
    assert header.has_mtp_head is True


def test_plain_file_has_no_mtp_head(tmp_path):
    path = tmp_path / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    path.write_bytes(_gguf_bytes([
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.40.attn_norm.weight",
    ]))
    header = read_gguf_header(path)
    assert header.mtp_blocks == frozenset()
    assert header.has_mtp_head is False


def test_header_defaults_when_field_absent():
    """Older callers build headers positionally; the field must stay optional."""
    header = GGUFHeader(path="x.gguf", version=3)
    assert header.has_mtp_head is False


def _tiny_profile(model_id: str):
    from hermes_cli.local_runtime.estimator import LayerKind, ModelProfile

    gib = 1 << 30
    return ModelProfile(
        name=model_id, weights_bytes=2 * gib, embd_table_bytes=0,
        n_ctx_train=131072, layers=[(LayerKind.FULL, 512)] * 4)


def _budget():
    from hermes_cli.local_runtime.estimator import HardwareBudget

    gib = 1 << 30
    return HardwareBudget(usable_vram_bytes=24 * gib, total_device_bytes=24 * gib,
                          ram_available_bytes=64 * gib)


def _preset_keys(tmp_path, monkeypatch, *, mtp_blocks: frozenset[int]):
    """Generate the preset for a staged model the catalog does not know, return its keys."""
    import hermes_cli.local_runtime.presets as presets_mod

    mdir = tmp_path / "models"
    mdir.mkdir()
    model_id = "Local-Finetune-35B-MTP"
    (mdir / f"{model_id}.gguf").write_bytes(b"GGUF" + b"\x00" * 64)

    class Stub:
        """Just enough of GGUFHeader for preset generation."""

        sampling_defaults: dict = {}

        def __init__(self, blocks: frozenset[int]) -> None:
            self.mtp_blocks = blocks

        @property
        def has_mtp_head(self) -> bool:
            return bool(self.mtp_blocks)

    monkeypatch.setattr(presets_mod, "read_gguf_header", lambda _p: Stub(mtp_blocks))
    monkeypatch.setattr(presets_mod, "profile_from_gguf", lambda h: _tiny_profile(model_id))

    from hermes_cli.local_runtime.catalog import entry_for_model
    assert entry_for_model(model_id) is None, "the staged id must be outside the catalog"

    entries = presets_mod.generate_presets(mdir, _budget(), tmp_path / "p.ini")
    assert len(entries) == 1
    return entries[0].keys or {}


def test_staged_mtp_file_gets_draft_keys(tmp_path, monkeypatch):
    keys = _preset_keys(tmp_path, monkeypatch, mtp_blocks=frozenset({40}))
    assert keys.get("spec-type") == "draft-mtp"
    assert keys.get("spec-draft-n-max") == "3"
    assert keys.get("backend-sampling") == "on"
    assert keys.get("spec-draft-backend-sampling") == "on"


def test_staged_plain_file_keeps_stock_decode(tmp_path, monkeypatch):
    keys = _preset_keys(tmp_path, monkeypatch, mtp_blocks=frozenset())
    assert "spec-type" not in keys
    assert "backend-sampling" not in keys
