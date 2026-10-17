"""Fixed CPU-offload placements must fit beside foreign GPU tenants (#135153)."""

from pathlib import Path
import struct

import pytest

from hermes_cli.local_runtime import hardware, presets
from hermes_cli.local_runtime.estimator import HardwareBudget, profile_from_gguf
from hermes_cli.local_runtime.gguf import read_gguf_header

GIB = 1 << 30


def _write_header(path, metadata, tensors):
    def string(text):
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    raw = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        raw += string(key) + (struct.pack("<I", 8) + string(value) if isinstance(value, str) else struct.pack("<II", 4, value))
    for name, nbytes in tensors:
        raw += string(name) + struct.pack("<IQIQ", 1, nbytes // 4, 0, 0)
    path.write_bytes(raw)


def _model(tmp_path, family):
    arch = "qwen2moe" if family == "moe" else "qwen35" if family == "hybrid" else "llama"
    meta = {"general.architecture": arch, f"{arch}.context_length": 65536,
            f"{arch}.block_count": 2, f"{arch}.attention.head_count": 1,
            f"{arch}.attention.head_count_kv": 0, f"{arch}.attention.key_length": 1,
            f"{arch}.attention.value_length": 1}
    if family == "moe":
        meta[f"{arch}.expert_count"] = 8
        tensors = [("blk.0.ffn_up_exps.weight", 3 * GIB), ("blk.1.ffn_down_exps.weight", 3 * GIB),
                   ("blk.0.ffn_up_shexp.weight", GIB), ("output.weight", 2 * GIB)]
    elif family == "hybrid":
        tensors = [("blk.0.ffn_up.weight", 3 * GIB), ("blk.1.ffn_down.weight", 3 * GIB), ("output.weight", 3 * GIB)]
    else:
        # No recurrent layers and no experts: stock fitter can still choose a smaller placement.
        meta[f"{arch}.attention.head_count_kv"] = 1
        tensors = [("output.weight", 9 * GIB)]
    path = tmp_path / f"fixture-{family}.gguf"
    _write_header(path, meta, tensors)
    return path


@pytest.mark.parametrize("family", ["dense", "hybrid", "moe"])
@pytest.mark.parametrize("state", ["foreign", "roomy", "reclaimed", "shared-ffn-boundary", "uma"])
def test_only_fixed_offload_shortfalls_refuse_preset_emission(tmp_path, monkeypatch, family, state):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    gguf = _model(tmp_path, family)
    capacity = HardwareBudget(7 * GIB, 8 * GIB, 64 * GIB, uma=state == "uma")
    free = (3 if state in {"foreign", "reclaimed", "uma"} else 8) * GIB
    if state == "shared-ffn-boundary":
        free = 4 * GIB + hardware._LAUNCH_HEADROOM
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: (8 * GIB, free, "fixture GPU", None))
    live = hardware.launch_budget(capacity, own_bytes=5 * GIB if state == "reclaimed" else 0)
    result = presets.preset_for_model(gguf, capacity, set(), requested_window=0, live=live)
    text = presets.render_presets([result])
    refused = family != "dense" and state in {"foreign", "shared-ffn-boundary"}
    if refused:
        assert result.refusal, "Fixed tensor overrides disable the launch-time fitter"
        assert result.keys is None
        assert f"[{result.model_id}]" not in text
        assert "GPU" in result.refusal and "MiB" in result.refusal
    else:
        assert result.refusal is None
        assert result.spilled
        assert result.keys["model"] == str(gguf)
        assert f"[{result.model_id}]" in text
        assert ("override-tensor" in result.keys) == (family != "dense" and state != "uma")


@pytest.mark.parametrize("split", [False, True])
def test_expert_accounting_excludes_shared_ffns_and_survives_split_headers(tmp_path, split):
    meta = {"general.architecture": "qwen2moe", "qwen2moe.expert_count": 8, "qwen2moe.block_count": 2}
    tensors = [("blk.0.ffn_up_exps.weight", 4096), ("blk.1.ffn_down_exps.weight", 2048),
               ("blk.0.ffn_up_shexp.weight", 8192), ("blk.0.ffn_norm.weight", 512)]
    if split:
        first = tmp_path / "fixture-00001-of-00002.gguf"
        second = tmp_path / "fixture-00002-of-00002.gguf"
        _write_header(first, {**meta, "split.count": 2, "split.no": 0}, tensors[:1])
        _write_header(second, {**meta, "split.count": 2, "split.no": 1}, tensors[1:])
    else:
        first = tmp_path / "fixture.gguf"
        _write_header(first, meta, tensors)
    profile = profile_from_gguf(read_gguf_header(first))
    assert getattr(profile, "expert_weight_bytes", None) == 6144
    assert profile.weights_bytes == sum(size for _, size in tensors)
    assert sum(profile.ffn_block_bytes.values()) > profile.expert_weight_bytes
