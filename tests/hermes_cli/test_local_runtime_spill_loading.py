"""Spilled weights must stay outside Vulkan's device-memory budget (#148261)."""

import struct
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import presets, supervisor
from hermes_cli.local_runtime.estimator import HardwareBudget


def _gguf_string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _stage_model(models, name, gib, *, moe=False):
    metadata = {"general.architecture": "llama", "llama.context_length": 65536,
                "llama.block_count": 0, "llama.expert_count": 8 if moe else 0}
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata))
    for key, value in metadata.items():
        data += _gguf_string(key)
        data += (struct.pack("<I", 8) + _gguf_string(value) if isinstance(value, str)
                 else struct.pack("<II", 4, value))
    tensor = "blk.0.ffn_gate_exps.weight" if moe else "blk.0.ffn_up.weight"
    # Only the header is read: price large weights without allocating their payload.
    data += _gguf_string(tensor) + struct.pack("<IQIQ", 1, (gib << 30) // 4, 0, 0)
    (models / f"{name}.gguf").write_bytes(data)


@pytest.mark.parametrize("help_text, resident_args, spilled_args", [
    ("--load-mode MODE", ["--load-mode", "dio"], ["--load-mode", "mmap"]),
    ("-dio, --direct-io", ["-dio"], []),
])
def test_spilled_library_uses_mmap_and_resident_library_keeps_direct_io(
        tmp_path, monkeypatch, help_text, resident_args, spilled_args):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    models = tmp_path / "models"
    models.mkdir()
    _stage_model(models, "resident", 1)
    _stage_model(models, "spilled-moe", 12, moe=True)
    _stage_model(models, "spilled-dense", 12)
    _stage_model(models, "refused", 128)
    ini = tmp_path / "presets.ini"
    entries = presets.generate_presets(models, HardwareBudget(8 << 30, 8 << 30, 32 << 30), ini)
    decisions = presets.read_preset_decisions(ini)
    assert decisions["spilled-moe"].spilled and decisions["spilled-dense"].spilled
    assert not decisions["resident"].spilled
    assert decisions["refused"].refusal

    monkeypatch.setattr(supervisor.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=help_text, stderr=""))
    commands = []
    monkeypatch.setattr(supervisor, "spawn_server", lambda argv, **kwargs: (
        commands.append(argv) or SimpleNamespace(pid=123), None))
    monkeypatch.setattr(supervisor.LlamaServerSupervisor, "_write_state", lambda self: None)
    sup = supervisor.LlamaServerSupervisor(tmp_path / "llama-server", models,
                                           port=19001, preset_path=ini)
    try:
        sup._spawn()
        assert "-dio" not in commands[-1] and "dio" not in commands[-1]
        if spilled_args:
            index = commands[-1].index(spilled_args[0])
            assert commands[-1][index:index + len(spilled_args)] == spilled_args
        else:
            assert "--load-mode" not in commands[-1]
        assert commands[-1][commands[-1].index("--models-preset") + 1] == str(ini)

        # A restart re-reads placement, even though the engine help probe is cached.
        presets.write_presets([e for e in entries if not e.spilled], ini)
        sup._spawn()
        index = commands[-1].index(resident_args[0])
        assert commands[-1][index:index + len(resident_args)] == resident_args
    finally:
        if sup._log_handle is not None:
            sup._log_handle.close()
