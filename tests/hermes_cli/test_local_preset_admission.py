"""The router must serve only admitted presets, retaining refusal and spill facts on read-back."""
from pathlib import Path
from types import SimpleNamespace

from hermes_cli.local_runtime import presets, supervisor
from hermes_cli.local_runtime.estimator import HardwareBudget, ModelProfile


def test_preset_roundtrip_keeps_refusals_and_dense_spill(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    mdir = tmp_path / "models"
    mdir.mkdir()
    for name in ("allowed", "refused"):
        (mdir / f"{name}.gguf").touch()
    monkeypatch.setattr(presets, "read_gguf_header", lambda p: SimpleNamespace(path=p, sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda h: ModelProfile(
        name=h.path.stem, weights_bytes=(4 if h.path.stem == "allowed" else 40) << 30,
        embd_table_bytes=0, n_ctx_train=65536, layers=[]))
    ini = tmp_path / "presets.ini"
    generated = presets.generate_presets(mdir, HardwareBudget(2 << 30, 2 << 30, 8 << 30), ini)
    reread = presets.read_preset_decisions(ini)
    assert set(reread) == {p.model_id for p in generated}
    assert reread["refused"].refusal
    assert reread["allowed"].spilled
    assert reread["allowed"].keys["model"] == str(mdir / "allowed.gguf")
    assert "override-tensor" not in reread["allowed"].keys  # Dense spill has no tensor-pattern override.


def test_optional_draft_is_enabled_only_with_room_at_the_selected_window(tmp_path, monkeypatch):
    from dataclasses import replace
    from hermes_cli.local_runtime import bootstrap, catalog

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entry = next(e for e in catalog.CATALOG if e.draft)
    main = tmp_path / f"{entry.variants[0].model_id}.gguf"
    draft = bootstrap.assets_dir() / entry.draft.local_name
    draft.parent.mkdir(parents=True, exist_ok=True)
    draft.touch()
    main_profile = ModelProfile("main", 12 << 30, 0, 65536, [], moe=True)
    draft_profile = ModelProfile("draft", 1 << 30, 0, 65536, [])
    monkeypatch.setattr(presets, "read_gguf_header", lambda p: SimpleNamespace(path=p, sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda h: draft_profile if h.path == draft else main_profile)
    tight = HardwareBudget(8 << 30, 8 << 30, 6 << 30)
    result = presets.preset_for_model(main, tight, set())
    assert result.window == 65536 and result.spilled
    assert "model-draft" not in result.keys
    roomy = replace(tight, ram_available_bytes=16 << 30)
    with_draft = presets.preset_for_model(main, roomy, set())
    assert with_draft.window == result.window
    assert with_draft.keys["model-draft"] == str(draft)
    assert with_draft.keys["spec-type"] == "draft-dspark"
    # Full target-window f16 state and logits count even above the draft's native window.
    from hermes_cli.local_runtime.context_policy import RUNTIME_OVERHEAD_BYTES, ub_logits_bytes
    from hermes_cli.local_runtime.estimator import LayerKind, ctx_bytes

    draft_profile = replace(draft_profile, n_ctx_train=32768,
                            layers=[(LayerKind.FULL, 4096)] * 4, n_vocab=32768)
    draft_cost = (draft_profile.weights_bytes
                  + ctx_bytes(draft_profile, result.window, flash_attention=False)
                  + RUNTIME_OVERHEAD_BYTES
                  + ub_logits_bytes(draft_profile.n_vocab, mtp_capable=False))
    device_boundary = RUNTIME_OVERHEAD_BYTES + draft_cost
    exact = replace(roomy, usable_vram_bytes=device_boundary, total_device_bytes=device_boundary)
    assert "model-draft" in presets.preset_for_model(main, exact, set()).keys
    below = replace(exact, usable_vram_bytes=device_boundary - 1)
    assert "model-draft" not in presets.preset_for_model(main, below, set()).keys

    # A draft too large for GPU memory is optional, not permission to move its buffers to RAM.
    draft_profile = replace(draft_profile, weights_bytes=9 << 30)
    assert "model-draft" not in presets.preset_for_model(main, roomy, set()).keys


def test_supervisor_with_presets_does_not_scan_unadmitted_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ini = tmp_path / "presets.ini"
    ini.write_text("[allowed]\nmodel = allowed.gguf\nctx-size = 65536\n")
    calls = []
    monkeypatch.setattr(supervisor, "server_binary", lambda p: Path("llama-server"))
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(pid=123))
    sup = supervisor.LlamaServerSupervisor(tmp_path, tmp_path, port=1234, preset_path=ini)
    try:
        sup._spawn()
    finally:
        sup._log_handle.close()
    assert "--models-preset" in calls[0]
    assert "--models-dir" not in calls[0]
