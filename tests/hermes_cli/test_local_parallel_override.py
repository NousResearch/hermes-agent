"""Per-model parallel-slots overrides reach generated presets.

window_overrides.json is the established operator seam for one launch-policy
input (growth.py + the presets override seam); this file pins its sibling: a
parallel_overrides.json entry renders the server slot count into the model's
preset, an operator choice the physics never prices, and the file reads with
the window file's tolerance (one bad value drops the whole file)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _stage_fake_gguf(mdir, name):
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"{name}.gguf").write_bytes(b"GGUF" + b"\x00" * 64)


def _tiny_profile(name):
    from hermes_cli.local_runtime.estimator import LayerKind, ModelProfile

    gib = 1 << 30
    return ModelProfile(name=name, weights_bytes=2 * gib, embd_table_bytes=0,
                        n_ctx_train=131072, layers=[(LayerKind.FULL, 512)] * 4)


def test_parallel_override_splits_the_granted_window(hermes_home, tmp_path, monkeypatch):
    from hermes_cli.local_runtime import presets as presets_mod
    from hermes_cli.local_runtime.estimator import HardwareBudget
    from hermes_cli.local_runtime.growth import parallel_overrides_path

    mdir = tmp_path / "models"
    _stage_fake_gguf(mdir, "split-me")
    monkeypatch.setattr(presets_mod, "read_gguf_header",
                        lambda p: SimpleNamespace(path=p, sampling_defaults={}))
    monkeypatch.setattr(presets_mod, "profile_from_gguf",
                        lambda h: _tiny_profile(h.path.stem))
    gib = 1 << 30
    budget = HardwareBudget(usable_vram_bytes=24 * gib, total_device_bytes=24 * gib,
                            ram_available_bytes=64 * gib)

    path = parallel_overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    plain = presets_mod.preset_for_model(mdir / "split-me.gguf", budget, set())
    assert plain.refusal is None and plain.window > 0
    assert "parallel" not in plain.keys

    path.write_text(json.dumps({plain.model_id: 2}), encoding="utf-8")
    split = presets_mod.preset_for_model(mdir / "split-me.gguf", budget, set())
    assert split.keys["parallel"] == "2"

    path.write_text("{}", encoding="utf-8")
    restored = presets_mod.preset_for_model(mdir / "split-me.gguf", budget, set())
    assert "parallel" not in restored.keys


def test_parallel_override_file_reads_like_the_window_file(hermes_home):
    from hermes_cli.local_runtime.growth import (
        load_parallel_overrides,
        parallel_overrides_path,
    )

    assert load_parallel_overrides() == {}
    path = parallel_overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"model-a": 2, "model-b": 4}', encoding="utf-8")
    assert load_parallel_overrides() == {"model-a": 2, "model-b": 4}
    path.write_text('{"model-a": "two"}', encoding="utf-8")
    assert load_parallel_overrides() == {}
    path.write_text("{not json", encoding="utf-8")
    assert load_parallel_overrides() == {}
