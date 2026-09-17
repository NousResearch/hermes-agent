"""The mtp override forces the spec-decode posture regardless of catalog knowledge."""

from pathlib import Path
from types import SimpleNamespace

from hermes_cli.local_runtime import growth, presets
from hermes_cli.local_runtime.estimator import HardwareBudget, ModelProfile


def _stage(tmp_path, monkeypatch, name):
    mdir = tmp_path / "models"
    mdir.mkdir(exist_ok=True)
    gguf = mdir / f"{name}.gguf"
    gguf.touch()
    monkeypatch.setattr(
        presets,
        "read_gguf_header",
        lambda p: SimpleNamespace(path=p, sampling_defaults={}),
    )
    monkeypatch.setattr(
        presets,
        "profile_from_gguf",
        lambda h: ModelProfile(
            name=h.path.stem,
            weights_bytes=4 << 30,
            embd_table_bytes=0,
            n_ctx_train=65536,
            layers=[],
        ),
    )
    return gguf


def test_mtp_override_enables_spec_decode_for_uncatalogued_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    model_id = "Swift-Qwen3.8-27B-Uncensored-Dynamic-MTP-UD-Q6_K_XL"
    gguf = _stage(tmp_path, monkeypatch, model_id)
    budget = HardwareBudget(8 << 30, 8 << 30, 8 << 30)

    # No override and no catalog entry: plain launch, no spec-decode flags.
    plain = presets.preset_for_model(gguf, budget, set())
    assert plain is not None and plain.keys is not None
    assert plain.keys.get("spec-type") != "draft-mtp"
    assert "backend-sampling" not in plain.keys

    growth.save_mtp_override(model_id, True)
    forced = presets.preset_for_model(gguf, budget, set())
    assert forced is not None and forced.keys is not None
    assert forced.keys["spec-type"] == "draft-mtp"
    assert forced.keys["backend-sampling"] == "on"
    assert forced.keys["spec-draft-backend-sampling"] == "on"

    # Clearing returns to the catalog's verdict (off for an unknown model).
    growth.clear_mtp_override(model_id)
    cleared = presets.preset_for_model(gguf, budget, set())
    assert cleared is not None and cleared.keys is not None
    assert cleared.keys.get("spec-type") != "draft-mtp"


def test_mtp_override_false_disables_catalog_mtp_model(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import catalog

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    mtp_model = next(e for e in catalog.CATALOG if e.mtp)
    gguf = _stage(tmp_path, monkeypatch, mtp_model.variants[0].model_id)
    budget = HardwareBudget(16 << 30, 16 << 30, 16 << 30)

    stock = presets.preset_for_model(gguf, budget, set())
    assert stock is not None and stock.keys is not None
    assert stock.keys["spec-type"] == "draft-mtp"  # catalog default is MTP

    growth.save_mtp_override(mtp_model.variants[0].model_id, False)
    off = presets.preset_for_model(gguf, budget, set())
    assert off is not None and off.keys is not None
    assert off.keys.get("spec-type") != "draft-mtp"
    assert "backend-sampling" not in off.keys
