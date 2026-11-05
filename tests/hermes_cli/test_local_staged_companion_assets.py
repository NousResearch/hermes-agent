"""Companion GGUFs (mmproj projectors, TTS codecs) must never surface as servable models.

The picker, ``plan_presets()``/``generate_presets()`` (router autoload + ``GET /v1/models``) and
the ``--models-max`` admission cap all read ``staged_in()``; a projector staged in ``models/``
used to get its own launch section and a wasted slot (#133037)."""

import hermes_cli.local_runtime.bootstrap as bs
from hermes_cli.local_runtime.gguf import COMPANION_ASSET_RE


def test_companion_prefixes_are_recognized():
    """Well-known companion naming shapes match; look-alike model names do not."""
    companions = [
        "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
        "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",   # TTS audio codec, not a chat model
        "Qwen3-VL-mmproj-BF16.gguf",
        "projector-tiny.gguf",
    ]
    models = [
        "Single-Q4_K_M.gguf",
        "mmprojmodel-Q4.gguf",          # no separator after the prefix token
        "My-projector-model-Q4.gguf",   # infix, not the leading "projector-" token
    ]
    assert all(COMPANION_ASSET_RE.match(c.removesuffix(".gguf")) for c in companions)
    assert not any(COMPANION_ASSET_RE.match(m.removesuffix(".gguf")) for m in models)


def test_staged_models_skip_companion_assets(tmp_path, monkeypatch):
    """A projector/codec staged in models/ is not a staged model, while real models — single,
    split, and look-alike-named — still are. This is the seam behind the picker row, the preset
    INI sections the router autoloads, and /v1/models."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    mdir = bs.models_dir()
    mdir.mkdir(parents=True, exist_ok=True)

    (mdir / "mmproj-Qwen3.8-Flash-Next-BF16.gguf").touch()
    (mdir / "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf").touch()
    (mdir / "Qwen3-VL-mmproj-BF16.gguf").touch()

    (mdir / "Single-Q4_K_M.gguf").touch()
    (mdir / "Whole-Q4-00001-of-00002.gguf").touch()
    (mdir / "Whole-Q4-00002-of-00002.gguf").touch()
    (mdir / "mmprojmodel-Q4.gguf").touch()

    assert bs.staged_model_ids() == ["Single-Q4_K_M", "Whole-Q4", "mmprojmodel-Q4"]


def test_staged_in_filters_split_companion_first_part(tmp_path):
    """The companion check runs on the split-stripped stem, so a split-packaged projector never
    counts even with require_complete=False (the predicate is independent of completeness)."""
    d = tmp_path / "models"
    d.mkdir()
    (d / "mmproj-Big-00001-of-00002.gguf").touch()
    (d / "mmproj-Big-00002-of-00002.gguf").touch()
    (d / "Real-Q4-00001-of-00002.gguf").touch()
    (d / "Real-Q4-00002-of-00002.gguf").touch()

    assert [p.name for p in bs.staged_in(d)] == ["Real-Q4-00001-of-00002.gguf"]
    assert [p.name for p in bs.staged_in(d, require_complete=False)] == ["Real-Q4-00001-of-00002.gguf"]
