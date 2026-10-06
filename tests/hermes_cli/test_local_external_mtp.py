"""External MTP recipes must download the right head and price the model they launch."""

from dataclasses import replace
import hashlib

import pytest

from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401
from tests.hermes_cli.test_local_runtime_gguf import _write_gguf


def test_companion_repository_and_digest_survive_the_download_plan(tmp_path, monkeypatch, dl_server):
    from hermes_cli.local_runtime import catalog
    from hermes_cli.web_routers import local_models as lm

    body = b"correct external MTP head"
    digest = hashlib.sha256(body).hexdigest()
    entry = replace(catalog.CATALOG[0], repo="publisher/target", mmproj=None,
                    draft=catalog.AssetFile("head.gguf", len(body), local="other-head.gguf",
                                            repo="other/head", revision="pinned", sha256=digest))
    variant = replace(entry.variants[0], files=(catalog.AssetFile("target.gguf", 4),))
    monkeypatch.setattr(lm.bootstrap, "models_dir", lambda: tmp_path / "models")
    monkeypatch.setattr(lm.bootstrap, "assets_dir", lambda: tmp_path / "models" / "assets")
    plan = lm._download_plan(entry, variant)
    assert plan[0][0] == "https://huggingface.co/publisher/target/resolve/main/target.gguf"
    assert plan[1][0] == "https://huggingface.co/other/head/resolve/pinned/head.gguf"
    assert plan[1][1].name == entry.draft.local_name
    RangeHandler.payloads["/target"] = b"main"
    RangeHandler.payloads["/head"] = body
    local_plan = [(url(dl_server, path), *item[1:]) for path, item in zip(("/target", "/head"), plan)]
    # A stale file with the same destination must not satisfy a pinned transfer.
    plan[1][1].parent.mkdir(parents=True)
    plan[1][1].write_bytes(b"old incompatible head")
    lm._run_download_plan(lm._job("model-download", "test"), local_plan, "test")
    assert plan[1][1].read_bytes() == body
    plan[1][1].unlink()
    RangeHandler.payloads["/head"] = b"corrupted download"
    with pytest.raises(Exception, match="(?i)(sha256|checksum|digest)"):
        lm._run_download_plan(lm._job("model-download", "test"), local_plan, "test")
    assert not plan[1][1].exists()


def test_external_mtp_split_model_is_priced_and_launched_as_one_recipe(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import bootstrap, catalog, presets
    from hermes_cli.local_runtime.estimator import HardwareBudget, profile_from_gguf, PhysicsRefusal
    from hermes_cli.local_runtime.gguf import read_gguf_header

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(bootstrap, "assets_dir", lambda: models / "assets")
    entry = next(e for e in catalog.CATALOG if e.mtp and e.draft)
    variant = entry.variants[0]
    first = models / variant.files[0].local_name
    # The first shard carries metadata only; all resident weights and the lazy PLE table
    # are in later shards. Pricing only the first shard would admit impossible models.
    metadata = {"general.architecture": "qwen4exp", "qwen4exp.block_count": 48,
                "qwen4exp.context_length": 262144, "qwen4exp.full_attention_interval": 4,
                "qwen4exp.attention.head_count": 24, "qwen4exp.attention.head_count_kv": 2,
                "qwen4exp.attention.key_length": 256, "qwen4exp.attention.value_length": 256,
                "qwen4exp.attention.indexer.key_length": 128,
                "qwen4exp.expert_count": 512, "qwen4exp.vocab_size": 248320}
    _write_gguf(first, metadata, [])
    _write_gguf(models / variant.files[1].local_name, {}, [("blk.0.ffn_up_exps.weight", 15 << 30)])
    _write_gguf(models / variant.files[2].local_name, {}, [("per_layer_token_embd.weight", 7 << 30)])
    profile = profile_from_gguf(read_gguf_header(first))
    assert profile.weights_bytes == 60 << 30  # lazy PLE is mapped, not fully resident
    roomy = HardwareBudget(81 << 30, 102 << 30, 0, uma=True)
    assert not isinstance(entry.launch_plan(variant, roomy).decision, PhysicsRefusal)
    if not entry.auto_recommend:
        assert catalog.recommended_entry(roomy, (entry,)) is None
    missing = presets.preset_for_model(first, roomy, set())
    assert missing.refusal and "draft" in missing.refusal.lower()
    assets = bootstrap.assets_dir()
    assets.mkdir()
    (assets / entry.draft.local_name).touch()
    (assets / entry.mmproj.local_name).touch()
    result = presets.preset_for_model(first, roomy, set())
    assert not result.refusal and not result.spilled
    assert result.keys["model-draft"] == str(assets / entry.draft.local_name)
    assert result.keys["spec-type"] == "draft-mtp"
    assert result.keys["spec-draft-n-max"] == str(entry.mtp_draft_depth)
    assert result.keys["lazy-mode"] == "on"
    assert result.keys["load-mode"] == "mmap"
    assert result.keys["cache-type-k"] == result.keys["cache-type-v"] == "f16"
    assert result.keys["ubatch-size"] == "512"
    assert presets.resident_footprint(first, roomy, result.window) > profile.weights_bytes + entry.draft.size_bytes
    assert presets.admitted_residency_count(models, roomy, 4) == 1
    too_small = replace(roomy, usable_vram_bytes=40 << 30, total_device_bytes=40 << 30)
    assert presets.preset_for_model(first, too_small, set()).refusal
    # A missing continuation shard cannot be priced as a smaller model.
    (models / variant.files[2].local_name).unlink()
    assert presets.preset_for_model(first, roomy, set()) is None
