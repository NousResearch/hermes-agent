"""A model whose MTP head ships as its own file.

Qwen3.8 Flash Next carries no MTP layers. ggml-org publishes a head for it in its own repo
(``ggml-org/Qwen3.8-Flash-Next-GGUF``, ``mtp-Qwen3.8-Flash-Next-Q8_0.gguf``), and llama.cpp drafts
from it with ``--spec-type draft-mtp --spec-draft-model <head>``. The head downloads with the model
from that repo, is priced before and after download, and is the only thing that turns MTP on:
``draft-mtp`` without a head asks the engine for MTP layers this model does not have.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from hermes_cli.local_runtime import bootstrap, catalog, presets
from hermes_cli.local_runtime.catalog import AssetFile, CatalogEntry, QuantVariant
from hermes_cli.local_runtime.estimator import HardwareBudget
from hermes_cli.web_routers import local_models

GIB = 1 << 30
UNIFIED_128 = HardwareBudget(usable_vram_bytes=int(128 * GIB * 0.8), total_device_bytes=128 * GIB,
                             ram_available_bytes=0, uma=True)

BUILD = QuantVariant(quant="UD-IQ4_XS", files=(AssetFile("m/Head-Model-00001-of-00002.gguf", 10 << 20),
                                               AssetFile("m/Head-Model-00002-of-00002.gguf", 40 * GIB)))
ENTRY = CatalogEntry(
    id="head-model", display_name="Head Model", description="", repo="org/Head-Model-GGUF",
    variants=(BUILD,), n_ctx_train=262144, full_layers=12, recurrent_layers=36, per_layer_f16=2048,
    moe=True, n_vocab=248320, mtp_draft_depth=3,
    mtp_head=AssetFile("mtp-Head-Model-Q8_0.gguf", 4 * GIB, repo="other/Head-Model-GGUF", revision="0123abc"))


def test_the_head_downloads_with_the_model_from_its_own_repo_and_is_priced_before_download():
    without_head = replace(ENTRY, mtp_head=None)

    urls = [url for url, _, _ in local_models._download_plan(ENTRY, BUILD)]
    assert "https://huggingface.co/other/Head-Model-GGUF/resolve/0123abc/mtp-Head-Model-Q8_0.gguf" in urls
    assert all("/org/Head-Model-GGUF/resolve/main/" in url for url in urls[:2])
    assert ENTRY.mtp_capable and not without_head.mtp_capable
    with_plan = ENTRY.launch_plan(BUILD, UNIFIED_128)
    without_plan = without_head.launch_plan(BUILD, UNIFIED_128)
    assert with_plan.overhead_bytes >= without_plan.overhead_bytes + ENTRY.mtp_head.size_bytes


def test_the_preset_drafts_from_the_head_once_it_is_on_disk(tmp_path, monkeypatch):
    gguf = tmp_path / "Head-Model-00001-of-00002.gguf"
    monkeypatch.setattr(catalog, "entry_for_model",
                        lambda model_id: ENTRY if model_id == BUILD.model_id else None)
    monkeypatch.setattr(presets, "read_gguf_header", lambda p: SimpleNamespace(sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda h: ENTRY.profile(BUILD))
    head = bootstrap.assets_dir() / ENTRY.mtp_head.local_name

    before = presets.preset_for_model(gguf, UNIFIED_128, set())
    head.parent.mkdir(parents=True, exist_ok=True)
    head.touch()
    after = presets.preset_for_model(gguf, UNIFIED_128, set())

    assert "spec-type" not in before.keys and "model-draft" not in before.keys
    assert after.keys["model-draft"] == str(head)
    assert after.keys["spec-type"] == "draft-mtp"
    assert after.keys["spec-draft-n-max"] == "3"
