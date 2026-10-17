"""Catalog aliases: a sideloaded GGUF inherits the launch policy of the catalog build it derives from.

A fine-tune, abliteration or re-quant of a listed model shares its architecture, companions
(mmproj, MTP head) and sampling defaults, but its file stem matches no catalog variant. Without
an alias it takes the generic path; with one it launches like its base. Identity lookups
(download, delete) must keep ignoring aliases so the derivative never deletes the base's assets.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import bootstrap, catalog, presets
from hermes_cli.local_runtime.estimator import HardwareBudget

GIB = 1024 ** 3
MACHINE = HardwareBudget(int(128 * GIB * 0.8), 128 * GIB, 0, True)  # 128 GB unified memory


def _base_with_companions():
    """A catalog entry that ships both a vision projector and an MTP head (the policy with the
    most to lose on the generic path)."""
    for entry in catalog.CATALOG:
        if entry.mmproj is not None and entry.mtp_head is not None:
            return entry, entry.variants[0]
    pytest.skip("catalog has no entry with both mmproj and mtp_head")


@pytest.fixture
def dev_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(catalog, "_alias_cache", None, raising=False)
    return tmp_path


def _write_aliases(payload) -> None:
    path = catalog.catalog_aliases_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")


def test_no_alias_file_means_no_aliases(dev_home):
    assert catalog.load_catalog_aliases() == {}
    assert catalog.entry_for_model("Some-Finetune-Q4_K_M") is None


def test_alias_governs_policy_but_not_identity(dev_home):
    entry, variant = _base_with_companions()
    _write_aliases({"Some-Finetune-Q4_K_M": variant.model_id})

    assert catalog.entry_for_model("Some-Finetune-Q4_K_M") is entry
    # Identity stays exact: the derivative is not a catalog build, so the download route 404s
    # and the delete route never collects the base's mmproj / MTP head among its files.
    assert catalog.find_entry_for_model("Some-Finetune-Q4_K_M") is None
    # A real catalog id is never redirected, even if someone aliases it.
    assert catalog.entry_for_model(variant.model_id) is entry


def test_alias_to_unknown_build_is_reported_and_ignored(dev_home, caplog):
    _write_aliases({"Some-Finetune-Q4_K_M": "No-Such-Build-Q4_K_M"})
    with caplog.at_level(logging.WARNING, logger=catalog.logger.name):
        assert catalog.entry_for_model("Some-Finetune-Q4_K_M") is None
    assert "No-Such-Build-Q4_K_M" in caplog.text
    assert str(catalog.catalog_aliases_path()) in caplog.text


@pytest.mark.parametrize("payload", ["{not json", '["a", "b"]', '{"x": 1}'])
def test_malformed_alias_file_is_reported_not_raised(dev_home, caplog, payload):
    _write_aliases(payload)
    with caplog.at_level(logging.WARNING, logger=catalog.logger.name):
        assert catalog.load_catalog_aliases() == {}
    assert str(catalog.catalog_aliases_path()) in caplog.text


def test_alias_file_is_reread_when_it_changes(dev_home):
    entry, variant = _base_with_companions()
    _write_aliases({})
    assert catalog.entry_for_model("Some-Finetune-Q4_K_M") is None
    _write_aliases({"Some-Finetune-Q4_K_M": variant.model_id})
    assert catalog.entry_for_model("Some-Finetune-Q4_K_M") is entry


def test_aliased_preset_matches_its_base(dev_home, monkeypatch):
    """The whole point: same launch keys (draft head, projector, sampling, buffers) and the same
    window as the base — only the served file differs."""
    entry, variant = _base_with_companions()
    profile = entry.profile(variant)
    monkeypatch.setattr(presets, "read_gguf_header", lambda p: SimpleNamespace(sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda h: profile)
    for companion in (entry.mmproj, entry.mtp_head):
        asset = bootstrap.assets_dir() / companion.local_name
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.touch()
    base_path = dev_home / f"{variant.model_id}.gguf"
    alias_path = dev_home / "Some-Finetune-Q4_K_M.gguf"

    before = presets.preset_for_model(alias_path, MACHINE, set())
    _write_aliases({"Some-Finetune-Q4_K_M": variant.model_id})
    base = presets.preset_for_model(base_path, MACHINE, set())
    after = presets.preset_for_model(alias_path, MACHINE, set())

    assert base.refusal is None
    assert "mmproj" in base.keys and base.keys.get("spec-type") == "draft-mtp"
    # Generic path before the alias: no projector, no draft head.
    assert "mmproj" not in before.keys and "spec-type" not in before.keys
    # With it: identical policy, own model id.
    assert after.model_id == "Some-Finetune-Q4_K_M"
    assert after.window == base.window and after.spilled == base.spilled
    assert {k: v for k, v in after.keys.items() if k != "model"} == \
        {k: v for k, v in base.keys.items() if k != "model"}
