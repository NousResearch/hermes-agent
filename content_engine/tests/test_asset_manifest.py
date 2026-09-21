"""Tests for P11 planned/generated asset provenance manifests."""

import hashlib
import json

import pytest

from blog import asset_manifest as am


def _digest(value="x"):
    return hashlib.sha256(value.encode()).hexdigest()


def _reference_input(reference_id="layout-hero", role="layout"):
    return {
        "reference_id": reference_id,
        "sha256": _digest(reference_id),
        "provenance_class": "sahil_curated",
        "visual_role": role,
    }


def _planned_record(**overrides):
    record = {
        "asset_key": "hero",
        "article_id": "article-1",
        "state": "planned",
        "visual_plan_schema_version": "1",
        "visual_plan_digest": _digest("plan"),
        "prompt": "A brass control hall showing the article mechanism.",
        "prompt_digest": _digest("A brass control hall showing the article mechanism."),
        "reference_inputs": [_reference_input()],
        "provider": None,
        "model": None,
        "output_path": "hero.png",
        "output_digest": None,
        "requested_dimensions": {"width": 1600, "height": 900},
        "actual_dimensions": None,
        "generated_at": None,
        "text_ocr": {"policy": "none", "result": "not-run"},
        "visual_qa": {"status": "pending", "rejection_reasons": []},
        "review_status": "pending",
    }
    record.update(overrides)
    return record


def _generated_record(**overrides):
    record = _planned_record(
        state="generated",
        provider="local-comfyui-rest",
        model="unbound-in-p11",
        output_digest=_digest("output"),
        actual_dimensions={"width": 1600, "height": 900},
        generated_at="2026-07-23T11:00:00+00:00",
        text_ocr={"policy": "none", "result": "pass"},
        visual_qa={"status": "approved", "rejection_reasons": []},
        review_status="approved",
    )
    record.update(overrides)
    return record


def test_planned_record_is_valid_without_provider_or_output():
    manifest = am.build_asset_manifest("article-1", [_planned_record()])
    record = manifest.records[0]
    assert record.state == "planned"
    assert record.provider is None
    assert record.output_digest is None
    assert record.reference_inputs[0].visual_role == "layout"


def test_generated_record_requires_completion_evidence():
    manifest = am.build_asset_manifest("article-1", [_generated_record()])
    assert manifest.records[0].state == "generated"
    assert manifest.records[0].actual_dimensions == {"height": 900, "width": 1600}


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"state": "planned", "output_digest": _digest("wrong")}, "planned"),
        ({"state": "generated", "provider": None}, "provider"),
        ({"state": "generated", "actual_dimensions": None}, "actual_dimensions"),
        ({"prompt_digest": _digest("different")}, "prompt_digest"),
    ],
)
def test_invalid_lifecycle_evidence_is_rejected(overrides, message):
    base = _generated_record() if overrides.get("state") == "generated" else _planned_record()
    base.update(overrides)
    with pytest.raises(am.AssetManifestError, match=message):
        am.build_asset_manifest("article-1", [base])


def test_reference_input_must_bind_hash_provenance_and_role():
    ref = _reference_input()
    del ref["provenance_class"]
    with pytest.raises(am.AssetManifestError, match="provenance_class"):
        am.build_asset_manifest("article-1", [_planned_record(reference_inputs=[ref])])


def test_credential_fields_are_rejected_but_prompt_is_preserved():
    data = _planned_record(api_key="[REDACTED]")
    with pytest.raises(am.AssetManifestError, match="credential"):
        am.build_asset_manifest("article-1", [data])


def test_duplicate_asset_keys_and_escaping_output_path_are_rejected():
    with pytest.raises(am.AssetManifestError, match="duplicate asset keys"):
        am.build_asset_manifest("article-1", [_planned_record(), _planned_record()])
    with pytest.raises(am.AssetManifestError, match="must be relative"):
        am.build_asset_manifest("article-1", [_planned_record(output_path="../hero.png")])


def test_manifest_json_round_trip_is_deterministic_and_save_is_explicit(tmp_path):
    manifest = am.build_asset_manifest("article-1", [_planned_record(), _generated_record(asset_key="section")])
    restored = am.AssetManifest.from_json(manifest.to_json())
    assert restored.to_json() == manifest.to_json()
    output = tmp_path / "article.asset-manifest.json"
    am.save_asset_manifest(manifest, output)
    assert json.loads(output.read_text()) == manifest.to_dict()
    assert output.read_text() == manifest.to_json() + "\n"


def test_empty_article_id_and_asset_key_are_rejected():
    with pytest.raises(am.AssetManifestError, match="article_id is required"):
        am.build_asset_manifest("", [_planned_record()])
    with pytest.raises(am.AssetManifestError, match="asset_key is required"):
        am.build_asset_manifest("article-1", [_planned_record(asset_key="")])


def test_absolute_output_path_is_rejected():
    with pytest.raises(am.AssetManifestError, match="must be relative"):
        am.build_asset_manifest("article-1", [_planned_record(output_path="/tmp/hero.png")])


def test_missing_visual_plan_digest_is_rejected():
    with pytest.raises(am.AssetManifestError, match="visual_plan_digest"):
        am.build_asset_manifest("article-1", [_planned_record(visual_plan_digest="")])


def test_invalid_digest_format_is_rejected():
    with pytest.raises(am.AssetManifestError, match="64-character"):
        am.build_asset_manifest("article-1", [_planned_record(visual_plan_digest="not-a-digest")])


def test_generated_record_requires_an_output_digest():
    record = _generated_record(output_digest=None)
    with pytest.raises(am.AssetManifestError, match="output_digest"):
        am.build_asset_manifest("article-1", [record])


def test_rejected_visual_qa_requires_reasons():
    record = _generated_record(
        visual_qa={"status": "rejected", "rejection_reasons": []},
        review_status="rejected",
    )
    with pytest.raises(am.AssetManifestError, match="rejection_reasons"):
        am.build_asset_manifest("article-1", [record])


def test_invalid_review_status_is_rejected():
    with pytest.raises(am.AssetManifestError, match="review_status"):
        am.build_asset_manifest("article-1", [_planned_record(review_status="published")])


def test_unknown_manifest_version_is_rejected():
    data = am.build_asset_manifest("article-1", [_planned_record()]).to_dict()
    data["version"] = "999"
    with pytest.raises(am.AssetManifestError, match="unsupported manifest version"):
        am.AssetManifest.from_dict(data)
