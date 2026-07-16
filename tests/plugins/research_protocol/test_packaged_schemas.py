"""Generated schema and policy packaging contracts."""

import json
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from plugins.research_protocol.models.schema_export import render_schema_documents
from plugins.research_protocol.models.policy import POLICY_MODELS

PLUGIN_ROOT = Path(__file__).resolve().parents[3] / "plugins" / "research_protocol"
SCHEMA_ROOT = PLUGIN_ROOT / "schemas"
POLICY_ROOT = PLUGIN_ROOT / "policies"
EXPECTED_SCHEMAS = {
    "plan.v1.json",
    "run_config.v1.json",
    "manifest_record.v1.json",
    "source_candidate.v1.json",
    "evidence_snapshot.v1.json",
    "source_quality.v1.json",
    "conflict_record.v1.json",
    "dedup_cluster.v1.json",
    "verdict_record.v1.json",
    "figure_spec.v1.json",
}
EXPECTED_POLICIES = {
    "source_quality.v1.yaml",
    "contradiction.v1.yaml",
    "dedup.v1.yaml",
}
BOUNDARY_COMMENT = (
    "JSON Schema validation is necessary but insufficient; artifact bytes must be "
    "validated with load_artifact_from_json_bytes and Pydantic model_validate_json."
)


def test_json_schemas_are_generated_deterministically_from_models():
    rendered = render_schema_documents()

    assert set(rendered) == EXPECTED_SCHEMAS
    for filename, expected_bytes in rendered.items():
        packaged_bytes = (SCHEMA_ROOT / filename).read_bytes()
        assert packaged_bytes == expected_bytes
        parsed = json.loads(packaged_bytes)
        assert parsed["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert "schema_version" in parsed["properties"]
        assert "producer_version" in parsed["properties"]
        assert "run_id" in parsed["properties"]
        assert "created_at" in parsed["properties"]


def test_all_ten_schemas_declare_the_pydantic_boundary_contract():
    schemas = {
        filename: json.loads((SCHEMA_ROOT / filename).read_bytes())
        for filename in EXPECTED_SCHEMAS
    }

    assert len(schemas) == 10
    for schema in schemas.values():
        assert BOUNDARY_COMMENT in schema["$comment"]
        assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("schema_name", "field_name"),
    [
        ("run_config.v1.json", "policy_hashes"),
        ("source_quality.v1.json", "dimensions"),
        ("figure_spec.v1.json", "units"),
        ("figure_spec.v1.json", "labels"),
    ],
)
def test_mapping_schemas_reject_keys_outside_identifier_pattern(
    schema_name, field_name
):
    schema = json.loads((SCHEMA_ROOT / schema_name).read_bytes())
    mapping_schema = schema["properties"][field_name]

    assert "patternProperties" in mapping_schema
    assert mapping_schema["additionalProperties"] is False


def test_schema_expresses_path_shape_and_documents_semantic_path_validation():
    schema = json.loads((SCHEMA_ROOT / "manifest_record.v1.json").read_bytes())
    path_schema = schema["properties"]["path_relative"]
    path_pattern = re.compile(path_schema["pattern"])

    assert path_schema["maxLength"] == 8192
    assert path_pattern.fullmatch("evidence/evidence-001.json")
    for unsafe in (
        "..\\\\secret.json",
        "foo\\\\..\\\\secret.json",
        ".",
        "..",
        "/absolute.json",
        "foo//bar.json",
        "foo/./bar.json",
        "foo/../bar.json",
        "foo/bar.json/",
        "foo\x00bar.json",
        " evidence/evidence-001.json",
        "evidence/evidence-001.json ",
    ):
        assert path_pattern.fullmatch(unsafe) is None
    assert "Pydantic" in path_schema["$comment"]


def test_schema_annotations_cover_cross_field_and_nested_collection_invariants():
    plan = json.loads((SCHEMA_ROOT / "plan.v1.json").read_bytes())
    conflict = json.loads((SCHEMA_ROOT / "conflict_record.v1.json").read_bytes())
    dedup = json.loads((SCHEMA_ROOT / "dedup_cluster.v1.json").read_bytes())

    capabilities = plan["properties"]["capabilities"]
    assert capabilities["uniqueItems"] is True
    assert "capability" in capabilities["x-hermes-validation"]
    assert "left_provenance_id" in conflict["$comment"]
    assert "right_provenance_id" in conflict["$comment"]
    assert "distinct" in conflict["$comment"]

    members = dedup["properties"]["member_provenance_ids"]
    assert members["uniqueItems"] is True
    assert "canonical_provenance_id" in dedup["$comment"]
    assert "member" in dedup["$comment"]


def test_versioned_policy_documents_validate_strictly():
    assert {path.name for path in POLICY_ROOT.glob("*.yaml")} == EXPECTED_POLICIES

    for filename, model in POLICY_MODELS.items():
        payload = yaml.safe_load((POLICY_ROOT / filename).read_text(encoding="utf-8"))
        validated = model.model_validate(payload)
        assert validated.schema_version.endswith(".v1")

        payload["unknown_top_level_key"] = True
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            model.model_validate(payload)


def test_packaging_metadata_covers_all_research_protocol_data_channels():
    pyproject = (PLUGIN_ROOT.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    manifest = (PLUGIN_ROOT.parents[1] / "MANIFEST.in").read_text(encoding="utf-8")

    for suffix in ("*.json", "*.yaml", "*.sql", "*.md"):
        assert f'"research_protocol/**/*{suffix[1:]}"' in pyproject
    assert (
        "recursive-include plugins/research_protocol *.json *.yaml *.sql *.md"
        in manifest
    )
