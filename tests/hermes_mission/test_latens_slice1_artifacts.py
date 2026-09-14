import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLICE = ROOT / "hermes-mission" / "latens" / "slice-1"


def load_json(name: str):
    with (SLICE / name).open(encoding="utf-8") as fh:
        return json.load(fh)


def test_required_slice_artifacts_exist():
    expected = {
        "README.md",
        "site-routes.json",
        "lexicon-routes.json",
        "provenance.schema.json",
        "provenance-example.public.json",
        "qa-security-checklist.md",
    }
    assert expected <= {path.name for path in SLICE.iterdir()}


def test_site_routes_are_unique_and_public_private_scoped():
    manifest = load_json("site-routes.json")
    paths = [route["path"] for route in manifest["routes"]]

    assert len(paths) == len(set(paths))
    assert manifest["route_policy"]["deploy_requires_founder_yes"] is True
    assert "/review/{run_id}" in paths

    for route in manifest["routes"]:
        assert route["path"].startswith("/")
        assert route["data_scope"] in {"public", "private_review"}
        if route["audience"] == "public":
            assert route["data_scope"] == "public"
        if route["data_scope"] == "private_review":
            assert route["audience"] == "internal"


def test_lexicon_terms_map_to_canonical_urls_and_known_routes():
    routes = {route["path"] for route in load_json("site-routes.json")["routes"]}
    dynamic_routes = {"/provenance/{run_id}", "/review/{run_id}", "/editions/{edition_id}", "/artifacts/{artifact_id}"}
    allowed_concept_routes = routes | dynamic_routes
    lexicon = load_json("lexicon-routes.json")

    terms = {entry["term"] for entry in lexicon["terms"]}
    assert {"LATENS", "FERROGLYPH", "OZVENA", "provenance spine"} <= terms

    slugs = [entry["slug"] for entry in lexicon["terms"]]
    canonical_paths = [entry["canonical_path"] for entry in lexicon["terms"]]
    assert len(slugs) == len(set(slugs))
    assert len(canonical_paths) == len(set(canonical_paths))

    for entry in lexicon["terms"]:
        assert entry["canonical_path"] == f"/lexicon/{entry['slug']}"
        assert entry["concept_route"] in allowed_concept_routes
        assert entry["definition"].strip()


def test_provenance_schema_covers_required_spine_fields_and_scopes():
    schema = load_json("provenance.schema.json")
    assert "packet_scope" in schema["required"]
    assert set(schema["properties"]["packet_scope"]["enum"]) == {"public", "internal_review"}

    public_required = set(schema["properties"]["public"]["required"])
    assert {
        "model",
        "prompt_seed",
        "run_id",
        "emotion_signature",
        "source_refs",
        "edition_id",
        "artifact_id",
        "approvals",
        "public_private_split",
    } <= public_required

    approvals_required = set(
        schema["properties"]["public"]["properties"]["approvals"]["required"]
    )
    assert {"pm", "qa", "security", "founder_publication"} <= approvals_required

    private_required = set(schema["properties"]["private_review"]["required"])
    assert {"access", "raw_prompt_ref", "operator_notes_ref", "credential_handling"} <= private_required

    schema_text = json.dumps(schema)
    assert '"const": "internal_review"' in schema_text
    assert '"required": ["private_review"]' in schema_text
    assert '"const": "public"' in schema_text
    assert '"not": {"required": ["private_review"]}' in schema_text


def test_public_example_omits_private_review_and_keeps_approvals_pending():
    example = load_json("provenance-example.public.json")
    public = example["public"]

    assert set(example) == {"schema_version", "packet_scope", "public"}
    assert example["schema_version"] == "slice-1"
    assert example["packet_scope"] == "public"
    assert "private_review" not in example

    for approval in public["approvals"].values():
        assert approval["status"] == "pending"

    split = public["public_private_split"]
    private_fields = set(split["private_fields"])
    public_fields = set(split["public_fields"])
    assert private_fields
    assert public_fields
    assert private_fields.isdisjoint(public_fields)
    assert "credentials" in split["redaction_summary"].lower()
    assert "private_review" in split["redaction_summary"]
