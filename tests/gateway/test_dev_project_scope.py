from gateway.dev_control.project_scope import (
    DEFAULT_PROJECT_ID,
    normalize_project_id,
    project_id_from_payload,
    resolve_project_id,
)


def test_normalize_project_id_strips_and_rejects_empty():
    assert normalize_project_id("  OrynPlatform  ") == "OrynPlatform"
    assert normalize_project_id("") is None
    assert normalize_project_id(None) is None


def test_resolve_project_id_uses_first_non_empty_candidate():
    assert resolve_project_id(None, "", "CustomProject") == "CustomProject"
    assert resolve_project_id("Primary", "Secondary") == "Primary"
    assert resolve_project_id(None, None) == DEFAULT_PROJECT_ID


def test_project_id_from_payload_prefers_top_level_then_context():
    assert project_id_from_payload({
        "project_id": "TopLevel",
        "project_context": {"project_id": "ContextOnly"},
    }) == "TopLevel"
    assert project_id_from_payload({
        "project_context": {
            "project_id": "ContextOnly",
            "project_name": "Context Project",
        },
    }) == "ContextOnly"
    assert project_id_from_payload({}) == DEFAULT_PROJECT_ID
