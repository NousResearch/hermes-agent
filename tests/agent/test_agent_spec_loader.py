from pathlib import Path

import pytest

from agent.agent_spec import (
    load_agent_spec,
    preview_agent_spec,
    render_json,
    render_text,
    validate_agent_spec,
    validate_mcp_references,
)

FIX = Path("tests/fixtures/agent_specs")


def test_parses_agent_markdown_yaml_and_toml_imports():
    md = load_agent_spec(FIX / "valid/minimal.agent.md")
    assert md.raw["id"] == "minimal-agent"
    assert md.body.strip() == "Minimal instructions."
    assert load_agent_spec(FIX / "valid/import-example.yaml").raw["id"] == "yaml-import"
    assert load_agent_spec(FIX / "valid/import-example.toml").raw["id"] == "toml-import"


def test_validation_accepts_valid_fixtures(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "skills" / "kanban-worker").mkdir(parents=True)
    context_profile = home / "profiles" / "context-manager"
    (context_profile / "skills" / "kanban-worker").mkdir(parents=True)
    (context_profile / "config.yaml").write_text(
        "mcp_servers:\n  context7: {}\n  github: {}\n",
        encoding="utf-8",
    )

    minimal = validate_agent_spec(load_agent_spec(FIX / "valid/minimal.agent.md"))
    assert minimal.status == "pass"

    canonical_doc = load_agent_spec(FIX / "valid/full-preview.agent.md")
    canonical = validate_agent_spec(canonical_doc)
    assert canonical.status == "pass"
    assert canonical_doc.raw["id"] == "context-preview"


@pytest.mark.parametrize(
    ("fixture_name", "expected_codes"),
    [
        ("missing-schema.agent.md", {"unsupported_schema_version"}),
        ("bad-frontmatter.agent.md", {"invalid_frontmatter", "unsupported_schema_version"}),
        ("unknown-toolset.agent.md", {"unknown_toolset"}),
        ("bad-reasoning.agent.md", {"invalid_reasoning_effort"}),
        ("unknown-profile.agent.md", {"unknown_profile"}),
        ("missing-required-skill.agent.md", {"missing_required_skill"}),
        ("unsafe-artifact-path.agent.md", {"unsafe_artifact_path"}),
        ("unknown-gate.agent.md", {"unknown_gate_id"}),
        ("sandbox-overclaims-enforced.agent.md", {"sandbox_overclaimed"}),
        ("unknown-mcp.agent.md", {"unknown_server_id"}),
        ("malformed-structured-fields.agent.md", {"invalid_toolsets", "invalid_mcp", "invalid_gates"}),
    ],
)
def test_validation_rejects_negative_fixtures_with_expected_codes(tmp_path, monkeypatch, fixture_name, expected_codes):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "skills" / "kanban-worker").mkdir(parents=True)

    report = validate_agent_spec(load_agent_spec(FIX / "invalid" / fixture_name), profile_id="default")

    assert report.status == "fail", fixture_name
    actual_codes = {error.code for error in report.errors}
    assert expected_codes <= actual_codes


def test_unknown_fields_warn_and_strict_fails():
    doc = load_agent_spec(FIX / "valid/minimal.agent.md")
    doc.raw["surprise"] = True
    assert validate_agent_spec(doc, strict=False).status == "warn"
    assert validate_agent_spec(doc, strict=True).status == "fail"


def test_mcp_state_machine_covers_exact_states():
    catalog = {
        "configured": {"allowed_tools": ["ok"]},
        "optional": {},
        "required": {},
        "no_tools": {},
        "tools": {"allowed_tools": ["known"]},
    }
    refs = [
        {"server_id": "configured"},
        {"server_id": "optional", "required": False},
        {"server_id": "required", "required": True},
        {"server_id": "mystery"},
        {"server_id": "no_tools", "tool": "anything"},
        {"server_id": "tools", "tool": "missing"},
    ]
    states = [m.state for m in validate_mcp_references(refs, catalog=catalog, configured_ids={"configured"})]
    assert states == [
        "known_in_catalog_and_configured",
        "known_in_catalog_but_not_configured_optional",
        "known_in_catalog_but_required_missing",
        "unknown_server_id",
        "tool_discovery_unavailable",
        "tool_not_in_catalog_or_discovery",
    ]


def test_preview_is_deterministic_read_only_and_redacts_secrets(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  provider: test-provider\n  model: test-model\n"
        "mcp_servers:\n  github:\n    headers:\n      Authorization: Bearer dummy-token-value\n",
        encoding="utf-8",
    )
    (home / "SOUL.md").write_text("secret_key: dummy-secret-value", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    before = {p.relative_to(home): (p.stat().st_mtime_ns, p.read_bytes()) for p in home.rglob("*") if p.is_file()}
    report1 = preview_agent_spec("default", spec_path=FIX / "valid/full-preview.agent.md")
    report2 = preview_agent_spec("default", spec_path=FIX / "valid/full-preview.agent.md")
    payload = render_json(report1)
    assert payload == render_json(report2)
    assert "dummy-token-value" not in payload
    assert "dummy-secret-value" not in payload
    assert "dummy-should-not-leak" not in payload
    assert '"read_only_guarantee": true' in payload
    assert '"enforcement_enabled": false' in payload
    assert "Read-only guarantee: true" in render_text(report1)
    assert "Runtime enforcement: disabled" in render_text(report1)
    after = {p.relative_to(home): (p.stat().st_mtime_ns, p.read_bytes()) for p in home.rglob("*") if p.is_file()}
    assert before == after
