"""AgentSpec parsing and validation."""

from __future__ import annotations

import pytest

from nova.errors import SpecError
from nova.spec import AgentSpec


def test_minimal_spec_defaults_name_to_id():
    spec = AgentSpec.parse({"id": "support"})
    assert spec.id == "support"
    assert spec.name == "support"
    assert spec.enabled is True


def test_unknown_field_is_rejected():
    """A misspelled key must fail loudly, not produce a subtly wrong agent."""
    with pytest.raises(SpecError, match="unknown field"):
        AgentSpec.parse({"id": "support", "instuctions": "typo.md"})


def test_unknown_nested_field_is_rejected():
    with pytest.raises(SpecError, match="unknown field"):
        AgentSpec.parse({"id": "support", "limits": {"max_trns": 4}})


@pytest.mark.parametrize("bad", ["Bad_ID", "-leading", "trailing-", "has space", "UPPER", ""])
def test_invalid_ids_rejected(bad):
    with pytest.raises(SpecError):
        AgentSpec.parse({"id": bad})


def test_id_must_be_present():
    with pytest.raises(SpecError, match="required"):
        AgentSpec.parse({"name": "no id"})


def test_wrong_type_names_the_field():
    with pytest.raises(SpecError) as excinfo:
        AgentSpec.parse({"id": "support", "limits": {"max_turns": "sixty"}})
    assert "limits.max_turns" in str(excinfo.value)


def test_bool_is_not_accepted_as_int():
    with pytest.raises(SpecError, match="whole number"):
        AgentSpec.parse({"id": "support", "limits": {"max_turns": True}})


def test_limit_bounds_enforced():
    with pytest.raises(SpecError, match="at least"):
        AgentSpec.parse({"id": "support", "limits": {"max_retries": -1}})


def test_env_interpolation_with_default(monkeypatch):
    monkeypatch.delenv("NOVA_TEST_PROVIDER", raising=False)
    spec = AgentSpec.parse({"id": "s", "model": {"provider": "${NOVA_TEST_PROVIDER:-bedrock}"}})
    assert spec.model.provider == "bedrock"


def test_env_interpolation_prefers_environment(monkeypatch):
    monkeypatch.setenv("NOVA_TEST_PROVIDER", "anthropic")
    spec = AgentSpec.parse({"id": "s", "model": {"provider": "${NOVA_TEST_PROVIDER:-bedrock}"}})
    assert spec.model.provider == "anthropic"


def test_missing_env_without_default_is_an_error(monkeypatch):
    """A blank model name would fail much later, in another process, in front of a customer."""
    monkeypatch.delenv("NOVA_TEST_MISSING", raising=False)
    with pytest.raises(SpecError, match="not set"):
        AgentSpec.parse({"id": "s", "model": {"name": "${NOVA_TEST_MISSING}"}})


def test_allow_and_deny_overlap_rejected():
    with pytest.raises(SpecError, match="both allow and deny"):
        AgentSpec.parse({"id": "s", "tools": {"allow": ["email"], "deny": ["email"]}})


def test_self_delegation_rejected():
    with pytest.raises(SpecError, match="may not delegate to itself"):
        AgentSpec.parse({"id": "s", "delegation": {"may_assign_to": ["s"]}})


def test_duplicate_list_entry_rejected():
    with pytest.raises(SpecError, match="more than once"):
        AgentSpec.parse({"id": "s", "permissions": ["read", "read"]})


def test_instructions_file_and_inline_are_mutually_exclusive(tmp_path):
    (tmp_path / "p.md").write_text("x", encoding="utf-8")
    with pytest.raises(SpecError, match="not both"):
        AgentSpec.parse(
            {"id": "s", "instructions": "p.md", "instructions_text": "inline"}, base_dir=tmp_path
        )


def test_instructions_path_escape_is_rejected(tmp_path):
    with pytest.raises(SpecError, match="inside the bundle"):
        AgentSpec.parse({"id": "s", "instructions": "../secrets.md"}, base_dir=tmp_path)


def test_absolute_instructions_path_is_rejected(tmp_path):
    with pytest.raises(SpecError, match="inside the bundle"):
        AgentSpec.parse({"id": "s", "instructions": "/etc/passwd"}, base_dir=tmp_path)


def test_missing_instructions_file_is_reported(tmp_path):
    with pytest.raises(SpecError, match="not found"):
        AgentSpec.parse({"id": "s", "instructions": "nope.md"}, base_dir=tmp_path)


def test_instructions_are_read_into_the_spec(tmp_path):
    (tmp_path / "p.md").write_text("You are a test agent.\n", encoding="utf-8")
    spec = AgentSpec.parse({"id": "s", "instructions": "p.md"}, base_dir=tmp_path)
    assert spec.instructions == "You are a test agent."
    assert spec.instructions_path == "p.md"


def test_digest_is_stable_across_equal_specs():
    a = AgentSpec.parse({"id": "s", "model": {"provider": "bedrock"}})
    b = AgentSpec.parse({"id": "s", "model": {"provider": "bedrock"}})
    assert a.digest() == b.digest()


def test_digest_changes_with_meaning():
    a = AgentSpec.parse({"id": "s", "limits": {"max_turns": 10}})
    b = AgentSpec.parse({"id": "s", "limits": {"max_turns": 11}})
    assert a.digest() != b.digest()


def test_digest_ignores_source_path(tmp_path):
    a = AgentSpec.parse({"id": "s"}, source=tmp_path / "one.yaml")
    b = AgentSpec.parse({"id": "s"}, source=tmp_path / "two.yaml")
    assert a.digest() == b.digest()


def test_reasoning_effort_is_constrained():
    with pytest.raises(SpecError, match="must be one of"):
        AgentSpec.parse({"id": "s", "model": {"reasoning_effort": "extreme"}})
