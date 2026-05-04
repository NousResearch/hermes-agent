"""Unit tests for ``hermes_cli.personas`` discovery + config helpers.

Personas live under ``~/.hermes/personas/<category>/<name>.md``.  The
fake-personas fixture builds that exact layout in a tmp dir, then routes
discovery to it via the ``personas_path=`` arg.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from hermes_cli import personas


# ── Frontmatter parser ────────────────────────────────────────────────────


def test_strip_frontmatter_drops_yaml_block():
    text = textwrap.dedent("""
        ---
        name: foo
        description: bar
        ---

        # Body

        Content.
    """).lstrip()
    body = personas._strip_frontmatter(text)
    assert body.startswith("# Body")
    assert "name: foo" not in body


def test_strip_frontmatter_passes_through_when_missing():
    text = "# No Frontmatter\n\nJust body."
    assert personas._strip_frontmatter(text) == text


def test_strip_frontmatter_handles_unclosed_block():
    text = "---\nname: incomplete\nbody\n"
    assert personas._strip_frontmatter(text) == text


def test_parse_frontmatter_simple_keys():
    text = textwrap.dedent("""
        ---
        name: researcher
        description: Investigates patterns
        ---

        body
    """).lstrip()
    meta = personas._parse_frontmatter(text)
    assert meta["name"] == "researcher"
    assert meta["description"] == "Investigates patterns"


def test_parse_frontmatter_strips_quotes():
    text = textwrap.dedent("""
        ---
        name: "quoted-name"
        description: 'single-quoted description'
        ---
        body
    """).lstrip()
    meta = personas._parse_frontmatter(text)
    assert meta["name"] == "quoted-name"
    assert meta["description"] == "single-quoted description"


def test_parse_frontmatter_joins_continuation_lines():
    text = textwrap.dedent("""
        ---
        name: foo
        description: line one
          continued on line two
        ---
        body
    """).lstrip()
    meta = personas._parse_frontmatter(text)
    assert meta["description"] == "line one continued on line two"


def test_parse_frontmatter_missing_returns_empty():
    assert personas._parse_frontmatter("# No frontmatter\nbody") == {}


# ── Discovery ─────────────────────────────────────────────────────────────


@pytest.fixture
def fake_personas(tmp_path: Path) -> Path:
    """Build a personas tree: <root>/<category>/<name>.md and root .md files."""
    # Root-level persona (category="general").
    (tmp_path / "researcher.md").write_text(
        textwrap.dedent("""
            ---
            name: researcher
            description: Investigates patterns
            ---

            # Researcher
            Body content.
        """).lstrip(),
        encoding="utf-8",
    )
    # Subdir persona (category="swarm").
    swarm = tmp_path / "swarm"
    swarm.mkdir()
    (swarm / "coordinator.md").write_text(
        textwrap.dedent("""
            ---
            name: coordinator
            description: Coordinates swarm topology
            ---

            # Coordinator
        """).lstrip(),
        encoding="utf-8",
    )
    # README at root — should be filtered by _NON_AGENT_BASENAMES.
    (tmp_path / "README.md").write_text("# README\n", encoding="utf-8")
    return tmp_path


def test_discover_returns_filtered_personas(fake_personas: Path):
    found = personas.discover_personas(fake_personas)
    names = sorted(p.name for p in found)
    assert names == ["coordinator", "researcher"]


def test_discover_assigns_categories(fake_personas: Path):
    found = personas.discover_personas(fake_personas)
    by_name = {p.name: p for p in found}
    assert by_name["researcher"].category == "general"  # at root
    assert by_name["coordinator"].category == "swarm"   # under swarm/


def test_discover_returns_empty_for_missing_path(tmp_path: Path):
    missing = tmp_path / "nope"
    assert personas.discover_personas(missing) == []


def test_load_prompt_strips_frontmatter(fake_personas: Path):
    found = personas.discover_personas(fake_personas)
    researcher = next(p for p in found if p.name == "researcher")
    body = researcher.load_prompt()
    assert body.startswith("# Researcher")
    assert "name:" not in body
    assert body.strip() != ""


def test_group_by_category_preserves_within_group_order(fake_personas: Path):
    found = personas.discover_personas(fake_personas)
    groups = personas.group_by_category(found)
    assert sorted(groups.keys()) == ["general", "swarm"]
    assert [p.name for p in groups["general"]] == ["researcher"]
    assert [p.name for p in groups["swarm"]] == ["coordinator"]


def test_lookup_agent_via_discovery(fake_personas: Path, monkeypatch):
    monkeypatch.setattr(personas, "get_personas_path", lambda: fake_personas)
    p = personas.lookup_agent("researcher")
    assert p is not None
    assert p.name == "researcher"
    assert personas.lookup_agent("ghost") is None
    assert personas.lookup_agent("") is None


# ── sync_from_ruflo ───────────────────────────────────────────────────────


@pytest.fixture
def fake_ruflo(tmp_path: Path) -> Path:
    """Minimal ruflo-shaped tree (.claude/agents/...) for sync_from_ruflo."""
    a1 = tmp_path / ".claude" / "agents"
    a1.mkdir(parents=True)
    (a1 / "researcher.md").write_text(
        "---\nname: researcher\n---\n# Researcher\n", encoding="utf-8"
    )
    sub = a1 / "swarm"
    sub.mkdir()
    (sub / "coordinator.md").write_text(
        "---\nname: coordinator\n---\n# Coordinator\n", encoding="utf-8"
    )
    # Should be filtered out by sync (cloud-integration category).
    fn = a1 / "flow-nexus"
    fn.mkdir()
    (fn / "auth.md").write_text("---\nname: auth\n---\n# Auth\n", encoding="utf-8")
    # Legacy v2 tree — filtered.
    legacy = tmp_path / "v2" / ".claude" / "agents"
    legacy.mkdir(parents=True)
    (legacy / "old.md").write_text("---\nname: old\n---\n# Old\n", encoding="utf-8")
    return tmp_path


def test_sync_copies_filtered_personas(fake_ruflo: Path, tmp_path: Path):
    dst = tmp_path / "personas-out"
    copied, skipped = personas.sync_from_ruflo(fake_ruflo, dest=dst)
    assert copied == 2  # researcher + coordinator; flow-nexus and v2 filtered
    assert skipped == 0
    assert (dst / "general" / "researcher.md").is_file()
    assert (dst / "swarm" / "coordinator.md").is_file()


def test_sync_skips_existing_when_no_overwrite(fake_ruflo: Path, tmp_path: Path):
    dst = tmp_path / "personas-out"
    personas.sync_from_ruflo(fake_ruflo, dest=dst)  # first sync
    copied, skipped = personas.sync_from_ruflo(fake_ruflo, dest=dst)  # second
    assert copied == 0
    assert skipped == 2


def test_sync_overwrites_when_requested(fake_ruflo: Path, tmp_path: Path):
    dst = tmp_path / "personas-out"
    personas.sync_from_ruflo(fake_ruflo, dest=dst)
    # Modify the dest copy, then re-sync with overwrite to verify it gets reset.
    target = dst / "general" / "researcher.md"
    target.write_text("LOCAL EDIT", encoding="utf-8")
    copied, _ = personas.sync_from_ruflo(fake_ruflo, dest=dst, overwrite=True)
    assert copied == 2
    assert "name: researcher" in target.read_text(encoding="utf-8")


def test_sync_missing_root_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        personas.sync_from_ruflo(tmp_path / "nope", dest=tmp_path / "out")


# ── Role-model map (config-backed) ────────────────────────────────────────
#
# These tests stub the load/save plumbing so they don't touch the real
# ~/.hermes/config.yaml.


def test_get_role_model_map_empty_when_no_delegation(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    assert personas.get_role_model_map() == {}


def test_get_role_model_map_reads_delegation_section(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "delegation": {
                "model_by_role": {
                    "researcher": "claude-haiku-4-5",
                    "architect": "claude-sonnet-4-6",
                }
            }
        },
    )
    m = personas.get_role_model_map()
    assert m == {
        "researcher": "claude-haiku-4-5",
        "architect": "claude-sonnet-4-6",
    }


def test_get_role_model_map_filters_non_string_values(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "delegation": {
                "model_by_role": {
                    "researcher": "claude-haiku-4-5",
                    "bogus": 42,         # non-string value — drop
                    "blank": "   ",      # whitespace-only — drop
                    "good": "claude-opus-4-7",
                }
            }
        },
    )
    m = personas.get_role_model_map()
    assert m == {
        "researcher": "claude-haiku-4-5",
        "good": "claude-opus-4-7",
    }


def test_set_role_model_writes_through(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert personas.set_role_model("researcher", "claude-haiku-4-5") is True
    written = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "researcher:" in written
    assert "claude-haiku-4-5" in written


def test_set_role_model_clears_when_model_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "delegation": {
                "model_by_role": {"researcher": "claude-haiku-4-5"}
            }
        },
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "delegation:\n  model_by_role:\n    researcher: claude-haiku-4-5\n",
        encoding="utf-8",
    )
    assert personas.set_role_model("researcher", None) is True
    written = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "researcher" not in written


def test_lookup_model_for_role_returns_none_when_unset(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"delegation": {"model_by_role": {"researcher": "claude-haiku-4-5"}}},
    )
    assert personas.lookup_model_for_role("researcher") == "claude-haiku-4-5"
    assert personas.lookup_model_for_role("unset_role") is None
    assert personas.lookup_model_for_role("") is None
    assert personas.lookup_model_for_role(None) is None


# ── apply_suggested_defaults ──────────────────────────────────────────────


def test_apply_suggested_defaults_fills_empties(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    applied, skipped = personas.apply_suggested_defaults()
    assert applied == len(personas.SUGGESTED_ROLE_MODELS)
    assert skipped == 0
    written = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "researcher: claude-haiku-4-5" in written
    assert "security-architect: claude-opus-4-7" in written


def test_apply_suggested_defaults_preserves_user_pins(monkeypatch, tmp_path):
    user_pin = "claude-opus-4-7"  # not the suggested default for researcher
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"delegation": {"model_by_role": {"researcher": user_pin}}},
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"delegation:\n  model_by_role:\n    researcher: {user_pin}\n",
        encoding="utf-8",
    )
    applied, skipped = personas.apply_suggested_defaults(overwrite=False)
    assert skipped >= 1
    assert applied == len(personas.SUGGESTED_ROLE_MODELS) - 1
    written = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert f"researcher: {user_pin}" in written


def test_apply_suggested_defaults_force_overwrites(monkeypatch, tmp_path):
    user_pin = "claude-opus-4-7"
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"delegation": {"model_by_role": {"researcher": user_pin}}},
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"delegation:\n  model_by_role:\n    researcher: {user_pin}\n",
        encoding="utf-8",
    )
    applied, skipped = personas.apply_suggested_defaults(overwrite=True)
    assert applied == len(personas.SUGGESTED_ROLE_MODELS)
    written = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "researcher: claude-haiku-4-5" in written
    assert f"researcher: {user_pin}" not in written


def test_apply_suggested_defaults_idempotent(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    applied1, _ = personas.apply_suggested_defaults()

    map_after_first = dict(personas.SUGGESTED_ROLE_MODELS)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"delegation": {"model_by_role": map_after_first}},
    )
    applied2, skipped2 = personas.apply_suggested_defaults()
    assert applied2 == 0
    assert skipped2 == len(personas.SUGGESTED_ROLE_MODELS)
    assert applied1 == len(personas.SUGGESTED_ROLE_MODELS)


def test_suggested_role_models_only_uses_known_models():
    """Sanity: every suggested model is one of the three curated choices."""
    valid = {"claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"}
    bad = {
        role: model
        for role, model in personas.SUGGESTED_ROLE_MODELS.items()
        if model not in valid
    }
    assert not bad, f"Unknown model in defaults: {bad}"


# ── Back-compat shim ──────────────────────────────────────────────────────


def test_ruflo_agents_shim_reexports():
    """The legacy ``hermes_cli.ruflo_agents`` shim re-exports everything we
    need for the old import paths to keep working without churn."""
    from hermes_cli import ruflo_agents

    # Public API
    assert ruflo_agents.Persona is personas.Persona
    assert ruflo_agents.RufloAgent is personas.Persona
    assert ruflo_agents.SUGGESTED_ROLE_MODELS is personas.SUGGESTED_ROLE_MODELS
    assert ruflo_agents.discover_ruflo_agents is personas.discover_ruflo_agents
    assert ruflo_agents.lookup_agent is personas.lookup_agent
    assert ruflo_agents.get_role_model_map is personas.get_role_model_map
    assert ruflo_agents.set_role_model is personas.set_role_model
    assert ruflo_agents.lookup_model_for_role is personas.lookup_model_for_role
    assert ruflo_agents.apply_suggested_defaults is personas.apply_suggested_defaults
    # Private helpers re-exported for older test imports
    assert ruflo_agents._parse_frontmatter is personas._parse_frontmatter
    assert ruflo_agents._strip_frontmatter is personas._strip_frontmatter
