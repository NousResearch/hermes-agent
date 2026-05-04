"""Unit tests for ``hermes_cli.ruflo_agents`` discovery + config helpers."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from hermes_cli import ruflo_agents


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
    body = ruflo_agents._strip_frontmatter(text)
    assert body.startswith("# Body")
    assert "name: foo" not in body


def test_strip_frontmatter_passes_through_when_missing():
    text = "# No Frontmatter\n\nJust body."
    assert ruflo_agents._strip_frontmatter(text) == text


def test_strip_frontmatter_handles_unclosed_block():
    # If the closing --- is absent, return the original text unchanged so
    # we don't accidentally trim a real markdown body.
    text = "---\nname: incomplete\nbody\n"
    assert ruflo_agents._strip_frontmatter(text) == text


def test_parse_frontmatter_simple_keys():
    text = textwrap.dedent("""
        ---
        name: researcher
        description: Investigates patterns
        ---

        body
    """).lstrip()
    meta = ruflo_agents._parse_frontmatter(text)
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
    meta = ruflo_agents._parse_frontmatter(text)
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
    meta = ruflo_agents._parse_frontmatter(text)
    assert meta["description"] == "line one continued on line two"


def test_parse_frontmatter_missing_returns_empty():
    assert ruflo_agents._parse_frontmatter("# No frontmatter\nbody") == {}


# ── Discovery ─────────────────────────────────────────────────────────────


@pytest.fixture
def fake_ruflo(tmp_path: Path) -> Path:
    """Build a minimal ruflo-shaped tree for discovery tests."""
    # Two .claude/agents/ trees, one at root and one under v3/.
    a1 = tmp_path / ".claude" / "agents"
    a1.mkdir(parents=True)
    (a1 / "researcher.md").write_text(
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
    sub = a1 / "swarm"
    sub.mkdir()
    (sub / "coordinator.md").write_text(
        textwrap.dedent("""
            ---
            name: coordinator
            description: Coordinates swarm topology
            ---

            # Coordinator
        """).lstrip(),
        encoding="utf-8",
    )

    # A flow-nexus agent — should be filtered.
    fn = a1 / "flow-nexus"
    fn.mkdir()
    (fn / "auth.md").write_text("---\nname: auth\n---\n# Auth\n", encoding="utf-8")

    # A README.md at the agents root — filtered by basename.
    (a1 / "README.md").write_text("# Index\n", encoding="utf-8")

    # A second tree under v3/@claude-flow/cli/.claude/agents/ that has the
    # same researcher.md — should dedupe (first encounter wins).
    a2 = tmp_path / "v3" / "@claude-flow" / "cli" / ".claude" / "agents"
    a2.mkdir(parents=True)
    (a2 / "researcher.md").write_text(
        "---\nname: researcher\ndescription: dup\n---\n\n# Dup\n", encoding="utf-8"
    )

    # A v2 legacy file — should be skipped by the v2 filter.
    legacy = tmp_path / "v2" / ".claude" / "agents"
    legacy.mkdir(parents=True)
    (legacy / "legacy.md").write_text(
        "---\nname: legacy\n---\n# Legacy\n", encoding="utf-8"
    )

    return tmp_path


def test_discover_returns_filtered_unique_agents(fake_ruflo: Path):
    agents = ruflo_agents.discover_ruflo_agents(fake_ruflo)
    names = sorted(a.name for a in agents)
    # researcher and coordinator only — README, auth (flow-nexus), legacy (v2) all filtered.
    assert names == ["coordinator", "researcher"]


def test_discover_dedupes_by_name(fake_ruflo: Path):
    """Same agent name in two trees should appear exactly once.

    The "first encounter wins" claim in the docstring is real, but
    Path.rglob() doesn't guarantee directory walk order across platforms,
    so we just assert the dedupe + that the description came from one of
    the two known sources (not garbled by accidental concatenation).
    """
    agents = ruflo_agents.discover_ruflo_agents(fake_ruflo)
    by_name = {a.name: a for a in agents}
    # Dedupe: exactly one researcher even though it lives in two trees.
    matches = [a for a in agents if a.name == "researcher"]
    assert len(matches) == 1
    # Description from one of the two definitions, not corrupted.
    assert by_name["researcher"].description in {
        "Investigates patterns",
        "dup",
    }


def test_discover_assigns_categories(fake_ruflo: Path):
    agents = ruflo_agents.discover_ruflo_agents(fake_ruflo)
    by_name = {a.name: a for a in agents}
    assert by_name["researcher"].category == "general"  # at agents/ root
    assert by_name["coordinator"].category == "swarm"


def test_discover_returns_empty_for_missing_path(tmp_path: Path):
    # Subdirectory of tmp_path that doesn't exist
    missing = tmp_path / "nope"
    assert ruflo_agents.discover_ruflo_agents(missing) == []


def test_load_prompt_strips_frontmatter(fake_ruflo: Path):
    agents = ruflo_agents.discover_ruflo_agents(fake_ruflo)
    researcher = next(a for a in agents if a.name == "researcher")
    body = researcher.load_prompt()
    # Body could be either # Researcher or # Dup depending on rglob walk
    # order — both are valid post-frontmatter-strip outputs, what matters
    # is that no YAML leaked through.
    assert body.startswith("#")
    assert "name:" not in body
    assert body.strip() != ""


def test_group_by_category_preserves_within_group_order(fake_ruflo: Path):
    agents = ruflo_agents.discover_ruflo_agents(fake_ruflo)
    groups = ruflo_agents.group_by_category(agents)
    assert sorted(groups.keys()) == ["general", "swarm"]
    assert [a.name for a in groups["general"]] == ["researcher"]
    assert [a.name for a in groups["swarm"]] == ["coordinator"]


# ── Role-model map (config-backed) ────────────────────────────────────────
#
# These tests stub the load/save plumbing so they don't touch the real
# ~/.hermes/config.yaml.


def test_get_role_model_map_empty_when_no_delegation(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {},
    )
    assert ruflo_agents.get_role_model_map() == {}


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
    m = ruflo_agents.get_role_model_map()
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
    m = ruflo_agents.get_role_model_map()
    assert m == {
        "researcher": "claude-haiku-4-5",
        "good": "claude-opus-4-7",
    }


def test_set_role_model_writes_through(monkeypatch, tmp_path):
    """`set_role_model` writes to the active config.yaml via the inline saver."""
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    # Redirect HERMES_HOME so the inline saver writes to a tmp file, not
    # the real ~/.hermes/config.yaml.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert ruflo_agents.set_role_model("researcher", "claude-haiku-4-5") is True
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
    # Pre-seed the config file so the saver can read+write it.
    (tmp_path / "config.yaml").write_text(
        "delegation:\n  model_by_role:\n    researcher: claude-haiku-4-5\n",
        encoding="utf-8",
    )
    assert ruflo_agents.set_role_model("researcher", None) is True
    written = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    # researcher entry should be gone (and the dict empty).
    assert "researcher" not in written


def test_lookup_model_for_role_returns_none_when_unset(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"delegation": {"model_by_role": {"researcher": "claude-haiku-4-5"}}},
    )
    assert ruflo_agents.lookup_model_for_role("researcher") == "claude-haiku-4-5"
    assert ruflo_agents.lookup_model_for_role("unset_role") is None
    assert ruflo_agents.lookup_model_for_role("") is None
    assert ruflo_agents.lookup_model_for_role(None) is None


# ── apply_suggested_defaults ──────────────────────────────────────────────


def test_apply_suggested_defaults_fills_empties(monkeypatch, tmp_path):
    """First-run case: no existing pins → all suggested defaults applied."""
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    applied, skipped = ruflo_agents.apply_suggested_defaults()
    assert applied == len(ruflo_agents.SUGGESTED_ROLE_MODELS)
    assert skipped == 0
    written = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "researcher: claude-haiku-4-5" in written
    assert "security-architect: claude-opus-4-7" in written


def test_apply_suggested_defaults_preserves_user_pins(monkeypatch, tmp_path):
    """User pin on `researcher` should NOT be overwritten by default mode."""
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
    applied, skipped = ruflo_agents.apply_suggested_defaults(overwrite=False)
    # researcher kept; everything else freshly applied.
    assert skipped >= 1
    assert applied == len(ruflo_agents.SUGGESTED_ROLE_MODELS) - 1
    written = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    # User's researcher pin still present, unchanged.
    assert f"researcher: {user_pin}" in written


def test_apply_suggested_defaults_force_overwrites(monkeypatch, tmp_path):
    """`overwrite=True` clobbers existing pins back to the curated default."""
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
    applied, skipped = ruflo_agents.apply_suggested_defaults(overwrite=True)
    # researcher gets reset to suggested (haiku); count = full size.
    assert applied == len(ruflo_agents.SUGGESTED_ROLE_MODELS)
    written = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "researcher: claude-haiku-4-5" in written
    assert f"researcher: {user_pin}" not in written


def test_apply_suggested_defaults_idempotent(monkeypatch, tmp_path):
    """Running defaults twice is a no-op the second time."""
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    applied1, _ = ruflo_agents.apply_suggested_defaults()

    # Second call sees the now-populated map; reload via load_config patch.
    map_after_first = dict(ruflo_agents.SUGGESTED_ROLE_MODELS)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"delegation": {"model_by_role": map_after_first}},
    )
    applied2, skipped2 = ruflo_agents.apply_suggested_defaults()
    assert applied2 == 0
    assert skipped2 == len(ruflo_agents.SUGGESTED_ROLE_MODELS)
    assert applied1 == len(ruflo_agents.SUGGESTED_ROLE_MODELS)


def test_suggested_role_models_only_uses_known_models():
    """Sanity: every suggested model is one of the three curated choices."""
    valid = {"claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"}
    bad = {
        role: model
        for role, model in ruflo_agents.SUGGESTED_ROLE_MODELS.items()
        if model not in valid
    }
    assert not bad, f"Unknown model in defaults: {bad}"
