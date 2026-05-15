"""Tests for gateway.agent_registry — multi-agent profile enumeration."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.agent_registry import (
    DEFAULT_AGENT_NAME,
    AgentProfile,
    AgentRegistry,
    default_registry,
    reset_default_registry,
)


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """Lay out a fake hermes root with default + 2 named profiles."""
    root = tmp_path / ".hermes"
    root.mkdir()
    (root / "profiles").mkdir()
    (root / "profiles" / "coder").mkdir()
    (root / "profiles" / "data-sci").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    reset_default_registry()
    yield root
    reset_default_registry()


class TestAgentRegistryBasics:
    def test_lists_default_first(self, fake_root):
        r = AgentRegistry()
        names = [p.name for p in r.list()]
        assert names[0] == DEFAULT_AGENT_NAME
        assert set(names) == {DEFAULT_AGENT_NAME, "coder", "data-sci"}

    def test_named_profiles_sorted(self, fake_root):
        # add another profile out of alphabetical order
        (fake_root / "profiles" / "alpha").mkdir()
        r = AgentRegistry()
        named = [p.name for p in r.list() if not p.is_default]
        assert named == sorted(named)

    def test_default_always_present(self, tmp_path, monkeypatch):
        """Even when ~/.hermes is empty, default agent is in the registry."""
        empty = tmp_path / ".hermes"
        empty.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        reset_default_registry()
        try:
            r = AgentRegistry()
            d = r.default()
            assert d.name == DEFAULT_AGENT_NAME
            assert d.is_default is True
            assert d.home == empty
        finally:
            reset_default_registry()

    def test_get_returns_named_profile(self, fake_root):
        r = AgentRegistry()
        p = r.get("coder")
        assert p is not None
        assert p.name == "coder"
        assert p.is_default is False
        assert p.home == fake_root / "profiles" / "coder"

    def test_get_default_alias(self, fake_root):
        r = AgentRegistry()
        for variant in ("default", "Default", "DEFAULT", "  default  "):
            p = r.get(variant)
            assert p is not None, variant
            assert p.name == DEFAULT_AGENT_NAME

    def test_get_unknown_returns_none(self, fake_root):
        r = AgentRegistry()
        assert r.get("nonexistent") is None
        assert r.get("") is None
        assert r.get("  ") is None

    def test_get_invalid_id_returns_none(self, fake_root):
        r = AgentRegistry()
        # Capitals / special chars are not valid profile ids — must reject
        assert r.get("Bad/Name") is None
        assert r.get("@coder") is None

    def test_get_case_insensitive_for_named(self, fake_root):
        r = AgentRegistry()
        # The on-disk dir is lowercase; user-typed "Coder" should canonicalise.
        p = r.get("Coder")
        assert p is not None
        assert p.name == "coder"

    def test_names(self, fake_root):
        r = AgentRegistry()
        names = r.names()
        assert names[0] == DEFAULT_AGENT_NAME
        assert "coder" in names


class TestAgentRegistryRefresh:
    def test_refresh_picks_up_new_profile(self, fake_root):
        r = AgentRegistry()
        assert r.get("newbie") is None
        (fake_root / "profiles" / "newbie").mkdir()
        # Without refresh, cached list does not see the new profile
        assert r.get("newbie") is None
        r.refresh()
        assert r.get("newbie") is not None

    def test_refresh_picks_up_deletion(self, fake_root):
        r = AgentRegistry()
        assert r.get("coder") is not None
        # Simulate deletion
        import shutil

        shutil.rmtree(fake_root / "profiles" / "coder")
        # Without refresh, still cached
        assert r.get("coder") is not None
        r.refresh()
        assert r.get("coder") is None


class TestDisplayNameAndDescription:
    def test_display_name_falls_back_to_id(self, fake_root):
        r = AgentRegistry()
        coder = r.get("coder")
        assert coder.display_name == "coder"

    def test_display_name_from_gateway_config(self, fake_root):
        (fake_root / "profiles" / "coder" / "config.yaml").write_text(
            "gateway:\n  agent_display_name: Coder Bot\n",
            encoding="utf-8",
        )
        r = AgentRegistry()
        coder = r.get("coder")
        assert coder.display_name == "Coder Bot"

    def test_display_name_from_branding_fallback(self, fake_root):
        (fake_root / "profiles" / "coder" / "config.yaml").write_text(
            "display:\n  branding:\n    agent_name: Coder Bot\n",
            encoding="utf-8",
        )
        r = AgentRegistry()
        coder = r.get("coder")
        assert coder.display_name == "Coder Bot"

    def test_description_from_soul(self, fake_root):
        (fake_root / "profiles" / "coder" / "SOUL.md").write_text(
            "# Coder\n\nI write Python.\n", encoding="utf-8"
        )
        r = AgentRegistry()
        coder = r.get("coder")
        assert coder.description == "I write Python."

    def test_description_empty_when_no_soul(self, fake_root):
        r = AgentRegistry()
        assert r.get("coder").description == ""

    def test_bad_yaml_does_not_crash(self, fake_root):
        (fake_root / "profiles" / "coder" / "config.yaml").write_text(
            ": : : not valid yaml at all : :", encoding="utf-8"
        )
        r = AgentRegistry()
        coder = r.get("coder")
        assert coder is not None
        assert coder.display_name == "coder"  # falls back


class TestProfileFiltering:
    def test_ignores_non_directory_entries(self, fake_root):
        (fake_root / "profiles" / "notes.txt").write_text("ignore me")
        r = AgentRegistry()
        assert r.get("notes.txt") is None

    def test_ignores_invalid_profile_names(self, fake_root):
        # Profile names must match [a-z0-9][a-z0-9_-]{0,63}.
        (fake_root / "profiles" / "Bad-Caps").mkdir()
        (fake_root / "profiles" / "1starts-numeric").mkdir()  # valid
        (fake_root / "profiles" / "_starts-underscore").mkdir()  # invalid
        r = AgentRegistry()
        names = r.names()
        assert "1starts-numeric" in names
        assert "Bad-Caps" not in names
        assert "_starts-underscore" not in names


class TestDefaultRegistry:
    def test_singleton_identity(self, fake_root):
        r1 = default_registry()
        r2 = default_registry()
        assert r1 is r2

    def test_reset_creates_new_instance(self, fake_root):
        r1 = default_registry()
        reset_default_registry()
        r2 = default_registry()
        assert r1 is not r2


class TestAgentProfile:
    def test_to_dict(self):
        p = AgentProfile(
            name="x", home=Path("/tmp/x"), display_name="X", description="y", is_default=False
        )
        assert p.to_dict() == {
            "name": "x",
            "home": str(Path("/tmp/x")),
            "display_name": "X",
            "description": "y",
            "is_default": False,
        }

    def test_frozen(self):
        p = AgentProfile(name="x", home=Path("/tmp/x"), display_name="X")
        with pytest.raises(Exception):  # FrozenInstanceError
            p.name = "y"  # type: ignore[misc]
