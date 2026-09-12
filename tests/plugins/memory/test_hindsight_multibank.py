"""Tests for per-project and multi-bank Hindsight routing (acme/lead/shared
generic fixtures — no real projects, orgs, agents, or user paths).

Covers: .hindsight/config.toml walk-up discovery, the full bank precedence
(CLI flag → TOML → template → static), the optional write-set fan-out
(mirror_to_own_bank + additional_banks), and multi-bank recall merge.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.memory.hindsight import HindsightMemoryProvider, _discover_cwd_bank_id


def _make_mock_client():
    """Minimal mock Hindsight client (same shape as the sibling test file)."""
    client = SimpleNamespace()
    async def _arecall(**kwargs):
        return SimpleNamespace(results=[])
    client.arecall = AsyncMock(side_effect=_arecall)
    return client


def _make_provider(tmp_path, monkeypatch, **config_overrides):
    """Canonical construction (mirrors test_hindsight_provider.provider):
    a real config file on disk under the patched HERMES_HOME, then
    initialize(). config_overrides merge into the base config."""
    config = {
        "mode": "cloud",
        "apiKey": "test-key",
        "api_url": "http://localhost:9999",
        "bank_id": "lead",
        "bank_id_template": "{workspace}",
        "budget": "mid",
        "memory_mode": "hybrid",
    }
    config.update(config_overrides)
    config_path = tmp_path / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)
    p = HindsightMemoryProvider()
    p.initialize(session_id="s1", hermes_home=str(tmp_path), platform="cli",
                 agent_identity="lead", agent_workspace="acme")
    p._client = _make_mock_client()
    return p


@pytest.fixture()
def toml_project(tmp_path):
    """A fake project tree: /tmp/xx/acme/.hindsight/config.toml -> bank_id acme."""
    proj = tmp_path / "acme"
    (proj / ".hindsight").mkdir(parents=True)
    (proj / ".hindsight" / "config.toml").write_text('bank_id = "acme"\n')
    return proj


class TestCwdBankDiscovery:
    def test_discovers_closest_toml(self, toml_project):
        assert _discover_cwd_bank_id(start_dir=str(toml_project)) == "acme"

    def test_walks_up_to_parent(self, toml_project):
        nested = toml_project / "src" / "views"
        nested.mkdir(parents=True)
        assert _discover_cwd_bank_id(start_dir=str(nested)) == "acme"

    def test_absent_toml_returns_none(self, tmp_path):
        assert _discover_cwd_bank_id(start_dir=str(tmp_path)) is None

    def test_malformed_toml_skipped_not_fatal(self, tmp_path):
        proj = tmp_path / "acme"
        (proj / ".hindsight").mkdir(parents=True)
        (proj / ".hindsight" / "config.toml").write_text("not [valid toml")
        # Invalid file is skipped; walk continues up and finds nothing.
        assert _discover_cwd_bank_id(start_dir=str(proj)) is None


class TestBankPrecedence:
    def test_cli_flag_beats_toml(self, tmp_path, monkeypatch, toml_project):
        from plugins.memory.hindsight import _HINDSIGHT_BANK_OVERRIDE_ENV
        monkeypatch.setenv(_HINDSIGHT_BANK_OVERRIDE_ENV, "override-bank")
        # TOML would resolve to acme if it were consulted — the flag must win.
        monkeypatch.setattr("plugins.memory.hindsight._discover_cwd_bank_id",
                            lambda *a, **k: "acme")
        p = _make_provider(tmp_path, monkeypatch)
        assert p._bank_id == "override-bank"

    def test_toml_beats_template(self, tmp_path, monkeypatch):
        monkeypatch.setattr("plugins.memory.hindsight._discover_cwd_bank_id",
                            lambda *a, **k: "acme")
        p = _make_provider(tmp_path, monkeypatch, bank_id_template="{workspace}")
        assert p._bank_id == "acme"

    def test_template_renders_workspace(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch)
        assert p._bank_id == "acme"  # agent_workspace="acme" threaded via initialize

    def test_template_empty_workspace_falls_back_to_static(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch)
        # Simulate "outside any project": clear the workspace and re-resolve.
        p._agent_workspace = ""
        p._apply_connection_settings(p._config)
        assert p._bank_id == "lead"


class TestWriteSet:
    def test_no_multi_bank_config_is_single(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch)
        assert p._write_bank_ids == ["acme"]  # primary only; no mirror

    def test_mirror_appends_own_bank(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch,
                           mirror_to_own_bank=True)
        assert p._write_bank_ids == ["acme", "lead"]

    def test_additional_banks_appended_in_order(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch,
                           mirror_to_own_bank=True,
                           additional_banks=["shared", "team"])
        assert p._write_bank_ids == ["acme", "lead", "shared", "team"]

    def test_additional_banks_deduplicated(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch,
                           mirror_to_own_bank=True,
                           additional_banks=["acme", "shared", "acme"])
        assert p._write_bank_ids == ["acme", "lead", "shared"]


class TestMultiBankRecall:
    def test_recall_queries_all_banks_primary_first(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch,
                           mirror_to_own_bank=True,
                           additional_banks=["shared"])
        calls = []

        async def _recall(**kwargs):
            calls.append(kwargs["bank_id"])
            text = "acme-mem" if kwargs["bank_id"] == "acme" else "shared-mem"
            return SimpleNamespace(results=[SimpleNamespace(text=text)])

        p._client.arecall = AsyncMock(side_effect=_recall)
        results = p._recall("q")
        assert calls == ["acme", "lead", "shared"]
        assert [getattr(r, "text", None) for r in results] == ["acme-mem", "shared-mem"]

    def test_recall_dedupes_across_banks(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch, mirror_to_own_bank=True)

        async def _recall(**kwargs):
            return SimpleNamespace(results=[SimpleNamespace(text="same")])

        p._client.arecall = AsyncMock(side_effect=_recall)
        results = p._recall("q")
        assert len([getattr(r, "text", None) for r in results]) == 1

    def test_recall_single_bank_when_no_multi_config(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch)
        calls = []

        async def _recall(**kwargs):
            calls.append(kwargs["bank_id"])
            return SimpleNamespace(results=[])

        p._client.arecall = AsyncMock(side_effect=_recall)
        p._recall("q")
        assert calls == ["acme"]