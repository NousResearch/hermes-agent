"""Tests for the bundled Recall memory provider plugin."""

from __future__ import annotations

import json
from pathlib import Path

from plugins.memory import discover_memory_providers, load_memory_provider
from plugins.memory.recall import RecallMemoryProvider


def test_recall_is_discovered_as_available_bundled_provider():
    providers = {name: (description, available) for name, description, available in discover_memory_providers()}

    assert "recall" in providers
    description, available = providers["recall"]
    assert "searchable memory archive" in description.lower()
    assert available is True


def test_recall_initializes_local_sqlite_archive_and_reports_build(tmp_path):
    provider = RecallMemoryProvider({"db_path": "$HERMES_HOME/recall-test.sqlite"})
    provider.initialize("session-1", hermes_home=tmp_path, cwd="/workspace/project")
    try:
        build_info = json.loads(provider.handle_tool_call("memory_recall_build_info", {}))
        stats = json.loads(provider.handle_tool_call("memory_archive_stats", {}))

        assert build_info["name"] == "recall"
        assert build_info["version"] == "0.3.7"
        assert Path(build_info["db_path"]) == tmp_path / "recall-test.sqlite"
        assert "hash-chain-audit" in build_info["capabilities"]
        assert stats["observations_by_status"] == {}
        assert stats["episode_count"] == 0
    finally:
        provider.shutdown()


def test_recall_captures_turns_as_lower_trust_archive_rows(tmp_path):
    provider = RecallMemoryProvider({"db_path": "$HERMES_HOME/recall.sqlite"})
    provider.initialize("session-1", hermes_home=tmp_path, cwd="/workspace/project")
    try:
        provider.sync_turn(
            "Remember the unusual marker alpha_beta_12345 for this test.",
            "Noted in the archive.",
            session_id="session-1",
        )

        search = json.loads(
            provider.handle_tool_call(
                "memory_archive_search",
                {"query": "alpha_beta_12345", "limit": 5},
            )
        )
        assert len(search["results"]) == 1
        row = search["results"][0]
        assert row["trust_level"] == "archive"
        assert row["confidence"] == 0.35
    finally:
        provider.shutdown()


def test_recall_save_config_persists_under_plugins_namespace(monkeypatch, tmp_path):
    saved = {}
    config = {"memory": {"provider": "recall"}}

    def fake_save_config(new_config):
        saved.update(new_config)

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config.copy())
    monkeypatch.setattr("hermes_cli.config.save_config", fake_save_config)

    provider = RecallMemoryProvider()
    provider.save_config(
        {
            "db_path": "$HERMES_HOME/recall.sqlite",
            "auto_capture": "true",
            "prefetch_enabled": "false",
            "max_prefetch_results": "2",
            "audit_enabled": "true",
        },
        str(tmp_path),
    )

    assert saved["memory"]["provider"] == "recall"
    assert saved["plugins"]["recall"] == {
        "db_path": "$HERMES_HOME/recall.sqlite",
        "auto_capture": "true",
        "prefetch_enabled": "false",
        "max_prefetch_results": "2",
        "audit_enabled": "true",
    }


def test_load_memory_provider_returns_recall_instance():
    provider = load_memory_provider("recall")

    assert isinstance(provider, RecallMemoryProvider)
    schema_names = {schema["name"] for schema in provider.get_tool_schemas()}
    assert "memory_archive_search" in schema_names
    assert "memory_promote_candidate" in schema_names
