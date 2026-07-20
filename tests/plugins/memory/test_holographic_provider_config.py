"""Behavior tests for Holographic memory provider configuration."""

from pathlib import Path

import yaml

from plugins.memory.holographic import HolographicMemoryProvider


def test_auto_extract_string_false_disables_session_end_extraction(tmp_path):
    provider = HolographicMemoryProvider(
        config={
            "db_path": str(tmp_path / "memory_store.db"),
            "auto_extract": "false",
        }
    )
    provider.initialize("session-1")

    try:
        provider.on_session_end(
            [{"role": "user", "content": "I prefer concise answers in every conversation."}]
        )

        store = provider._store
        assert store is not None
        assert store.list_facts() == []
    finally:
        provider.shutdown()


def test_auto_extract_string_true_enables_session_end_extraction(tmp_path):
    provider = HolographicMemoryProvider(
        config={
            "db_path": str(tmp_path / "memory_store.db"),
            "auto_extract": "true",
        }
    )
    provider.initialize("session-1")

    try:
        provider.on_session_end(
            [{"role": "user", "content": "I prefer concise answers in every conversation."}]
        )

        store = provider._store
        assert store is not None
        assert [fact["content"] for fact in store.list_facts()] == [
            "I prefer concise answers in every conversation."
        ]
    finally:
        provider.shutdown()


def test_plugin_declares_numpy_for_hrr_algebra():
    plugin_path = (
        Path(__file__).resolve().parents[3]
        / "plugins"
        / "memory"
        / "holographic"
        / "plugin.yaml"
    )
    metadata = yaml.safe_load(plugin_path.read_text(encoding="utf-8"))

    assert "numpy==2.4.3" in metadata.get("pip_dependencies", [])
