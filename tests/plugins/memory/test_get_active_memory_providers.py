"""Direct unit tests for get_active_memory_providers().

These test the actual config-reading logic by monkeypatching load_config
rather than monkeypatching get_active_memory_providers itself.
"""

import pytest
from plugins.memory import get_active_memory_providers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_load_config(monkeypatch, config_dict):
    """Monkeypatch hermes_cli.config.load_config to return *config_dict*."""
    import hermes_cli.config
    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: config_dict)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestGetActiveMemoryProviders:
    """Tests for get_active_memory_providers()."""

    def test_providers_list_returned(self, monkeypatch):
        """Case 1: memory.providers: ['honcho', 'mem0'] -> ['honcho', 'mem0']"""
        _patch_load_config(monkeypatch, {
            "memory": {"providers": ["honcho", "mem0"]}
        })
        assert get_active_memory_providers() == ["honcho", "mem0"]

    def test_legacy_fallback_when_providers_empty(self, monkeypatch):
        """Case 2: memory.providers: [] + memory.provider: 'honcho' -> ['honcho']"""
        _patch_load_config(monkeypatch, {
            "memory": {"providers": [], "provider": "honcho"}
        })
        assert get_active_memory_providers() == ["honcho"]

    def test_falsy_entries_filtered(self, monkeypatch):
        """Case 3: memory.providers: ['honcho', ''] -> ['honcho'] (empty strings filtered)"""
        _patch_load_config(monkeypatch, {
            "memory": {"providers": ["honcho", ""]}
        })
        assert get_active_memory_providers() == ["honcho"]

    def test_providers_list_wins_over_provider_string(self, monkeypatch):
        """Case 4: Both providers list and provider string set -> providers list wins"""
        _patch_load_config(monkeypatch, {
            "memory": {"providers": ["mem0"], "provider": "honcho"}
        })
        assert get_active_memory_providers() == ["mem0"]

    def test_neither_set_returns_empty(self, monkeypatch):
        """Case 5: Neither memory.providers nor memory.provider set -> []"""
        _patch_load_config(monkeypatch, {})
        assert get_active_memory_providers() == []

    def test_no_memory_key_returns_empty(self, monkeypatch):
        """Case 5b: Config has no 'memory' key at all -> []"""
        _patch_load_config(monkeypatch, {"other": "value"})
        assert get_active_memory_providers() == []

    def test_exception_in_load_config_returns_empty(self, monkeypatch):
        """Case 6: Exception in load_config -> []"""
        import hermes_cli.config
        monkeypatch.setattr(
            hermes_cli.config,
            "load_config",
            lambda: (_ for _ in ()).throw(RuntimeError("config broken")),
        )
        assert get_active_memory_providers() == []

    def test_legacy_single_provider_without_list(self, monkeypatch):
        """Legacy format only (no providers key) -> [provider]"""
        _patch_load_config(monkeypatch, {
            "memory": {"provider": "supermemory"}
        })
        assert get_active_memory_providers() == ["supermemory"]

    def test_legacy_empty_string_provider(self, monkeypatch):
        """Legacy format with empty string provider -> []"""
        _patch_load_config(monkeypatch, {
            "memory": {"provider": ""}
        })
        assert get_active_memory_providers() == []

    def test_providers_list_all_falsy(self, monkeypatch):
        """providers list with all falsy entries -> [] (empty list after filter, no legacy fallback)"""
        _patch_load_config(monkeypatch, {
            "memory": {"providers": ["", None], "provider": "honcho"}
        })
        # Note: providers list is truthy (non-empty), so it takes precedence.
        # After filtering falsy entries, result is [].
        # Legacy fallback does NOT happen because providers was non-empty.
        assert get_active_memory_providers() == []
