"""Tests for auto-compact mode for local providers (issue #43028).

When a local model server (Ollama, localhost) is detected, the CLI should
automatically enable compact mode and suppress tool progress to prevent the
spinner rendering loop from starving slow local inference.
"""

import pytest


# ---------------------------------------------------------------------------
# _is_local_provider tests
# ---------------------------------------------------------------------------

class TestIsLocalProvider:
    """Test the _is_local_provider helper function."""

    @pytest.fixture(autouse=True)
    def _import_helper(self):
        from cli import _is_local_provider
        self.is_local = _is_local_provider

    def test_ollama_provider(self):
        assert self.is_local("ollama", None) is True

    def test_ollama_case_insensitive(self):
        assert self.is_local("Ollama", None) is True
        assert self.is_local("OLLAMA", None) is True

    def test_llama_provider(self):
        assert self.is_local("llama", None) is True

    def test_llamacpp_provider(self):
        assert self.is_local("llamacpp", None) is True

    def test_llama_cpp_provider(self):
        assert self.is_local("llama-cpp", None) is True

    def test_local_provider(self):
        assert self.is_local("local", None) is True

    def test_localhost_base_url(self):
        assert self.is_local("custom", "http://localhost:11434/v1") is True

    def test_127_base_url(self):
        assert self.is_local("custom", "http://127.0.0.1:8080/v1") is True

    def test_ipv6_loopback(self):
        assert self.is_local("custom", "http://[::1]:8080/v1") is True

    def test_all_interfaces_base_url(self):
        assert self.is_local("custom", "http://0.0.0.0:11434/v1") is True

    def test_mdns_hostname(self):
        assert self.is_local("custom", "http://myserver.local:8080/v1") is True

    def test_remote_provider_remote_url(self):
        assert self.is_local("openrouter", "https://openrouter.ai/api/v1") is False

    def test_custom_provider_remote_url(self):
        assert self.is_local("custom", "https://api.example.com/v1") is False

    def test_none_provider_none_url(self):
        assert self.is_local(None, None) is False

    def test_empty_provider_none_url(self):
        assert self.is_local("", None) is False

    def test_anthropic_provider(self):
        assert self.is_local("anthropic", "https://api.anthropic.com") is False

    def test_openai_provider(self):
        assert self.is_local("openai", "https://api.openai.com/v1") is False


# ---------------------------------------------------------------------------
# Auto-compact logic tests (unit-level, no HermesCLI init)
# ---------------------------------------------------------------------------

class TestAutoCompactLogic:
    """Test the auto-compact decision logic extracted from HermesCLI.__init__."""

    @pytest.fixture(autouse=True)
    def _import_helpers(self):
        from cli import _is_local_provider
        self.is_local = _is_local_provider

    def _should_auto_compact(self, compact, auto_compact_local, provider, base_url):
        """Reproduce the 4-line logic from HermesCLI.__init__."""
        if not compact:
            if auto_compact_local and self.is_local(provider, base_url):
                return True
        return False

    def test_local_provider_triggers_auto_compact(self):
        assert self._should_auto_compact(
            compact=False, auto_compact_local=True,
            provider="ollama", base_url=None
        ) is True

    def test_remote_provider_no_auto_compact(self):
        assert self._should_auto_compact(
            compact=False, auto_compact_local=True,
            provider="openrouter", base_url="https://openrouter.ai/api/v1"
        ) is False

    def test_config_disabled_no_auto_compact(self):
        assert self._should_auto_compact(
            compact=False, auto_compact_local=False,
            provider="ollama", base_url=None
        ) is False

    def test_already_compact_no_change(self):
        # When compact is already True, the gate `if not self.compact` prevents
        # the auto-compact branch from running (compact stays True).
        assert self._should_auto_compact(
            compact=True, auto_compact_local=True,
            provider="ollama", base_url=None
        ) is False  # returns False because the gate blocks


# ---------------------------------------------------------------------------
# Config key tests
# ---------------------------------------------------------------------------

class TestAutoCompactLocalConfig:
    """Test that the config key exists with correct default."""

    def test_default_value(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["display"]["auto_compact_local"] is True

    def test_config_key_in_dump_source(self):
        """Verify the key is listed in dump.py's interesting_paths."""
        import pathlib
        dump_src = pathlib.Path("hermes_cli/dump.py").read_text()
        assert '("display", "auto_compact_local")' in dump_src
