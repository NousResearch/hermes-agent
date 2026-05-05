"""Tests for _get_configured_providers() and _set_configured_providers()."""
import pytest
from hermes_cli.memory_setup import _get_configured_providers, _set_configured_providers


# ── _get_configured_providers ────────────────────────────────────────────

class TestGetConfiguredProviders:
    def test_providers_list(self):
        """New list format is returned directly."""
        config = {"memory": {"providers": ["a", "b"]}}
        assert _get_configured_providers(config) == ["a", "b"]

    def test_legacy_single_provider(self):
        """Old single-provider string is wrapped in a list."""
        config = {"memory": {"provider": "a"}}
        assert _get_configured_providers(config) == ["a"]

    def test_list_wins_over_legacy(self):
        """When both providers and provider exist, the list takes precedence."""
        config = {"memory": {"providers": ["a", "b"], "provider": "c"}}
        assert _get_configured_providers(config) == ["a", "b"]

    def test_empty_config(self):
        """Empty top-level config returns empty list."""
        assert _get_configured_providers({}) == []

    def test_empty_memory_section(self):
        """Empty memory section returns empty list."""
        assert _get_configured_providers({"memory": {}}) == []

    def test_legacy_empty_string(self):
        """Legacy provider='' (empty string) should return empty list."""
        config = {"memory": {"provider": ""}}
        assert _get_configured_providers(config) == []

    def test_providers_list_with_blanks_filtered(self):
        """Blank strings in the providers list are filtered out."""
        config = {"memory": {"providers": ["a", "", "b"]}}
        assert _get_configured_providers(config) == ["a", "b"]


# ── _set_configured_providers ────────────────────────────────────────────

class TestSetConfiguredProviders:
    def test_set_multiple(self):
        """Multiple providers written to both new and legacy keys."""
        config = {}
        _set_configured_providers(config, ["a", "b"])
        assert config["memory"]["providers"] == ["a", "b"]
        assert config["memory"]["provider"] == "a"

    def test_set_empty(self):
        """Empty list clears both keys."""
        config = {}
        _set_configured_providers(config, [])
        assert config["memory"]["providers"] == []
        assert config["memory"]["provider"] == ""

    def test_set_single(self):
        """Single provider appears in both keys identically."""
        config = {}
        _set_configured_providers(config, ["x"])
        assert config["memory"]["providers"] == ["x"]
        assert config["memory"]["provider"] == "x"

    def test_creates_memory_key_if_missing(self):
        """If config has no 'memory' key, it is created."""
        config = {}
        _set_configured_providers(config, ["a"])
        assert "memory" in config
        assert config["memory"]["providers"] == ["a"]
        assert config["memory"]["provider"] == "a"

    def test_overwrites_existing_memory(self):
        """Existing memory values are overwritten, not merged."""
        config = {"memory": {"providers": ["old"], "provider": "old"}}
        _set_configured_providers(config, ["new1", "new2"])
        assert config["memory"]["providers"] == ["new1", "new2"]
        assert config["memory"]["provider"] == "new1"

    def test_does_not_mutate_input_list(self):
        """The caller's list is copied, not stored by reference."""
        names = ["a", "b"]
        config = {}
        _set_configured_providers(config, names)
        names.append("c")
        assert config["memory"]["providers"] == ["a", "b"]


# ── cmd_setup add-vs-replace integration ─────────────────────────────────

class _DummyProvider:
    """Minimal MemoryProvider stub with no post_setup hook."""
    @property
    def name(self): return "dummy"

    def is_available(self): return True

    def get_config_schema(self): return []

    def initialize(self, session_id, **kwargs): pass


class TestCmdSetupAddVsReplace:
    """When existing active providers exist, the wizard asks add-vs-replace."""

    def _mock_providers(self, monkeypatch, *provider_tuples, config_providers=None):
        """Mock the provider discovery, config loading, and dependencies.

        provider_tuples: (name, desc, provider_instance) as from _get_available_providers.
        config_providers: list of already-active provider names.
        """
        monkeypatch.setattr(
            "hermes_cli.memory_setup._get_available_providers",
            lambda: list(provider_tuples)
        )
        monkeypatch.setattr(
            "hermes_cli.memory_setup._install_dependencies",
            lambda name: None
        )

        mem = {}
        if config_providers:
            mem.update({"providers": list(config_providers),
                        "provider": config_providers[0]})
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": mem}
        )

        # Capture what gets saved
        saved_config = {}
        def fake_save(cfg):
            saved_config["memory"] = cfg.get("memory", {})
        monkeypatch.setattr("hermes_cli.config.save_config", fake_save)

        # Silence input() for schema prompts (press Enter past all)
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "")

        return saved_config

    def test_first_provider_no_prompt(self, tmp_path, monkeypatch):
        """No active providers → selected provider configured without asking."""
        monkeypatch.setattr("hermes_cli.memory_setup.get_hermes_home",
                            lambda: tmp_path)
        saved = self._mock_providers(
            monkeypatch,
            ("x", "local", _DummyProvider()),
            config_providers=None
        )

        # Pick provider "x" (index 0) — no add-vs-replace needed
        monkeypatch.setattr(
            "hermes_cli.memory_setup._curses_select",
            lambda *args, **kwargs: 0
        )

        from hermes_cli.memory_setup import cmd_setup
        cmd_setup(None)

        assert saved.get("memory", {}).get("providers") == ["x"]
        assert saved.get("memory", {}).get("provider") == "x"

    def test_add_alongside_existing(self, tmp_path, monkeypatch):
        """Active providers ['a','b'] + pick 'Add c alongside' → ['a','b','c']."""
        monkeypatch.setattr("hermes_cli.memory_setup.get_hermes_home",
                            lambda: tmp_path)
        saved = self._mock_providers(
            monkeypatch,
            ("c", "local", _DummyProvider()),
            config_providers=["a", "b"]
        )

        # Two _curses_select calls:
        #   1st = provider picker (pick "c", index 0)
        #   2nd = add-vs-replace (pick "Add c alongside", index 0)
        selections = iter([0, 0])
        monkeypatch.setattr(
            "hermes_cli.memory_setup._curses_select",
            lambda *args, **kwargs: next(selections)
        )

        from hermes_cli.memory_setup import cmd_setup
        cmd_setup(None)

        assert saved.get("memory", {}).get("providers") == ["a", "b", "c"]
        assert saved.get("memory", {}).get("provider") == "a"

    def test_replace_existing(self, tmp_path, monkeypatch):
        """Active providers ['a','b'] + pick 'Replace all' → ['c'] only."""
        monkeypatch.setattr("hermes_cli.memory_setup.get_hermes_home",
                            lambda: tmp_path)
        saved = self._mock_providers(
            monkeypatch,
            ("c", "local", _DummyProvider()),
            config_providers=["a", "b"]
        )

        # Two _curses_select calls:
        #   1st = provider picker (pick "c", index 0)
        #   2nd = add-vs-replace (pick "Replace all", index 1)
        selections = iter([0, 1])
        monkeypatch.setattr(
            "hermes_cli.memory_setup._curses_select",
            lambda *args, **kwargs: next(selections)
        )

        from hermes_cli.memory_setup import cmd_setup
        cmd_setup(None)

        assert saved.get("memory", {}).get("providers") == ["c"]
        assert saved.get("memory", {}).get("provider") == "c"

    def test_reselect_same_provider_no_duplicate(self, tmp_path, monkeypatch):
        """Picking a provider that's already active doesn't duplicate or prompt."""
        monkeypatch.setattr("hermes_cli.memory_setup.get_hermes_home",
                            lambda: tmp_path)
        saved = self._mock_providers(
            monkeypatch,
            ("a", "local", _DummyProvider()),
            ("b", "local", _DummyProvider()),
            config_providers=["a"]
        )

        # Pick "b" (index 1) — a is already active, b is new
        # Then add-vs-replace: "Add b alongside" (index 0)
        selections = iter([1, 0])
        monkeypatch.setattr(
            "hermes_cli.memory_setup._curses_select",
            lambda *args, **kwargs: next(selections)
        )

        from hermes_cli.memory_setup import cmd_setup
        cmd_setup(None)

        assert saved.get("memory", {}).get("providers") == ["a", "b"]
        # "a" appears once — not duplicated
