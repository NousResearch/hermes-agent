from types import SimpleNamespace
from unittest.mock import MagicMock

import hermes_cli.memory_setup as memory_setup
from hermes_cli.memory_setup import _CANCELLED, _curses_select








def test_cmd_setup_generic_choice_cancel_writes_nothing(tmp_path, monkeypatch):
    class ChoiceProvider:
        def __init__(self):
            self.save_config = MagicMock()

        def get_config_schema(self):
            return [{
                "key": "mode",
                "description": "Mode",
                "default": "one",
                "choices": ["one", "two"],
            }]

    provider = ChoiceProvider()
    selections = iter([0, _CANCELLED])
    save_config = MagicMock()
    install_dependencies = MagicMock()

    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [("fake", "local", provider)])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: next(selections))
    monkeypatch.setattr(memory_setup, "_install_dependencies", install_dependencies)
    monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {}})
    monkeypatch.setattr("hermes_cli.config.save_config", save_config)

    memory_setup.cmd_setup(SimpleNamespace())

    install_dependencies.assert_called_once_with("fake")
    save_config.assert_not_called()
    provider.save_config.assert_not_called()
    assert not (tmp_path / ".env").exists()


# _write_env_vars's CR/LF-stripping, denylist, and plain-value-roundtrip
# behavior is covered by tests/hermes_cli/test_memory_setup_env_denylist.py,
# which exercises the current save_env_value-routed signature
# (env_writes, hermes_home=None) \u2014 these three tests pinned the prior direct
# Path.write_text(env_path, env_writes) signature/implementation and were
# removed along with it (#60587).


# ---------------------------------------------------------------------------
# _provider_pip_dependencies — mode-aware dep expansion (#70636)
# ---------------------------------------------------------------------------





def test_install_dependencies_force_reinstalls_versioned_specs(tmp_path, monkeypatch):
    """force=True hands every declared spec (version ranges intact) to pip,
    so a downgraded/stripped bridge package is restored on hermes update."""
    import yaml as _yaml

    plugin_dir = tmp_path / "mem0"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text(
        _yaml.safe_dump({"pip_dependencies": ["mem0ai>=2.0.10,<3"]}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "plugins.memory.find_provider_dir", lambda name: plugin_dir
    )

    installed = []

    def fake_install_specs(specs, timeout=120):
        installed.append(list(specs))
        return SimpleNamespace(ok=True, blocked=False, reason="", stderr="")

    monkeypatch.setattr("tools.lazy_deps.install_specs", fake_install_specs)

    memory_setup._install_dependencies("mem0", force=True)

    assert installed, "force=True must reach the install step"
    assert any("mem0ai>=2.0.10,<3" in specs for specs in installed)


def test_cmd_status_memory_tool_gate_disabled(capsys, monkeypatch):
    """When both memory stores are disabled, Memory status reports memory tool as disabled."""
    _cfg = {"memory": {"memory_enabled": False, "user_profile_enabled": False}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _cfg)
    # check_memory_requirements() reads the readonly loader, not load_config.
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: _cfg, raising=False
    )
    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [])

    memory_setup.cmd_status(SimpleNamespace())

    captured = capsys.readouterr().out
    assert "Memory tool:        disabled ✗" in captured
    assert "Memory injection:   disabled ✗" in captured
    assert "User profile:       disabled ✗" in captured


def test_cmd_status_memory_tool_gate_enabled(capsys, monkeypatch):
    """When at least one memory store is enabled, Memory status reports memory tool as enabled."""
    _cfg = {"memory": {"memory_enabled": True, "user_profile_enabled": False}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _cfg)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: _cfg, raising=False
    )
    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [])

    memory_setup.cmd_status(SimpleNamespace())

    captured = capsys.readouterr().out
    assert "Memory tool:        enabled ✓" in captured
    assert "Memory injection:   enabled ✓" in captured
    assert "User profile:       disabled ✗" in captured


def _config_less_provider():
    """A provider that stores nothing itself: its non-secret answers live under ``memory.<name>``."""
    from types import SimpleNamespace

    return SimpleNamespace(get_config_schema=lambda: [
        {"key": "project", "description": "Project identifier"},
        {"key": "api_key", "description": "API key", "secret": True, "env_var": "FAKE_API_KEY"},
    ])


def _run_generic_setup(monkeypatch, tmp_path, run):
    """Drive one setup flow against a provider without save_config; return the saved config.yaml dict."""
    from unittest.mock import MagicMock

    provider = _config_less_provider()
    saved = {}
    prompts = iter(["proj-1", "sk-secret"])
    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [("fake", "local", provider)])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: 0)
    monkeypatch.setattr(memory_setup, "_prompt", lambda *args, **kwargs: next(prompts))
    monkeypatch.setattr(memory_setup, "_install_dependencies", MagicMock())
    monkeypatch.setattr(memory_setup, "_write_env_vars", lambda writes, *a, **k: saved.setdefault("env", dict(writes)))
    monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {}})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))
    run()
    return saved


def test_cmd_setup_persists_non_secret_answers_for_a_provider_without_save_config(tmp_path, monkeypatch):
    """First-time setup: the ``memory.<name>`` block did not exist yet, so the answers went into a dict
    nothing referenced — config.yaml was saved without them while the summary said "Provider config saved"."""
    from types import SimpleNamespace

    saved = _run_generic_setup(monkeypatch, tmp_path, lambda: memory_setup.cmd_setup(SimpleNamespace()))

    assert saved["config"]["memory"]["provider"] == "fake"
    assert saved["config"]["memory"]["fake"] == {"project": "proj-1"}, "the non-secret answer must reach config.yaml"
    assert saved["env"] == {"FAKE_API_KEY": "sk-secret"}, "the secret still goes to .env, never config.yaml"


def test_cmd_setup_provider_runs_the_same_schema_prompts(tmp_path, monkeypatch):
    """``hermes memory setup <name>`` used to activate without asking a single schema question."""
    saved = _run_generic_setup(monkeypatch, tmp_path, lambda: memory_setup.cmd_setup_provider("fake"))

    assert saved["config"]["memory"]["provider"] == "fake"
    assert saved["config"]["memory"]["fake"] == {"project": "proj-1"}
    assert saved["env"] == {"FAKE_API_KEY": "sk-secret"}
