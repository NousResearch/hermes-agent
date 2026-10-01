"""`hermes memory status` must not call a present-but-unloadable provider absent.

Regression coverage for NousResearch/hermes-agent#130072: every loader failure
(import, ``register()``, subclass instantiation) collapses to ``None``, and
status used to print "NOT installed" plus an install hint pointing at the very
directory the plugin already lives in — while the real load error stayed
invisible at DEBUG level.
"""

import logging
from pathlib import Path

import hermes_cli.memory_setup as memory_setup
import plugins.memory as memory_plugins


def _configure(monkeypatch, provider):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"provider": provider}},
    )


def test_present_but_failing_provider_reports_load_failed(monkeypatch, capsys):
    _configure(monkeypatch, "mnemosyne")
    # Discovery silently drops the provider (its load returns None).
    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [])
    monkeypatch.setattr(
        memory_plugins,
        "find_provider_dir",
        lambda name: Path("/home/u/.hermes/plugins/mnemosyne"),
    )
    monkeypatch.setattr(memory_plugins, "find_provider_entry_point", lambda name: None)

    seen_register_skills = []

    def fake_load(name, *, register_skills=None):
        seen_register_skills.append(register_skills)
        # The reason real loads fail only ever logs at DEBUG.
        logging.getLogger("plugins.plugin_loader").debug(
            "Failed to exec_module _hermes_user_memory.mnemosyne: "
            "RuntimeError: runtime Python 3.14; selected Mnemosyne environment Python 3.11"
        )
        return None

    monkeypatch.setattr(memory_plugins, "load_memory_provider", fake_load)

    memory_setup.cmd_status(object())
    out = capsys.readouterr().out

    assert "LOAD FAILED" in out
    assert "NOT installed" not in out
    assert "Install the 'mnemosyne'" not in out  # wrong-direction hint is gone
    assert "runtime Python 3.14" in out  # the swallowed reason is surfaced
    assert seen_register_skills == [False]  # diagnostic re-run: no skill side effects


def test_absent_provider_keeps_not_installed_hint(monkeypatch, capsys):
    _configure(monkeypatch, "ghost")
    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [])
    monkeypatch.setattr(memory_plugins, "find_provider_dir", lambda name: None)
    monkeypatch.setattr(memory_plugins, "find_provider_entry_point", lambda name: None)
    monkeypatch.setattr(memory_plugins, "load_memory_provider", lambda name, **kw: None)

    memory_setup.cmd_status(object())
    out = capsys.readouterr().out

    assert "NOT installed" in out
    assert "Install the 'ghost' memory plugin" in out
    assert "LOAD FAILED" not in out


def test_diagnostics_restores_logger_levels(monkeypatch):
    _loader = logging.getLogger("plugins.plugin_loader")
    _memory = logging.getLogger("plugins.memory")
    monkeypatch.setattr(memory_plugins, "load_memory_provider", lambda name, **kw: None)
    saved = (_loader.level, _memory.level)

    memory_setup._provider_load_diagnostics("mnemosyne")

    assert (_loader.level, _memory.level) == saved
