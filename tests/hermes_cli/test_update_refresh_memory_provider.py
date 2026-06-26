"""Regression tests for active memory-provider dependency refresh during ``hermes update``.

Issue #53272: ``hermes update`` rebuilds the venv and drops memory-provider
plugin ``pip_dependencies`` (e.g. ``mem0ai``) installed by ``hermes memory
setup``, but never reinstalls them — so the active provider's backend fails to
import after an update.

These tests cover ``_refresh_active_memory_provider``: the function the update
flow calls (alongside ``_refresh_active_lazy_features``) to reinstall the
*active* memory provider's missing deps. The config read and the install
subprocess are mocked; the routing logic under test is real.
"""

from hermes_cli import main as hermes_main


def test_refresh_calls_install_dependencies_when_provider_set(monkeypatch):
    """A configured ``memory.provider`` must trigger a reinstall of its deps."""
    calls = []

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"provider": "mem0"}},
    )
    monkeypatch.setattr(
        "hermes_cli.memory_setup._install_dependencies",
        lambda name: calls.append(name),
    )

    hermes_main._refresh_active_memory_provider()

    assert calls == ["mem0"]


def test_refresh_skips_when_no_provider_configured(monkeypatch):
    """An empty provider string must not trigger any install."""
    calls = []

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"provider": ""}},
    )
    monkeypatch.setattr(
        "hermes_cli.memory_setup._install_dependencies",
        lambda name: calls.append(name),
    )

    hermes_main._refresh_active_memory_provider()

    assert calls == []


def test_refresh_skips_when_memory_section_absent(monkeypatch):
    """A config without a ``memory`` section must not trigger any install."""
    calls = []

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.memory_setup._install_dependencies",
        lambda name: calls.append(name),
    )

    hermes_main._refresh_active_memory_provider()

    assert calls == []


def test_refresh_never_raises_on_install_failure(monkeypatch):
    """An exception from the installer must not escape into the update flow."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"provider": "mem0"}},
    )

    def boom(name):
        raise RuntimeError("install exploded")

    monkeypatch.setattr("hermes_cli.memory_setup._install_dependencies", boom)

    # Must not raise — update must complete even if the provider refresh fails.
    hermes_main._refresh_active_memory_provider()


def test_refresh_never_raises_on_config_read_failure(monkeypatch):
    """A broken config read must not escape into the update flow."""
    calls = []

    def bad_load_config():
        raise OSError("config file vanished")

    monkeypatch.setattr("hermes_cli.config.load_config", bad_load_config)
    monkeypatch.setattr(
        "hermes_cli.memory_setup._install_dependencies",
        lambda name: calls.append(name),
    )

    # Must not raise.
    hermes_main._refresh_active_memory_provider()

    assert calls == []
