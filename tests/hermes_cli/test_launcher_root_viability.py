"""A service definition must never name a code root that cannot resolve its dependencies.

Regression for the 2026-09-26 outage: the launchd plist was regenerated from a pinned environment copy
(`<installs>/<key>/environments/<env>/workspace`), whose install key owns no state under
`~/.hermes/installs`, so every launch died with "no dependency environment is committed for this
install" (exit 1) and launchd respawned it forever.
"""

from __future__ import annotations

from pathlib import Path

from hermes_cli import gateway_launchd


def _patch_resolvers(monkeypatch, *, store_python, committed, payload=None):
    # Inline imports inside the predicate read the DEFINING modules at call time, so patch there.
    import hermes_cli._launchers as launchers
    import pm.environments as environments

    monkeypatch.setattr(launchers, "resolve_store_python", lambda root: store_python)
    monkeypatch.setattr(environments, "committed_venv", lambda root: committed)
    monkeypatch.setattr(environments, "payload_venv", lambda root: payload)


def test_install_root_with_committed_generation_is_viable(tmp_path, monkeypatch):
    _patch_resolvers(monkeypatch, store_python=tmp_path / "bin/python3", committed=tmp_path / "venv")
    assert gateway_launchd.launcher_root_is_viable(tmp_path) is True


def test_pinned_environment_workspace_is_not_viable(tmp_path, monkeypatch):
    """The exact outage shape: store Python, no committed generation, no payload venv."""
    _patch_resolvers(monkeypatch, store_python=tmp_path / "bin/python3", committed=None, payload=None)
    assert gateway_launchd.launcher_root_is_viable(tmp_path) is False


def test_externally_owned_runtime_is_viable(tmp_path, monkeypatch):
    """Nix / developer venv: no store Python, so bootstrap keeps the interpreter's own packages."""
    _patch_resolvers(monkeypatch, store_python=None, committed=None, payload=None)
    assert gateway_launchd.launcher_root_is_viable(tmp_path) is True


def test_payload_venv_is_viable(tmp_path, monkeypatch):
    """Sealed payload: payload_venv is enough even with no committed generation."""
    _patch_resolvers(monkeypatch, store_python=tmp_path / "bin/python3", committed=None, payload=tmp_path / "payload-venv")
    assert gateway_launchd.launcher_root_is_viable(tmp_path) is True


def test_assert_refuses_an_unviable_root(tmp_path, monkeypatch):
    _patch_resolvers(monkeypatch, store_python=tmp_path / "bin/python3", committed=None, payload=None)
    import pytest

    with pytest.raises(RuntimeError, match="cannot resolve its dependencies"):
        gateway_launchd.assert_launcher_root_is_viable(tmp_path)


def test_installed_launcher_root_is_recovered_from_a_definition():
    definition = (
        'do shell script "exec /Users/example/.hermes/hermes-agent/.hermes/bin/hermes '
        '--run-module hermes_cli.stderr_timestamp --error-log /tmp/e.log -- '
        '/Users/example/.hermes/hermes-agent/.hermes/bin/hermes gateway run --external-supervisor"'
    )
    assert gateway_launchd.installed_service_launcher_root(definition) == Path(
        "/Users/example/.hermes/hermes-agent"
    )


def test_installed_launcher_root_is_recovered_from_a_workspace_definition():
    definition = (
        'exec /Users/example/.hermes/installs/abc/environments/def/workspace/.hermes/bin/hermes '
        "gateway run"
    )
    assert gateway_launchd.installed_service_launcher_root(definition) == Path(
        "/Users/example/.hermes/installs/abc/environments/def/workspace"
    )


def test_installed_launcher_root_is_none_without_a_launcher():
    assert gateway_launchd.installed_service_launcher_root("<plist/>") is None