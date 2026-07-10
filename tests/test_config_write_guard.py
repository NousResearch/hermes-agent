"""Regression tests for the conftest real-config write tripwire.

The autouse ``_guard_real_config_writes`` fixture (tests/conftest.py) must
fail any test whose un-repointed ``cli.save_config_value()`` call would land
on the developer's live config.yaml — or on the repo-root ``cli-config.yaml``
fallback used when that home has no config.yaml — while leaving tests that
persist into their own tmp homes untouched.

Regression for the 2026-07-08 incident where five config.set tests run via
plain ``pytest tests/`` clobbered the live user config fleet-wide.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

cli = pytest.importorskip("cli")


def test_guard_trips_on_repo_root_fallback(
    tmp_path, monkeypatch, _guard_real_config_writes
):
    """A config-less home falls through to repo-root cli-config.yaml — forbidden."""
    repo_cfg = _guard_real_config_writes.repo_root_cli_config
    pre = repo_cfg.read_bytes() if repo_cfg.exists() else None

    empty_home = tmp_path / "empty_home"
    empty_home.mkdir()
    monkeypatch.setattr(cli, "_hermes_home", empty_home)

    with pytest.raises(pytest.fail.Exception, match="real Hermes config"):
        cli.save_config_value("model.default", "guard-proof")

    # The guard failed the call BEFORE the write landed.
    post = repo_cfg.read_bytes() if repo_cfg.exists() else None
    assert post == pre


def test_guard_trips_on_pre_isolation_user_config(
    monkeypatch, _guard_real_config_writes
):
    """Persisting from the pre-isolation home is forbidden in either shape.

    If that home has a config.yaml, the target IS the live user config; if it
    does not, the write falls through to the repo-root cli-config.yaml.  Both
    are forbidden, so the guard trips regardless of how pytest was invoked.
    """
    real_user_config = _guard_real_config_writes.real_user_config
    pre = real_user_config.read_bytes() if real_user_config.exists() else None

    monkeypatch.setattr(cli, "_hermes_home", real_user_config.parent)

    with pytest.raises(pytest.fail.Exception, match="real Hermes config"):
        cli.save_config_value("model.default", "guard-proof")

    post = real_user_config.read_bytes() if real_user_config.exists() else None
    assert post == pre


def test_guard_allows_writes_to_test_tmp_home(tmp_path, monkeypatch):
    """A test that repoints cli._hermes_home at its own tmp home may persist."""
    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text("model:\n  default: old\n")
    monkeypatch.setattr(cli, "_hermes_home", home)

    result = cli.save_config_value("model.default", "new-model")

    # The guard let the call through to the real implementation.  The write
    # itself succeeds only where ruamel.yaml is installed; where it is, the
    # tmp config must carry the new value — and only the tmp config changed.
    assert isinstance(result, bool)
    if result:
        assert "new-model" in config.read_text()
