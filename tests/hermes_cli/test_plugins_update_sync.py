"""Every update surface validates staged code before changing an active plugin."""
from __future__ import annotations

import subprocess

import pytest

from tests.hermes_cli.test_plugin_update_transaction import installed, _version  # noqa: F401
from tests.pm.test_plugin_survival_contract import admission_env  # noqa: F401


@pytest.mark.parametrize("installed", ["catalog", "custom"], indirect=True)
def test_cli_update_scan_refusal_keeps_the_installed_tree(installed, monkeypatch):
    from hermes_cli import plugins_cmd as pc

    _, home, repo, target, state = installed
    old = (target / "__init__.py").read_bytes()
    before = (home / "config.yaml").read_bytes()
    state["sha"] = _version(repo, "2.0.0")
    scans = []

    def refuse(staged, source, **kwargs):
        scans.append(staged)
        assert staged != target
        assert (target / "__init__.py").read_bytes() == old
        raise pc.PluginScanBlocked("fixture refuses candidate")

    monkeypatch.setattr(pc, "_scan_plugin_tree", refuse)
    with pytest.raises(SystemExit):
        pc.cmd_update("transactional", interactive=False)
    assert len(scans) == 1
    assert (target / "__init__.py").read_bytes() == old
    assert (home / "config.yaml").read_bytes() == before


@pytest.mark.parametrize("installed", ["catalog", "custom"], indirect=True)
def test_disabled_update_does_not_change_dependencies_or_enablement(installed, monkeypatch):
    from hermes_cli import plugins_cmd as pc
    from pm import paths

    _, home, repo, target, state = installed
    pc._set_plugin_enabled("transactional", enable=False)
    config = (home / "config.yaml").read_bytes()
    facts = paths.runtime_facts_path().read_bytes()
    state["sha"] = _version(repo, "2.0.0")
    monkeypatch.setattr("pm.client.sync_venv",
                        lambda **kwargs: pytest.fail("disabled plugin changed the dependency selection"))
    result = pc.dashboard_update_user_plugin("transactional")
    assert result["ok"], result
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip() == state["sha"]
    assert (home / "config.yaml").read_bytes() == config
    assert paths.runtime_facts_path().read_bytes() == facts


@pytest.mark.parametrize("installed", ["catalog", "custom"], indirect=True)
def test_enablement_before_lock_acquisition_cannot_skip_validation(installed, monkeypatch):
    from contextlib import contextmanager
    from hermes_cli import plugins_cmd as pc, runtime_state
    from pm import paths

    _, home, repo, target, state = installed
    pc._set_plugin_enabled("transactional", enable=False)
    watched = [target / "__init__.py", target / "pyproject.toml", home / "plugins/.install-metadata.json"]
    before = {path: path.read_bytes() for path in watched}
    state["sha"] = _version(repo, "2.0.0", broken=True)
    lock = runtime_state.runtime_lock
    enabled = {}

    @contextmanager
    def enable_before_acquiring_lock(project):
        if not enabled:
            enabled["begun"] = True
            pc._set_plugin_enabled("transactional", enable=True)
            enabled["config"] = (home / "config.yaml").read_bytes()
            enabled["facts"] = paths.runtime_facts_path().read_bytes()
        with lock(project):
            yield

    monkeypatch.setattr(runtime_state, "runtime_lock", enable_before_acquiring_lock)
    result = pc.dashboard_update_user_plugin("transactional")
    assert result["ok"] is False, result
    assert {path: path.read_bytes() for path in watched} == before
    assert (home / "config.yaml").read_bytes() == enabled["config"]
    assert paths.runtime_facts_path().read_bytes() == enabled["facts"]
