"""An install made while agent-browser was a pm package keeps working after it left.

The store, its facts and the recorded opt-outs of such an install still name
agent-browser. Every reader must skip that fact, `pm gc` reclaims its entry, and
an opt-out recorded as `--without agent-browser` (what the installers'
`--skip-browser` passes) keeps skipping the browser, which is Chromium alone now.

Real pm code paths over a fixture store under a scratch HERMES_HOME and
HERMES_RUNTIME_DIR; the real registry and pm/lock.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

import pm
import pm.cli
from pm import paths
from pm.lock import Facts, Lockfile
from pm.store import Store, current_target, tree_digest
from tests.pm.test_install_default_closure import _bare_install, install_spy as install_spy  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import


def _record(facts: Facts, store: Path, name: str, version: str, entry: Path, env: dict,
            artifacts: list[str]) -> None:
    facts.record(name, version, entry.name, env, store, target=current_target(),
                 artifacts=artifacts, digest=tree_digest(entry))


@pytest.fixture()
def upgraded_store(tmp_path, monkeypatch):
    """A store as the agent-browser era left it: both entries, the old chromium env, the opt-out."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    target = current_target()
    chromium = pm.get_package("chromium")
    if chromium.missing_reason(target) is not None:
        pytest.skip(f"pm builds no Chromium for {target}")
    store = paths.store_root()
    lock = Lockfile(paths.lockfile_path())
    version = lock.version("chromium")
    assert version is not None
    entry = store / chromium.store_entry(version, target)
    binary = entry / "chrome-fixture" / ("chrome.exe" if target.startswith("win32") else "chrome")
    binary.parent.mkdir(parents=True)
    binary.write_text("inert browser fixture", encoding="utf-8")
    binary.chmod(0o755)
    (entry / "INSTALLATION_COMPLETE").touch()
    facts = Facts(paths.facts_path())
    _record(facts, store, "chromium", version, entry,
            {"PLAYWRIGHT_BROWSERS_PATH": str(store), "AGENT_BROWSER_EXECUTABLE_PATH": str(binary)},
            [row["sha256"] for row in lock.artifacts("chromium", target)])
    retired = store / "agent-browser-0.26.0"
    retired_bin = retired / "bin"
    retired_bin.mkdir(parents=True)
    (retired_bin / f"agent-browser-{target}").write_text("retired", encoding="utf-8")
    _record(facts, store, "agent-browser", "0.26.0", retired, {"PATH": [str(retired_bin)]}, ["0" * 64])
    from pm.defaults import declined_path

    declined_path().parent.mkdir(parents=True, exist_ok=True)
    declined_path().write_text(json.dumps({"schema": 1, "declined": ["agent-browser"]}), encoding="utf-8")
    return {"store": store, "binary": binary, "retired": retired, "retired_bin": retired_bin}


def test_every_reader_skips_the_retired_fact(upgraded_store, monkeypatch):
    from pm.install import activate, drift, env_for
    from pm.registry import all_packages

    assert pm.installed_package("chromium").binary == upgraded_store["binary"]
    assert "agent-browser" not in drift(include_venv=False)
    env = env_for(*all_packages(), base_env={})
    assert str(upgraded_store["retired_bin"]) not in env.get("PATH", "").split(os.pathsep)
    assert env["PLAYWRIGHT_BROWSERS_PATH"] == str(upgraded_store["store"])
    monkeypatch.setenv("PATH", "")
    activate(allow_incomplete=True)
    assert str(upgraded_store["retired_bin"]) not in os.environ["PATH"].split(os.pathsep)


def test_ensure_rerecords_the_old_chromium_env_without_installing(upgraded_store, monkeypatch):
    """`hermes update` skips installed defaults, so nothing reinstalls Chromium to re-record its env: an
    ensure() that finds it installed re-derives the env from the package code, without a fetch."""
    import pm.install

    monkeypatch.setattr(pm.install, "_install", lambda *a, **k: pytest.fail("an installed package was installed"))
    pm.install.ensure("chromium")
    assert pm.install.env_for("chromium", base_env={}) == {"PLAYWRIGHT_BROWSERS_PATH": str(upgraded_store["store"])}
    assert "AGENT_BROWSER_EXECUTABLE_PATH" not in Facts(paths.facts_path()).env_for("chromium", paths.store_root())
    assert pm.installed_package("chromium").binary == upgraded_store["binary"]


def test_gc_reclaims_the_retired_entry_and_keeps_chromium(upgraded_store):
    store = Store(paths.store_root())
    pm.cli._gc_store(store, Facts(paths.facts_path()))
    facts = Facts(paths.facts_path())
    assert facts.get("agent-browser") is None
    assert not upgraded_store["retired"].exists()
    assert facts.get("chromium") is not None
    assert pm.installed_package("chromium").binary == upgraded_store["binary"]


def test_a_recorded_agent_browser_opt_out_still_skips_the_browser(upgraded_store, install_spy):
    from pm.defaults import declined

    assert declined() == frozenset({"chromium"})
    assert "chromium" not in _bare_install(install_spy)
    # The opt-back-in old install logs print restores Chromium.
    install_spy["names"] = None
    assert pm.cli.cmd_install(argparse.Namespace(names=["agent-browser"], tools_only=False)) == 0
    assert install_spy["names"] == ["chromium"]
    assert declined() == frozenset()
    assert "chromium" in _bare_install(install_spy)


def test_an_old_installer_without_agent_browser_declines_chromium(upgraded_store, install_spy):
    from pm.defaults import declined, declined_path

    declined_path().unlink()
    assert "chromium" not in _bare_install(install_spy, without=["agent-browser"])
    assert declined() == frozenset({"chromium"})
    assert json.loads(declined_path().read_text(encoding="utf-8"))["declined"] == ["chromium"]
