"""Browser resolution uses installed PM facts without provisioning a browser."""

import os

import pytest

import pm
from pm import paths


@pytest.mark.parametrize("record_executable", [False, True])
def test_chromium_resolves_installed_binary_without_mutation(tmp_path, monkeypatch, record_executable):
    from hermes_cli.browser_runtime import chromium_executable

    home = tmp_path / "home"
    store = home / "tools"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
    lock = pm.Lockfile(paths.lockfile_path())
    target = pm.current_target()
    package = pm.get_package("chromium")
    version = lock.version("chromium")
    assert version is not None
    entry = store / package.store_entry(version, target)
    binary = entry / "browser" / ("chrome.exe" if target.startswith("win32") else "chrome")
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"browser-fixture")
    binary.chmod(0o755)
    env = {"PLAYWRIGHT_BROWSERS_PATH": str(store)}
    if record_executable:
        env["AGENT_BROWSER_EXECUTABLE_PATH"] = str(binary)
    facts = pm.Facts(store / "facts.json")
    facts.record(
        "chromium", version, entry.name, env, store,
        target=target, artifacts=[a["sha256"] for a in lock.artifacts("chromium", target)],
    )
    before = facts.path.read_bytes()
    environment = dict(os.environ)

    assert chromium_executable() == str(binary)
    assert facts.path.read_bytes() == before
    assert dict(os.environ) == environment


def test_chromium_override_wins_without_installing(tmp_path, monkeypatch):
    from hermes_cli.browser_runtime import chromium_executable

    store = tmp_path / "missing-store"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    override = str(tmp_path / "external browser")
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", override)

    def forbidden(*args, **kwargs):
        pytest.fail("browser resolution must not provision or activate packages")

    monkeypatch.setattr(pm, "ensure", forbidden)
    monkeypatch.setattr(pm, "activate", forbidden)
    with monkeypatch.context() as scoped:
        scoped.setattr(pm, "installed_package", forbidden)
        assert chromium_executable() == override
    monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH")
    assert chromium_executable() is None
    assert not store.exists()
    assert "AGENT_BROWSER_EXECUTABLE_PATH" not in os.environ
