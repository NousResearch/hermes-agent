"""A memory provider that left core is installed from the catalog, config untouched; a provider the
catalog does not know is reported with the one-liner instead of silently dropping memory."""

from pathlib import Path

import pytest

from hermes_cli import memory_provider_migration as mig


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("memory:\n  provider: honcho\n  honcho:\n    workspace: keep-me\n")
    monkeypatch.setattr(mig, "provider_present", lambda name, home: (home / "plugins" / name).is_dir())
    return tmp_path


def test_missing_provider_installs_its_catalog_plugin_and_keeps_config(home, monkeypatch):
    monkeypatch.setattr(mig, "catalog_source", lambda name: name)
    calls: list[str] = []
    said: list[str] = []

    def fake_install(name: str) -> dict:
        calls.append(name)
        (home / "plugins" / name).mkdir(parents=True)
        return {"ok": True}

    assert mig.migrate_home(home, install=fake_install, say=said.append) == "honcho"
    assert calls == ["honcho"]
    assert "settings and data are unchanged" in said[0]
    assert "workspace: keep-me" in (home / "config.yaml").read_text()
    # present now → nothing to do, nothing said
    assert mig.migrate_home(home, install=fake_install, say=said.append) is None
    assert calls == ["honcho"]


def test_presence_is_checked_in_the_home_being_migrated(tmp_path, monkeypatch):
    """The update hook walks several profile homes from one process; a provider installed in profile B
    must count as present for B even when the process-level home (A) lacks it. Real lookup, no mock."""
    a, b = tmp_path / "a", tmp_path / "b"
    for h in (a, b):
        h.mkdir(); (h / "config.yaml").write_text("memory:\n  provider: twin\n")
    (b / "plugins" / "twin").mkdir(parents=True)
    (b / "plugins" / "twin" / "__init__.py").write_text("class Twin(MemoryProvider): ...\n")
    monkeypatch.setenv("HERMES_HOME", str(a))
    monkeypatch.setattr(mig, "catalog_source", lambda name: name)
    installs: list[Path] = []
    assert mig.migrate_home(b, install=lambda n: installs.append(b) or {"ok": True}, say=lambda s: None) is None
    assert installs == []
    assert mig.migrate_home(a, install=lambda n: installs.append(a) or {"ok": True}, say=lambda s: None) == "twin"


def test_provider_unknown_to_catalog_is_reported_not_installed(home, monkeypatch):
    monkeypatch.setattr(mig, "catalog_source", lambda name: None)
    said: list[str] = []
    assert mig.migrate_home(home, install=lambda n: pytest.fail("must not install"), say=said.append) is None
    assert "not in the plugin catalog" in said[0] and "memory.provider" in said[0]


@pytest.mark.parametrize("provider", ["default", "builtin", "none"])
def test_core_memory_providers_never_trip_migration(home, monkeypatch, provider):
    """The built-in file store is not a plugin: ``default``/``builtin``/``none`` early-return
    with no install and no warning rather than a dead-end catalog notice (#115113)."""
    (home / "config.yaml").write_text(f"memory:\n  provider: {provider}\n", encoding="utf-8")
    monkeypatch.setattr(mig, "catalog_source", lambda name: None)
    said: list[str] = []
    assert mig.migrate_home(
        home, install=lambda n: pytest.fail("must not install"), say=said.append
    ) is None
    assert said == []
