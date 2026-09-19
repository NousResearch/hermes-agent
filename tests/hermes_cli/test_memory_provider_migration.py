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


def test_recover_at_startup_one_shot_is_scoped_per_profile_home(tmp_path, monkeypatch):
    """A multiplex gateway serves several profile homes; two of them configuring the same provider
    name must each get their own one-shot recovery attempt — recovering it for one must not skip
    the other. The one-shot must still hold when the SAME home retries the SAME name."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setattr(mig, "_attempted", set())
    monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: True)

    attempted_homes: list[Path] = []

    def fake_migrate_home(home, *, install, say):
        attempted_homes.append(home)
        return "honcho"

    monkeypatch.setattr(mig, "migrate_home", fake_migrate_home)

    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()

    token = set_hermes_home_override(home_a)
    try:
        assert mig.recover_at_startup("honcho") is True
        # same home, same name, second call: the one-shot fires, no second attempt
        assert mig.recover_at_startup("honcho") is False
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(home_b)
    try:
        assert mig.recover_at_startup("honcho") is True
    finally:
        reset_hermes_home_override(token)

    assert attempted_homes == [home_a, home_b]
