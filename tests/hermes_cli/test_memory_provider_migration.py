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


def _four_homes_needing_hindsight(tmp_path):
    homes = [tmp_path / name for name in ("default", "work-a", "work-b", "personal")]
    for home in homes:
        home.mkdir()
        (home / "config.yaml").write_text("memory:\n  provider: hindsight\n")
    return homes


def test_migrate_all_homes_asks_once_for_several_profiles_sharing_a_provider(tmp_path, monkeypatch):
    """Four profiles all missing the same provider must face ONE consent prompt, not
    four (#125794): accepting it installs every affected home without reprompting."""
    homes = _four_homes_needing_hindsight(tmp_path)
    monkeypatch.setattr("pm.plugins_state.dependency_homes", lambda: homes)
    monkeypatch.setattr(mig, "provider_present", lambda name, home: (home / "plugins" / name).is_dir())
    monkeypatch.setattr(mig, "catalog_source", lambda name: name)
    monkeypatch.setattr(mig.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(mig.sys.stdout, "isatty", lambda: True)
    prompts: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt: (prompts.append(prompt), "y")[1])

    installed_homes: list[Path] = []
    consent_flags: list[bool] = []

    def fake_install_into(home, *, consented=False):
        consent_flags.append(consented)

        def _install(name: str) -> dict:
            installed_homes.append(home)
            (home / "plugins" / name).mkdir(parents=True)
            return {"ok": True}
        return _install

    monkeypatch.setattr(mig, "_install_into", fake_install_into)

    installed = mig.migrate_all_homes(say=lambda s: None)

    assert len(prompts) == 1
    assert installed == ["hindsight"] * 4
    assert installed_homes == homes
    assert consent_flags == [True] * 4


def test_migrate_all_homes_declined_batch_touches_no_home(tmp_path, monkeypatch):
    homes = _four_homes_needing_hindsight(tmp_path)
    monkeypatch.setattr("pm.plugins_state.dependency_homes", lambda: homes)
    monkeypatch.setattr(mig, "provider_present", lambda name, home: (home / "plugins" / name).is_dir())
    monkeypatch.setattr(mig, "catalog_source", lambda name: name)
    monkeypatch.setattr(mig.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(mig.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    def must_not_install(home, *, consented=False):
        def _install(name: str) -> dict:
            pytest.fail("declining the batch prompt must not install anything")
        return _install

    monkeypatch.setattr(mig, "_install_into", must_not_install)

    assert mig.migrate_all_homes(say=lambda s: None) == []
    assert not any((home / "plugins").exists() for home in homes)


def test_migrate_all_homes_non_interactive_skips_without_prompting(tmp_path, monkeypatch):
    homes = _four_homes_needing_hindsight(tmp_path)
    monkeypatch.setattr("pm.plugins_state.dependency_homes", lambda: homes)
    monkeypatch.setattr(mig, "provider_present", lambda name, home: (home / "plugins" / name).is_dir())
    monkeypatch.setattr(mig, "catalog_source", lambda name: name)
    monkeypatch.setattr(mig.sys.stdin, "isatty", lambda: False)

    def fail_on_input(prompt):
        pytest.fail("non-interactive migration must not block on input()")

    monkeypatch.setattr("builtins.input", fail_on_input)

    def must_not_install(home, *, consented=False):
        def _install(name: str) -> dict:
            pytest.fail("a non-interactive decline must not install anything")
        return _install

    monkeypatch.setattr(mig, "_install_into", must_not_install)

    assert mig.migrate_all_homes(say=lambda s: None) == []


def test_migrate_all_homes_keyboard_interrupt_at_prompt_is_a_clean_decline(tmp_path, monkeypatch):
    homes = _four_homes_needing_hindsight(tmp_path)
    monkeypatch.setattr("pm.plugins_state.dependency_homes", lambda: homes)
    monkeypatch.setattr(mig, "provider_present", lambda name, home: (home / "plugins" / name).is_dir())
    monkeypatch.setattr(mig, "catalog_source", lambda name: name)
    monkeypatch.setattr(mig.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(mig.sys.stdout, "isatty", lambda: True)

    def raise_interrupt(prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", raise_interrupt)

    def must_not_install(home, *, consented=False):
        def _install(name: str) -> dict:
            pytest.fail("a cancelled batch consent must not install anything")
        return _install

    monkeypatch.setattr(mig, "_install_into", must_not_install)

    # Must return cleanly — no KeyboardInterrupt escapes to the caller.
    assert mig.migrate_all_homes(say=lambda s: None) == []


def test_startup_recovery_attempts_each_profile_home(tmp_path, monkeypatch):
    """One multiplexed process can start agents for two homes missing the same provider."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from pm import install as pm_install

    homes = [tmp_path / "a", tmp_path / "b"]
    for profile_home in homes:
        profile_home.mkdir()
        (profile_home / "config.yaml").write_text("memory:\n  provider: twin\n", encoding="utf-8")
    monkeypatch.setattr(mig, "_attempted", set())
    monkeypatch.setattr(mig, "catalog_source", lambda name: name)
    monkeypatch.setattr(pm_install, "lazy_installs_allowed", lambda: True)
    installed = []

    def fake_installer(profile_home):
        def install(name):
            plugin_dir = profile_home / "plugins" / name
            plugin_dir.mkdir(parents=True)
            (plugin_dir / "__init__.py").write_text("class Twin(MemoryProvider): ...\n", encoding="utf-8")
            installed.append(profile_home)
            return {"ok": True}

        return install

    monkeypatch.setattr(mig, "_install_into", fake_installer)
    outcomes = []
    for profile_home in (homes[0], homes[1], homes[0]):
        token = set_hermes_home_override(profile_home)
        try:
            outcomes.append(mig.recover_at_startup("twin"))
        finally:
            reset_hermes_home_override(token)

    assert outcomes == [True, True, False]
    assert installed == homes
