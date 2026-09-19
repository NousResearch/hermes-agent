"""A memory provider that left core is installed from the catalog, config untouched; a provider the
catalog does not know is reported with the one-liner instead of silently dropping memory."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import SimpleNamespace

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


@pytest.fixture
def recovery_profiles(tmp_path, monkeypatch):
    # Bind before cold imports: neither config nor plugin discovery may see a real profile.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("HERMES_HOME", str(a))
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    for home in (a, b):
        home.mkdir()
        (home / "config.yaml").write_text(
            "memory:\n  provider: recovery_fixture\n"
            "  recovery_fixture:\n    workspace: ${RECOVERY_WORKSPACE}\n"
            "security:\n  allow_lazy_installs: true\n"
            f"terminal:\n  cwd: {home}\n"
        )
        (home / ".env").write_text(f"RECOVERY_WORKSPACE={home.name}\n")
    from agent import secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli import plugin_catalog, plugins_cmd
    from tools.terminal_scope import install_and_reset_profile_terminal_scope

    monkeypatch.setattr(mig, "_attempted", set())
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr(
        plugin_catalog, "get_live_catalog_entry", lambda name: SimpleNamespace(name=name)
    )

    @contextmanager
    def scope(home):
        home_token = set_hermes_home_override(home)
        try:
            secret_token = secret_scope.set_secret_scope(
                secret_scope.build_profile_secret_scope(home)
            )
            try:
                with install_and_reset_profile_terminal_scope(home):
                    yield
            finally:
                secret_scope.reset_secret_scope(secret_token)
        finally:
            reset_hermes_home_override(home_token)

    calls = []

    def install(identifier, *, force, enable, catalog_name):
        from hermes_constants import get_hermes_home
        from hermes_cli.config import load_config_readonly
        from tools.terminal_scope import terminal_env

        home = get_hermes_home().resolve()
        assert identifier == "" and force is False and enable is True
        assert catalog_name == "recovery_fixture"
        assert secret_scope.get_secret("RECOVERY_WORKSPACE") == home.name
        assert load_config_readonly()["memory"][catalog_name]["workspace"] == home.name
        assert Path(terminal_env("TERMINAL_CWD")) == home
        calls.append(home)
        plugin = home / "plugins" / catalog_name
        plugin.mkdir(parents=True, exist_ok=True)
        (plugin / "__init__.py").write_text(
            "# MemoryProvider discovery fixture; never initialized.\n"
        )
        return {"ok": True}

    monkeypatch.setattr(plugins_cmd, "dashboard_install_plugin", install)
    return SimpleNamespace(a=a, b=b, scope=scope, calls=calls, install=install,
                           plugins_cmd=plugins_cmd, catalog=plugin_catalog)


@pytest.mark.parametrize("first_outcome", ["installed", "failed", "raised", "refused", "unknown"])
def test_startup_budget_belongs_to_resolved_home(recovery_profiles, monkeypatch, caplog, first_outcome):
    profiles = recovery_profiles
    a, b = profiles.a, profiles.b
    name = "recovery_fixture"
    alias = a / "child" / ".."
    (a / "child").mkdir()
    if first_outcome == "refused":
        config = a / "config.yaml"
        config.write_text(config.read_text().replace("allow_lazy_installs: true", "allow_lazy_installs: false"))
    before = {home: ((home / "config.yaml").read_bytes(), (home / ".env").read_bytes())
              for home in (a, b)}
    attempts = []

    def install(*args, **kwargs):
        from hermes_constants import get_hermes_home
        home = get_hermes_home().resolve()
        attempts.append(home)
        if home == a and first_outcome == "failed":
            return {"ok": False, "error": "offline fixture"}
        if home == a and first_outcome == "raised":
            raise OSError("offline fixture")
        return profiles.install(*args, **kwargs)

    def catalog_entry(provider):
        from hermes_constants import get_hermes_home
        if get_hermes_home().resolve() == a and first_outcome == "unknown":
            return None
        return SimpleNamespace(name=provider)

    monkeypatch.setattr(profiles.plugins_cmd, "dashboard_install_plugin", install)
    monkeypatch.setattr(profiles.catalog, "get_live_catalog_entry", catalog_entry)
    results = []
    for home in (a, b, alias, a):
        with profiles.scope(home):
            results.append(mig.recover_at_startup(name))
            if home == a and len(results) == 1 and first_outcome == "installed":
                # Disappearance must not replenish this process's automatic budget.
                (a / "plugins" / name / "__init__.py").unlink()
    assert results == [first_outcome == "installed", True, False, False]
    assert attempts == ([a, b] if first_outcome in {"installed", "failed", "raised"} else [b])
    assert mig.provider_present(name, b)
    assert not mig.provider_present(name, a)
    if first_outcome != "installed":
        assert "hermes plugins install" in caplog.text

    # Explicit repair is still usable after success, failure or policy refusal.
    monkeypatch.setattr(profiles.plugins_cmd, "dashboard_install_plugin", profiles.install)
    monkeypatch.setattr(profiles.catalog, "get_live_catalog_entry", lambda provider: SimpleNamespace(name=provider))
    with profiles.scope(a):
        assert mig.migrate_home(a, install=mig._install_into(a)) == name
    assert mig.provider_present(name, a)
    assert before == {home: ((home / "config.yaml").read_bytes(), (home / ".env").read_bytes())
                      for home in (a, b)}


@pytest.mark.parametrize("same_home", [True, False], ids=["same-home", "other-home"])
def test_startup_claim_does_not_hold_install_lock(recovery_profiles, monkeypatch, same_home):
    profiles = recovery_profiles
    started, release = Event(), Event()
    name = "recovery_fixture"

    def install(*args, **kwargs):
        from hermes_constants import get_hermes_home
        if get_hermes_home().resolve() == profiles.a:
            started.set()
            assert release.wait(10), "test did not release the first install"
        return profiles.install(*args, **kwargs)

    def recover(home):
        with profiles.scope(home):
            return mig.recover_at_startup(name)

    monkeypatch.setattr(profiles.plugins_cmd, "dashboard_install_plugin", install)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(recover, profiles.a)
        try:
            assert started.wait(10), "first startup did not reach installation"
            second = pool.submit(recover, profiles.a if same_home else profiles.b)
            # A duplicate returns without waiting; another home can install while A is blocked.
            assert second.result(timeout=10) is (not same_home)
        finally:
            release.set()
        assert first.result(timeout=10) is True
    expected = [profiles.a] if same_home else [profiles.b, profiles.a]
    assert profiles.calls == expected
    assert all(mig.provider_present(name, home) for home in expected)
