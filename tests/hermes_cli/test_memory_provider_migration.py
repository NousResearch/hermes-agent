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
    # Must not imply config.yaml's memory.<name> keys still apply: the plugin reads its own (#124038).
    assert "memory.honcho" not in said[0] and "hermes memory status" in said[0]
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
    monkeypatch.setattr(mig, "_LEFT_CORE", frozenset({"twin"}))
    monkeypatch.setattr(pm_install, "lazy_installs_allowed", lambda: True)
    installed = []

    def fake_installer(profile_home, **_kw):
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


@pytest.mark.parametrize(("name", "tty", "lazy", "consent"), [
    ("hindsight", False, True, True),    # Desktop/gateway/scripted update: nobody can answer the prompt
    ("hindsight", False, False, False),  # allow_lazy_installs off: still refused, with the install hint
    ("hindsight", True, True, False),    # a terminal: the user is asked
    ("someplugin", False, True, False),  # never bundled: a config naming it is not consent to install it
])
def test_unattended_migration_install_carries_lazy_install_consent(tmp_path, monkeypatch, name, tty, lazy, consent):
    """Every provider still in core declares Python deps; without this, no non-interactive migration
    (the whole point of the agent-start hook) can ever publish one."""
    import io

    from hermes_cli import plugins_cmd
    from pm import install as pm_install

    class _Stream(io.StringIO):
        def isatty(self):
            return tty

    monkeypatch.setattr("sys.stdin", _Stream())
    monkeypatch.setattr("sys.stdout", _Stream())
    monkeypatch.setattr(pm_install, "lazy_installs_allowed", lambda: lazy)
    seen = {}
    monkeypatch.setattr(plugins_cmd, "dashboard_install_plugin", lambda *a, **kw: seen.update(kw) or {"ok": True})
    assert mig._install_into(tmp_path)(name) == {"ok": True}
    assert seen["catalog_name"] == name
    assert seen.get("assume_deps_consent", False) is consent


@pytest.mark.parametrize(("name", "installs"), [("hindsight", True), ("someplugin", False)])
def test_startup_recovery_never_asks_a_dependency_question(tmp_path, monkeypatch, name, installs):
    """Agent init cannot answer a prompt: in the CLI the question hangs the turn behind the chat input,
    with no terminal it is refused on every process start. A provider that shipped in core installs
    with its built-in's consent even on a terminal; any other one is not attempted and names the command."""
    import io

    from hermes_cli import plugins_cmd
    from pm import install as pm_install

    class _Tty(io.StringIO):
        def isatty(self):
            return True

    (tmp_path / "config.yaml").write_text(f"memory:\n  provider: {name}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("sys.stdin", _Tty())
    monkeypatch.setattr("sys.stdout", _Tty())
    monkeypatch.setattr(mig, "_attempted", set())
    monkeypatch.setattr(mig, "catalog_source", lambda n: n)
    monkeypatch.setattr(pm_install, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_a: pytest.fail("agent init asked a question"))
    seen: list[dict] = []
    monkeypatch.setattr(plugins_cmd, "dashboard_install_plugin", lambda *a, **kw: seen.append(kw) or {"ok": False})
    said: list[str] = []
    mig.recover_at_startup(name, say=said.append)
    assert [kw["assume_deps_consent"] for kw in seen] == ([True] if installs else [])
    assert installs or f"`hermes plugins install {name}`" in said[-1]


def test_update_asks_once_and_names_each_profile(tmp_path, monkeypatch):
    """Profiles sharing one environment face one dependency question for a migrating provider
    (#125794); every line names its profile, and a decline names the rest once with their commands."""
    from hermes_cli import plugins_cmd_install

    root = tmp_path / ".hermes"
    homes = [root, root / "profiles" / "work-a", root / "profiles" / "work-b"]
    for profile_home in homes:
        profile_home.mkdir(parents=True, exist_ok=True)
        (profile_home / "config.yaml").write_text("memory:\n  provider: twin\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.setattr("pm.plugins_state.dependency_homes", lambda: homes)
    monkeypatch.setattr(mig, "provider_present", lambda name, home: (home / "plugins" / name).is_dir())
    monkeypatch.setattr(mig, "catalog_source", lambda name: name)
    monkeypatch.setattr(plugins_cmd_install.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(plugins_cmd_install.sys.stdout, "isatty", lambda: True)

    class Console:
        def print(self, *args, **kwargs):
            pass

    def installer(profile_home):
        def install(name):  # the real consent gate every catalog install passes through
            consented, reason = plugins_cmd_install._consent_python_deps(name, ("twin-client",), Console())
            if not consented:
                return {"ok": False, "error": reason}
            (profile_home / "plugins" / name).mkdir(parents=True)
            return {"ok": True}
        return install

    monkeypatch.setattr(mig, "_install_into", installer)
    for answer, expected in (("n", []), ("y", ["twin"] * 3)):
        prompts, said = [], []
        monkeypatch.setattr("builtins.input", lambda prompt, a=answer: prompts.append(prompt) or a)
        assert mig.migrate_all_homes(say=said.append) == expected
        assert len(prompts) == 1
        if answer == "n":
            assert "[profile 'default']" in said[1] and "dependency install declined" in said[1]
            assert "hermes -p work-a plugins install twin" in said[2]
            assert "hermes -p work-b plugins install twin" in said[2]
            assert not any((h / "plugins").exists() for h in homes)
        else:
            assert [line.split("]")[0] for line in said[1:]] == [
                "  [profile 'default'", "  [profile 'work-a'", "  [profile 'work-b'"]


def _profile_homes(tmp_path, monkeypatch, *names):
    root = tmp_path / ".hermes"
    homes = [root, *(root / "profiles" / n for n in names)]
    for profile_home in homes:
        profile_home.mkdir(parents=True, exist_ok=True)
        (profile_home / "config.yaml").write_text("memory:\n  provider: twin\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.setattr(mig, "provider_present", lambda name, home: (home / "plugins" / name).is_dir())
    monkeypatch.setattr(mig, "catalog_source", lambda name: name)
    monkeypatch.setattr(mig, "_LEFT_CORE", frozenset({"twin"}))  # stands in for a formerly bundled provider
    return homes


def test_unattended_update_refusal_in_one_profile_does_not_skip_the_others(tmp_path, monkeypatch):
    """Unattended consent is each home's own allow_lazy_installs: the default home refusing says
    nothing about a profile that allows it, which must still migrate (not be reported as failed)."""
    import io

    from hermes_cli import plugins_cmd
    from hermes_constants import get_hermes_home
    from pm import install as pm_install

    default, work = _profile_homes(tmp_path, monkeypatch, "work")
    monkeypatch.setattr("pm.plugins_state.dependency_homes", lambda: [default, work])
    monkeypatch.setattr("sys.stdin", io.StringIO())  # Desktop / scripted `hermes update`: no terminal
    monkeypatch.setattr(pm_install, "lazy_installs_allowed", lambda: get_hermes_home() == work)

    def install(*_a, catalog_name, assume_deps_consent, **_kw):
        if not assume_deps_consent:
            return {"ok": False, "error": "dependency install skipped (non-interactive)"}
        (get_hermes_home() / "plugins" / catalog_name).mkdir(parents=True)
        return {"ok": True}

    monkeypatch.setattr(plugins_cmd, "dashboard_install_plugin", install)
    said: list[str] = []
    assert mig.migrate_all_homes(say=said.append) == ["twin"]
    assert (work / "plugins" / "twin").is_dir() and not (default / "plugins").exists()
    assert not any("either" in line for line in said)


def test_startup_hint_installs_into_the_profile_that_printed_it(tmp_path, monkeypatch):
    """A bare `hermes plugins install` from a shell targets the sticky profile, so the hint an agent
    started for a named profile (Desktop, gateway, `hermes -p`) prints must carry `-p`."""
    from pm import install as pm_install

    default, work = _profile_homes(tmp_path, monkeypatch, "work")
    monkeypatch.setattr(pm_install, "lazy_installs_allowed", lambda: False)
    for profile_home, command in ((work, "hermes -p work plugins install twin"),
                                  (default, "hermes plugins install twin")):
        monkeypatch.setattr(mig, "_attempted", set())
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        said: list[str] = []
        assert mig.recover_at_startup("twin", say=said.append) is False
        assert f"`{command}`" in said[0]


def test_missing_catalog_provider_recovery_names_the_profile_install_command(tmp_path, monkeypatch, capsys):
    """Offline, lazy installs off, or a failed migration: every surface a user turns to (doctor,
    ``memory status``, ``memory setup honcho``, ``hermes honcho``) names the one command that installs
    the catalog plugin into THIS profile. Real in-tree catalog, nothing installed."""
    from types import SimpleNamespace
    from hermes_cli import doctor_state, memory_setup
    from hermes_cli._parser import build_top_level_parser

    home = tmp_path / "profiles" / "work"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("memory:\n  provider: honcho\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    want = "hermes -p work plugins install honcho"

    doctor_state._memory_provider_generic("honcho")
    memory_setup.cmd_status(SimpleNamespace())
    memory_setup.cmd_setup_provider("honcho")
    assert capsys.readouterr().out.count(want) == 3
    parser, _subparsers, _chat = build_top_level_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["honcho", "status"])
    assert want in capsys.readouterr().err


def test_recovery_copy_without_a_catalog_memory_entry_keeps_the_generic_hint(tmp_path, monkeypatch, capsys):
    from hermes_cli import doctor_state

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # not a profile home: no -p to add
    assert mig.catalog_install_hint("honcho", category="memory") == "hermes plugins install honcho"
    assert mig.catalog_install_hint("honcho", category="tools") is None
    assert mig.catalog_install_hint("no-such-provider") is None
    doctor_state._memory_provider_generic("no-such-provider")
    assert "run: hermes memory setup" in capsys.readouterr().out


def test_startup_recovery_backs_off_after_a_failed_install(tmp_path, monkeypatch):
    """A failed agent-start install is not retried by every following process start (each `hermes chat`,
    cron run, Desktop restart): one failure, then a quiet hour; `hermes update` still retries."""
    from pm import install as pm_install

    (tmp_path / "config.yaml").write_text("memory:\n  provider: hindsight\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(mig, "catalog_source", lambda n: n)
    monkeypatch.setattr(pm_install, "lazy_installs_allowed", lambda: True)
    attempts: list[str] = []
    monkeypatch.setattr(mig, "_install_into", lambda home, **_kw: (lambda n: attempts.append(n) or {"ok": False, "error": "uv lock exited 1"}))
    said: list[str] = []
    for _process in range(3):
        monkeypatch.setattr(mig, "_attempted", set())  # a fresh process each time
        assert mig.recover_at_startup("hindsight", say=said.append) is False
    assert attempts == ["hindsight"]
    assert "`hermes plugins install hindsight`" in said[-1]  # still told memory is off, and the fix
    monkeypatch.setattr(mig, "STARTUP_RETRY_SECONDS", 0.0)
    monkeypatch.setattr(mig, "_attempted", set())
    mig.recover_at_startup("hindsight")
    assert attempts == ["hindsight", "hindsight"]
