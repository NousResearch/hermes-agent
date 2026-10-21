"""Behaviour of the machine-level install probe (``hermes_cli.install_state``)."""

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import install_state
from hermes_cli.install_state import Evidence, Places


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """An empty active root (``HERMES_HOME``) plus one other root the probe is told about."""
    active = tmp_path / "active"
    other = tmp_path / "other-root"
    active.mkdir()
    other.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(active))
    places = Places(markers=(tmp_path / "state" / ".machine",),
                    sources=(install_state._active_root, lambda: (other,)))
    monkeypatch.setattr(install_state, "machine_places", lambda: places)
    install_state.probe.cache_clear()
    yield other, places
    install_state.probe.cache_clear()


def _messages_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, content TEXT)")
        conn.execute("INSERT INTO messages (session_id, content) VALUES ('s', 'hi')")
    conn.close()


USE = {
    Evidence.ONBOARDING_DONE: lambda r: (r / "config.yaml").write_text("onboarding:\n  run: false\n", encoding="utf-8"),
    Evidence.NAMED_PROFILE: lambda r: ((r / "profiles" / "work").mkdir(parents=True),
                                       (r / "profiles" / "work" / "config.yaml").write_text("{}\n", encoding="utf-8")),
    Evidence.PROVIDER_CONFIG: lambda r: (r / "config.yaml").write_text("model:\n  provider: openrouter\n", encoding="utf-8"),
    Evidence.PROVIDER_KEY: lambda r: (r / ".env").write_text("OPENROUTER_API_KEY=sk-test\n", encoding="utf-8"),
    Evidence.PROVIDER_LOGIN: lambda r: (r / "auth.json").write_text(json.dumps(
        {"providers": {"nous": {"auth_method": "anonymous", "anon_token": "anon_x"}}}), encoding="utf-8"),
    Evidence.MESSAGES: lambda r: _messages_db(r / "state.db"),
}


def test_a_machine_with_only_install_time_files_is_fresh(machine):
    other, places = machine
    # What installers and a first boot write: template config, example .env keys, an empty state.db,
    # pool rows Hermes seeded from the shell, the PM boot record.
    (other / "config.yaml").write_text("_config_version: 50\nmodel:\n  default: x\n  provider: auto\n", encoding="utf-8")
    (other / ".env").write_text('TERMINAL_ENV=local\n# OPENROUTER_API_KEY=\nOPENROUTER_API_KEY="" # optional\n',
                                encoding="utf-8")
    (other / "auth.json").write_text(json.dumps({"credential_pool": {"openrouter": [{"source": "env:OPENROUTER_API_KEY"}]}}), encoding="utf-8")
    sqlite3.connect(other / "state.db").execute("CREATE TABLE messages (id INTEGER)").connection.close()
    (other / "installs" / "k" / "bootstrap").mkdir(parents=True)
    (other / "installs" / "k" / "bootstrap" / "default.json").write_text("{}", encoding="utf-8")

    assert install_state.probe().fresh
    assert not places.markers[0].exists()


@pytest.mark.parametrize("evidence", list(USE))
def test_use_in_another_root_beats_an_empty_hermes_home_and_marks_the_machine(machine, evidence):
    other, places = machine
    USE[evidence](other)

    state = install_state.probe()

    assert [(f.evidence, f.where) for f in state.findings] == [(evidence, other.resolve())]
    assert places.markers[0].is_file()
    # The marker answers for the machine from now on, even once that root is gone.
    for child in sorted(other.rglob("*"), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    install_state.probe.cache_clear()
    assert install_state.probe().findings[0].evidence is Evidence.MACHINE_MARKER


def test_an_unreadable_root_counts_as_existing_but_never_marks(machine):
    other, places = machine
    (other / "config.yaml").write_text("model: [unclosed\n", encoding="utf-8")

    state = install_state.probe()

    assert state.existing and state.findings[0].evidence is Evidence.UNREADABLE
    assert not places.markers[0].exists()


def test_a_root_in_a_protected_folder_is_existing_without_being_opened(machine, tmp_path, monkeypatch):
    _other, places = machine
    protected = tmp_path / "Documents"
    places = Places(markers=places.markers, sources=(lambda: (protected / "hermes-home",),),
                    protected=(protected,))
    monkeypatch.setattr(install_state, "machine_places", lambda: places)

    assert install_state.probe().findings[0].evidence is Evidence.UNREADABLE


@pytest.mark.parametrize("name", ["config.yaml", ".env", "auth.json", "state.db", "profiles"])
def test_a_file_linked_into_a_protected_folder_is_existing_without_being_opened(machine, tmp_path, monkeypatch, name):
    """A root outside the protected folders whose file links into one: following the link would raise
    the macOS permission prompt, so the target counts as unreadable."""
    other, places = machine
    protected = tmp_path / "Documents"
    target = protected / "hermes" / name
    target.parent.mkdir(parents=True)
    if name == "profiles":
        target.mkdir()
    elif name == "state.db":
        sqlite3.connect(target).execute("CREATE TABLE messages (id INTEGER)").connection.close()
    else:
        target.write_text("{}\n", encoding="utf-8")
    (other / name).symlink_to(target)
    monkeypatch.setattr(install_state, "machine_places",
                        lambda: Places(markers=places.markers, sources=places.sources, protected=(protected,)))

    assert [(f.evidence, f.where) for f in install_state.probe().findings] == [(Evidence.UNREADABLE, other.resolve())]


def test_a_launcher_in_a_protected_folder_names_its_folder_without_being_opened(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import hermes_platform.resolver
    from hermes_cli import _launchers
    from hermes_platform.resolver import known_dirs

    protected = tmp_path / "Documents"
    launcher = protected / "hermes-src" / "hermes"
    monkeypatch.setattr(known_dirs, "mac_privacy_protected_dirs", lambda: (str(protected),))
    monkeypatch.setattr(hermes_platform.resolver, "locate_command",
                        lambda _name: SimpleNamespace(command=(str(launcher),)))

    def opened(_target):
        pytest.fail("the launcher inside a protected folder was opened")

    monkeypatch.setattr(_launchers, "_launcher_python", opened)

    assert list(install_state._launcher_root()) == [launcher.parent]


def test_claiming_marks_the_machine_and_the_next_probe_sees_it(machine):
    assert install_state.probe().fresh

    install_state.mark_machine_used()

    assert install_state.existing_install()


def test_gateway_service_definitions_name_their_homes(tmp_path, monkeypatch):
    home = tmp_path / "gw-home"
    unit = tmp_path / "hermes-gateway.service"
    unit.write_text(f'[Service]\nEnvironment="HERMES_HOME={home}"\n', encoding="utf-8")
    vbs = tmp_path / "Hermes_Gateway.vbs"
    vbs.write_text(f'env.Item("HERMES_HOME") = "{home}-win"\r\n', encoding="utf-8")
    from hermes_platform.resolver import known_dirs
    monkeypatch.setattr(known_dirs, "gateway_service_definition_globs", lambda: (str(tmp_path / "*Gateway*"),
                                                                                str(tmp_path / "*gateway*")))

    assert set(install_state._service_homes()) == {home, Path(f"{home}-win")}


def test_a_gateway_started_by_a_scheduled_task_or_the_startup_fallback_names_its_home(tmp_path, monkeypatch):
    """A Scheduled Task install removes the Startup-folder entries, and the Startup fallback only starts
    the home's own launcher: both name their home through the ``gateway.vbs`` they start."""
    from hermes_cli import gateway_windows
    from hermes_platform.resolver import known_dirs

    def launcher(home: Path) -> Path:  # what _write_task_script leaves in the home
        script = home / "gateway-service" / "Hermes_Gateway.vbs"
        script.parent.mkdir(parents=True)
        script.write_text(f'env.Item("HERMES_HOME") = "{home}"\r\n', encoding="utf-8")
        return script

    by_task, by_startup = tmp_path / "task home", tmp_path / "startup home"
    startup = tmp_path / "Startup" / "Hermes_Gateway.vbs"
    startup.parent.mkdir()
    startup.write_text(gateway_windows._build_startup_launcher(launcher(by_startup).with_suffix(".cmd")), encoding="utf-8")
    monkeypatch.setattr(known_dirs, "gateway_service_definition_globs", lambda: (str(startup.parent / "Hermes_Gateway*"),))
    monkeypatch.setattr(gateway_windows, "registered_task_launchers", lambda: [str(launcher(by_task))], raising=False)

    assert {by_task, by_startup} <= set(install_state._service_homes())


@pytest.mark.platforms("posix")
def test_a_gateway_service_counts_only_for_the_account_it_runs_as(tmp_path, monkeypatch):
    """A machine-wide unit is another account's install unless it runs as this one (``User=``, root
    when it names none): that account's gateway must not cost a new desktop user the trial."""
    import getpass

    from hermes_platform.resolver import known_dirs

    me = getpass.getuser()
    units = {"mine": f"User={me}\n", "theirs": "User=nobody\n" if me != "nobody" else "User=root\n", "root": ""}
    for name, identity in units.items():
        (tmp_path / f"hermes-gateway-{name}.service").write_text(
            f'[Service]\n{identity}Environment="HERMES_HOME={tmp_path / name}"\n', encoding="utf-8")
    monkeypatch.setattr(known_dirs, "gateway_service_definition_globs",
                        lambda: (str(tmp_path / "hermes-gateway-[mt]*.service"),))
    monkeypatch.setattr(known_dirs, "system_gateway_service_definition_globs",
                        lambda: (str(tmp_path / "hermes-gateway-root.service"),), raising=False)

    homes = set(install_state._service_homes())

    assert tmp_path / "mine" in homes
    assert tmp_path / "theirs" not in homes
    assert (tmp_path / "root" in homes) == (me == "root")


def test_a_state_db_another_process_holds_locked_is_not_use(machine, monkeypatch):
    other, _places = machine
    db = other / "state.db"
    sqlite3.connect(db).execute("CREATE TABLE messages (id INTEGER)").connection.close()
    monkeypatch.setattr(install_state, "_DB_BUSY_SECONDS", 0.05, raising=False)
    # A first boot creating or healing this state.db holds a write lock while the probe runs.
    holder = sqlite3.connect(db, isolation_level=None)
    holder.execute("PRAGMA journal_mode=DELETE")
    holder.execute("BEGIN EXCLUSIVE")
    try:
        assert install_state.probe().fresh
    finally:
        holder.execute("ROLLBACK")
        holder.close()
