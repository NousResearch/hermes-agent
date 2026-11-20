"""Preflight refusals for cases that used to change the machine first and roll back (or lose a record) after:
an unrecognised record state, an occupied restart-guard path, a managed Webapp overlay, a relative launcher
without a working directory, and a turn that started after the preflight."""

from __future__ import annotations

import json

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from hermes_cli.backend_migration_marker import backend_retired_by_migration
from tests.hermes_cli.test_gateway_migrate_backends_apply import FakeHost, _args


def test_an_unrecognised_record_state_is_refused_and_kept(home, capsys):
    """A complete record whose state is none of the engine's is not a new home: the apply refuses (exit 3),
    keeps the record and its backups, touches no service; the update marker keeps the old backend down."""
    host = FakeHost(home)
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    path = engine.manifest_path(home)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["state"] = "paused"
    path.write_text(json.dumps(record), encoding="utf-8")
    before = path.read_bytes()
    host.old.update(enabled=True, active=True)  # an old backend is discoverable again
    host.calls.clear()
    capsys.readouterr()

    assert engine.cmd_apply(host, _args()) == engine.EXIT_BLOCKED
    assert path.read_bytes() == before and host.calls == []
    assert "unrecognized migration state" in capsys.readouterr().out
    assert backend_retired_by_migration(home) is True


def test_an_existing_restart_guard_drop_in_is_refused_and_left_alone(home, capsys):
    """`UNIT.d/hermes-migrated.conf` already exists (the user's own file of that name): the apply refuses
    (exit 3) before any change instead of overwriting it and removing it on --restore."""
    from pathlib import Path
    host = FakeHost(home)
    guard = Path(host.old_unit_dropin_path({"supervisor": "systemd", "scope": "user",
                                            "unit": "hermes-dashboard.service"}))
    guard.parent.mkdir(parents=True)
    guard.write_text("[Service]\nNice=5\n", encoding="utf-8")

    code = engine.cmd_apply(host, _args())
    assert guard.read_text(encoding="utf-8") == "[Service]\nNice=5\n"
    assert code == engine.EXIT_BLOCKED
    assert host.calls == [] and not engine.manifest_path(home).exists()
    assert f"{guard} already exists" in capsys.readouterr().out


def _turn_starts_at_gateway_install(host, turns):
    real = host.install_gateway

    def install_then_a_turn_starts():
        real()
        host.turns = turns  # a chat turn starts on the old backend after the preflight read 0

    host.install_gateway = install_then_a_turn_starts


def test_a_turn_started_after_preflight_stops_the_move_before_the_old_backend_stops(home, capsys):
    host = FakeHost(home)
    _turn_starts_at_gateway_install(host, 1)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK
    out = capsys.readouterr().out
    assert "old_stop" not in host.calls and host.old == {"enabled": True, "active": True}
    assert "1 turn(s) are running on the old backend. Wait for them to finish" in out


def test_the_check_before_the_stop_decides_as_the_preflight_did_outside_its_network_namespace(home):
    """An automatic plan reads the old backend's markers whatever its network namespace, in the preflight and
    again right before the stop: a backend this planner cannot see the namespace of still migrates when idle."""
    host = FakeHost(home)
    host.same_netns = False

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    assert "old_stop" in host.calls


def test_a_turn_started_after_preflight_is_interrupted_with_force_interrupt(home):
    host = FakeHost(home)
    _turn_starts_at_gateway_install(host, 1)

    assert engine.cmd_apply(host, _args(force_interrupt=True)) == engine.EXIT_OK
    assert "old_stop" in host.calls


@pytest.fixture
def managed(home, tmp_path, monkeypatch):
    """A real managed scope (HERMES_MANAGED_DIR/config.yaml), set up as the managed-config tests do."""
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope
    folder = tmp_path / "managed"
    folder.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(folder))

    def write(body):
        (folder / "config.yaml").write_text(body, encoding="utf-8")
        cfg._LOAD_CONFIG_CACHE.clear()
        cfg._RAW_CONFIG_CACHE.clear()
        managed_scope.invalidate_managed_cache()
    yield folder, write
    managed_scope.invalidate_managed_cache()


@pytest.mark.parametrize(("body", "key"), [("webapp:\n  enabled: false\n", "webapp.enabled"),
                                           ("webapp:\n  port: 8000\n", "webapp.port")])
def test_a_managed_webapp_setting_that_overrides_the_move_is_refused_before_any_change(
        home, managed, capsys, body, key):
    folder, write = managed
    write(body)
    host = FakeHost(home)
    before = {n: (home / n).read_bytes() for n in ("config.yaml", ".env")}

    code = engine.cmd_apply(host, _args())

    out = capsys.readouterr().out
    assert "old_stop" not in host.calls and host.old == {"enabled": True, "active": True}
    assert code == engine.EXIT_BLOCKED, out
    assert host.calls == [] and not engine.manifest_path(home).exists()
    assert {n: (home / n).read_bytes() for n in ("config.yaml", ".env")} == before
    assert key in out and str(folder / "config.yaml") in out, out


def test_a_managed_webapp_block_that_agrees_with_the_move_does_not_block(home, managed):
    _folder, write = managed
    write("webapp:\n  enabled: true\n  host: 0.0.0.0\n  port: 9119\n")  # what the move writes
    assert engine.cmd_apply(FakeHost(home), _args()) == engine.EXIT_OK


_TAIL = ["dashboard", "--no-open", "--host", "0.0.0.0", "--port", "9119"]


@pytest.mark.parametrize("argv", [["python3", "hermes_cli/main.py", *_TAIL],   # a relative script after python
                                  ["./venv/bin/hermes", *_TAIL],              # a relative launcher
                                  ["/venv/bin/python", "-u", "-m", "hermes_cli.main", *_TAIL]])
def test_a_launch_that_needs_its_working_directory_is_refused_when_it_cannot_be_read(home, capsys, argv):
    """Started again without its working directory, a relative launcher or script (or `python -m`) would
    not be found: refused (exit 3) before the hand-started backend is stopped."""
    host = FakeHost(home, supervisor="manual")
    host.argv = argv
    host.process_cwd = lambda pid: None

    code = engine.cmd_apply(host, _args())

    out = capsys.readouterr().out
    assert "old_stop" not in host.calls and host.old["active"] is True
    assert code == engine.EXIT_BLOCKED, out
    assert "working directory could not be read" in out


def test_a_hermes_model_flag_is_not_python_module_mode(home):
    """`hermes -m <model> dashboard` from an absolute entry point runs anywhere: not refused."""
    host = FakeHost(home, supervisor="manual")
    host.argv = ["/opt/hermes/bin/hermes", "-m", "local-model", *_TAIL]
    host.process_cwd = lambda pid: None

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
