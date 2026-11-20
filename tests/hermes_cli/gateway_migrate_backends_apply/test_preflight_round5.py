"""Preflight refusals for cases that used to change the machine first and roll back (or lose a record) after:
an unrecognised record state, an occupied restart-guard path, a managed Webapp overlay, a relative launcher
without a working directory, and a turn that started after the preflight."""

from __future__ import annotations

import json

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
