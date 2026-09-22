from types import SimpleNamespace

import pytest

from hermes_cli.subcommands import fleet


class _Client:
    def health(self):
        return {"ok": True}

    def list_tasks(self):
        return []


def test_status_redacts_the_bearer_token(monkeypatch, capsys):
    monkeypatch.setattr(fleet, "_config", lambda: {"kanban": {"federated": {
        "enabled": True, "coordinator_url": "https://mac.example/fleet"
    }}})
    monkeypatch.setattr(fleet, "_secret", lambda: "super-secret-token")
    monkeypatch.setattr(fleet, "_client", lambda _config: _Client())

    rc = fleet.cmd_fleet(SimpleNamespace(fleet_action="status", json=False))
    output = capsys.readouterr().out

    assert rc == 0
    assert "https://mac.example/fleet" in output
    assert "configured" in output
    assert "super-secret-token" not in output


def test_runner_validation_fails_before_starting_when_executable_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(fleet, "_config", lambda: {"kanban": {"federated": {
        "coordinator_url": "https://mac.example/fleet"
    }}})
    monkeypatch.setattr(fleet, "_secret", lambda: "token")
    args = SimpleNamespace(
        fleet_action="runner",
        hermes_executable=str(tmp_path / "missing-hermes"),
        liveness_file=str(tmp_path / "live"),
        node_id="windows",
        profile=["coding-expert"],
        project=["LunaBot"],
        model=[],
        tool=["terminal", "git"],
        interval=1.0,
        once=True,
    )

    with pytest.raises(ValueError, match="executable does not exist"):
        fleet.cmd_fleet(args)
