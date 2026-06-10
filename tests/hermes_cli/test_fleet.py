"""Tests for the in-container fleet client (hermes_cli.fleet)."""
import pytest

from hermes_cli import fleet


def test_build_ssh_argv_shape(monkeypatch):
    monkeypatch.setattr(fleet, "SSH_KEY", "/k/id")
    monkeypatch.setattr(fleet, "KNOWN_HOSTS", "/k/known")
    monkeypatch.setattr(fleet, "OPS_USER", "hermes-ops")
    monkeypatch.setattr(fleet, "OPS_HOST", "127.0.0.1")
    argv = fleet.build_ssh_argv(["up", "acme"])
    assert argv[0] == "ssh"
    assert "-i" in argv and "/k/id" in argv
    # Strict host-key checking against the pinned known_hosts (no TOFU).
    assert "StrictHostKeyChecking=yes" in argv
    assert "UserKnownHostsFile=/k/known" in argv
    assert "IdentitiesOnly=yes" in argv
    assert argv[-2] == "hermes-ops@127.0.0.1"
    assert argv[-1] == "up acme"          # joined into $SSH_ORIGINAL_COMMAND


@pytest.mark.parametrize("bad", ["../etc", "a b", "ACME", "a;reboot", "-x"])
def test_remote_subcommands_validate_slug(bad, monkeypatch):
    # A bad client name is rejected locally, before any ssh is attempted.
    called = []
    monkeypatch.setattr(fleet, "_remote", lambda r: called.append(r) or 0)
    for cmd in ("up", "down", "restart", "status", "logs"):
        with pytest.raises(SystemExit):
            fleet.main([cmd, bad])
    assert called == []


def test_logs_line_bounds(monkeypatch):
    sent = []
    monkeypatch.setattr(fleet, "_remote", lambda r: sent.append(r) or 0)
    assert fleet.main(["logs", "acme", "50"]) == 0
    assert sent[-1] == ["logs", "acme", "50"]
    with pytest.raises(SystemExit):
        fleet.main(["logs", "acme", "9999"])


def test_remote_passthrough_builds_command(monkeypatch):
    sent = []
    monkeypatch.setattr(fleet, "_remote", lambda r: sent.append(r) or 0)
    fleet.main(["up", "acme"])
    fleet.main(["ps"])
    fleet.main(["apply"])
    assert sent == [["up", "acme"], ["ps"], ["apply"]]


def test_remote_errors_clearly_without_key(monkeypatch, tmp_path):
    monkeypatch.setattr(fleet, "SSH_KEY", str(tmp_path / "absent"))
    with pytest.raises(SystemExit) as e:
        fleet.main(["ps"])
    assert "broker" in str(e.value).lower()


def test_generate_delegates_to_compose_gen(monkeypatch):
    captured = {}
    def fake_compose_main(argv):
        captured["argv"] = argv
        return 0
    monkeypatch.setattr("hermes_cli.compose_gen.main", fake_compose_main)
    rc = fleet.main(["generate", "--data-root", "/opt/data", "--env", "prod"])
    assert rc == 0
    a = captured["argv"]
    assert "--data-root" in a and "/opt/data" in a
    assert "--output" in a and a[a.index("--output") + 1].endswith("docker-compose.clients.yml")
    assert "--env" in a and "prod" in a
