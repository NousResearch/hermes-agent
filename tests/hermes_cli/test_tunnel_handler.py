from argparse import Namespace
from hermes_cli import tunnel_commands as tc


def test_up_validates_no_origins_or_zone(monkeypatch, capsys):
    # No zone configured and no origin -> error, no supervisor started.
    monkeypatch.setattr(tc, "resolve_tunnel_config",
                        lambda cli_origins=None: {"zone": "", "tunnel_name": "",
                                                  "credentials_file": "", "metrics_port": 0,
                                                  "idle_timeout_seconds": 1800,
                                                  "drain_seconds": 15, "poll_interval_seconds": 5,
                                                  "admin": [], "routes": [], "enabled": False})
    rc = tc.tunnel_command(Namespace(tunnel_command="up", origins=[],
                                     hold_request=False, reason="", until=""))
    assert rc == 2
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "zone" in out.lower() or "origin" in out.lower()


def test_requests_lists_pending(monkeypatch, capsys, tmp_path):
    from hermes_cli import tunnel_approvals as ta
    p = str(tmp_path / "hold.jsonl")
    monkeypatch.setattr(tc, "_approvals_path", lambda: p)
    ta.file_request(p, user="alice", subdomains=["alice.noit2.com"],
                    reason="demo", requested_until=None)
    rc = tc.tunnel_command(Namespace(tunnel_command="requests"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "alice" in out
    assert "pending" in out


def test_approve_admin_ok(monkeypatch, tmp_path):
    from hermes_cli import tunnel_approvals as ta
    p = str(tmp_path / "hold.jsonl")
    monkeypatch.setattr(tc, "_approvals_path", lambda: p)
    monkeypatch.setattr(tc, "resolve_tunnel_config",
                        lambda cli_origins=None: {"admin": ["admin1"], "zone": "noit2.com",
                                                  "tunnel_name": "", "credentials_file": "",
                                                  "metrics_port": 0, "idle_timeout_seconds": 1800,
                                                  "drain_seconds": 15, "poll_interval_seconds": 5,
                                                  "routes": [], "enabled": False})
    monkeypatch.setattr(tc, "_current_user", lambda: "admin1")
    rid = ta.file_request(p, user="alice", subdomains=["alice.noit2.com"],
                          reason="demo", requested_until=None)
    rc = tc.tunnel_command(Namespace(tunnel_command="approve", id=rid, until="6h",
                                     reason="", kill_origins=False, origins=[],
                                     hold_request=False))
    assert rc == 0
    assert ta.is_approved(p, rid) is True


def test_approve_non_admin_denied(monkeypatch, tmp_path):
    from hermes_cli import tunnel_approvals as ta
    p = str(tmp_path / "hold.jsonl")
    monkeypatch.setattr(tc, "_approvals_path", lambda: p)
    monkeypatch.setattr(tc, "resolve_tunnel_config",
                        lambda cli_origins=None: {"admin": ["admin1"], "zone": "noit2.com",
                                                  "tunnel_name": "", "credentials_file": "",
                                                  "metrics_port": 0, "idle_timeout_seconds": 1800,
                                                  "drain_seconds": 15, "poll_interval_seconds": 5,
                                                  "routes": [], "enabled": False})
    monkeypatch.setattr(tc, "_current_user", lambda: "alice")
    rid = ta.file_request(p, user="alice", subdomains=["alice.noit2.com"],
                          reason="demo", requested_until=None)
    rc = tc.tunnel_command(Namespace(tunnel_command="approve", id=rid, until="6h",
                                     reason="", kill_origins=False, origins=[],
                                     hold_request=False))
    assert rc == 3
    assert ta.is_approved(p, rid) is False
