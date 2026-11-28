"""Regression for #46303: messaging-gateway leases do not inherit the gateway process's checkout."""

import gateway.run as run_mod
from gateway.config import Platform
from gateway.session import SessionSource
from hermes_cli.active_sessions import active_session_registry_snapshot


def test_gateway_lease_records_no_checkout_even_when_started_inside_one(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    # The gateway process was launched from inside a checkout; its chat sessions are not attached to it.
    monkeypatch.chdir(repo)
    monkeypatch.setenv("HERMES_HOME", str(home))

    runner = run_mod.GatewayRunner.__new__(run_mod.GatewayRunner)
    runner._running_agents = {}
    runner.config = {}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1")

    lease, refusal = runner._claim_active_session_slot("agent:main:telegram:dm:c1", source)
    assert refusal is None and lease is not None, refusal
    try:
        [entry] = active_session_registry_snapshot(registry_home=home)
        assert not (entry.get("metadata") or {}).get("repo_root")
    finally:
        lease.release()
