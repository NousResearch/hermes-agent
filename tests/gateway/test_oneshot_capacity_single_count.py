"""A ``hermes chat --oneshot`` turn consumes exactly one running-capacity slot.

The oneshot CLI is a GatewayAgent proxy: it claims a ``cli`` ownership lease
(cli_single_query -> HermesCLI._claim_active_session) and the gateway then
claims a ``gateway:local`` lease for the turn it actually executes. Before the
fix both rows counted, so with ``max_concurrent_sessions: 1`` the oneshot
refused its own turn, and with ``2`` one oneshot filled the whole cap.
"""

from cli import HermesCLI
from gateway.config import Platform
from gateway.session import SessionSource, build_session_key
from hermes_cli import active_sessions
from tests.gateway.test_max_concurrent_sessions import _make_runner, _make_source


def _oneshot_cli(cfg: dict) -> HermesCLI:
    cli = object.__new__(HermesCLI)
    cli.session_id = "oneshot-cli-session"
    cli.config = cfg
    cli._active_session_lease = None
    cli._console_print = lambda text: None
    return cli


def test_oneshot_proxy_and_gateway_turn_count_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    cfg = {"max_concurrent_sessions": 1}
    cli = _oneshot_cli(cfg)
    runner = _make_runner(max_concurrent_sessions=1)
    local = SessionSource(platform=Platform.LOCAL, chat_id="local", chat_type="dm", user_id="local-user")
    local_key = build_session_key(local)

    # The oneshot entrypoint claims with stderr=True (quiet mode).
    assert cli._claim_active_session("cli", stderr=True) is True
    assert "Active session limit" not in capsys.readouterr().err
    turn_lease, refusal = runner._claim_active_session_slot(local_key, local)
    try:
        assert refusal is None, refusal
        assert turn_lease is not None
        rows = {row["surface"]: row for row in active_sessions.active_session_registry_snapshot()}
        assert set(rows) == {"cli", "gateway:local"}
        assert rows["cli"]["capacity_active"] is False
        assert rows["gateway:local"]["capacity_active"] is True

        # The cap is still enforced: the oneshot's running turn holds the only slot.
        other = _make_source("telegram-chat")
        blocked, refusal = runner._claim_active_session_slot(build_session_key(other), other)
        assert blocked is None
        assert getattr(refusal, "reason", None) == active_sessions.MAX_CONCURRENT_SESSIONS
    finally:
        if turn_lease is not None:
            active_sessions.release_active_session(turn_lease)

    # Once the turn settles, the idle proxy ownership leaves room for other work.
    other = _make_source("telegram-chat")
    next_turn, refusal = runner._claim_active_session_slot(build_session_key(other), other)
    try:
        assert refusal is None and next_turn is not None
    finally:
        if next_turn is not None:
            active_sessions.release_active_session(next_turn)
        cli._active_session_lease.release()
    assert active_sessions.active_session_registry_snapshot() == []
