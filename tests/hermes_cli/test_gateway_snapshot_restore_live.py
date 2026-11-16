"""CLI-16 in #134889: restore offline, then keep chatting on the restored authority."""
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
from types import SimpleNamespace
import uuid

import pytest

from hermes_cli.backup import create_quick_snapshot, restore_quick_snapshot
from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.cli_init_mixin import CLIInitMixin
from hermes_cli.gateway_connection import GatewayConnection
from hermes_cli.gateway_runtime import discover_gateway_endpoint
from hermes_cli.gateway_session_store import GatewaySessionStore
from hermes_cli.profiles import parked_marker_path
from tests.gateway.fixtures.local_recovery_probe import Model, child_env
from tests.waiting import wait_until


@pytest.fixture
def local_gateway(tmp_path, monkeypatch):
    """Only replace process launch: sockets, authority, SQLite restore and model I/O are real."""
    user = tmp_path / "user"
    user.mkdir()
    peer = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{peer.server_port}/v1"
    for key, value in {"OPENAI_API_KEY": "loopback-only", "OPENAI_BASE_URL": url}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: user))
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    root = Path(__file__).resolve().parents[2]
    children, logs = [], []

    def spawn(home, *, deadline, idle_exit=False):
        env = child_env()
        env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
                   PYTHONPATH=str(root), OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=url,
                   # This child owns a disposable default root; it is reaped below.
                   HERMES_STATE_DB_GUARD_BYPASS="1")
        log = (home / f"fixture-gateway-{len(children)}.log").open("w+", encoding="utf-8")
        logs.append(log)
        process = subprocess.Popen([sys.executable, "-m", "gateway.run"], cwd=root, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        children.append(process)
        return process

    monkeypatch.setattr("hermes_cli.gateway_runtime_start.spawn_unmanaged_gateway", spawn)
    monkeypatch.setattr(GatewayConnection, "_shared", None)

    def configure(multiplex):
        home = user / ".hermes"
        config = {"gateway": {"multiplex_profiles": multiplex},
                  "model": {"provider": "custom", "default": "fixture", "base_url": url},
                  "auxiliary": {"title_generation": {"enabled": False}},
                  "platform_toolsets": {"cli": []}}
        homes = [home, home / "profiles" / "satellite"] if multiplex else [home]
        for target in homes:
            target.mkdir(parents=True, mode=0o700)
            (target / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
            (target / ".env").write_text("OPENAI_API_KEY=loopback-only\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(homes[-1]))
        return homes[-1]

    try:
        yield configure
    finally:
        connection = GatewayConnection._shared
        if connection is not None:
            connection.close()
        for process in children:
            if process.poll() is not None and process.returncode:
                log = logs[children.index(process)]
                log.seek(0)
                print(log.read())
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
        for log in logs:
            log.seek(0)
            print(log.read()[-6000:])
            log.close()
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)


def _turn(store, session_id, text):
    request_id = uuid.uuid4().hex
    store.call("prompt.submit", session_id=session_id, input_id=request_id, text=text)
    def settled():
        with store._db()._read_ctx() as db:
            row = db.execute("SELECT status FROM session_admissions WHERE request_id=?", (request_id,)).fetchone()
        return row is not None and row[0] == "terminal"
    wait_until(settled, timeout=40, interval=.05, what=f"gateway turn {text} to settle")
    assert any(text in str(row.get("content")) and row.get("role") == "assistant"
               for row in store.get_messages_as_conversation(session_id))


def _epoch(store):
    with store._db()._read_ctx() as db:
        return db.execute("SELECT epoch FROM runtime_epoch WHERE singleton=1").fetchone()[0]


def _default_profile_state(home):
    path = home.parent.parent / "state.db"
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        return (db.execute("SELECT epoch FROM runtime_epoch WHERE singleton=1").fetchone(),
                db.execute("SELECT id FROM sessions ORDER BY id").fetchall(),
                db.execute("SELECT session_id,role,content FROM messages ORDER BY id").fetchall())


def _legacy_row(home):
    """A pre-authority CLI row the gateway has not imported yet, as the old in-process CLI wrote it."""
    from hermes_state import SessionDB
    db = SessionDB(home / "state.db")
    try:
        db.create_session("legacy-cli", source="cli", model="fixture")
        db.append_message("legacy-cli", role="user", content="BEFORE_SNAPSHOT")
        db.append_message("legacy-cli", role="assistant", content="BEFORE_SNAPSHOT answer")
        db.end_session("legacy-cli", "cli_exit")
    finally:
        db.close()
    return "legacy-cli"


@pytest.mark.platforms("posix")  # The fixture exercises the POSIX default/named profile layout.
@pytest.mark.parametrize("multiplex, mode", [(False, "retained"), (True, "missing"), (False, "legacy-leased")],
                         ids=["standalone-resume-restored", "satellite-create-missing", "legacy-reimport-leased"])
def test_snapshot_restore_rejoins_the_restored_authority(local_gateway, multiplex, mode, capsys, monkeypatch):
    from cli import HermesCLI
    from types import MethodType
    home = local_gateway(multiplex)
    legacy = _legacy_row(home) if mode == "legacy-leased" else None
    # A legacy snapshot predates the gateway's import, so the rejoin must import the row again.
    snapshot = create_quick_snapshot(hermes_home=home) if legacy else None
    connection = GatewayConnection.shared()
    store = GatewaySessionStore(connection)
    sid = store.attach(legacy)["stored_session_id"] if legacy else store.create({"toolsets": []})
    if not legacy:
        _turn(store, sid, "BEFORE_SNAPSHOT")
        snapshot = create_quick_snapshot(hermes_home=home)
    assert snapshot
    if mode == "missing":
        store.end_session(sid)
        sid = store.create({"toolsets": []})
    _turn(store, sid, "AFTER_SNAPSHOT")
    # The direct API must still refuse a live owner, without rolling back its newer rows.
    history_before = store.get_messages_as_conversation(sid)
    assert not restore_quick_snapshot(snapshot, hermes_home=home)
    assert store.get_messages_as_conversation(sid) == history_before
    old_reader = store._reader
    assert old_reader is not None
    from hermes_state_registry import acquire
    shared_reader = acquire(home / "state.db")
    before = discover_gateway_endpoint(home).endpoint
    assert before is not None
    before_epoch = _epoch(store)
    default_before = _default_profile_state(home) if multiplex else None
    cli = SimpleNamespace(_session_db=store, _gateway_session_open=True, session_id=sid, config={},
                          conversation_history=[{"role": "user", "content": "AFTER_SNAPSHOT"}],
                          agent=object(), model="fixture", _active_session_lease=None)
    for name in ("_claim_active_session", "_release_active_session"):
        setattr(cli, name, MethodType(getattr(HermesCLI, name), cli))
    # The interactive CLI holds its real active-session lease for the whole chat.
    assert cli._claim_active_session("cli", stderr=True)
    retained = mode != "missing"
    rescans = []
    if multiplex:
        from gateway.control_socket import rescan_gateway_profiles
        from hermes_cli import profiles_remove
        unowned = profiles_remove.wait_profile_unowned

        def rescan_mid_handoff(target):
            # The host's periodic rescan lands between the unserve and the restore.
            rescans.append(rescan_gateway_profiles(target.parent.parent))
            return unowned(target)
        monkeypatch.setattr(profiles_remove, "wait_profile_unowned", rescan_mid_handoff)
    try:
        CLICommandsMixin._snapshot_restore(cli, ["/snapshot", "restore", snapshot])

        out = capsys.readouterr().out
        assert connection.final, out
        assert "Restored state from" in out and "failed" not in out and "not served again" not in out, out
        assert cli._active_session_lease is not None
        assert cli._active_session_lease.session_id == cli.session_id
    finally:
        cli._release_active_session()
    assert store.connection is not connection and not store.connection.final
    assert old_reader._wrapped._conn is None and shared_reader._conn is None
    assert cli.agent is None
    from hermes_cli import gateway_tools
    assert gateway_tools._view.connection is store.connection  # /agents, /stop and the status bar
    assert (cli.session_id == sid) is retained
    history = json.dumps(cli.conversation_history)
    assert "AFTER_SNAPSHOT" not in history
    assert ("BEFORE_SNAPSHOT" in history) is retained
    state = store.connection.state(cli.session_id)
    assert state.subscription_id and state.execution_generation is not None
    after = discover_gateway_endpoint(home).endpoint
    assert after is not None
    assert _epoch(store) > before_epoch
    if multiplex:
        assert rescans and home.name not in rescans[0]["added"], rescans
        assert not parked_marker_path(home).exists()  # served again after the restore
        assert before.instance_id == after.instance_id
        assert _default_profile_state(home) == default_before
    else:
        assert before.instance_id != after.instance_id
        assert after.authority_epoch > before.authority_epoch
    _turn(store, cli.session_id, "AFTER_RESTORE")
    context = store.call("session.context_breakdown", session_id=cli.session_id)
    assert context is not None
    previous = cli.session_id
    fresh = CLIInitMixin._new_session_id(cli, None)
    assert fresh != previous
    _turn(store, fresh, "AFTER_NEW")
