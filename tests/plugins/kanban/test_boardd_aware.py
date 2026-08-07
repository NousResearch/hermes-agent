"""End-to-end boardd custody tests for the kanban dashboard plugin.

The broker in these tests is the vendored production ``scripts/fleet/boardd.py``
running in a separate process against an isolated fleet board.  No in-process
socket fake is used.
"""

from __future__ import annotations

import importlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

# These are read when kb_client is imported. Keep loss tests bounded even when
# this module is collected outside scripts/run_tests.sh.
os.environ.setdefault("KB_CLIENT_RETRY_DEADLINE_S", "0.4")
os.environ.setdefault("KB_CLIENT_CONNECT_TIMEOUT_S", "0.2")
os.environ.setdefault("KB_CLIENT_READ_TIMEOUT_S", "1")

from hermes_cli import boardd_shim, kb_client
from hermes_cli import kanban_db as kb
from plugins.kanban.dashboard import plugin_api


REPO_ROOT = Path(__file__).resolve().parents[3]
BOARDD = REPO_ROOT / "scripts" / "fleet" / "boardd.py"


class TemporaryBoardd:
    """A real boardd subprocess with deterministic startup and teardown."""

    def __init__(self, db_path: Path, sock_path: Path) -> None:
        self.db_path = db_path
        self.sock_path = sock_path
        self.proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        assert BOARDD.is_file(), BOARDD
        env = os.environ.copy()
        # The daemon owns SQLite directly. Broker custody is a client-process
        # declaration and must not recursively route the daemon back to itself.
        env.pop("HERMES_KANBAN_BROKER", None)
        env["BOARDD_DISKGUARD_MIN_FREE_BYTES"] = "0"
        env["BOARDD_BACKUP_INTERVAL_S"] = "3600"
        env["BOARDD_CHECKPOINT_INTERVAL_S"] = "3600"
        self.proc = subprocess.Popen(
            [
                sys.executable,
                str(BOARDD),
                "--db",
                str(self.db_path),
                "--sock",
                str(self.sock_path),
                "--import-schema",
                "--log-level",
                "WARNING",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 8
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                stderr = (self.proc.stderr.read() if self.proc.stderr else b"").decode(
                    "utf-8", "replace"
                )
                raise RuntimeError(f"boardd exited during startup: {stderr}")
            try:
                client = kb_client.Client(sock_path=str(self.sock_path))
                client.ping()
                client.close()
                return
            except Exception as exc:  # broker socket may not exist yet
                last_error = exc
                time.sleep(0.03)
        self.stop()
        raise RuntimeError(f"boardd did not become ready: {last_error}")

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _reset_thread_client() -> None:
    client = getattr(kb_client._tl, "client", None)
    if client is not None:
        client.close()
    kb_client._tl.client = None


@pytest.fixture(autouse=True)
def short_client_deadline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(kb_client, "_RETRY_DEADLINE_S", 0.4)
    monkeypatch.setattr(kb_client, "_CONNECT_TIMEOUT_S", 0.2)
    monkeypatch.setattr(kb_client, "_READ_TIMEOUT_S", 1.0)
    _reset_thread_client()
    yield
    _reset_thread_client()


def _make_fleet_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_BROKER", raising=False)
    monkeypatch.delenv("BOARDD_SOCK", raising=False)
    kb.create_board("fleet")
    kb.set_current_board("fleet")
    return home, kb.kanban_db_path(board="fleet")


@pytest.fixture
def broker_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _home, db_path = _make_fleet_home(tmp_path, monkeypatch)
    sock_path = tmp_path / "boardd.sock"
    broker = TemporaryBoardd(db_path, sock_path)
    broker.start()
    monkeypatch.setenv("HERMES_KANBAN_BROKER", "1")
    monkeypatch.setenv("BOARDD_SOCK", str(sock_path))

    # Reload under the custody declaration. Import-time install_rebind is the
    # behavior under review, and must cover helpers imported elsewhere.
    mod = importlib.reload(plugin_api)
    assert kb.connect is boardd_shim.connect
    # Pin the runtime socket exactly as the dashboard's first custodied request
    # does. kb_client may have been imported before this fixture set BOARDD_SOCK.
    assert mod._boardd_active(board="fleet") is True
    try:
        yield mod, broker, db_path
    finally:
        broker.stop()


def _create_task(*, title: str, triage: bool = False) -> str:
    with kb.connect_closing(board="fleet") as conn:
        return kb.create_task(conn, title=title, triage=triage)


def _fake_llm(payload: dict) -> SimpleNamespace:
    message = SimpleNamespace(content=json.dumps(payload))
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_standalone_fleet_board_uses_sqlite_without_custody_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A standalone board named fleet remains ordinary writable SQLite."""
    _make_fleet_home(tmp_path, monkeypatch)
    mod = importlib.reload(plugin_api)
    conn = mod._conn(board="fleet")
    try:
        assert isinstance(conn, sqlite3.Connection)
        assert conn.execute("SELECT 1 AS n").fetchone()["n"] == 1
    finally:
        conn.close()


@pytest.mark.live_system_guard_bypass
def test_broker_active_routes_through_real_boardd(broker_runtime) -> None:
    mod, _broker, _db_path = broker_runtime
    conn = mod._conn(board="fleet")
    try:
        assert isinstance(conn, boardd_shim.BrokerConnection)
        row = conn.execute("SELECT 1 AS n").fetchone()
        assert row is not None
        assert row["n"] == 1
    finally:
        conn.close()

    task_id = _create_task(title="broker-owned")
    with kb.connect_closing(board="fleet") as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.title == "broker-owned"


def test_custody_fails_closed_when_real_broker_is_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_fleet_home(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_BROKER", "1")
    monkeypatch.setenv("BOARDD_SOCK", str(tmp_path / "missing.sock"))
    mod = importlib.reload(plugin_api)
    with pytest.raises(
        boardd_shim.BoarddUnavailableError,
        match="broker socket missing",
    ):
        mod._conn(board="fleet")


def test_custody_resolver_failure_is_not_sqlite_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_fleet_home(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_BROKER", "1")
    mod = importlib.reload(plugin_api)

    def broken_resolver(*_args, **_kwargs):
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr(boardd_shim, "routes_to_fleet", broken_resolver)
    with pytest.raises(
        boardd_shim.BoarddUnavailableError,
        match="custody routing could not be resolved.*resolver exploded",
    ):
        mod._conn(board="fleet")


@pytest.mark.live_system_guard_bypass
def test_nested_specify_and_decompose_helpers_use_real_broker(
    broker_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod, _broker, _db_path = broker_runtime
    specify_id = _create_task(title="rough", triage=True)
    decompose_id = _create_task(title="split me", triage=True)

    # If either nested helper bypasses the import-time rebind, this sentinel is
    # reached instead of boardd and the test fails immediately.
    def no_local_connect(*_args, **_kwargs):
        raise AssertionError("nested helper attempted direct SQLite")

    monkeypatch.setattr(boardd_shim, "_ORIG_CONNECT", no_local_connect)

    from agent import auxiliary_client
    from hermes_cli import kanban_decompose

    responses = {
        "triage_specifier": {
            "title": "Specified through boardd",
            "body": "broker-routed body",
        },
        "kanban_decomposer": {
            "fanout": True,
            "rationale": "one child proves the nested transaction path",
            "tasks": [
                {
                    "title": "Broker child",
                    "body": "created through boardd",
                    "assignee": "worker",
                    "parents": [],
                }
            ],
        },
    }
    monkeypatch.setattr(
        auxiliary_client,
        "call_llm",
        lambda *, task, **_kwargs: _fake_llm(responses[task]),
    )
    monkeypatch.setattr(
        kanban_decompose,
        "_load_config",
        lambda: {
            "kanban": {
                "orchestrator_profile": "orchestrator",
                "default_assignee": "worker",
                "auto_promote_children": True,
            }
        },
    )
    monkeypatch.setattr(
        kanban_decompose,
        "_build_roster",
        lambda: ([], {"orchestrator", "worker"}),
    )
    monkeypatch.setattr(
        kanban_decompose, "_resolve_orchestrator_profile", lambda _cfg: "orchestrator"
    )
    monkeypatch.setattr(
        kanban_decompose, "_resolve_default_assignee", lambda _cfg: "worker"
    )

    specified = mod.specify_task_endpoint(
        specify_id, mod.SpecifyBody(author="test"), board="fleet"
    )
    decomposed = mod.decompose_task_endpoint(
        decompose_id, mod.DecomposeBody(author="test"), board="fleet"
    )
    assert specified["ok"] is True
    assert decomposed["ok"] is True
    assert len(decomposed["child_ids"]) == 1

    with kb.connect_closing(board="fleet") as conn:
        specified_task = kb.get_task(conn, specify_id)
        assert specified_task is not None
        assert specified_task.title == "Specified through boardd"
        child = kb.get_task(conn, decomposed["child_ids"][0])
        assert child is not None
        assert child.title == "Broker child"


@pytest.mark.live_system_guard_bypass
def test_mid_bulk_real_broker_loss_never_falls_back_to_sqlite(
    broker_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod, broker, db_path = broker_runtime
    first = _create_task(title="first")
    second = _create_task(title="second")
    original_get_task = kb.get_task
    calls = 0

    def stop_before_second(conn, task_id):
        nonlocal calls
        calls += 1
        if calls == 2:
            broker.stop()
        return original_get_task(conn, task_id)

    monkeypatch.setattr(kb, "get_task", stop_before_second)
    result = mod.bulk_update(
        mod.BulkTaskBody(ids=[first, second], priority=77), board="fleet"
    )
    assert result["results"][0] == {"id": first, "ok": True}
    assert result["results"][1]["ok"] is False
    assert "boardd" in result["results"][1]["error"].lower()

    # Read the stopped broker's DB directly only after process teardown. The
    # first item committed through boardd; the second was never written locally.
    conn = sqlite3.connect(str(db_path))
    try:
        priorities = dict(
            conn.execute(
                "SELECT id, priority FROM tasks WHERE id IN (?, ?)",
                (first, second),
            ).fetchall()
        )
    finally:
        conn.close()
    assert priorities[first] == 77
    assert priorities[second] != 77
