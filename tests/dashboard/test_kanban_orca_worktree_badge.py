"""Orca worktree badge backend (ORCA-06) — state mapping, RPC client, reveal.

Pins the contract between the kanban dashboard badge and Orca's local runtime
RPC (verified against stablyai/orca@main source, 2026-10-26):

* ``orca-runtime.json`` metadata → unix-socket endpoint + auth token;
  newline-delimited JSON envelopes (``src/cli/runtime/transport.ts``).
* ``worktree.ps`` rows collapse Orca's 5-state ``RuntimeWorktreeStatus``
  (active/working/permission/done/inactive) onto the badge's mandated
  active/idle/complete vocabulary.
* Reveal (badge click) = ``worktree.activate`` RPC with selector
  ``id:<worktree_id>`` — the same call Orca's mobile companion issues for
  its worktree "open" action. There is no ``orca://worktree/...`` URL route
  (Orca's open-url handler only routes skill-share links), so the RPC IS
  the deep link.
* Unreachable runtime degrades to the orca-status skill journal, then to an
  honest ``unknown`` — never a fabricated state.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DASHBOARD = REPO / "plugins" / "kanban" / "dashboard"


def _load_orca_worktree():
    """Load orca_worktree.py by path (same import context independence the
    plugin_api loader uses)."""
    mod_name = "orca_worktree_under_test"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, DASHBOARD / "orca_worktree.py")
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError("cannot load orca_worktree.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


ow = _load_orca_worktree()


# ---------------------------------------------------------------------------
# Fake Orca runtime: unix socket speaking the verified envelope protocol
# ---------------------------------------------------------------------------

class FakeOrcaRuntime:
    """Serves worktree.ps / worktree.activate over a unix socket.

    The socket binds under a short /tmp path: macOS caps AF_UNIX paths at
    104 bytes and pytest's tmp_path is far deeper than that.
    """

    _seq = 0

    def __init__(self, tmp: Path, *, ps_rows=None, runtime_id="rt-test", auth_token="tok-test",
                 pre_frames=None, ps_error=None, activate_error=None):
        self.tmp = tmp
        FakeOrcaRuntime._seq += 1
        self._short_dir = Path(f"/tmp/orca-badge-t-{os.getpid()}-{FakeOrcaRuntime._seq}")
        self._short_dir.mkdir(exist_ok=True)
        self.sock_path = str(self._short_dir / "rt.sock")
        self.ps_rows = ps_rows if ps_rows is not None else []
        self.runtime_id = runtime_id
        self.auth_token = auth_token
        self.pre_frames = pre_frames or []
        self.ps_error = ps_error
        self.activate_error = activate_error
        self.requests: list[dict] = []
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.sock_path)
        self._server.listen(4)
        self._server.settimeout(5.0)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    # -- protocol helpers ---------------------------------------------------
    def _line(self, conn: socket.socket, obj: dict) -> None:
        conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))

    def _respond(self, req: dict, result=None, *, ok=True, error=None):
        frame = {"id": req.get("id"), "ok": ok, "_meta": {"runtimeId": self.runtime_id}}
        if ok:
            frame["result"] = result
        else:
            frame["error"] = error or {"code": "internal", "message": "boom"}
        return frame

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except (OSError, TimeoutError):
                continue
            with conn:
                conn.settimeout(5.0)
                try:
                    data = conn.recv(65536)
                except OSError:
                    continue
                try:
                    req = json.loads(data.decode("utf-8").strip())
                except ValueError:
                    continue
                self.requests.append(req)
                for frame in self.pre_frames:
                    try:
                        self._line(conn, frame)
                    except OSError:
                        return
                method = req.get("method")
                if method == "worktree.ps":
                    if self.ps_error is not None:
                        resp = self._respond(req, ok=False, error=self.ps_error)
                    else:
                        resp = self._respond(req, {"worktrees": self.ps_rows, "totalCount": len(self.ps_rows), "truncated": False})
                elif method == "worktree.activate":
                    if self.activate_error is not None:
                        resp = self._respond(req, ok=False, error=self.activate_error)
                    else:
                        resp = self._respond(req, {"ok": True})
                else:
                    resp = self._respond(req, ok=False, error={"code": "method_not_found", "message": f"no {method}"})
                try:
                    self._line(conn, resp)
                except OSError:
                    pass

    # -- metadata file --------------------------------------------------------
    def metadata_path(self) -> Path:
        path = self.tmp / "orca-runtime.json"
        path.write_text(json.dumps({
            "runtimeId": self.runtime_id,
            "pid": os.getpid(),
            "transports": [{"kind": "unix", "endpoint": self.sock_path}],
            "authToken": self.auth_token,
            "startedAt": int(time.time()),
        }), encoding="utf-8")
        return path

    def stop(self):
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass
        try:
            os.unlink(self.sock_path)
        except OSError:
            pass
        try:
            self._short_dir.rmdir()
        except OSError:
            pass


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    runtime = FakeOrcaRuntime(tmp_path, ps_rows=[
        {
            "worktreeId": "repo-42::/Users/me/orca/wt-active",
            "status": "working",
            "path": "/Users/me/orca/wt-active",
            "branch": "fix/login",
            "displayName": "fix login",
            "isActive": True,
            "workspaceStatus": "in-progress",
            "liveTerminalCount": 1,
            "lastActivityAt": 1792980000,
            "agents": [{"state": "working", "agentType": "claude-code", "stateStartedAt": 1792970000}],
        },
        {
            "worktreeId": "repo-42::/Users/me/orca/wt-idle",
            "status": "inactive",
            "path": "/Users/me/orca/wt-idle",
            "agents": [],
        },
        {
            "worktreeId": "repo-42::/Users/me/orca/wt-done",
            "status": "done",
            "path": "/Users/me/orca/wt-done",
            "agents": [{"state": "done", "agentType": "codex"}],
        },
    ])
    monkeypatch.setenv("ORCA_RUNTIME_METADATA", str(runtime.metadata_path()))
    # Isolate the journal fallback layer from the machine's real skill state.
    journal = tmp_path / "empty-journal"
    journal.mkdir(exist_ok=True)
    monkeypatch.setenv("ORCA_STATE_DIR", str(journal))
    yield runtime
    runtime.stop()


@pytest.fixture
def no_runtime(tmp_path, monkeypatch):
    """No runtime metadata (Orca not running) and no journal entries."""
    empty = tmp_path / "no-userdata"
    empty.mkdir()
    monkeypatch.setenv("ORCA_USER_DATA_DIR", str(empty))
    monkeypatch.delenv("ORCA_RUNTIME_METADATA", raising=False)
    journal = tmp_path / "no-journal"
    journal.mkdir(exist_ok=True)
    monkeypatch.setenv("ORCA_STATE_DIR", str(journal))
    return empty


# ---------------------------------------------------------------------------
# State mapping (Orca RuntimeWorktreeStatus -> badge vocabulary)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,expected", [
    ("active", ow.BADGE_ACTIVE),
    ("working", ow.BADGE_ACTIVE),
    ("permission", ow.BADGE_ACTIVE),
    ("inactive", ow.BADGE_IDLE),
    ("done", ow.BADGE_COMPLETE),
])
def test_derive_badge_state_maps_runtime_status(status, expected):
    state, detail = ow.derive_badge_state({"status": status, "agents": []})
    assert state == expected
    assert detail["orca_status"] == status


@pytest.mark.parametrize("agents,expected", [
    ([{"state": "working"}], ow.BADGE_ACTIVE),
    ([{"state": "blocked"}], ow.BADGE_ACTIVE),
    ([{"state": "waiting"}], ow.BADGE_ACTIVE),
    ([{"state": "done"}], ow.BADGE_COMPLETE),
    ([], ow.BADGE_IDLE),
])
def test_derive_badge_state_agent_fallback_when_status_missing(agents, expected):
    state, detail = ow.derive_badge_state({"agents": agents})
    assert state == expected
    assert detail["orca_status"] is None


# ---------------------------------------------------------------------------
# Layer 1: live worktree.ps RPC
# ---------------------------------------------------------------------------

def test_get_status_live_rpc_active(fake_runtime):
    out = ow.get_worktree_status("repo-42::/Users/me/orca/wt-active")
    assert out["state"] == ow.BADGE_ACTIVE
    assert out["source"] == "orca-runtime-rpc"
    assert out["orca_status"] == "working"
    assert out["agent_states"] == ["working"]
    assert out["model"] == "claude-code"
    assert out["is_active_view"] is True
    assert out["workspace_status"] == "in-progress"
    assert out["path"] == "/Users/me/orca/wt-active"
    assert out["last_activity_at"] == 1792980000
    assert out["error_code"] is None


def test_get_status_live_rpc_idle_and_complete(fake_runtime):
    idle = ow.get_worktree_status("repo-42::/Users/me/orca/wt-idle")
    assert idle["state"] == ow.BADGE_IDLE
    assert idle["agent_states"] == []
    done = ow.get_worktree_status("repo-42::/Users/me/orca/wt-done")
    assert done["state"] == ow.BADGE_COMPLETE
    assert done["agent_states"] == ["done"]


def test_get_status_unknown_when_runtime_does_not_list_worktree(fake_runtime):
    out = ow.get_worktree_status("repo-42::/Users/me/orca/wt-ghost")
    assert out["state"] == ow.BADGE_UNKNOWN
    assert out["error_code"] == ow.ERR_UNKNOWN_WORKTREE
    assert "repo-42::/Users/me/orca/wt-ghost" in out["error"]


def test_rpc_sends_verified_envelope(fake_runtime):
    ow.get_worktree_status("repo-42::/Users/me/orca/wt-active")
    req = fake_runtime.requests[-1]
    assert req["method"] == "worktree.ps"
    assert req["params"] == {}
    assert req["authToken"] == "tok-test"          # metadata auth token rides the frame
    assert isinstance(req["id"], str) and req["id"]  # correlation id present


def test_rpc_skips_keepalive_and_foreign_frames(tmp_path, monkeypatch):
    runtime = FakeOrcaRuntime(
        tmp_path,
        ps_rows=[{"worktreeId": "wt-1", "status": "active", "agents": []}],
        pre_frames=[
            {"_keepalive": True},
            {"id": "someone-elses", "ok": True, "result": {"noise": True}, "_meta": {"runtimeId": "rt-test"}},
        ],
    )
    monkeypatch.setenv("ORCA_RUNTIME_METADATA", str(runtime.metadata_path()))
    try:
        out = ow.get_worktree_status("wt-1")
        assert out["state"] == ow.BADGE_ACTIVE
    finally:
        runtime.stop()


def test_rpc_error_envelope_raises_structured(tmp_path, monkeypatch):
    runtime = FakeOrcaRuntime(tmp_path, ps_error={"code": "worktree_not_found", "message": "nope"})
    monkeypatch.setenv("ORCA_RUNTIME_METADATA", str(runtime.metadata_path()))
    try:
        with pytest.raises(ow.OrcaRuntimeError) as excinfo:
            ow.rpc_call(ow.read_runtime_metadata(), "worktree.ps", {})
        assert excinfo.value.code == "worktree_not_found"
    finally:
        runtime.stop()


def test_rpc_identity_change_rejected(tmp_path, monkeypatch):
    runtime = FakeOrcaRuntime(tmp_path, runtime_id="rt-other")  # metadata says rt-test? no — same obj
    # Build metadata manually with a DIFFERENT runtimeId than the server stamps.
    meta_path = tmp_path / "orca-runtime.json"
    meta_path.write_text(json.dumps({
        "runtimeId": "rt-client", "pid": os.getpid(),
        "transports": [{"kind": "unix", "endpoint": runtime.sock_path}],
        "authToken": "tok-test", "startedAt": int(time.time()),
    }), encoding="utf-8")
    monkeypatch.setenv("ORCA_RUNTIME_METADATA", str(meta_path))
    try:
        with pytest.raises(ow.OrcaRuntimeError) as excinfo:
            ow.rpc_call(ow.read_runtime_metadata(), "worktree.ps", {})
        assert excinfo.value.code == ow.ERR_RUNTIME_UNAVAILABLE
    finally:
        runtime.stop()


# ---------------------------------------------------------------------------
# Layer 2: orca-status skill journal fallback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("skill_status,expected", [
    ("running", ow.BADGE_ACTIVE),
    ("done", ow.BADGE_COMPLETE),
    ("failed", ow.BADGE_COMPLETE),
    ("pending", ow.BADGE_IDLE),
])
def test_journal_fallback_maps_skill_statuses(tmp_path, monkeypatch, no_runtime, skill_status, expected):
    state_dir = tmp_path / "journal"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(json.dumps({
        "wt-journal-1": {"status": skill_status, "model": "hermes"},
    }), encoding="utf-8")
    monkeypatch.setenv("ORCA_STATE_DIR", str(state_dir))
    out = ow.get_worktree_status("wt-journal-1")
    assert out["state"] == expected
    assert out["source"] == "orca-status-state"


def test_fully_unknown_when_no_runtime_and_no_journal(no_runtime):
    out = ow.get_worktree_status("wt-nothing")
    assert out["state"] == ow.BADGE_UNKNOWN
    assert out["source"] == "none"
    assert out["error_code"] == ow.ERR_RUNTIME_UNAVAILABLE


# ---------------------------------------------------------------------------
# Reveal (badge click -> focus worktree in Orca UI)
# ---------------------------------------------------------------------------

def test_reveal_sends_worktree_activate(fake_runtime):
    out = ow.reveal_worktree("repo-42::/Users/me/orca/wt-active")
    assert out["ok"] is True
    assert out["launched"] is False
    req = fake_runtime.requests[-1]
    assert req["method"] == "worktree.activate"
    assert req["params"] == {
        "worktree": "id:repo-42::/Users/me/orca/wt-active",
        "navigation": "caller",
        "notifyClients": False,
    }


def test_reveal_launch_fallback_end_to_end(tmp_path, monkeypatch):
    """Runtime metadata absent → launch Orca (fake CLI) → metadata appears →
    retry activate succeeds. The fake CLI writes the metadata file the real
    `orca open` sequence would make available once the app is up."""
    runtime = FakeOrcaRuntime(tmp_path, ps_rows=[])
    launcher = tmp_path / "fake-orca-cli"
    meta_path = tmp_path / "runtime-dir" / "orca-runtime.json"
    launcher.write_text(
        "#!/bin/sh\nmkdir -p " + str(meta_path.parent) + "\n"
        "cat > " + str(meta_path) + " <<'EOF'\n"
        + json.dumps({
            "runtimeId": "rt-test", "pid": os.getpid(),
            "transports": [{"kind": "unix", "endpoint": runtime.sock_path}],
            "authToken": "tok-test", "startedAt": int(time.time()),
        }) + "\nEOF\n",
        encoding="utf-8",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("ORCA_CLI_COMMAND", str(launcher))
    monkeypatch.delenv("ORCA_RUNTIME_METADATA", raising=False)
    monkeypatch.setenv("ORCA_USER_DATA_DIR", str(tmp_path / "runtime-dir"))
    try:
        out = ow.reveal_worktree("wt-1")
        assert out["ok"] is True
        assert out["launched"] is True
        req = runtime.requests[-1]
        assert req["method"] == "worktree.activate"
        assert req["params"]["worktree"] == "id:wt-1"
    finally:
        runtime.stop()


def test_reveal_reports_unreachable_structured(no_runtime):
    out = ow.reveal_worktree("wt-1")
    assert out["ok"] is False
    assert out["error_code"] in (ow.ERR_RUNTIME_UNAVAILABLE, ow.ERR_LAUNCH_FAILED)
    assert out["error"]  # human-readable, never a traceback


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "   ", None, 42, "has space", "·bad"])
def test_invalid_worktree_ids_rejected(bad):
    with pytest.raises(ow.OrcaRuntimeError) as excinfo:
        ow.validate_worktree_id(bad)
    assert excinfo.value.code == ow.ERR_INVALID_WORKTREE_ID


@pytest.mark.parametrize("good", [
    "wt-123", "repo-42::/Users/me/orca/wt-active", "orca-wt-a.b:c/d",
])
def test_valid_worktree_ids_accepted(good):
    assert ow.validate_worktree_id(good) == good


# ---------------------------------------------------------------------------
# HTTP endpoints (plugin_api router)
# ---------------------------------------------------------------------------

def _load_plugin_api(mod_name="hermes_dashboard_plugin_orca_badge_test"):
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, DASHBOARD / "plugin_api.py")
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError("cannot load plugin_api.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def api_client(fake_runtime):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    api = _load_plugin_api()
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app)


def test_status_endpoint_happy_path(api_client):
    r = api_client.get("/orca/worktree-status", params={"worktree_id": "repo-42::/Users/me/orca/wt-active"})
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "active"
    assert body["source"] == "orca-runtime-rpc"


def test_status_endpoint_invalid_id_is_structured(api_client):
    r = api_client.get("/orca/worktree-status", params={"worktree_id": "  "})
    assert r.status_code == 200  # degrades, never 500s
    body = r.json()
    assert body["state"] == "unknown"
    assert body["error_code"] == "invalid_worktree_id"


def test_reveal_endpoint_ok(api_client):
    r = api_client.post("/orca/worktree-reveal", json={"worktree_id": "repo-42::/Users/me/orca/wt-active"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["method"] == "worktree.activate"


def test_reveal_endpoint_unreachable_is_structured(no_runtime, monkeypatch):
    # no_runtime fixture only patches env inside its test; re-apply here.
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    api = _load_plugin_api()
    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app)
    r = client.post("/orca/worktree-reveal", json={"worktree_id": "wt-1"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["worktree_id"] == "wt-1"


# ---------------------------------------------------------------------------
# Regression: board payload carries worktree_id (badge visibility contract)
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def test_board_payload_carries_worktree_id(isolated_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        linked = kb.create_task(
            conn, title="linked card", created_by="test",
            worktree_id="repo-42::/Users/me/orca/wt-active")
        plain = kb.create_task(conn, title="plain card", created_by="test")

    api = _load_plugin_api()
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app)

    r = client.get("/board")
    assert r.status_code == 200
    tasks = [t for col in r.json()["columns"] for t in col["tasks"]]
    by_id = {t["id"]: t for t in tasks}
    assert by_id[linked]["worktree_id"] == "repo-42::/Users/me/orca/wt-active"
    assert by_id[plain]["worktree_id"] is None
