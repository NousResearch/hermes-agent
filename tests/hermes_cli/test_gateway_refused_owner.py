"""#139845 / #140252: a same-user gateway that refuses the client is named, not ``authorization``.

#139845: an elevated gateway's pipe/process refuses the non-elevated Desktop (``OpenProcess`` err=5).
#140252: a session-0 S4U boot-task gateway won the control pipe; the interactive Desktop can never
reach it. Both surfaced as ``Gateway inaccessible (authorization)``, which reads as a credential
problem. The probe facts and the host fact (``windows=True``) are passed in here, never faked; ``test_refused_owner_windows.py``
runs the real probes on Windows CI.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from gateway import runtime_bootstrap_windows as native
from hermes_cli import gateway_runtime as runtime
from hermes_cli.gateway_runtime_cli import ensure_exit_code

# The owner stand-in is a stub ``hermes gateway run`` child that each test reaps itself.
pytestmark = pytest.mark.spawns_gateway_lookalike


@pytest.mark.parametrize(("facts", "expected"), [
    # #140252: owner in session 0, Desktop in the interactive session 1.
    ({"own_session": 1, "owner_session": 0, "own_elevated": False, "owner_elevated": None}, "gateway_other_session"),
    ({"own_session": 1, "owner_session": 0, "own_elevated": True, "owner_elevated": True}, "gateway_other_session"),
    # #139845: same session, owner elevated (or its token unopenable from a filtered client).
    ({"own_session": 1, "owner_session": 1, "own_elevated": False, "owner_elevated": True}, "gateway_elevated"),
    ({"own_session": 1, "owner_session": 1, "own_elevated": False, "owner_elevated": None}, "gateway_elevated"),
    # Nothing to name: keep the generic code.
    ({"own_session": 1, "owner_session": 1, "own_elevated": False, "owner_elevated": False}, None),
    ({"own_session": 1, "owner_session": 1, "own_elevated": True, "owner_elevated": None}, None),
    ({"own_session": 1, "owner_session": None, "own_elevated": False, "owner_elevated": None}, None),
])
def test_classify_refused_owner(facts, expected):
    assert native.classify_refused_owner(**facts) == expected


def _gateway_child(tmp_path: Path):
    """A real live process whose command line is a gateway's (``.../hermes gateway run``)."""
    entry = tmp_path / "bin" / "hermes"
    entry.parent.mkdir(exist_ok=True)
    entry.write_text("import sys\nsys.stdin.read()\n", encoding="utf-8")
    return subprocess.Popen([sys.executable, str(entry), "gateway", "run"], stdin=subprocess.PIPE)


def _record(home: Path, pid: int, **extra) -> dict:
    from gateway.status import _get_process_start_time
    return {"pid": pid, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
            "start_time": _get_process_start_time(pid), "hermes_home": str(home), **extra}


@pytest.fixture()
def gateway_owner(tmp_path):
    child = _gateway_child(tmp_path)
    yield child
    child.stdin.close()
    child.wait(timeout=10)


def _refusing_owner(monkeypatch, tmp_path: Path, named, owner):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / "gateway.pid").write_text(json.dumps(_record(home, owner.pid)), encoding="utf-8")
    seen = []
    monkeypatch.setattr(native, "probe_refused_owner", lambda pid: seen.append(pid) or named)
    return home, seen


def test_refusal_names_the_other_session_owner_with_its_remedy(monkeypatch, tmp_path, gateway_owner):
    pid = gateway_owner.pid
    named = ("gateway_other_session", native.refused_owner_remedy("gateway_other_session", pid, 0))
    home, seen = _refusing_owner(monkeypatch, tmp_path, named, gateway_owner)
    result = runtime._refused(home, windows=True)
    assert seen == [pid]
    assert (result.state, result.reason_code) == ("inaccessible", "gateway_other_session")
    assert result.detail and f"PID {pid}" in result.detail and "session 0" in result.detail
    assert ensure_exit_code(result) == 4


def test_refusal_names_the_elevated_owner_with_its_remedy(monkeypatch, tmp_path, gateway_owner):
    named = ("gateway_elevated", native.refused_owner_remedy("gateway_elevated", gateway_owner.pid, 1))
    home, _ = _refusing_owner(monkeypatch, tmp_path, named, gateway_owner)
    result = runtime._refused(home, windows=True)
    assert result.reason_code == "gateway_elevated"
    assert result.detail and "administrator" in result.detail and "hermes gateway stop" in result.detail


def test_unnamed_or_unprobeable_refusal_keeps_authorization(monkeypatch, tmp_path, gateway_owner):
    home, _ = _refusing_owner(monkeypatch, tmp_path, None, gateway_owner)
    assert runtime._refused(home, windows=True).reason_code == "authorization"
    monkeypatch.setattr(native, "probe_refused_owner", lambda pid: (_ for _ in ()).throw(OSError("x")))
    assert runtime._refused(home, windows=True).reason_code == "authorization"
    (home / "gateway.pid").unlink()
    assert runtime._refused(home, windows=True).reason_code == "authorization"


def test_discovery_routes_a_refused_control_channel_through_refused(monkeypatch, tmp_path):
    """The real discovery path: ``query_identify`` raising PermissionError is what Windows' pipe
    ``CreateFile`` err=5 / peer-identity refusal produces; it reaches ``_refused`` for this home.
    The Windows naming itself is ``_refused(home, windows=True)`` above."""
    from hermes_cli import gateway_runtime_discovery as discovery
    home = tmp_path / "home"
    home.mkdir()
    seen = []
    monkeypatch.setattr(runtime, "_refused", lambda h: seen.append(Path(h)) or runtime.GatewayDiscovery(
        "inaccessible", reason_code="gateway_elevated"))
    monkeypatch.setattr(discovery, "query_identify",
                        lambda *_a, **_k: (_ for _ in ()).throw(PermissionError("runtime control pipe inaccessible")))
    result = runtime.discover_gateway_endpoint(home, timeout=1)
    assert (result.state, result.reason_code) == ("inaccessible", "gateway_elevated")
    assert [p.resolve() for p in seen] == [home.resolve()]


@pytest.mark.platforms("posix")
def test_off_windows_a_refusal_stays_authorization_without_probing(monkeypatch, tmp_path, gateway_owner):
    named = ("gateway_elevated", "x")
    home, seen = _refusing_owner(monkeypatch, tmp_path, named, gateway_owner)
    assert runtime._refused(home).reason_code == "authorization" and seen == []


# ── Review-9 (e): never name a PID that is not provably this home's gateway ─────────────────────


def _probe_recorder(monkeypatch):
    seen = []
    named = lambda pid: seen.append(pid) or ("gateway_elevated", native.refused_owner_remedy("gateway_elevated", pid, 1))
    monkeypatch.setattr(native, "probe_refused_owner", named)
    return seen


def test_a_recycled_pid_is_never_named(monkeypatch, tmp_path, gateway_owner):
    """A live process whose start time differs from the record's (the PID was recycled)."""
    home = tmp_path / "home"
    home.mkdir()
    record = _record(home, gateway_owner.pid)
    record["start_time"] = (record["start_time"] or 0) + 12345
    (home / "gateway.pid").write_text(json.dumps(record), encoding="utf-8")
    seen = _probe_recorder(monkeypatch)
    assert runtime._refused(home, windows=True).reason_code == "authorization" and seen == []


@pytest.mark.parametrize("shape", ["no start time", "not a gateway", "another home"])
def test_an_unverifiable_owner_is_never_named(monkeypatch, tmp_path, shape):
    home = tmp_path / "home"
    home.mkdir()
    if shape == "not a gateway":
        child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    else:
        child = _gateway_child(tmp_path)
    try:
        record = _record(home, child.pid)
        if shape == "no start time":
            del record["start_time"]
        elif shape == "another home":
            record["hermes_home"] = str(tmp_path / "other-profile")
        (home / "gateway.pid").write_text(json.dumps(record), encoding="utf-8")
        seen = _probe_recorder(monkeypatch)
        assert runtime._refused(home, windows=True).reason_code == "authorization" and seen == []
    finally:
        child.stdin.close()
        child.wait(timeout=10)


def test_a_stale_pid_file_falls_through_to_the_lock_owner(monkeypatch, tmp_path, gateway_owner):
    home = tmp_path / "home"
    home.mkdir()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (home / "gateway.pid").write_text(json.dumps({"pid": dead.pid, "start_time": 1}), encoding="utf-8")
    (home / "gateway.lock").write_text(json.dumps(_record(home, gateway_owner.pid)), encoding="utf-8")
    seen = _probe_recorder(monkeypatch)
    assert runtime._refused(home, windows=True).reason_code == "gateway_elevated" and seen == [gateway_owner.pid]
