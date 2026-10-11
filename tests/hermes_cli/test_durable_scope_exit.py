"""Issue #132286: a supervised caller must park when the other systemd scope is durable.

The host already serves this profile. Two installed scopes (user and system) will
not go away, so the caller must exit 78 (RestartPreventExitStatus). A peer that
can still disappear stays 75. An unsupervised attach stays 0.

Drives the real ``decide`` / ``_attach_to_host_gateway_or_guard`` exit. The host
record is the one ``publish_record`` writes; the served set comes from that
owner's real control-socket ``identify``. No MagicMock of the decision.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gateway.host_attach import ATTACH, decide, invalidate_host_gateway_cache
from hermes_cli import gateway as gw
from hermes_cli.config import get_hermes_home

_SUPERVISOR_KEYS = (
    "INVOCATION_ID",
    "HERMES_S6_SUPERVISED_CHILD",
    "XPC_SERVICE_NAME",
    "HERMES_LAUNCHD_LABEL",
    "HERMES_GATEWAY_EXTERNAL_SUPERVISOR",
    "HERMES_DESKTOP_MANAGED",
    "LAUNCHD_SOCKET",
)

_UNIT_BODY = (
    "[Service]\n"
    "Restart=always\n"
    "RestartForceExitStatus=75\n"
    "RestartPreventExitStatus=78\n"
)


def _serve_peer() -> int:
    """Other process: real host record + real identify. Not the decision under test."""
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    home = Path(os.environ["HERMES_HOME"]).resolve()
    ready = Path(os.environ["ISSUE_132286_READY"])
    stop = Path(os.environ["ISSUE_132286_STOP"])
    from gateway.control_socket import GatewayControlServer, build_identify_payload
    from gateway.host_rendezvous import ROLE_GATEWAY, publish_record

    async def serve() -> int:
        # Real identify: supervisor comes from this process's launch env
        # (INVOCATION_ID → systemd, otherwise manual). Not a hand-built dict.
        server = GatewayControlServer(home, verb_handlers={"identify": build_identify_payload})
        if not await server.start():
            print("issue 132286 peer: control socket did not start", file=sys.stderr)
            return 2
        record = publish_record(ROLE_GATEWAY, profiles=("default",), home=str(home))
        if record is None:
            print("issue 132286 peer: publish_record returned None", file=sys.stderr)
            return 2
        ready.write_text(str(os.getpid()), encoding="utf-8")
        try:
            while not stop.exists():
                await asyncio.sleep(0.05)
        finally:
            await server.stop()
        return 0

    return asyncio.run(serve())


def _install_units(tmp_path: Path, monkeypatch, *, both_scopes: bool) -> None:
    """Point the real unit-file detector at this test's dirs. Do not mock it."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(gw, "_SYSTEM_UNIT_DIR", tmp_path / "system-units")
    user_path = gw.get_systemd_unit_path(system=False)
    user_path.parent.mkdir(parents=True, exist_ok=True)
    user_path.write_text(_UNIT_BODY, encoding="utf-8")
    if both_scopes:
        system_path = gw.get_systemd_unit_path(system=True)
        system_path.parent.mkdir(parents=True, exist_ok=True)
        system_path.write_text(_UNIT_BODY, encoding="utf-8")
    scopes = gw.get_installed_systemd_scopes()
    if both_scopes:
        assert scopes == ["user", "system"], scopes
        assert gw.has_conflicting_systemd_units() is True
    else:
        assert scopes == ["user"], scopes
        assert gw.has_conflicting_systemd_units() is False


def _start_peer(tmp_path: Path, *, peer_supervisor: str) -> subprocess.Popen:
    ready = tmp_path / "peer-ready"
    stop = tmp_path / "peer-stop"
    env = os.environ.copy()
    for key in _SUPERVISOR_KEYS:
        env.pop(key, None)
    env["ISSUE_132286_PEER"] = "1"
    env["ISSUE_132286_READY"] = str(ready)
    env["ISSUE_132286_STOP"] = str(stop)
    if peer_supervisor == "systemd":
        env["INVOCATION_ID"] = "issue-132286-peer"
    elif peer_supervisor != "manual":
        raise ValueError(f"unsupported peer supervisor: {peer_supervisor}")
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    proc.ready = ready  # type: ignore[attr-defined]
    proc.stop = stop  # type: ignore[attr-defined]
    return proc


def _wait_ready(proc: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    ready = proc.ready  # type: ignore[attr-defined]
    while not ready.exists():
        if proc.poll() is not None or time.monotonic() > deadline:
            out, err = proc.communicate(timeout=5)
            raise AssertionError(
                f"issue 132286 peer did not publish a live host "
                f"(code={proc.returncode})\nstdout:\n{out}\nstderr:\n{err}"
            )
        time.sleep(0.05)


def _stop_peer(proc: subprocess.Popen) -> None:
    stop = proc.stop  # type: ignore[attr-defined]
    stop.write_text("stop", encoding="utf-8")
    try:
        proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=5)


def _attach_exit_code(
    tmp_path: Path,
    monkeypatch,
    *,
    both_scopes: bool,
    peer_supervisor: str,
    caller_supervised: bool,
) -> int:
    """Real already-serves verdict, then the real process exit that chooses 75 vs 78 vs 0."""
    for key in _SUPERVISOR_KEYS:
        monkeypatch.delenv(key, raising=False)
    _install_units(tmp_path, monkeypatch, both_scopes=both_scopes)
    proc = _start_peer(tmp_path, peer_supervisor=peer_supervisor)
    try:
        _wait_ready(proc)
        if caller_supervised:
            monkeypatch.setenv("INVOCATION_ID", "issue-132286-caller")
        invalidate_host_gateway_cache()
        home = get_hermes_home()
        decision = decide(home)
        assert decision.outcome == ATTACH, (
            f"expected the already-serves ATTACH path, got {decision.outcome!r}: {decision.message}"
        )
        assert "already serves profile" in decision.message
        with pytest.raises(SystemExit) as caught:
            gw._attach_to_host_gateway_or_guard(force=False, replace=False)
        code = caught.value.code
        assert isinstance(code, int)
        return code
    finally:
        _stop_peer(proc)


def test_supervised_durable_duplicate_scope_exits_78(tmp_path, monkeypatch):
    """Both scopes installed, caller supervised, host already serves this profile → 78.

    On current main the exit path does not read the scope detector, so this is RED
    (75, the transient bucket).
    """
    code = _attach_exit_code(
        tmp_path,
        monkeypatch,
        both_scopes=True,
        peer_supervisor="systemd",
        caller_supervised=True,
    )
    assert code == 78


def test_supervised_disappearing_peer_stays_75(tmp_path, monkeypatch):
    """One scope, manual foreground owner, supervised caller: the peer can still exit → 75."""
    code = _attach_exit_code(
        tmp_path,
        monkeypatch,
        both_scopes=False,
        peer_supervisor="manual",
        caller_supervised=True,
    )
    assert code == 75


def test_unsupervised_attach_stays_0(tmp_path, monkeypatch):
    """Same durable peer, but the caller is not a supervisor → exit 0, not 78 and not 75."""
    code = _attach_exit_code(
        tmp_path,
        monkeypatch,
        both_scopes=True,
        peer_supervisor="systemd",
        caller_supervised=False,
    )
    assert code == 0


def test_one_scope_systemd_peer_stays_75(tmp_path, monkeypatch):
    """One scope, systemd peer, supervised caller: a single scope must keep retrying → 75."""
    code = _attach_exit_code(
        tmp_path,
        monkeypatch,
        both_scopes=False,
        peer_supervisor="systemd",
        caller_supervised=True,
    )
    assert code == 75


def test_both_scopes_manual_peer_stays_75(tmp_path, monkeypatch):
    """Both scopes installed, manual peer, supervised caller: the peer can still exit → 75."""
    code = _attach_exit_code(
        tmp_path,
        monkeypatch,
        both_scopes=True,
        peer_supervisor="manual",
        caller_supervised=True,
    )
    assert code == 75


def test_peer_exits_after_decide_before_exit_code_stays_75(tmp_path, monkeypatch):
    """decide() returns ATTACH with a systemd owner and both scopes; the peer then exits.

    The exit code is chosen from that snapshot. A peer that is gone before
    ``_host_decision_exit_code`` must stay 75. Exit 78 would park an unowned profile.
    The host record is left as the peer left it (dead owner, not retracted here).
    """
    for key in _SUPERVISOR_KEYS:
        monkeypatch.delenv(key, raising=False)
    _install_units(tmp_path, monkeypatch, both_scopes=True)
    proc = _start_peer(tmp_path, peer_supervisor="systemd")
    decision = None
    try:
        _wait_ready(proc)
        monkeypatch.setenv("INVOCATION_ID", "issue-132286-caller")
        invalidate_host_gateway_cache()
        decision = decide(get_hermes_home())
        assert decision.outcome == ATTACH, (
            f"expected the already-serves ATTACH path, got {decision.outcome!r}: {decision.message}"
        )
        assert decision.owner is not None
        assert decision.owner.supervisor == "systemd"
        assert decision.transient is True
        assert gw.has_conflicting_systemd_units() is True
        _stop_peer(proc)
    finally:
        if proc.poll() is None:
            _stop_peer(proc)
    assert decision is not None
    # Snapshot still says systemd. The owner PID is not live. Must not park.
    assert gw._host_decision_exit_code(decision) == 75


if __name__ == "__main__" and os.environ.get("ISSUE_132286_PEER") == "1":
    raise SystemExit(_serve_peer())
