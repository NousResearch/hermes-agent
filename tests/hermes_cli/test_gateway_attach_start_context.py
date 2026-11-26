"""`hermes gateway attach` tells a remote client where a gateway it starts would run.

An administrator's OpenSSH login on Windows is elevated and in session 0; a gateway started there
refuses the user's desktop apps (``gateway_other_session``). The attach reply names that case and
``--start`` warns on stderr, while starting the gateway and exiting exactly as before.
"""
import json
from types import SimpleNamespace

import pytest

from hermes_cli.gateway_runtime import GatewayDiscovery, GatewayEndpoint
from hermes_cli.gateway_runtime_cli import cmd_gateway_attach

READY = GatewayDiscovery("ready", endpoint=GatewayEndpoint(
    profile_id="p", instance_id="owner", authority_epoch=1, runtime_protocol=1,
    api_origin="http://127.0.0.1:8765", supervisor="none", capabilities=frozenset({"session-authority-v1"})))
ABSENT = GatewayDiscovery("absent")


def _attach(monkeypatch, capsys, *flags, session=None, elevated=None, verdict=READY, timeout="60"):
    """Run the command with the Windows facts faked at the platform seam (None: the real host) and
    the gateway faked at ensure/discover. Returns (exit code, payload, stderr lines, what ran)."""
    ran = []
    if session is not None:
        monkeypatch.setattr("hermes_platform.host.windows_session_id", lambda: session)
    if elevated is not None:
        monkeypatch.setattr("hermes_dir_acl.process_is_elevated", lambda: elevated)
    monkeypatch.setattr("hermes_cli.gateway_runtime.ensure_gateway_runtime",
                        lambda home, **kw: ran.append("start") or verdict)
    monkeypatch.setattr("hermes_cli.gateway_runtime.discover_gateway_endpoint",
                        lambda home, **kw: ran.append("discover") or verdict)
    monkeypatch.setattr("hermes_cli.gateway_client._session_ticket", lambda home, endpoint: "one-ticket")
    args = SimpleNamespace(start="--start" in flags, recover=False, timeout=timeout)
    with pytest.raises(SystemExit) as exit_:
        cmd_gateway_attach(args)
    out = capsys.readouterr()
    return exit_.value.code, json.loads(out.out), out.err.splitlines(), ran


def _warnings(stderr):
    return [line for line in stderr if "gateway_other_session" in line]


def test_start_from_an_elevated_session_0_login_warns_and_still_starts_the_gateway(monkeypatch, capsys):
    code, payload, stderr, ran = _attach(monkeypatch, capsys, "--start", session=0, elevated=True)
    assert payload["start_context"] == {"windows_service_session": True}
    warning, = _warnings(stderr)
    assert warning.startswith("warning:") and "normal (non-administrator) window" in warning
    assert "gateway install" not in warning  # installing can start the gateway elevated (#152218)
    assert ran == ["start"] and code == 0
    assert payload["ticket"] == "one-ticket" and payload["url"] == "ws://127.0.0.1:8765/api/ws"


def test_the_reply_and_exit_code_differ_from_a_normal_login_only_by_the_flag(monkeypatch, capsys):
    for flags, verdict in ((("--start",), READY), (("--start",), ABSENT), ((), READY), ((), ABSENT)):
        service = _attach(monkeypatch, capsys, *flags, session=0, elevated=True, verdict=verdict)
        normal = _attach(monkeypatch, capsys, *flags, session=1, elevated=False, verdict=verdict)
        assert service[1].pop("start_context") == {"windows_service_session": True}
        assert normal[1].pop("start_context") == {"windows_service_session": False}
        assert (service[0], service[1], service[3]) == (normal[0], normal[1], normal[3])
        assert [line for line in service[2] if line not in _warnings(service[2])] == normal[2]


def test_a_failed_start_keeps_its_verdict_and_exit_code(monkeypatch, capsys):
    code, payload, stderr, ran = _attach(monkeypatch, capsys, "--start", session=0, elevated=True, verdict=ABSENT)
    assert code == 5 and ran == ["start"] and payload["message"] == "gateway absent: not ready"
    assert payload["start_context"] == {"windows_service_session": True}
    assert len(_warnings(stderr)) == 1 and "gateway absent: not ready" in stderr


def test_a_plain_attach_from_that_login_reports_the_flag_without_a_warning(monkeypatch, capsys):
    code, payload, stderr, ran = _attach(monkeypatch, capsys, session=0, elevated=True)
    assert payload["start_context"] == {"windows_service_session": True}
    assert not _warnings(stderr) and ran == ["discover"] and code == 0


@pytest.mark.parametrize("session, elevated", [
    (1, True),      # an administrator window on the user's own desktop
    (0, False),     # a non-administrator OpenSSH login
    (None, True),   # the session cannot be read
])
def test_only_elevated_and_session_0_together_count(monkeypatch, capsys, session, elevated):
    if session is None:
        monkeypatch.setattr("hermes_platform.host.windows_session_id", lambda: None)
    code, payload, stderr, ran = _attach(monkeypatch, capsys, "--start", session=session, elevated=elevated)
    assert payload["start_context"] == {"windows_service_session": False}
    assert not _warnings(stderr) and ran == ["start"] and code == 0


@pytest.mark.platforms("linux", "macos")
def test_off_windows_the_flag_is_present_and_false(monkeypatch, capsys):
    from hermes_platform.host import windows_session_id

    assert windows_session_id() is None
    for flags in ((), ("--start",)):
        code, payload, stderr, _ = _attach(monkeypatch, capsys, *flags)
        assert payload["start_context"] == {"windows_service_session": False}
        assert not _warnings(stderr) and code == 0


def test_an_invalid_timeout_still_exits_2_and_carries_the_flag(monkeypatch, capsys):
    code, payload, _, ran = _attach(monkeypatch, capsys, "--start", session=0, elevated=True, timeout="0")
    assert code == 2 and ran == []
    assert payload == {"state": "inaccessible", "reason_code": "invalid_invocation", "endpoint": None,
                       "start_context": {"windows_service_session": True}}
