"""Native Windows port and stop checks must work without weakening process identity guards."""

import socket

import pytest

from hermes_cli import gateway


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("host,family", [("127.0.0.1", socket.AF_INET), ("0.0.0.0", socket.AF_INET),
                                        ("::1", socket.AF_INET6), ("::", socket.AF_INET6)])
def test_windows_port_probe_stays_exclusive_until_listener_closes(host, family):
    with socket.socket(family, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((host, 0))
        except OSError:
            if family == socket.AF_INET6 and not socket.has_ipv6:
                pytest.skip("IPv6 sockets are unavailable on this machine")
            raise
        listener.listen(8)
        port = listener.getsockname()[1]
        assert gateway._wait_for_tcp_port_free(host, port, timeout=0.3) is False
    assert gateway._wait_for_tcp_port_free(host, port, timeout=5.0) is True


@pytest.mark.platforms("windows")
def test_windows_stop_keeps_pid_file_when_guarded_termination_fails(monkeypatch):
    from gateway import status
    from hermes_cli import gateway_windows

    pid, start_time = 12345, 100
    calls = []
    clock = {"now": 0.0}
    monkeypatch.setattr(status, "get_running_pid", lambda: pid)
    monkeypatch.setattr(status, "get_process_start_time", lambda target: start_time)
    monkeypatch.setattr(status, "_pid_exists", lambda target: True)
    monkeypatch.setattr(status, "write_planned_stop_marker", lambda target: calls.append(("marker", target)))
    monkeypatch.setattr(status, "remove_pid_file", lambda: calls.append(("remove",)))
    monkeypatch.setattr(gateway_windows, "_windows_stop_drain_timeout", lambda: 2.0)
    monkeypatch.setattr(gateway_windows.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(gateway_windows.time, "sleep", lambda delay: clock.__setitem__("now", clock["now"] + delay))
    monkeypatch.setattr(gateway.os, "kill", lambda *_: pytest.fail("Windows must drain before a guarded taskkill"))
    monkeypatch.setattr(gateway, "_reap_unsupervised_gateway_orphans",
                        lambda extra_exclude=None: calls.append(("reap", extra_exclude)))

    def refuse_termination(target, *, force=False, expected_start_time=None):
        calls.append(("terminate", target, force, expected_start_time))
        raise OSError("process identity changed")

    monkeypatch.setattr(status, "terminate_pid", refuse_termination)

    assert gateway.stop_profile_gateway() is True
    assert calls == [("marker", pid), ("terminate", pid, True, start_time), ("reap", {pid})]
