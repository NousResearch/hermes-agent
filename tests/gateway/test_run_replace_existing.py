import signal

from gateway import run as gateway_run


def test_terminate_existing_gateway_graceful_exit(monkeypatch):
    calls = []
    alive = {"value": True}

    def fake_kill(pid, sig):
        calls.append(sig)
        if sig == signal.SIGTERM:
            alive["value"] = False
            return None
        if sig == 0 and alive["value"]:
            return None
        if sig == 0 and not alive["value"]:
            raise ProcessLookupError
        return None

    monkeypatch.setattr(gateway_run.os, "kill", fake_kill)
    monkeypatch.setattr(gateway_run.time, "sleep", lambda _: None)

    assert gateway_run._terminate_existing_gateway(1234, timeout_seconds=0.2) is True
    assert calls[0] == signal.SIGTERM


def test_terminate_existing_gateway_permission_denied(monkeypatch):
    def fake_kill(_pid, _sig):
        raise PermissionError

    monkeypatch.setattr(gateway_run.os, "kill", fake_kill)

    assert gateway_run._terminate_existing_gateway(1234, timeout_seconds=0.1) is False
