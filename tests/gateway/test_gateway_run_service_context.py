import gateway.run as gateway_run


def test_is_noninteractive_service_context_true(monkeypatch):
    monkeypatch.setenv("INVOCATION_ID", "abc123")
    monkeypatch.setattr(gateway_run.sys.stdin, "isatty", lambda: False)
    assert gateway_run._is_noninteractive_service_context() is True


def test_is_noninteractive_service_context_false_without_service_env(monkeypatch):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("JOURNAL_STREAM", raising=False)
    monkeypatch.delenv("SYSTEMD_EXEC_PID", raising=False)
    monkeypatch.delenv("SUPERVISOR_ENABLED", raising=False)
    monkeypatch.setattr(gateway_run.sys.stdin, "isatty", lambda: False)
    assert gateway_run._is_noninteractive_service_context() is False
