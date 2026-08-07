from types import SimpleNamespace


def _scheduler_readiness_warning():
    from cron.scheduler_readiness import scheduler_readiness_warning

    return scheduler_readiness_warning()


def test_builtin_scheduler_without_gateway_warns(monkeypatch):
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: SimpleNamespace(name="builtin"),
    )
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])

    warning = _scheduler_readiness_warning()

    assert warning is not None
    assert "will NOT fire automatically" in warning
    assert "hermes gateway install" in warning
    assert "hermes cron status" in warning


def test_builtin_scheduler_with_gateway_is_ready(monkeypatch):
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: SimpleNamespace(name="builtin"),
    )
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [4242])

    assert _scheduler_readiness_warning() is None


def test_external_scheduler_does_not_require_local_gateway(monkeypatch):
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: SimpleNamespace(name="chronos"),
    )
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])

    assert _scheduler_readiness_warning() is None
