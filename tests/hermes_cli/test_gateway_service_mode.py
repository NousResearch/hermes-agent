from types import SimpleNamespace

from hermes_cli import gateway as gateway_cli


def test_gateway_run_enables_replace_in_service_context(monkeypatch):
    monkeypatch.setenv("INVOCATION_ID", "unit-test")

    called = {}

    def fake_run_gateway(verbose=False, replace_existing=False):
        called["verbose"] = verbose
        called["replace_existing"] = replace_existing

    monkeypatch.setattr(gateway_cli, "run_gateway", fake_run_gateway)

    args = SimpleNamespace(gateway_command="run", verbose=False, replace=False)
    gateway_cli.gateway_command(args)

    assert called == {"verbose": False, "replace_existing": True}


def test_gateway_run_respects_replace_outside_service_context(monkeypatch):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("JOURNAL_STREAM", raising=False)
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)

    called = {}

    def fake_run_gateway(verbose=False, replace_existing=False):
        called["verbose"] = verbose
        called["replace_existing"] = replace_existing

    monkeypatch.setattr(gateway_cli, "run_gateway", fake_run_gateway)

    args = SimpleNamespace(gateway_command="run", verbose=True, replace=False)
    gateway_cli.gateway_command(args)

    assert called == {"verbose": True, "replace_existing": False}
