"""The Foundry CLI wizard preserves explicit CLI account choices from Desktop."""

import pytest


@pytest.mark.parametrize("source", ["azure_cli", "default"])
def test_entra_preflight_selects_and_preserves_account_options(monkeypatch, source):
    from agent import azure_identity_adapter as adapter
    from hermes_cli import model_setup_flows_azure as flow

    current = {"credential_source": "azure_cli", "scope": "https://audience.example/.default",
               "tenant_id": "tenant-a"}
    answers = iter(["2", "", ""] if source == "azure_cli" else ["1"])
    monkeypatch.setattr(flow, "_ask", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr(adapter, "has_azure_identity_installed", lambda: True)
    probed = []
    monkeypatch.setattr(adapter, "describe_active_credential", lambda **kwargs: probed.append(kwargs["config"]) or {"ok": True})
    monkeypatch.setattr(adapter, "build_token_provider", lambda **kwargs: lambda: kwargs["config"].credential_source)
    token, saved = flow._azure_entra_preflight(current)
    assert token() == source
    assert probed[0] == adapter.EntraIdentityConfig.from_dict(saved)
    assert saved["scope"] == current["scope"]
    if source == "azure_cli":
        assert saved == current
    else:
        assert "tenant_id" not in saved and "subscription" not in saved


def test_malformed_endpoint_cancels_wizard_without_writing(monkeypatch, capsys):
    from hermes_cli import model_setup_flows_azure as flow
    from hermes_cli import azure_detect
    from hermes_cli.config import read_raw_config

    before = read_raw_config()
    monkeypatch.setattr(flow, "_ask", lambda *args, **kwargs: "https://[")
    def unexpected_probe(*args, **kwargs):
        raise AssertionError("Invalid URLs must not reach a credentialed network probe")
    monkeypatch.setattr(azure_detect, "detect", unexpected_probe)
    flow._model_flow_azure_foundry({"model": {}})
    assert "Invalid endpoint URL" in capsys.readouterr().out
    assert read_raw_config() == before
