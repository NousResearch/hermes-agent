"""Regression for #111323 (follow-up): a model served by the MANAGED local runtime must never
inherit the session's cloud provider.

Field symptom: switching the session to a managed llama.cpp model while the session sat on a
cloud provider kept the cloud provider — the switch was validated against the cloud catalog
("not found in this provider's model listing") and, once such a pairing was persisted, the
request itself went to the cloud API, surfacing as a baffling `Connection error` for a model
that only ever existed on this machine. Same story for an unrouted local provider whose server
is down: the alias only registers while its endpoint resolves, so the user was told
"Unknown provider 'llamacpp'" instead of "the local model server isn't running".

Contract (both directions):
  * a managed local model id routes to the local provider, canonicalised to the preset's id;
  * a local-runtime provider alias with no endpoint reports the runtime's own diagnosis.
"""

from __future__ import annotations


LOCAL_ID = "Spark-X2.5-4B-Q4_K_M"


def test_managed_local_model_routes_to_the_local_provider(tmp_path, monkeypatch):
    """Typed in any case, the local id must land on the local provider with its endpoint."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("hermes_cli.local_runtime.presets.read_preset_decisions",
                        lambda *a, **k: {LOCAL_ID: None})
    monkeypatch.setattr("hermes_cli.local_runtime.endpoint.resolve_llamacpp_endpoint",
                        lambda *a, **k: {"base_url": "http://127.0.0.1:18434/v1", "api_key": "sk-managed"})

    from hermes_cli.model_switch import switch_model

    result = switch_model("spark-x2.5-4b-q4_k_m", "deepseek", "deepseek-flash")

    assert result.success, result.error_message
    assert result.target_provider == "llamacpp"
    assert result.new_model == LOCAL_ID  # the router's own spelling, not the typed case
    assert result.base_url == "http://127.0.0.1:18434/v1"
    assert result.api_key == "sk-managed"


def test_local_provider_without_endpoint_reports_the_runtime_not_an_unknown_provider(tmp_path, monkeypatch):
    """A local-runtime alias whose server is down says why, instead of blaming the user's
    spelling with "Unknown provider"."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("hermes_cli.local_runtime.endpoint.resolve_llamacpp_endpoint",
                        lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"local_runtime": {"enabled": True}})

    from hermes_cli.model_switch import switch_model

    result = switch_model(LOCAL_ID, "deepseek", "deepseek-flash", explicit_provider="llamacpp")

    assert not result.success
    assert "isn't running" in result.error_message
    assert "Unknown provider" not in result.error_message
