"""Regression guard for the named custom Responses fallback."""

from unittest.mock import MagicMock, patch


def test_transport_failure_uses_named_custom_responses_client(tmp_path, monkeypatch):
    """A bare runtime custom class must recover its named Responses route."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  provider: custom:relay\n"
        "  default: fallback-model\n"
        "providers:\n"
        "  relay:\n"
        "    base_url: http://relay.test/backend-api/codex\n"
        "    key_env: RELAY_API_KEY\n"
        "    api_mode: codex_responses\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("RELAY_API_KEY", "test-key")

    from agent import auxiliary_client as mod

    class APIConnectionError(Exception):
        pass

    primary = MagicMock()
    primary.base_url = "http://primary.test/v1"
    primary.chat.completions.create.side_effect = APIConnectionError("transport down")
    captured = []

    def record_fallback(client, model, label, **_kwargs):
        captured.append((client, model, label))
        return {"fallback": True}

    mod.clear_runtime_main()
    mod.set_runtime_main(
        "custom", "fallback-model",
        base_url="http://relay.test/backend-api/codex",
        api_key="test-key",
    )
    try:
        with (
            patch.object(
                mod, "_resolve_task_provider_model",
                return_value=("primary", "primary-model", None, None, None),
            ),
            patch.object(mod, "_get_cached_client", return_value=(primary, "primary-model")),
            patch.object(mod, "_try_configured_fallback_chain", return_value=(None, None, "")),
            patch.object(mod, "_call_fallback_candidate_sync", side_effect=record_fallback),
        ):
            assert mod.call_llm(
                task="compression", messages=[{"role": "user", "content": "compress"}]
            ) == {"fallback": True}
    finally:
        mod.clear_runtime_main()

    client, model, label = captured.pop()
    assert isinstance(client, mod.CodexAuxiliaryClient)
    assert model == "fallback-model"
    assert label == "main-agent(custom:relay)"
    assert str(client.base_url).rstrip("/") == "http://relay.test/backend-api/codex"
