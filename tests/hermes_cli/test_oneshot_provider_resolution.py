from __future__ import annotations


def test_explicit_model_keeps_configured_custom_provider_key_cmd(monkeypatch):
    from hermes_cli import runtime_provider as rp
    from hermes_cli.oneshot import _resolve_model_and_provider

    config = {
        "model": {"provider": "custom:dbx", "default": "private-model"},
        "providers": {
            "dbx": {
                "base_url": "https://example.invalid/v1",
                "key_cmd": "printf minted-token",
                "models": ["gpt-5.4", "private-model"],
            }
        },
    }
    monkeypatch.setattr(
        "hermes_cli.models.detect_provider_for_model",
        lambda model, current: ("openai", model),
    )
    monkeypatch.setattr(rp, "load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *args, **kwargs: config)

    choice = _resolve_model_and_provider(config, "gpt-5.4", None)
    runtime = rp.resolve_runtime_provider(
        requested=choice.provider,
        target_model=choice.model,
    )

    assert choice.provider == "dbx"
    assert callable(runtime["api_key"])
