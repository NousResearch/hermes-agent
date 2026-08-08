"""Regression tests for `/model` picker provider overlays.

When a session has an explicit configured provider (for example Bedrock) but
the live runtime provider resolves to some other env-backed provider, the
picker must still use the explicit requested provider for inventory/current-row
state. Otherwise aws-sdk providers can disappear from the authenticated list.
"""

from __future__ import annotations

from hermes_cli.inventory import ConfigContext


class _StubCLI:
    model = "global.anthropic.claude-sonnet-4-6"
    provider = "openrouter"
    requested_provider = "bedrock"
    base_url = ""
    api_key = ""

    def __init__(self) -> None:
        self.opened = None

    def _open_model_picker(
        self,
        providers,
        current_model,
        current_provider,
        user_provs=None,
        custom_provs=None,
    ) -> None:
        self.opened = {
            "providers": providers,
            "current_model": current_model,
            "current_provider": current_provider,
            "user_provs": user_provs,
            "custom_provs": custom_provs,
        }


def test_model_picker_prefers_requested_provider_over_resolved_runtime(monkeypatch):
    import cli as cli_mod

    stub = _StubCLI()
    captured_ctx = {}

    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: ConfigContext(
            current_provider="bedrock",
            current_model="global.anthropic.claude-sonnet-4-6",
            current_base_url="",
            user_providers={},
            custom_providers=[],
        ),
    )

    def _build_models_payload(ctx, max_models=50):
        captured_ctx["provider"] = ctx.current_provider
        captured_ctx["model"] = ctx.current_model
        return {
            "providers": [
                {
                    "slug": "bedrock",
                    "name": "AWS Bedrock",
                    "is_current": True,
                    "is_user_defined": False,
                    "models": ["global.anthropic.claude-sonnet-4-6"],
                    "total_models": 1,
                    "source": "hermes",
                }
            ]
        }

    monkeypatch.setattr("hermes_cli.inventory.build_models_payload", _build_models_payload)
    monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)

    cli_mod.HermesCLI._handle_model_switch(stub, "/model")

    assert captured_ctx["provider"] == "bedrock"
    assert captured_ctx["model"] == "global.anthropic.claude-sonnet-4-6"
    assert stub.opened is not None
    assert stub.opened["current_provider"] == "AWS Bedrock"


def test_model_picker_keeps_resolved_provider_when_requested_is_auto(monkeypatch):
    import cli as cli_mod

    stub = _StubCLI()
    stub.requested_provider = "auto"
    captured_ctx = {}

    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: ConfigContext(
            current_provider="auto",
            current_model="gpt-5.4",
            current_base_url="",
            user_providers={},
            custom_providers=[],
        ),
    )

    def _build_models_payload(ctx, max_models=50):
        captured_ctx["provider"] = ctx.current_provider
        return {
            "providers": [
                {
                    "slug": "openrouter",
                    "name": "OpenRouter",
                    "is_current": True,
                    "is_user_defined": False,
                    "models": ["gpt-5.4"],
                    "total_models": 1,
                    "source": "built-in",
                }
            ]
        }

    monkeypatch.setattr("hermes_cli.inventory.build_models_payload", _build_models_payload)
    monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)

    cli_mod.HermesCLI._handle_model_switch(stub, "/model")

    assert captured_ctx["provider"] == "openrouter"
    assert stub.opened is not None
    assert stub.opened["current_provider"] == "OpenRouter"
