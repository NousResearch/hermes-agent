"""Regression tests for explicit-provider /model switches.

Discord's interactive picker calls switch_model(model_id, explicit_provider=provider).
When multiple direct aliases map to the same bare model id, the explicit provider
must win; otherwise a reverse alias match can leak another provider's base_url.
"""

from hermes_cli import model_switch


def test_explicit_provider_switch_does_not_inherit_other_alias_base_url(monkeypatch):
    monkeypatch.setattr(model_switch, "_ensure_direct_aliases", lambda: None)
    model_switch.DIRECT_ALIASES.clear()
    model_switch.DIRECT_ALIASES.update(
        {
            "clawbay-gpt5.5": model_switch.DirectAlias(
                model="gpt-5.5",
                provider="clawbay",
                base_url="https://api.theclawbay.com/v1",
            ),
            "codex-gpt5.5": model_switch.DirectAlias(
                model="gpt-5.5",
                provider="openai-codex",
                base_url="https://chatgpt.com/backend-api/codex",
            ),
        }
    )

    def fake_runtime_provider(*, requested=None, target_model=None, **_kwargs):
        assert requested == "openai-codex"
        assert target_model == "gpt-5.5"
        return {
            "provider": "openai-codex",
            "api_key": "sk-test",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_runtime_provider,
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *args, **kwargs: {"accepted": True, "persist": True, "recognized": True},
    )

    result = model_switch.switch_model(
        raw_input="gpt-5.5",
        current_provider="clawbay",
        current_model="gpt-5.5",
        current_base_url="https://api.theclawbay.com/v1",
        current_api_key="",
        explicit_provider="openai-codex",
        is_global=False,
        user_providers={
            "clawbay": {
                "name": "The Claw Bay",
                "base_url": "https://api.theclawbay.com/v1",
                "models": {"gpt-5.5": {}},
            }
        },
        custom_providers=[],
    )

    assert result.success
    assert result.target_provider == "openai-codex"
    assert result.new_model == "gpt-5.5"
    assert result.base_url == "https://chatgpt.com/backend-api/codex"
    assert result.api_mode == "codex_responses"
    assert result.resolved_via_alias == ""
