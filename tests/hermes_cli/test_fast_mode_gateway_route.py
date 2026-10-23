"""Fast-mode route gate: local gateways own the tier, third-party hosts stay gated.

The gate exists so a proxy never receives fast-mode params it cannot honour. A local router
(OmniRoute, `hp-omniroute`) is the opposite case: it applies or forwards the tier itself, so the
first-party host allowlist (`api.openai.com` / `chatgpt.com`) must not hide the toggle for the
models it serves.
"""

from hermes_cli.models import resolve_fast_mode_overrides


def _overrides(model, provider, base_url, tier):
    return resolve_fast_mode_overrides(model, provider=provider, base_url=base_url, tier=tier)


def test_local_gateway_gets_ultrafast_for_flagship_codex_models():
    assert _overrides(
        "codex/gpt-6.1-sol", "hp-omniroute", "http://localhost:20128/v1", "ultrafast"
    ) == {"service_tier": "ultrafast"}
    assert _overrides(
        "codex/gpt-6-astra", "hp-omniroute", "http://127.0.0.1:20128/v1", "ultrafast"
    ) == {"service_tier": "ultrafast"}


def test_local_gateway_gets_priority_tier():
    assert _overrides(
        "codex/gpt-6.1-sol", "hp-omniroute", "http://localhost:20128/v1", "priority"
    ) == {"service_tier": "priority"}


def test_third_party_hosts_remain_gated():
    assert (
        _overrides("gpt-6.1-sol", "some-proxy", "https://proxy.example/v1", "ultrafast") is None
    )
    assert _overrides("gpt-6.1-sol", None, "https://proxy.example/v1", "ultrafast") is None


def test_first_party_routes_keep_working():
    assert _overrides(
        "gpt-6.1-sol", "openai-codex", "https://chatgpt.com/backend-api/codex", "ultrafast"
    ) == {"service_tier": "ultrafast"}
    assert _overrides(
        "gpt-6.1-sol", "openai", "https://api.openai.com/v1", "priority"
    ) == {"service_tier": "priority"}


def test_ultrafast_still_requires_a_model_that_advertises_it():
    # gpt-5.6-sol is Fast-eligible but not Ultrafast-eligible -> nothing to send.
    assert (
        _overrides("codex/gpt-5.6-sol", "hp-omniroute", "http://localhost:20128/v1", "ultrafast")
        is None
    )


def test_anthropic_branch_is_not_opened_by_the_gateway_allowlist():
    # Anthropic Fast Mode stays first-party-only; the gateway exception applies to OpenAI routing.
    assert (
        _overrides(
            "claude/claude-opus-5-5", "hp-omniroute", "http://localhost:20128/v1", "ultrafast"
        )
        is None
    )
