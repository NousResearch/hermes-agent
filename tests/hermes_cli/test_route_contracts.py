import json

import pytest

from hermes_cli.route_contracts import (
    RouteContractError,
    build_agent_route_proof,
    build_route_hardening_plan,
    verify_agent_route_contract,
)


def _as_json(payload):
    return json.dumps(payload, sort_keys=True)


def test_surface_inference_covers_delegate_cron_tui_gateway_and_primary():
    from gateway.config import Platform

    cases = [
        ("primary", "cli", "primary"),
        ("delegation", "telegram", "delegation"),
        (None, "cron", "cron"),
        (None, "tui", "tui"),
        (None, "telegram", "gateway"),
    ]

    for platform in Platform:
        cases.append((None, platform.value, "gateway"))

    for surface, platform, expected_surface in cases:
        proof = build_agent_route_proof(
            surface=surface,
            platform=platform,
            provider="openai-codex",
            model="gpt-5.5",
            api_mode="codex_responses",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="eyJhbGciOiJSUzI1NiJ9.oauth-token-body",
        )
        assert proof["surface"] == expected_surface
        assert proof["contract"]["status"] == "ok"


def test_path_hint_redacts_untrusted_path_segments_and_query_secrets():
    proof = build_agent_route_proof(
        surface="primary",
        provider="custom",
        model="local/model",
        api_mode="chat_completions",
        base_url="https://proxy.example.com/token-is-hunter2/v1?api_key=query-secret",
        api_key="opaque-secret-value",
    )

    encoded = _as_json(proof)
    assert proof["base_url_host"] == "proxy.example.com"
    assert proof["base_url_path_hint"] == "/<redacted-path>"
    assert "token-is-hunter2" not in encoded
    assert "query-secret" not in encoded
    assert "opaque-secret-value" not in encoded


def test_hermes_recommended_codex_oauth_route_is_explicit_baseline():
    proof = build_agent_route_proof(
        surface="primary",
        provider="openai-codex",
        model="gpt-5.5",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="eyJhbG...body",
        reasoning_config={"enabled": True, "effort": "xhigh"},
        service_tier="priority",
    )

    assert proof["setup_mode"] == "hermes_recommended_codex_oauth"
    assert proof["runtime"] == "codex_api"
    assert proof["route_owner"] == "hermes"
    assert proof["requires_external_cli"] is False
    assert proof["auth_surface"] == "oauth"
    assert proof["cost_surface"] == "subscription"
    assert proof["contract"]["status"] == "ok"



def test_codex_app_server_route_proof_is_content_safe_and_contract_complete():
    proof = build_agent_route_proof(
        surface="primary",
        provider="openai-codex",
        model="gpt-5.5",
        api_mode="codex_app_server",
        base_url="https://chatgpt.com/backend-api/codex?token=super-secret",
        api_key="eyJhbG...body",
        reasoning_config={"enabled": True, "effort": "high"},
        service_tier="priority",
        fallback_model=[{"provider": "openrouter", "model": "openai/gpt-5.5"}],
    )

    assert proof["surface"] == "primary"
    assert proof["provider"] == "openai-codex"
    assert proof["model"] == "gpt-5.5"
    assert proof["api_mode"] == "codex_app_server"
    assert proof["setup_mode"] == "codex_app_server_opt_in"
    assert proof["runtime"] == "codex_app_server"
    assert proof["route_owner"] == "hermes_outer_codex_inner"
    assert proof["requires_external_cli"] is True
    assert proof["auth_surface"] == "oauth"
    assert proof["cost_surface"] == "subscription"
    assert proof["credential_present"] is True
    assert proof["credential_kind"] == "oauth_jwt"
    assert proof["base_url_host"] == "chatgpt.com"
    assert proof["base_url_path_hint"] == "/backend-api/codex"
    assert proof["reasoning_effort"] == "high"
    assert proof["service_tier"] == "priority"
    assert proof["fallback_chain_count"] == 1
    assert proof["contract"]["status"] == "ok"

    encoded = _as_json(proof)
    assert "super-secret" not in encoded
    assert "must-not-leak" not in encoded
    assert "eyJhbGci" not in encoded


def test_openai_provider_codex_app_server_is_still_explicit_opt_in():
    proof = build_agent_route_proof(
        surface="primary",
        provider="openai",
        model="gpt-5.5",
        api_mode="codex_app_server",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="eyJhbG...body",
    )

    assert proof["contract"]["status"] == "ok"
    assert proof["setup_mode"] == "codex_app_server_opt_in"
    assert proof["runtime"] == "codex_app_server"
    assert proof["requires_external_cli"] is True


def test_route_hardening_plan_maps_recommended_codex_route_across_all_7_tiers():
    proof = build_agent_route_proof(
        surface="primary",
        provider="openai-codex",
        model="gpt-5.5",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex?token=super-secret",
        api_key="eyJhbG...body",
        reasoning_config={"enabled": True, "effort": "xhigh"},
        service_tier="priority",
    )

    plan = build_route_hardening_plan(proof)

    assert plan["content_policy"] == "metadata_only"
    assert plan["setup_mode"] == "hermes_recommended_codex_oauth"
    assert plan["recommended_baseline"]["openai_runtime"] == "auto"
    assert plan["recommended_baseline"]["api_mode"] == "codex_responses"
    assert plan["recommended_baseline"]["requires_external_cli"] is False
    assert plan["tier_count"] == 7
    assert [item["tier"] for item in plan["tiers"]] == [1, 2, 3, 4, 5, 6, 7]
    assert plan["tiers"][1]["status"] == "ok"
    assert plan["tiers"][1]["required_action"] == "keep as canonical Hermes-recommended Codex OAuth/API-call route"
    assert plan["tiers"][4]["status"] == "ok"
    assert plan["tiers"][5]["status"] == "ok"

    encoded = _as_json(plan)
    assert "super-secret" not in encoded
    assert "eyJhbG" not in encoded



def test_route_hardening_plan_marks_codex_app_server_as_opt_in_exception():
    proof = build_agent_route_proof(
        surface="cron",
        provider="openai-codex",
        model="gpt-5.5",
        api_mode="codex_app_server",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="eyJhbG...body",
    )

    plan = build_route_hardening_plan(proof)

    assert plan["setup_mode"] == "codex_app_server_opt_in"
    assert plan["tiers"][1]["status"] == "attention"
    assert plan["tiers"][1]["required_action"] == "document and prove app-server as an explicit opt-in exception"
    assert plan["tiers"][4]["status"] == "attention"
    assert "MCP" in plan["tiers"][4]["required_action"]
    assert plan["tiers"][5]["status"] == "attention"
    assert plan["tiers"][6]["status"] == "ok"



def test_codex_app_server_rejects_openai_platform_api_key_fallback_without_leaking_key():
    api_key = "sk-proj-this-secret-must-not-appear"

    with pytest.raises(RouteContractError) as exc_info:
        verify_agent_route_contract(
            surface="primary",
            provider="openai-codex",
            model="gpt-5.5",
            api_mode="codex_app_server",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key=api_key,
            raise_on_error=True,
        )

    err = exc_info.value
    assert err.code == "codex_app_server_requires_oauth"
    assert api_key not in str(err)
    assert "sk-proj" not in str(err)
    assert err.proof["contract"]["status"] == "blocked"
    assert err.proof["contract"]["violations"][0]["code"] == "codex_app_server_requires_oauth"


def test_route_policy_rejects_forbidden_per_token_fallback_surface():
    proof = verify_agent_route_contract(
        surface="cron",
        provider="openrouter",
        model="openai/gpt-5.5",
        api_mode="chat_completions",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-secret",
        fallback_activated=True,
        policy={"allowed_cost_surfaces": ["subscription", "local"]},
        raise_on_error=False,
    )

    assert proof["surface"] == "cron"
    assert proof["cost_surface"] == "per_token_api"
    assert proof["fallback_activated"] is True
    assert proof["contract"]["status"] == "blocked"
    assert [v["code"] for v in proof["contract"]["violations"]] == ["per_token_api_forbidden"]


def test_agent_init_enforces_route_contract_before_work(monkeypatch):
    from run_agent import AIAgent

    monkeypatch.setattr("run_agent.OpenAI", lambda **kwargs: object())

    with pytest.raises(RouteContractError) as exc_info:
        AIAgent(
            model="gpt-5.5",
            provider="openai-codex",
            api_mode="codex_app_server",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="sk-proj-this-secret-must-not-appear",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert exc_info.value.code == "codex_app_server_requires_oauth"
    assert "sk-proj" not in str(exc_info.value)
