"""Selected-credential speed ceiling regression for #135024 (no live accounts)."""
import base64
import json
from types import SimpleNamespace

from agent import model_metadata as mm
from agent.client_lifecycle import ClientLifecycleMixin
from agent.fast_mode import effective_request_overrides
from agent.transports.codex import ResponsesApiTransport

BASE = "https://chatgpt.com/backend-api/codex"


def token(account):
    payload = {"https://api.openai.com/auth": {"chatgpt_account_id": account}}
    return "test." + base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=") + ".test"


def test_rotation_restores_requested_tier(monkeypatch):
    monkeypatch.setattr(mm, "_codex_oauth_context_cache", {})
    seen = []

    def get(url, **kwargs):
        account = kwargs["headers"]["ChatGPT-Account-ID"]
        seen.append(account)
        tiers = ["priority", "ultrafast"] if account == "B" else ["priority"]
        return SimpleNamespace(status_code=200, json=lambda: {"models": [{
            "slug": "gpt-6-astra", "context_window": 272000,
            "service_tiers": [{"id": tier} for tier in tiers],
        }]})

    monkeypatch.setattr(mm.model_metadata_http, "get", get)
    agent = SimpleNamespace(
        provider="openai-codex", api_mode="codex_responses", base_url=BASE,
        api_key=token("B"), model="gpt-6-astra-900k", service_tier="ultrafast",
        request_overrides={"service_tier": "ultrafast"}, _client_kwargs={},
        _reapply_route_client_config=lambda **kwargs: None,
        _replace_primary_openai_client=lambda **kwargs: None,
    )
    emitted = []
    for account in ("B", "A", "B"):
        entry = SimpleNamespace(id=account, access_token=token(account), base_url=BASE)
        assert ClientLifecycleMixin._swap_credential(agent, entry)
        overrides = effective_request_overrides(agent)
        wire = ResponsesApiTransport().build_kwargs(
            model=agent.model, messages=[{"role": "user", "content": "hello"}], tools=[],
            request_overrides=overrides, provider=agent.provider, base_url=BASE, is_codex_backend=True,
        )
        emitted.append(wire.get("service_tier"))
        assert wire["model"] == "gpt-6-astra"
        assert agent.request_overrides == {"service_tier": "ultrafast"}
        assert agent.service_tier == "ultrafast"
    assert emitted == ["ultrafast", "priority", "ultrafast"]
    assert seen == ["B", "A"]


def test_catalog_ceiling_unknown_expiry_and_route_isolation(monkeypatch):
    from agent import codex_service_tiers as tiers
    from agent.fast_mode import effective_request_overrides

    monkeypatch.setattr(mm, "_codex_oauth_context_cache", {})
    monkeypatch.setattr(tiers, "_catalogs", {})
    clock = [10000.0]
    monkeypatch.setattr(tiers.time, "time", lambda: clock[0])
    catalog = [{"slug": "gpt-6-astra", "context_window": 272000, "service_tiers": []}]
    calls = []
    failed = [False]

    def get(url, **kwargs):
        calls.append((url, kwargs["headers"]["ChatGPT-Account-ID"]))
        if failed[0]:
            raise RuntimeError("offline")
        return SimpleNamespace(status_code=200, json=lambda: {"models": catalog})

    monkeypatch.setattr(mm.model_metadata_http, "get", get)
    agent = SimpleNamespace(provider="openai-codex", api_mode="codex_responses", base_url=BASE,
                            api_key=token("C"), model="gpt-6-astra", request_overrides={"service_tier": "ultrafast"})
    assert "service_tier" not in effective_request_overrides(agent)
    # Empty lists are explicit Standard-only; absent fields mean unknown.
    catalog[0] = {"slug": "gpt-6-astra", "context_window": 272000}
    agent.api_key = token("D")
    assert effective_request_overrides(agent)["service_tier"] == "ultrafast"
    catalog[0]["additional_speed_tiers"] = ["fast", "ultrafast"]
    agent.api_key = token("E")
    agent.request_overrides = {"service_tier": "priority"}
    assert effective_request_overrides(agent)["service_tier"] == "priority"
    agent.request_overrides = {}
    assert effective_request_overrides(agent) == {}
    agent.request_overrides = {"service_tier": "ultrafast"}
    agent.model = "another-model"
    assert effective_request_overrides(agent)["service_tier"] == "ultrafast"
    agent.model = "gpt-6-astra"
    agent.api_key = token("C")
    assert "service_tier" not in effective_request_overrides(agent)
    clock[0] += 3601
    failed[0] = True
    assert effective_request_overrides(agent)["service_tier"] == "ultrafast"
    count = len(calls)
    assert effective_request_overrides(agent)["service_tier"] == "ultrafast"
    assert len(calls) == count  # negative discovery cache, no per-request probing
    for mode, provider, base in [("codex", "openai-codex", BASE),
                                 ("codex_responses", "openai", BASE),
                                 ("codex_responses", "openai-codex", "https://proxy.example/codex")]:
        agent.api_mode, agent.provider, agent.base_url = mode, provider, base
        assert effective_request_overrides(agent)["service_tier"] == "ultrafast"
    assert len(calls) == count

    # Codex's legacy ModelPreset fixture explicitly has an empty modern list
    # alongside legacy fast; serde defaults can produce this transitional shape.
    assert tiers._advertised_tiers({"service_tiers": [], "additional_speed_tiers": ["fast"]}) == frozenset({"priority"})
    assert tiers._advertised_tiers({"service_tiers": [{"id": "priority"}],
                                    "additional_speed_tiers": ["fast", "ultrafast"]}) == frozenset({"priority"})
