"""Offline service-to-default-planner pipeline; no live app or model requests."""

import importlib
import json
from types import SimpleNamespace

import pytest

from tests.tools.test_connected_source_capabilities import SourceTransport


def completion(value):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))])


def test_default_pipeline_discovers_reads_and_judges_existing_connections(monkeypatch):
    from agent import auxiliary_client
    from tools.connectors.gateway import bridge, client, config
    from tools.connectors.portal import client as portal

    service = SourceTransport()
    monkeypatch.setattr(bridge, "guest_identity_pending", lambda: False)
    monkeypatch.setattr(config, "connectors_available", lambda: True)
    for module in (client, portal):
        monkeypatch.setattr(module, "_default_transport", lambda: service)
        monkeypatch.setattr(module, "_default_endpoint_resolver", lambda: "https://fixture.test")
        monkeypatch.setattr(module, "_default_header_provider", lambda _: {"Authorization": "Bearer fixture"})
    calls = []

    def model(**kwargs):
        calls.append(kwargs)
        payload = json.loads(kwargs["messages"][1]["content"])
        assert kwargs.get("tools") is None
        if len(calls) == 1:
            cap = payload["capabilities"][0]
            return completion({"reads": [{"capability_id": cap["capability_id"], "arguments": {"query": "to:me"}}]})
        record = payload["records"][0]
        assert kwargs["messages"][1]["data_use"] == ["no_training"]
        return completion({"findings": [{"kind": "direct_request", "entity_key": "Atlas",
            "title": "Atlas review", "summary": "A review was requested",
            "why_now": "Please review Atlas today", "suggested_action": "review", "evidence": [{
                "source_id": record["source_id"], "record_id": record["record_id"],
                "json_pointer": record["json_pointer"], "excerpt": "Please review Atlas today",
                "url": "https://mail.test/m1"}]}]})

    monkeypatch.setattr(auxiliary_client, "call_llm", model)
    assistance = importlib.import_module("agent.connected_assistance")
    result = assistance.scan_connected_sources()
    assert result["connected"] is True
    assert result["errors"] == []
    assert len(result["findings"]) == 1
    assert result["findings"][0]["evidence"][0]["record_id"] == "m1"
    assert result["evidence"][0]["data_use"] == ["no_training"]
    assert len(calls) == 2
    assert len([url for _, url, _ in service.requests if url.endswith("/execute")]) == 1
    assert assistance.revalidate_evidence(result["findings"][0]["evidence"])
    service.connected = False
    assert not assistance.revalidate_evidence(result["findings"][0]["evidence"])


def selected(payload, count=1):
    cap = payload["capabilities"][0]
    return {"reads": [{"capability_id": cap["capability_id"], "arguments": {"query": "to:me"}}
                       for _ in range(count)]}


def test_seventh_read_is_rejected_before_any_dispatch():
    from agent.connected_assistance import scan_connected_sources

    service = SourceTransport()
    calls = []

    def model(**kw):
        calls.append(kw)
        return completion(selected(json.loads(kw["messages"][1]["content"]), count=7))

    result = scan_connected_sources(sources=service.sources(), llm=model)
    assert result["findings"] == []
    assert result["errors"] == ["invalid_plan"]
    assert len(calls) == 1
    assert not any(url.endswith("/execute") for _, url, _ in service.requests)


def test_six_reads_and_two_model_calls_are_the_hard_upper_bound():
    from agent.connected_assistance import scan_connected_sources

    service = SourceTransport()
    calls = []

    def model(**kw):
        calls.append(kw)
        if len(calls) == 1:
            return completion(selected(json.loads(kw["messages"][1]["content"]), count=6))
        return completion({"findings": []})

    result = scan_connected_sources(sources=service.sources(), llm=model)
    assert result["errors"] == []
    assert len(calls) == 2
    assert len([url for _, url, _ in service.requests if url.endswith("/execute")]) == 6


def test_oversized_source_payload_does_not_reach_judge():
    from agent.connected_assistance import scan_connected_sources

    service = SourceTransport()
    service.data["text"] = "sensitive" * 9000
    calls = []

    def model(**kw):
        calls.append(kw)
        return completion(selected(json.loads(kw["messages"][1]["content"])))

    result = scan_connected_sources(sources=service.sources(), llm=model)
    assert result["errors"] == ["payload_limit"]
    assert result["findings"] == result["evidence"] == []
    assert len(calls) == 1


def test_foreign_or_revoked_prior_evidence_is_not_sent_to_planner():
    from agent.connected_assistance import scan_connected_sources

    service = SourceTransport()
    calls = []

    def model(**kw):
        calls.append(kw)
        assert json.loads(kw["messages"][1]["content"])["prior_evidence"] == []
        return completion({"reads": []})

    result = scan_connected_sources([{"source_id": "another-profile", "record_id": "secret",
                                      "excerpt": "foreign", "data_use": ["no_training"]}],
                                    sources=service.sources(), llm=model)
    assert result["errors"] == []
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["unknown_id", "unknown_name", "extra_authority", "invalid_arguments"])
def test_entire_plan_is_validated_before_any_read(mutation):
    from agent.connected_assistance import scan_connected_sources

    service = SourceTransport()

    def model(**kw):
        plan = selected(json.loads(kw["messages"][1]["content"]), count=2)
        if mutation == "unknown_id":
            plan["reads"][1]["capability_id"] = "send_message"
        elif mutation == "unknown_name":
            plan["reads"][1]["name"] = "terminal"
        elif mutation == "extra_authority":
            plan["config"] = {"enabled": True}
        else:
            plan["reads"][1]["arguments"] = {"query": 123}
        return completion(plan)

    result = scan_connected_sources(sources=service.sources(), llm=model)
    assert result["findings"] == []
    assert result["errors"] == ["invalid_plan"]
    assert not any(url.endswith("/execute") for _, url, _ in service.requests)


def test_expired_deadline_makes_no_source_or_model_requests():
    from agent.connected_assistance import scan_connected_sources

    service = SourceTransport()
    result = scan_connected_sources(sources=service.sources(), deadline=0,
                                    llm=lambda **_: pytest.fail("model called after deadline"))
    assert result["errors"] == ["deadline_exceeded"]
    assert service.requests == []


def test_no_identity_is_never_created_by_default_discovery(monkeypatch):
    from agent.connected_assistance import scan_connected_sources
    from tools.connectors.gateway import bridge, config

    monkeypatch.setattr(config, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "guest_identity_pending", lambda: True)
    monkeypatch.setattr(config, "ensure_guest_identity", lambda: pytest.fail("authorization initiated"))
    result = scan_connected_sources(llm=lambda **_: pytest.fail("model called without connection"))
    assert result["connected"] is False
    assert result["findings"] == []
    assert "manage_connections" not in json.dumps(result)


def test_prior_and_observed_data_use_survive_both_prompts_and_derived_findings():
    from agent.connected_assistance import scan_connected_sources

    service = SourceTransport()
    sources = service.sources()
    cap = sources.discover()[0]
    prior = [{"source_id": cap.source_id, "capability_id": cap.capability_id,
              "record_id": "old", "excerpt": "previous review", "data_use": ["zero_retention"]}]
    calls = []

    def model(**kw):
        calls.append(kw)
        assert "zero_retention" in kw["messages"][1]["data_use"]
        payload = json.loads(kw["messages"][1]["content"])
        if len(calls) == 1:
            return completion(selected(payload))
        assert kw["messages"][1]["data_use"] == ["no_training", "zero_retention"]
        record = payload["records"][0]
        return completion({"findings": [{"kind": "direct_request", "entity_key": "Atlas", "title": "Atlas review",
            "summary": "Review requested", "why_now": "Please review Atlas today", "suggested_action": "review",
            "evidence": [{"source_id": record["source_id"], "record_id": record["record_id"],
                          "json_pointer": record["json_pointer"], "excerpt": "Please review Atlas today"}]}]})

    result = scan_connected_sources(prior, sources=sources, llm=model)
    assert result["errors"] == []
    assert result["findings"][0]["data_use"] == ["no_training", "zero_retention"]


def test_inflight_revocation_prevents_a_derived_judge_prompt():
    from agent.connected_assistance import scan_connected_sources

    service = SourceTransport()
    original = service.request
    calls = []

    def request(method, url, **kwargs):
        reply = original(method, url, **kwargs)
        if url.endswith("/execute"):
            service.connected = False
        return reply

    def model(**kw):
        calls.append(kw)
        return completion(selected(json.loads(kw["messages"][1]["content"])))

    service.request = request
    result = scan_connected_sources(sources=service.sources(), llm=model)
    assert result["findings"] == result["evidence"] == []
    assert result["errors"] == ["source_access_changed"]
    assert len(calls) == 1


@pytest.mark.parametrize("data", [{"id": "big", "text": "x" * 17_000},
                                  [{"id": str(i), "text": "x"} for i in range(201)]])
def test_retained_evidence_respects_store_record_and_count_limits(data):
    from agent.connected_assistance import scan_connected_sources

    service = SourceTransport()
    service.data = data
    calls = []

    def model(**kw):
        calls.append(kw)
        return completion(selected(json.loads(kw["messages"][1]["content"])))

    result = scan_connected_sources(sources=service.sources(), llm=model)
    assert result["errors"] == ["payload_limit"]
    assert result["evidence"] == []
    assert len(calls) == 1


def test_revocation_during_malformed_judgment_discards_retained_evidence():
    from agent.connected_assistance import scan_connected_sources

    service = SourceTransport()
    calls = []

    def model(**kw):
        calls.append(kw)
        if len(calls) == 1:
            return completion(selected(json.loads(kw["messages"][1]["content"])))
        service.connected = False
        return completion({"findings": [], "send_message": "not authorized"})

    result = scan_connected_sources(sources=service.sources(), llm=model)
    assert result["findings"] == result["evidence"] == []
    assert result["errors"] == ["source_access_changed"]
    assert len(calls) == 2


def test_real_auxiliary_recovery_obeys_scan_deadline(tmp_path, monkeypatch):
    from agent import auxiliary_client as aux
    from agent import connected_assistance as pipeline
    from tools.connectors import assistance as source_module
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    service = SourceTransport()
    clock = [100.0]
    monkeypatch.setattr(pipeline, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(source_module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    attempts = []
    def create(**kwargs):
        attempts.append({'at':clock[0], 'timeout':kwargs.get('timeout')})
        if len(attempts) == 1:
            clock[0] = 161.0
            raise ConnectionError('connection reset by peer')
        return completion({'reads':[]})
    client = SimpleNamespace(base_url='https://fixture.test/v1', chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(aux, '_resolve_task_provider_model', lambda *a, **kw: ('custom','fixture','https://fixture.test/v1','fixture',None))
    monkeypatch.setattr(aux, '_get_cached_client', lambda *a, **kw: (client,'fixture'))
    monkeypatch.setattr(aux, '_TRANSIENT_RETRY_BACKOFF_BASE', 0)
    result = pipeline.scan_connected_sources(sources=service.sources(), timeout_seconds=60)
    print('AUX_AFTER_DEADLINE', {'attempts':attempts, 'result':result})
    assert len(attempts) == 1


def test_real_auxiliary_retries_fit_two_physical_requests(tmp_path, monkeypatch):
    from agent import auxiliary_client as aux
    from agent.connected_assistance import scan_connected_sources
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    service = SourceTransport()
    attempts = []
    def create(**kwargs):
        attempts.append(kwargs)
        if len(attempts) < 3:
            raise ConnectionError('connection reset by peer')
        payload = json.loads(kwargs['messages'][1]['content'])
        if 'capabilities' in payload:
            return completion({'reads':[{'capability_id':payload['capabilities'][0]['capability_id'], 'arguments':{'query':'to:me'}}]})
        return completion({'findings':[]})
    client = SimpleNamespace(base_url='https://fixture.test/v1', chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(aux, '_resolve_task_provider_model', lambda *a, **kw: ('custom','fixture','https://fixture.test/v1','fixture',None))
    monkeypatch.setattr(aux, '_get_cached_client', lambda *a, **kw: (client,'fixture'))
    monkeypatch.setattr(aux, '_TRANSIENT_RETRY_BACKOFF_BASE', 0)
    result = scan_connected_sources(sources=service.sources())
    print('AUX_PHYSICAL_BUDGET', {'attempts':len(attempts),'errors':result['errors']})
    assert len(attempts) <= 2
