"""Canonical observed provenance must retain derived data-use constraints."""
import json


def test_persisted_citations_keep_requirements_from_prior_context(tmp_path):
    from agent.connected_assistance import scan_connected_sources
    from gateway.connected_assistance_store import AssistanceStore
    from tests.tools.test_connected_source_capabilities import SourceTransport
    from tests.agent.test_connected_assistance import completion, selected

    sources = SourceTransport().sources()
    capability = sources.discover()[0]
    prior = [{"source_id": capability.source_id, "capability_id": capability.capability_id,
              "record_id": "old", "json_pointer": "", "content": {"id": "old", "text": "Earlier context"},
              "data_use": ["zero_retention"]}]

    def model(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        if "capabilities" in payload:
            return completion(selected(payload))
        record = payload["records"][0]
        citation = {key: record[key] for key in ("source_id", "record_id", "json_pointer")}
        return completion({"findings": [{"kind": "direct_request", "entity_key": "Atlas",
            "title": "Atlas review", "summary": "Review requested", "why_now": "Please review Atlas today",
            "suggested_action": "review", "evidence": [citation | {"excerpt": "Please review Atlas today"}]}]})

    result = scan_connected_sources(prior, sources=sources, llm=model)
    assert result["errors"] == []
    before = result["findings"][0]["evidence"][0]["data_use"]
    assert "zero_retention" in before
    saved = AssistanceStore(tmp_path).ingest_scan(result)[0]
    assert saved["evidence"][0]["data_use"] == before
    assert set(saved["data_use"]) <= set(saved["evidence"][0]["data_use"])
