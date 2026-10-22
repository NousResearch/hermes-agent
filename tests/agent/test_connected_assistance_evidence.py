"""Ground citations in actual records, never in a judge's invented provenance."""

import importlib

import pytest


def observation(source="connector:profile:gmail:account-1", text="Please review Atlas today"):
    return {"source_id": source, "capability_id": source + ":READ",
            "response": {"messages": [{"id": "m1", "text": text, "url": "https://mail.test/m1"}]},
            "data_use": ["no_training"]}


def candidate(source="connector:profile:gmail:account-1"):
    return {"kind": "direct_request", "entity_key": "Atlas", "title": "Atlas needs review",
            "summary": "A review was requested", "why_now": "Please review Atlas today",
            "suggested_action": "review", "evidence": [{"source_id": source, "record_id": "m1",
                "json_pointer": "/messages/0", "excerpt": "Please review Atlas today",
                "url": "https://mail.test/m1"}]}


def test_validated_citations_keep_real_record_location_and_data_use():
    evidence = importlib.import_module("agent.connected_assistance_evidence")
    records = evidence.observed_records([observation()])
    assert records[0]["excerpt"] == "Please review Atlas today"
    assert records[0]["url"] == "https://mail.test/m1"
    findings = evidence.validate_findings([candidate()], records)
    assert len(findings) == 1
    assert findings[0]["suggested_action"] == "Review the cited records."
    assert findings[0]["evidence"][0]["json_pointer"] == "/messages/0"
    assert findings[0]["evidence"][0]["data_use"] == ["no_training"]
    assert findings[0]["evidence"][0]["capability_id"] == records[0]["capability_id"]
    assert findings[0]["data_use"] == ["no_training"]


@pytest.mark.parametrize(("key", "value"), [
    ("record_id", "invented"), ("source_id", "another-account"),
    ("json_pointer", "/messages/9"), ("excerpt", "invented quote"), ("excerpt", ""),
    ("url", "https://evil.test/invented"),
])
def test_fabricated_or_empty_citations_are_silent(key, value):
    from agent.connected_assistance_evidence import observed_records, validate_findings

    finding = candidate()
    finding["evidence"][0][key] = value
    assert validate_findings([finding], observed_records([observation()])) == []


@pytest.mark.parametrize("mutation", [
    {"suggested_action": "send_email"}, {"kind": "cross_source"},
    {"summary": "See https://invented.test/secret"}, {"evidence": []},
    {"destination": "public-channel"}, {"kind": "deadline"},
])
def test_unsupported_actions_or_claims_are_silent(mutation):
    from agent.connected_assistance_evidence import observed_records, validate_findings

    finding = candidate() | mutation
    assert validate_findings([finding], observed_records([observation()])) == []


def test_duplicate_records_do_not_prove_two_sources():
    from agent.connected_assistance_evidence import observed_records, validate_findings

    finding = candidate() | {"kind": "cross_source"}
    finding["evidence"] *= 2
    assert validate_findings([finding], observed_records([observation()])) == []


def test_cross_source_accepts_two_observed_accounts_not_a_model_source_count():
    from agent.connected_assistance_evidence import observed_records, validate_findings

    second = "connector:profile:calendar:account-2"
    finding = candidate() | {"kind": "cross_source"}
    finding["evidence"].append(candidate(second)["evidence"][0])
    findings = validate_findings([finding], observed_records([observation(), observation(second)]))
    assert len(findings) == 1
    assert {e["source_id"] for e in findings[0]["evidence"]} == {observation()["source_id"], second}


def test_nonurgent_single_source_and_injected_data_use_are_not_authority():
    from agent.connected_assistance_evidence import observed_records, validate_findings

    finding = candidate()
    finding["why_now"] = "Normal project status"
    finding["evidence"][0]["excerpt"] = "Normal project status"
    assert validate_findings([finding], observed_records([observation(text="Normal project status")])) == []
    finding = candidate()
    finding["evidence"][0]["data_use"] = []
    assert validate_findings([finding], observed_records([observation()]))[0]["data_use"] == ["no_training"]


@pytest.mark.parametrize("value", [None, {}, "not JSON findings", [None], [{}]])
def test_malformed_judgment_fails_closed(value):
    from agent.connected_assistance_evidence import observed_records, validate_findings

    assert validate_findings(value, observed_records([observation()])) == []


def test_observed_content_digest_ignores_model_quote_and_forged_digest():
    from agent.connected_assistance_evidence import observed_records, validate_findings

    records = observed_records([observation(text="Please review Atlas today before lunch.")])
    first = candidate()
    first["evidence"][0]["content_digest"] = "model-selected"
    short = validate_findings([first], records)[0]["evidence"][0]
    first["evidence"][0]["excerpt"] += " before lunch."
    first["evidence"][0].pop("url")
    long = validate_findings([first], records)[0]["evidence"][0]
    assert len(records[0]["content_digest"]) == 64
    assert short["content_digest"] == long["content_digest"] == records[0]["content_digest"]
    # Even a forged digest on a retained record is not authority over its actual content.
    records[0]["content_digest"] = "forged"
    assert validate_findings([first], records)[0]["evidence"][0]["content_digest"] == short["content_digest"]
    updated = observed_records([observation(text="Please review Atlas today before dinner.")])
    assert updated[0]["content_digest"] != short["content_digest"]
