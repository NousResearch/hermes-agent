"""Frozen final-wire contracts exercised with inert physical egress."""
import httpx
import pytest

from agent.final_wire_admission import (
    COVERED_MAIN, FinalAttemptIdentity, ProviderBoundUnsupportedAccounting,
    bind_attempt_identity, wrap_httpx_client_transports,
)


def identity(family="chat_completions", window=1000):
    return FinalAttemptIdentity(COVERED_MAIN, family, "inert-model", "https://inert.invalid", window, "inert-attempt")


def test_unknown_context_field_valid_json_refuses_before_physical_delegate():
    delegated = []
    with httpx.Client(transport=httpx.MockTransport(lambda r: delegated.append(r) or httpx.Response(200, json={}))) as client:
        wrap_httpx_client_transports(client)
        with bind_attempt_identity(identity()):
            with pytest.raises(ProviderBoundUnsupportedAccounting):
                client.post("https://inert.invalid/chat/completions", json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "unknown_context": {"prompt": "unclassified"},
                })
    assert delegated == []


@pytest.mark.parametrize("family,body", [
    ("chat_completions", {"messages": [{"role": "user", "content": "hi"}]}),
    ("codex_responses", {"input": "hi"}),
    ("bedrock_converse", {"messages": [{"role": "user", "content": [{"text": "hi"}]}]}),
])
def test_optional_omission_has_no_numeric_reservation(family, body):
    from agent.final_wire_admission import project_final_body, PROVIDER_DEFAULT_UNRESOLVED, admit_snapshot
    snapshot = project_final_body(body, identity(family))
    assert snapshot.reservation_state == PROVIDER_DEFAULT_UNRESOLVED
    assert snapshot.resolved_r is None
    admit_snapshot(snapshot)
