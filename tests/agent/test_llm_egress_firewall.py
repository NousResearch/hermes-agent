"""Focused contract tests for the source-bound LLM egress firewall."""

from __future__ import annotations

import base64
import json
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.llm_egress_firewall import (
    AuthorizedEgress,
    DestinationClass,
    EgressBlocked,
    EgressDecision,
    GeneratedContextSegment,
    LLMEgressFirewall,
    LiteralSegment,
    OutboundText,
    SanitizedSegment,
    SourceGrant,
    SourceBoundSegment,
    SourcePresentationSegment,
    TypedOutboundRequest,
    ValidatedToolSyntaxSegment,
    classify_destination,
    redact_remote_unsafe_text,
    source_grant_digest,
    static_literal_sha256,
)


def _route(**overrides):
    values = {"provider": "custom", "model": "test-model", "base_url": "https://llm.example.test/v1", "api_mode": None}
    values.update(overrides)
    return SimpleNamespace(**values)


def _source_grant(path: Path, *, request_id: str = "req-1", start: int = 1, end: int = 1):
    content = path.read_bytes().splitlines(keepends=True)[start - 1 : end]
    payload = b"".join(content)
    return SourceGrant(
        canonical_path=path.resolve(),
        display_path="src/private.py",
        line_start=start,
        line_end=end,
        content_sha256=sha256(payload).hexdigest(),
        byte_count=len(payload),
        session_id="session-1",
        turn_id="turn-1",
        request_id=request_id,
        policy_digest="policy-1",
    )


def _request(payload: str = "ordinary prompt"):
    return {
        "messages": [{"role": "user", "content": payload}],
        "session_id": "session-1",
        "turn_id": "turn-1",
        "request_id": "req-1",
        "policy_digest": "policy-1",
    }


def _typed_request(request, *, source_grant: SourceGrant | None = None):
    def typed(value):
        if isinstance(value, str):
            return LiteralSegment(value)
        if isinstance(value, list):
            return [typed(item) for item in value]
        if isinstance(value, dict):
            return {key: typed(item) for key, item in value.items()}
        return value

    # Request identity is control-plane metadata for grant binding and receipts,
    # not provider payload. Keep only the logical provider request here.
    payload = typed({"messages": request["messages"]})
    if source_grant is not None:
        payload["messages"][0]["content"] = OutboundText(
            (SourceBoundSegment(source_grant_digest(source_grant)),)
        )
    return TypedOutboundRequest(
        payload=payload,
        session_id=request["session_id"],
        turn_id=request["turn_id"],
        request_id=request["request_id"],
        policy_digest=request["policy_digest"],
    )


def _sanitized_request(text: str) -> TypedOutboundRequest:
    return TypedOutboundRequest(
        payload={"messages": [{"role": LiteralSegment("user"), "content": SanitizedSegment(text)}]},
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )


_COMMON_STATIC_LITERALS = {
    "messages",
    "role",
    "content",
    "user",
    "session_id",
    "session-1",
    "turn_id",
    "turn-1",
    "request_id",
    "req-1",
    "policy_digest",
    "policy-1",
}


def firewall(tmp_path: Path, *, static_literals=(), **kwargs) -> LLMEgressFirewall:
    allowed = _COMMON_STATIC_LITERALS | set(static_literals)
    return LLMEgressFirewall(
        tmp_path,
        static_literal_hashes_by_policy={
            "policy-1": frozenset(static_literal_sha256(value) for value in allowed)
        },
        **kwargs,
    )


def test_lan_and_unknown_are_remote_while_numeric_loopback_is_loopback():
    assert classify_destination("ollama", "http://127.0.0.1:11434", None).value == "loopback"
    assert classify_destination("custom", "http://192.168.1.9:8000", None).value == "remote"
    assert classify_destination("custom", None, None).value == "unknown"


def test_destination_classification_does_not_trust_dns_or_provider_name():
    assert classify_destination("ollama", "http://localhost:11434", None) in {
        DestinationClass.REMOTE,
        DestinationClass.UNKNOWN,
    }
    assert classify_destination("ollama", "http://ollama:11434", None) == DestinationClass.REMOTE
    assert classify_destination("custom", "not-a-url", None) == DestinationClass.UNKNOWN
    assert classify_destination("custom", "http://[::1]:8000", None) == DestinationClass.LOOPBACK
    assert classify_destination("custom", "http://100.64.0.1:8000", None) == DestinationClass.REMOTE


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:99999",
        "http://127.0.0.1:not-a-port",
        "http://[::1]:99999",
    ],
)
def test_malformed_loopback_ports_are_unknown(url):
    assert classify_destination("custom", url, None) == DestinationClass.UNKNOWN


def test_explicit_in_process_api_mode_is_local_process():
    assert classify_destination("custom", None, "local_process") == DestinationClass.LOCAL_PROCESS
    assert classify_destination("custom", None, "in_process") == DestinationClass.LOCAL_PROCESS


def test_source_grant_is_immutable_and_has_the_required_binding_fields(tmp_path):
    path = tmp_path / "private.py"
    path.write_text("private source\n", encoding="utf-8")
    grant = _source_grant(path)
    with pytest.raises(FrozenInstanceError):
        grant.request_id = "forged"  # type: ignore[misc]
    assert grant.canonical_path == path.resolve()
    assert grant.line_start == 1 and grant.line_end == 1
    assert grant.byte_count == len(b"private source\n")


def test_loopback_request_is_allowed_without_source_grant(tmp_path):
    decision = firewall(tmp_path).preflight(_request("local prompt"), _route(base_url="http://127.0.0.1:11434"))
    assert decision.allowed is True
    assert decision.destination_class == DestinationClass.LOOPBACK


def test_remote_request_requires_an_exact_source_grant(tmp_path):
    path = tmp_path / "private.py"
    path.write_text("x=1\npublic source\n", encoding="utf-8")
    grant = _source_grant(path)
    request = _request("private source\n")
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(_typed_request(request), _route(), grants=())
    assert "untrusted_provenance" in exc_info.value.decision.reason_codes

    decision = firewall(tmp_path).preflight(
        _typed_request(request, source_grant=grant),
        _route(),
        grants=(grant,),
    )
    assert isinstance(decision, EgressDecision)
    assert decision.allowed is True
    assert decision.payload_sha256


def test_unrelated_grant_and_incomplete_typed_source_request_are_denied(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("first source\n", encoding="utf-8")
    second.write_text("second source\n", encoding="utf-8")
    first_grant = _source_grant(first)
    second_grant = _source_grant(second)
    request = _request("first source\n")
    with pytest.raises(EgressBlocked) as unrelated_exc:
        firewall(tmp_path).preflight(
            _typed_request(request, source_grant=second_grant),
            _route(),
            grants=(first_grant,),
        )
    assert "source_segment_grant_mismatch" in unrelated_exc.value.decision.reason_codes

    with pytest.raises(EgressBlocked) as incomplete_exc:
        firewall(tmp_path).preflight(
            _typed_request(request, source_grant=first_grant),
            _route(),
            grants=(first_grant, second_grant),
        )
    assert "source_grant_unbound" in incomplete_exc.value.decision.reason_codes


def test_remote_raw_request_is_never_authorized_even_with_a_valid_grant(tmp_path):
    path = tmp_path / "private.py"
    path.write_text("private source\n", encoding="utf-8")
    grant = _source_grant(path)
    raw_request = _request("private source\n")
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(raw_request, _route(), grants=(grant,))
    assert "typed_request_required" in exc_info.value.decision.reason_codes


def test_source_bytes_cannot_be_smuggled_as_a_sanitized_literal(tmp_path):
    path = tmp_path / "private.py"
    path.write_text("private source\n", encoding="utf-8")
    grant = _source_grant(path)
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(
            _typed_request(_request("private source\n")),
            _route(),
            grants=(grant,),
        )
    assert "source_bytes_in_literal" in exc_info.value.decision.reason_codes
    assert "source_grant_unbound" in exc_info.value.decision.reason_codes


def test_only_exact_policy_allowlisted_static_wrappers_accompany_source(tmp_path):
    path = tmp_path / "private.py"
    path.write_text("x=1\n", encoding="utf-8")
    grant = _source_grant(path)
    typed_request = _typed_request(_request("ignored"), source_grant=grant)
    typed_request.payload["messages"][0]["content"] = OutboundText(
        (
            LiteralSegment("<source>\n"),
            SourceBoundSegment(source_grant_digest(grant)),
            LiteralSegment("</source>"),
        )
    )
    gate = firewall(tmp_path, static_literals={"<source>\n", "</source>"})
    authorization = gate.authorize(typed_request, _route(), grants=(grant,))
    assert json.loads(authorization.payload_bytes)["messages"][0]["content"] == (
        "<source>\nx=1\n</source>"
    )

    typed_request.payload["messages"][0]["content"] = OutboundText(
        (
            LiteralSegment("<changed>\n"),
            SourceBoundSegment(source_grant_digest(grant)),
            LiteralSegment("</source>"),
        )
    )
    with pytest.raises(EgressBlocked) as changed_exc:
        gate.preflight(typed_request, _route(), grants=(grant,))
    assert "static_literal_not_allowed" in changed_exc.value.decision.reason_codes

    typed_request.payload["messages"][0]["content"] = OutboundText(
        (
            SanitizedSegment("<source>\n"),
            SourceBoundSegment(source_grant_digest(grant)),
            LiteralSegment("</source>"),
        )
    )
    sanitized = gate.authorize(typed_request, _route(), grants=(grant,))
    assert json.loads(sanitized.payload_bytes)["messages"][0]["content"] == (
        "<source>\nx=1\n</source>"
    )


def test_bounded_sanitized_remote_text_needs_no_source_grant(tmp_path):
    request = TypedOutboundRequest(
        payload={"messages": [{"role": SanitizedSegment("user"), "content": SanitizedSegment("Fix CI now.")}]},
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )
    authorization = firewall(tmp_path).authorize(request, _route(), grants=())
    assert json.loads(authorization.payload_bytes) == {
        "messages": [{"content": "Fix CI now.", "role": "user"}]
    }
    assert authorization.decision.source_grant_count == 0


def test_remote_sanitized_request_requires_complete_request_identity(tmp_path):
    request = TypedOutboundRequest(
        payload={"messages": [SanitizedSegment("Fix CI now.")]},
        session_id="session-1",
        turn_id="",
        request_id="req-1",
        policy_digest="policy-1",
    )

    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).authorize(request, _route(), grants=())

    assert "missing_request_identity" in exc_info.value.decision.reason_codes


def test_sanitized_remote_text_has_an_independent_exact_byte_cap(tmp_path):
    request = TypedOutboundRequest(
        payload={"messages": [SanitizedSegment("CI fix!!")]},
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )
    assert firewall(tmp_path, max_sanitized_bytes=8).preflight(request, _route()).allowed
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path, max_sanitized_bytes=7).preflight(request, _route())
    assert "sanitized_bytes_exceeded" in exc_info.value.decision.reason_codes


def test_sanitized_segment_cap_remains_independent_from_larger_aggregate_cap(tmp_path):
    request = _sanitized_request("ordinary sentence. " * 2_000)
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(
            tmp_path,
            max_sanitized_segment_bytes=32_768,
            max_sanitized_bytes=2_000_000,
            max_serialized_bytes=2_000_000,
            max_conservative_tokens=666_667,
        ).preflight(request, _route())
    assert "sanitized_segment_bytes_exceeded" in exc_info.value.decision.reason_codes


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("token=super-secret-value", "secret_detected"),
        (base64.b64encode(b"encoded private detail").decode("ascii"), "base64_payload"),
        ("Read /Users/private/repository/file.py", "private_absolute_path"),
        (r"Read C:\\Users\\private\\secrets.txt", "private_absolute_path"),
    ],
)
def test_sanitized_remote_text_denies_secrets_encoding_and_private_paths(
    tmp_path,
    text,
    reason,
):
    request = TypedOutboundRequest(
        payload={"messages": [SanitizedSegment(text)]},
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(request, _route())
    assert reason in exc_info.value.decision.reason_codes


def test_source_bytes_cannot_be_relabelled_as_sanitized_text(tmp_path):
    path = tmp_path / "private.py"
    path.write_text("private source\n", encoding="utf-8")
    grant = _source_grant(path)
    request = TypedOutboundRequest(
        payload={"messages": [SanitizedSegment("private source\n")]},
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(request, _route(), grants=(grant,))
    assert "source_bytes_in_sanitized_segment" in exc_info.value.decision.reason_codes


def test_firewall_binds_requests_to_its_configured_policy_digest(tmp_path):
    request = TypedOutboundRequest(
        payload={"messages": [LiteralSegment("ordinary prompt")]},
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(
            tmp_path,
            policy_digest="policy-2",
            static_literals={"ordinary prompt"},
        ).preflight(
            request,
            _route(base_url="http://127.0.0.1:1234"),
            grants=(),
        )
    assert "policy_digest_mismatch" in exc_info.value.decision.reason_codes


@pytest.mark.parametrize(
    ("wrapper", "reason"),
    [
        ("token=super-secret-value\n", "secret_detected"),
        ("AQID\n", "base64_payload"),
    ],
)
def test_allowlisted_static_wrapper_still_passes_final_send_scan(tmp_path, wrapper, reason):
    path = tmp_path / "private.py"
    path.write_text("x=1\n", encoding="utf-8")
    grant = _source_grant(path)
    typed_request = _typed_request(_request("ignored"), source_grant=grant)
    typed_request.payload["messages"][0]["content"] = OutboundText(
        (
            LiteralSegment(wrapper),
            SourceBoundSegment(source_grant_digest(grant)),
        )
    )
    gate = firewall(tmp_path, static_literals={wrapper})
    with pytest.raises(EgressBlocked) as exc_info:
        gate.authorize(typed_request, _route(), grants=(grant,))
    assert reason in exc_info.value.decision.reason_codes
    assert not exc_info.value.decision.allowed


def test_second_source_cannot_be_mislabeled_as_allowlisted_literal(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("x=1\n", encoding="utf-8")
    second.write_text("y=2\n", encoding="utf-8")
    first_grant = _source_grant(first)
    second_grant = _source_grant(second)
    typed_request = _typed_request(_request("ignored"), source_grant=first_grant)
    typed_request.payload["messages"][0]["content"] = OutboundText(
        (
            SourceBoundSegment(source_grant_digest(first_grant)),
            LiteralSegment("y=2\n"),
        )
    )
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(
            typed_request,
            _route(),
            grants=(first_grant, second_grant),
        )
    assert "static_literal_not_allowed" in exc_info.value.decision.reason_codes
    assert "source_bytes_in_literal" in exc_info.value.decision.reason_codes
    assert "source_grant_unbound" in exc_info.value.decision.reason_codes


def test_source_bytes_are_constructed_from_grant_and_authorized_bytes_reject_tampering(tmp_path):
    path = tmp_path / "private.py"
    path.write_text("x=1\n", encoding="utf-8")
    grant = _source_grant(path)
    raw_request = _request("ignored raw bytes")
    typed_request = _typed_request(raw_request, source_grant=grant)
    authorization = firewall(tmp_path).authorize(typed_request, _route(), grants=(grant,))
    authorized_payload = json.loads(authorization.payload_bytes)
    assert authorized_payload["messages"][0]["content"] == "x=1\n"
    assert "ignored raw bytes" not in authorization.payload_bytes.decode("utf-8")

    typed_request.payload["messages"][0]["content"] = LiteralSegment("tampered")
    assert authorization.payload_bytes == authorization.verify_payload(authorization.payload_bytes)
    with pytest.raises(EgressBlocked) as exc_info:
        authorization.verify_payload(authorization.payload_bytes.replace(b"x=1", b"x=2"))
    assert "payload_digest_mismatch" in exc_info.value.decision.reason_codes


def test_dynamic_grammar_atom_still_requires_an_exact_current_source_grant(tmp_path):
    path = tmp_path / "private.py"
    path.write_text("user\n", encoding="utf-8")
    grant = _source_grant(path)
    typed_request = _typed_request(_request("ignored"), source_grant=grant)

    with pytest.raises(EgressBlocked) as missing_exc:
        firewall(tmp_path).preflight(typed_request, _route(), grants=())
    assert "untrusted_provenance" in missing_exc.value.decision.reason_codes

    authorization = firewall(tmp_path).authorize(typed_request, _route(), grants=(grant,))
    assert json.loads(authorization.payload_bytes)["messages"][0]["content"] == "user\n"

    path.write_text("tool\n", encoding="utf-8")
    with pytest.raises(EgressBlocked) as changed_exc:
        firewall(tmp_path).preflight(typed_request, _route(), grants=(grant,))
    assert "source_hash_mismatch" in changed_exc.value.decision.reason_codes


def test_typed_payload_identity_cannot_diverge_from_grant_binding_identity(tmp_path):
    path = tmp_path / "private.py"
    path.write_text("verified source\n", encoding="utf-8")
    grant = _source_grant(path)
    typed_request = _typed_request(_request("ignored"), source_grant=grant)
    typed_request.payload["request_id"] = LiteralSegment("different-request")
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(typed_request, _route(), grants=(grant,))
    assert "request_identity_mismatch" in exc_info.value.decision.reason_codes


def test_modified_source_and_adjacent_slice_are_denied(tmp_path):
    path = tmp_path / "private.py"
    path.write_text("private source\nneighbor\n", encoding="utf-8")
    grant = _source_grant(path)
    request = _request("private source\n")
    path.write_text("changed source\nneighbor\n", encoding="utf-8")
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(
            _typed_request(request, source_grant=grant),
            _route(),
            grants=(grant,),
        )
    assert "source_hash_mismatch" in exc_info.value.decision.reason_codes


def test_sensitive_path_and_forged_grant_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("API_KEY=sk-test-secret\n", encoding="utf-8")
    grant = _source_grant(path)
    sensitive_request = _request("API_KEY=sk-test-secret\n")
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(
            _typed_request(sensitive_request, source_grant=grant),
            _route(),
            grants=(grant,),
        )
    assert "sensitive_path" in exc_info.value.decision.reason_codes

    missing = tmp_path / "missing.py"
    forged = SourceGrant(
        canonical_path=missing,
        display_path="src/private.py",
        line_start=1,
        line_end=1,
        content_sha256=sha256(b"private source\n").hexdigest(),
        byte_count=len(b"private source\n"),
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )
    missing_request = _request("private source\n")
    with pytest.raises(EgressBlocked) as missing_exc:
        firewall(tmp_path).preflight(
            _typed_request(missing_request, source_grant=forged),
            _route(),
            grants=(forged,),
        )
    assert "source_unavailable" in missing_exc.value.decision.reason_codes


def test_forced_secret_redaction_rejects_instead_of_rewriting(tmp_path):
    path = tmp_path / "secret.txt"
    path.write_text("token=super-secret-value\n", encoding="utf-8")
    grant = _source_grant(path)
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(
            _typed_request(_request("ignored"), source_grant=grant),
            _route(),
            grants=(grant,),
        )
    assert "secret_detected" in exc_info.value.decision.reason_codes


@pytest.mark.parametrize(
    "payload",
    [
        base64.b64encode(b"sk-proj-abc123def456ghi789jkl012").decode("ascii"),
        base64.urlsafe_b64encode(b"harmless text that is still encoded").decode("ascii"),
        base64.b64encode(b"sk-x").decode("ascii"),
        base64.b64encode(b"sk-short").decode("ascii").rstrip("="),
        base64.urlsafe_b64encode(b"\xfb\xffshort-secret").decode("ascii").rstrip("="),
        base64.b64encode(b"\x01\x02\x03").decode("ascii"),
        base64.urlsafe_b64encode(b"\xfb\xff\x00").decode("ascii").rstrip("="),
    ],
)
def test_canonical_base64_payloads_are_rejected_even_when_decoded_content_is_benign(
    tmp_path,
    payload,
):
    path = tmp_path / "encoded.txt"
    path.write_text(payload + "\n", encoding="utf-8")
    grant = _source_grant(path)
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(
            _typed_request(_request("ignored"), source_grant=grant),
            _route(),
            grants=(grant,),
        )
    assert "base64_payload" in exc_info.value.decision.reason_codes


def test_base64_payload_split_by_whitespace_is_still_rejected(tmp_path):
    encoded = base64.b64encode(b"private source that must not leave the host").decode("ascii")
    split = " ".join(encoded[index : index + 2] for index in range(0, len(encoded), 2))
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(_sanitized_request(split), _route())
    assert "base64_payload" in exc_info.value.decision.reason_codes


def _source_presentation_request(grant: SourceGrant, text: str) -> TypedOutboundRequest:
    return TypedOutboundRequest(
        payload={
            "messages": [
                {
                    "role": LiteralSegment("user"),
                    "content": OutboundText(
                        (
                            SourcePresentationSegment(
                                source_grant_digest(grant),
                                text,
                                "read_file_json_v1",
                            ),
                        )
                    ),
                }
            ]
        },
        session_id=grant.session_id,
        turn_id=grant.turn_id,
        request_id=grant.request_id,
        policy_digest=grant.policy_digest,
    )


def test_source_presentation_allows_bounded_code_and_config_atoms(tmp_path):
    path = tmp_path / "pyproject.toml"
    source = (
        "[tool.mypy]\n"
        "warn_unused_ignores = true\n"
        "# ADVISORY: retain F401 and multi-value-repeated-key-literal\n"
    )
    path.write_text(source, encoding="utf-8")
    grant = _source_grant(path, end=3)
    presentation = json.dumps(
        {
            "content": "\n".join(
                f"{number}|{line}"
                for number, line in enumerate(source.split("\n"), start=1)
            )
        }
    )

    decision = firewall(tmp_path).preflight(
        _source_presentation_request(grant, presentation),
        _route(),
        grants=(grant,),
    )

    assert decision.allowed is True
    assert "base64_payload" not in decision.reason_codes


def test_source_presentation_scans_raw_source_not_json_line_number_artifacts(tmp_path):
    source_lines = ["PR_CI_RECEIPT_V1 = True", *[f"value_{number} = {number}" for number in range(1, 121)]]
    source = "\n".join(source_lines) + "\n"
    path = tmp_path / "receipt.py"
    path.write_text(source, encoding="utf-8")
    grant = _source_grant(path, end=len(source_lines))
    presentation = json.dumps(
        {
            "content": "\n".join(
                f"{number}|{line}"
                for number, line in enumerate(source.split("\n"), start=1)
            )
        }
    )

    decision = firewall(tmp_path).preflight(
        _source_presentation_request(grant, presentation),
        _route(),
        grants=(grant,),
    )

    assert decision.allowed is True
    assert "base64_payload" not in decision.reason_codes


def test_source_presentation_does_not_join_ordinary_short_prose_as_base64(tmp_path):
    source = 'message = "add the next step --"\n'
    path = tmp_path / "instructions.py"
    path.write_text(source, encoding="utf-8")
    grant = _source_grant(path)
    presentation = json.dumps({"content": '1|message = "add the next step --"\n2|'})

    decision = firewall(tmp_path).preflight(
        _source_presentation_request(grant, presentation),
        _route(),
        grants=(grant,),
    )

    assert decision.allowed is True
    assert "base64_payload" not in decision.reason_codes


def test_source_presentation_still_rejects_actual_base64_payload(tmp_path):
    encoded = base64.b64encode(b"private source that must not leave the host").decode(
        "ascii"
    )
    path = tmp_path / "encoded.txt"
    source = encoded + "\n"
    path.write_text(source, encoding="utf-8")
    grant = _source_grant(path)
    presentation = json.dumps({"content": f"1|{encoded}\n2|"})

    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(
            _source_presentation_request(grant, presentation),
            _route(),
            grants=(grant,),
        )

    assert "base64_payload" in exc_info.value.decision.reason_codes


@pytest.mark.parametrize(
    "text",
    [
        "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "commit 0123456789abcdef0123456789abcdef01234567 is already reviewed",
        "The prose mentions commit 0123456789abcdef0123456789abcdef01234567 here.",
    ],
)
def test_sha_and_commit_references_in_prose_are_not_base64_false_positives(tmp_path, text):
    decision = firewall(tmp_path).preflight(_sanitized_request(text), _route())
    assert "base64_payload" not in decision.reason_codes


@pytest.mark.parametrize(
    "text",
    [
        "Use systematic-debugging when a fix keeps not sticking.",
        "Call get_symbols_overview before reading a large file.",
        "The 50KB cap applies per attachment; retry after 1800 seconds.",
        "Templates live under references/templates/scripts in this skill.",
        "Bind ids/goals/status/transcripts before writing the summary.",
        "TODO: revisit this once the WAIT state clears; do not SKIP the check.",
        "PASS WARN SUMMARY REQUIREMENTS AVAILABILITY; 0x104e0860.",
    ],
)
def test_tool_and_skill_identifier_shapes_are_not_base64_false_positives(tmp_path, text):
    """live incident, 2026-08-28: moving profiles from a local (loopback)
    provider to a remote one (Nous) exposed them to egress scanning for the
    first time, and every one of them failed on message 1 -- the shared
    tool/skill descriptions are full of kebab-case slugs, snake_case
    function names, bare small numbers, and all-caps emphasis words that
    round-trip as valid unpadded Base64 by coincidence."""
    decision = firewall(tmp_path).preflight(_sanitized_request(text), _route())
    assert "base64_payload" not in decision.reason_codes


@pytest.mark.parametrize(
    "text",
    [
        "find smoke_rl_normalizer_20260823.py",
        "pytest tests/test_rl_agent_backtest_autosave.py -xvs",
        "Model fallback: devstral-small-2:24b",
        "read file L214-363; output 200p; result passed/1",
        "I missed part of the SHA1; commit a0a7a6dc1f82ef7a1309864a9eb0b41",
        "python -e ./package.json; rendered n--- marker",
    ],
)
def test_normal_worker_diagnostics_do_not_trigger_base64_block(tmp_path, text):
    """Worker diagnostics seen in the live crash log are not payloads."""
    decision = firewall(tmp_path).preflight(_sanitized_request(text), _route())
    assert "base64_payload" not in decision.reason_codes


def test_generated_kanban_context_allows_normal_worker_vocabulary(tmp_path):
    """Generated worker context must not trip the payload detector on prose."""
    request = TypedOutboundRequest(
        payload={
            "messages": [
                {
                    "role": LiteralSegment("user"),
                    "content": GeneratedContextSegment(
                        "profile ci-hygiene-fixer follows non-gate-weakening rules; "
                        "reproduction_command is recorded for HTTP HYGIENE checks."
                    ),
                }
            ]
        },
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )

    decision = firewall(tmp_path).preflight(request, _route())

    assert decision.allowed is True
    assert "base64_payload" not in decision.reason_codes
    assert redact_remote_unsafe_text(request.payload["messages"][0]["content"].text) == request.payload["messages"][0]["content"].text


def test_generated_kanban_context_still_rejects_real_base64(tmp_path):
    encoded = base64.b64encode(b"private source that must not leave the host").decode(
        "ascii"
    )
    request = TypedOutboundRequest(
        payload={
            "messages": [
                {"role": LiteralSegment("user"), "content": GeneratedContextSegment(encoded)}
            ]
        },
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )

    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(request, _route())

    assert "base64_payload" in exc_info.value.decision.reason_codes


def test_generated_context_attribution_prefix_does_not_skip_redaction(tmp_path):
    """Required attribution must not bypass sanitization of the rest of a prompt."""
    encoded = base64.b64encode(b"generated context that must not leave the host").decode(
        "ascii"
    )
    text = (
        "product=hermes-agent client=hermes-client-worker\n"
        f"encrypted replay: {encoded}\n"
    )
    redacted = redact_remote_unsafe_text(text)
    request = TypedOutboundRequest(
        payload={
            "messages": [
                {
                    "role": LiteralSegment("user"),
                    "content": GeneratedContextSegment(redacted),
                }
            ]
        },
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )

    decision = firewall(tmp_path).preflight(request, _route())

    assert decision.allowed is True
    assert encoded not in redacted
    assert "<redacted-base64>" in redacted


def test_bounded_kanban_show_structural_atoms_with_exact_receipt_ids_are_not_base64(
    tmp_path,
):
    tool_result = json.dumps(
        {
            "task": {
                "body": (
                    "expected head c2ba2e718019b2a9dbd044e89dd1d900290f5b5d; "
                    "receipt 0123456789abcdef0123456789abcdef"
                    "0123456789abcdef0123456789abcdef"
                ),
                "workspace_access": "dispatcher_current_directory",
            }
        },
        sort_keys=True,
    )

    decision = firewall(tmp_path).preflight(_sanitized_request(tool_result), _route())

    assert "base64_payload" not in decision.reason_codes


def test_representative_kanban_worker_evidence_is_not_mistaken_for_base64(tmp_path):
    tool_result = json.dumps(
        {
            "events": [{"payload": {"lock": 1831}}],
            "runs": [{"error": "runtime-executed elapsed 903s > limit 900s"}],
            "worker_context": (
                "Use PRAGMA under WSL1. Check _is_git_worktree, then "
                "claim/finalize/retry through prepare_receipt_worktree. "
                "Classify a logic-regression."
            ),
        },
        sort_keys=True,
    )

    decision = firewall(tmp_path).preflight(_sanitized_request(tool_result), _route())

    assert "base64_payload" not in decision.reason_codes
    assert "secret_detected" not in decision.reason_codes


def test_response_item_id_structural_key_is_not_mistaken_for_base64(tmp_path):
    request = TypedOutboundRequest(
        payload={
            "messages": [
                {
                    "role": LiteralSegment("user"),
                    "content": SanitizedSegment("continue the assigned work"),
                }
            ],
            "response_item_id": LiteralSegment("item-1"),
        },
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )

    decision = firewall(
        tmp_path,
        static_literals=("response_item_id", "item-1"),
    ).preflight(request, _route())

    assert "base64_payload" not in decision.reason_codes


def test_similar_unrecognized_underscore_atom_remains_base64_blocked(tmp_path):
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(
            _sanitized_request('{"workspace_access":"workspace_backup"}'),
            _route(),
        )

    assert "base64_payload" in exc_info.value.decision.reason_codes


def test_sanitized_proper_substring_of_grant_text_is_rejected(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("private source\n", encoding="utf-8")
    grant = _source_grant(path)
    request = TypedOutboundRequest(
        payload={"messages": [SanitizedSegment("private source")]},
        session_id=grant.session_id,
        turn_id=grant.turn_id,
        request_id=grant.request_id,
        policy_digest=grant.policy_digest,
    )
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(request, _route(), grants=(grant,))
    assert "source_bytes_in_sanitized_segment" in exc_info.value.decision.reason_codes


def test_common_short_word_in_grant_is_not_treated_as_source_excerpt(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("def run():\n    return 1\n", encoding="utf-8")
    grant = _source_grant(path, end=2)
    request = TypedOutboundRequest(
        payload={
            "messages": [
                SourceBoundSegment(source_grant_digest(grant)),
                SanitizedSegment("return"),
            ]
        },
        session_id=grant.session_id,
        turn_id=grant.turn_id,
        request_id=grant.request_id,
        policy_digest=grant.policy_digest,
    )

    decision = firewall(tmp_path).preflight(request, _route(), grants=(grant,))

    assert decision.allowed is True
    assert "source_bytes_in_sanitized_segment" not in decision.reason_codes


def test_common_eight_byte_overlap_is_not_treated_as_a_source_excerpt(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("private checkout_path implementation\n", encoding="utf-8")
    grant = _source_grant(path)
    request = TypedOutboundRequest(
        payload={
            "messages": [
                SourceBoundSegment(source_grant_digest(grant)),
                SanitizedSegment("checkout completed with clean status"),
            ]
        },
        session_id=grant.session_id,
        turn_id=grant.turn_id,
        request_id=grant.request_id,
        policy_digest=grant.policy_digest,
    )

    decision = firewall(tmp_path).preflight(request, _route(), grants=(grant,))

    assert decision.allowed is True
    assert "source_bytes_in_sanitized_segment" not in decision.reason_codes


def test_symlink_state_directory_is_rejected(tmp_path):
    target = tmp_path / "receipts-target"
    target.mkdir()
    state = tmp_path / "egress"
    state.symlink_to(target, target_is_directory=True)
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(state).preflight(_sanitized_request("safe request"), _route())
    assert "receipt_unavailable" in exc_info.value.decision.reason_codes


def test_ordinary_short_base64_alphabet_words_are_not_false_positives(tmp_path):
    decision = firewall(tmp_path).preflight(
        _request("ordinary short words remain usable"),
        _route(base_url="http://127.0.0.1:11434"),
    )
    assert "base64_payload" not in decision.reason_codes


def test_short_alphabetic_words_are_allowed_in_sanitized_remote_text(tmp_path):
    request = TypedOutboundRequest(
        payload={"messages": [SanitizedSegment("ordinary review carefully")]},
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )
    assert firewall(tmp_path).preflight(request, _route()).allowed


def test_json_protocol_acronym_is_not_a_base64_false_positive(tmp_path):
    request = TypedOutboundRequest(
        payload={"messages": [SanitizedSegment("Return JSON only.")]},
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )
    assert firewall(tmp_path).preflight(request, _route()).allowed


@pytest.mark.parametrize("diagnostic_code", ["E501", "F821", "F401", "W391"])
def test_linter_diagnostic_codes_are_not_base64_false_positives(
    tmp_path, diagnostic_code
):
    request = TypedOutboundRequest(
        payload={
            "messages": [
                SanitizedSegment(
                    f"hygiene findings: {diagnostic_code} E501 F821 W391"
                )
            ]
        },
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )

    decision = firewall(tmp_path).preflight(request, _route())

    assert decision.allowed is True
    assert "base64_payload" not in decision.reason_codes


def test_python_dunder_names_and_failure_status_are_not_base64_false_positives(
    tmp_path,
):
    request = _sanitized_request("__file__ __name__ __main__ FAIL")

    decision = firewall(tmp_path).preflight(request, _route())

    assert decision.allowed is True
    assert "base64_payload" not in decision.reason_codes


def test_bounded_relative_ci_path_is_not_a_base64_false_positive(tmp_path):
    request = _sanitized_request("venv/lib/python3")

    decision = firewall(tmp_path).preflight(request, _route())

    assert decision.allowed is True
    assert "base64_payload" not in decision.reason_codes


def test_source_presentation_allows_python_private_names(tmp_path):
    source = "_adapter _class_stack _format_findings _MutationVisitor\n"
    path = tmp_path / "owners.py"
    path.write_text(source, encoding="utf-8")
    grant = _source_grant(path)
    presentation = json.dumps({"content": f"1|{source}2|"})

    decision = firewall(tmp_path).preflight(
        _source_presentation_request(grant, presentation),
        _route(),
        grants=(grant,),
    )

    assert decision.allowed is True
    assert "base64_payload" not in decision.reason_codes


def test_source_presentation_allows_mixed_case_python_names(tmp_path):
    source = "visit_ImportFrom\n"
    path = tmp_path / "imports.py"
    path.write_text(source, encoding="utf-8")
    grant = _source_grant(path)
    presentation = json.dumps({"content": f"1|{source}2|"})

    decision = firewall(tmp_path).preflight(
        _source_presentation_request(grant, presentation),
        _route(),
        grants=(grant,),
    )

    assert decision.allowed is True
    assert "base64_payload" not in decision.reason_codes


@pytest.mark.parametrize(
    "schema_atom",
    [
        "HERMES_KANBAN_DB",
        "kanban_heartbeat",
        "machine-readable",
        "parent/child",
        "path/to/file",
        "skills/plugins/cron/memories",
        "MIME",
        "REQUIRED",
        "2000",
        "2026",
        "4dae",
        "HERMES_KANBAN_BRANCH",
        "HERMES_KANBAN_WORKSPACE",
        "max_runtime_seconds",
        "optional-profile",
        "HERMES_CONTROL_HOME",
        "HERMES_HOME=",
        "--repository",
        "already-resolved",
        "protected-remote",
    ],
)
def test_builtin_tool_schema_atoms_are_not_base64_false_positives(
    tmp_path, schema_atom
):
    request = TypedOutboundRequest(
        payload={"messages": [SanitizedSegment(schema_atom)]},
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )
    assert firewall(tmp_path).preflight(request, _route()).allowed


def test_fixed_hermes_task_id_is_not_a_base64_false_positive(tmp_path):
    request = TypedOutboundRequest(
        payload={"messages": [SanitizedSegment("work kanban task t_8c0aa909")]},
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )
    assert firewall(tmp_path).preflight(request, _route()).allowed


def test_fixed_prompt_cache_key_is_not_a_base64_false_positive(tmp_path):
    request = TypedOutboundRequest(
        payload={
            "parallel_tool_calls": True,
            "prompt_cache_key": SanitizedSegment("pck_e3aec8aaa5993646a80a5660"),
        },
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )
    assert firewall(
        tmp_path,
        static_literals={"parallel_tool_calls", "prompt_cache_key", "true"},
    ).preflight(request, _route()).allowed


@pytest.mark.parametrize(
    "tool_result",
    [
        '{"id":1117,"run_id":1125}',
        "HEAD OPEN/MERGEABLE/CLEAN d3b218473cc --noEmit n_error_",
    ],
)
def test_kanban_protocol_evidence_is_not_a_base64_false_positive(
    tmp_path, tool_result
):
    request = TypedOutboundRequest(
        payload={"messages": [SanitizedSegment(tool_result)]},
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )
    assert firewall(tmp_path).preflight(request, _route()).allowed


def test_content_free_violation_locations_never_return_keys_or_values():
    from agent.llm_egress_firewall import content_free_violation_locations

    private_path = "/Users/private/repository/file.py"
    encoded = base64.b64encode(b"encoded private detail").decode("ascii")
    result = content_free_violation_locations(
        {"private-key-name": [{"content-key": private_path}], "metadata-key": encoded}
    )
    rendered = repr(result)
    assert result == (
        ("$.map[0].key", ("base64_payload",)),
        ("$.map[0].value.sequence[0].map[0].value", ("private_absolute_path",)),
        ("$.map[1].value", ("base64_payload",)),
    )
    assert "private-key-name" not in rendered
    assert "content-key" not in rendered
    assert "metadata-key" not in rendered
    assert private_path not in rendered
    assert encoded not in rendered


def test_scanner_error_fails_closed(monkeypatch, tmp_path):
    def explode(*args, **kwargs):
        raise RuntimeError("scanner unavailable")

    monkeypatch.setattr("agent.llm_egress_firewall.redact_sensitive_text", explode)
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(
            _typed_request(_request()),
            _route(),
            grants=(object(),),
        )
    assert "redaction_failed" in exc_info.value.decision.reason_codes


def test_adjacent_sanitized_segments_cannot_hide_base64_across_boundaries(tmp_path):
    encoded = base64.b64encode(b"secret-material").decode("ascii")
    assert len(encoded) == 20
    request = TypedOutboundRequest(
        payload={
            "messages": [
                {
                    "role": LiteralSegment("user"),
                    "content": OutboundText(
                        tuple(
                            SanitizedSegment(encoded[offset : offset + 5])
                            for offset in range(0, len(encoded), 5)
                        )
                    ),
                }
            ]
        },
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )

    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).authorize(request, _route())

    assert "base64_payload" in exc_info.value.decision.reason_codes


def test_byte_and_token_caps_fail_closed(tmp_path):
    with pytest.raises(EgressBlocked) as bytes_exc:
        firewall(tmp_path, max_serialized_bytes=32).preflight(
            _typed_request(_request()),
            _route(),
            grants=(object(),),
        )
    assert "serialized_bytes_exceeded" in bytes_exc.value.decision.reason_codes

    with pytest.raises(EgressBlocked) as token_exc:
        firewall(tmp_path, max_serialized_bytes=100_000, max_conservative_tokens=2).preflight(
            _typed_request(_request("ordinary prompt")),
            _route(),
            grants=(object(),),
        )
    assert "token_cap_exceeded" in token_exc.value.decision.reason_codes


def test_byte_and_token_caps_allow_the_exact_boundary(tmp_path):
    request = _request("boundary prompt")
    serialized = json.dumps(
        request,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    token_count = (len(serialized) + 2) // 3
    decision = firewall(
        tmp_path,
        max_serialized_bytes=len(serialized),
        max_conservative_tokens=token_count,
    ).preflight(request, _route(base_url="http://127.0.0.1:11434"))
    assert decision.allowed is True


def test_larger_granted_caps_require_a_fully_valid_bound_source_segment(tmp_path):
    path = tmp_path / "large-source.txt"
    path.write_text("plain source sentence\n" * 24, encoding="utf-8")
    grant = _source_grant(path, end=24)
    request = _typed_request(_request("ignored"), source_grant=grant)
    gate = firewall(
        tmp_path,
        max_serialized_bytes=128,
        max_conservative_tokens=64,
        max_sanitized_bytes=2_048,
        max_granted_serialized_bytes=2_048,
        max_granted_conservative_tokens=1_024,
    )

    decision = gate.preflight(request, _route(), grants=(grant,))

    assert decision.allowed is True
    assert decision.serialized_bytes > 128
    assert decision.estimated_tokens > 64

    raw_request = _request(path.read_text(encoding="utf-8"))
    with pytest.raises(EgressBlocked) as raw_exc:
        gate.preflight(raw_request, _route(), grants=(grant,))
    assert "typed_request_required" in raw_exc.value.decision.reason_codes
    assert "serialized_bytes_exceeded" in raw_exc.value.decision.reason_codes

    sanitized_request = _sanitized_request(path.read_text(encoding="utf-8"))
    with pytest.raises(EgressBlocked) as sanitized_exc:
        gate.preflight(sanitized_request, _route())
    assert "serialized_bytes_exceeded" in sanitized_exc.value.decision.reason_codes
    assert "token_cap_exceeded" in sanitized_exc.value.decision.reason_codes


def test_invalid_grant_cannot_select_larger_granted_caps(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("plain source sentence\n", encoding="utf-8")
    mismatched_grant = _source_grant(path, request_id="another-request")
    request = _sanitized_request("bounded sanitized sentence " * 12)
    gate = firewall(
        tmp_path,
        max_serialized_bytes=128,
        max_conservative_tokens=64,
        max_sanitized_bytes=2_048,
        max_granted_serialized_bytes=2_048,
        max_granted_conservative_tokens=1_024,
    )

    with pytest.raises(EgressBlocked) as exc_info:
        gate.preflight(request, _route(), grants=(mismatched_grant,))

    reasons = exc_info.value.decision.reason_codes
    assert "grant_binding_mismatch" in reasons
    assert "serialized_bytes_exceeded" in reasons
    assert "token_cap_exceeded" in reasons


@pytest.mark.parametrize(
    ("source_text", "reason"),
    [
        ("token=super-secret-value\n", "secret_detected"),
        ("cHJpdmF0ZSBzb3VyY2UgdGhhdCBtdXN0IG5vdCBsZWF2ZQ==\n", "base64_payload"),
        ("Read /Users/private/repository/file.py\n", "private_absolute_path"),
    ],
)
def test_larger_granted_caps_do_not_bypass_content_scans(tmp_path, source_text, reason):
    path = tmp_path / "unsafe-source.txt"
    path.write_text(source_text, encoding="utf-8")
    grant = _source_grant(path)
    gate = firewall(
        tmp_path,
        max_serialized_bytes=8,
        max_conservative_tokens=4,
        max_granted_serialized_bytes=2_048,
        max_granted_conservative_tokens=1_024,
    )

    with pytest.raises(EgressBlocked) as exc_info:
        gate.preflight(
            _typed_request(_request("ignored"), source_grant=grant),
            _route(),
            grants=(grant,),
        )

    assert reason in exc_info.value.decision.reason_codes


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_typed_number_is_blocked_with_content_free_receipt(tmp_path, value):
    path = tmp_path / "private.py"
    path.write_text("x=1\n", encoding="utf-8")
    grant = _source_grant(path)
    typed_request = _typed_request(_request("ignored"), source_grant=grant)
    typed_request.payload["temperature"] = value
    gate = firewall(tmp_path, static_literals={"temperature"})

    with pytest.raises(EgressBlocked) as exc_info:
        gate.preflight(typed_request, _route(), grants=(grant,))
    assert exc_info.value.decision.reason_codes == ("non_finite_number",)

    receipt_text = (tmp_path / "llm-egress-receipts.jsonl").read_text()
    receipt = json.loads(receipt_text)
    assert receipt["decision"] == "block"
    assert receipt["reason_codes"] == ["non_finite_number"]
    assert "NaN" not in receipt_text
    assert "Infinity" not in receipt_text


def test_allow_receipt_contains_hashes_and_counts_but_no_payload(tmp_path):
    path = tmp_path / "private.py"
    path.write_text("x=1\n", encoding="utf-8")
    grant = _source_grant(path)
    request = _request("private source\n")
    decision = firewall(tmp_path).preflight(
        _typed_request(request, source_grant=grant),
        _route(),
        grants=(grant,),
    )
    receipt = json.loads((tmp_path / "llm-egress-receipts.jsonl").read_text().splitlines()[0])
    assert receipt["payload_sha256"] == decision.payload_sha256
    assert receipt["serialized_bytes"] == decision.serialized_bytes
    assert receipt["estimated_tokens"] == decision.estimated_tokens
    assert receipt["source_grants"][0]["content_sha256"] == grant.content_sha256
    assert "private source" not in json.dumps(receipt)
    assert "canonical_path" not in receipt["source_grants"][0]


def test_receipt_hashes_unsafe_identity_and_route_labels(tmp_path):
    secret_request_id = "sk-proj-abc123def456ghi789jkl012"
    private_model_label = str(Path.home() / "private-model")
    with pytest.raises(EgressBlocked):
        firewall(tmp_path).preflight(
            _typed_request({**_request(), "request_id": secret_request_id}),
            _route(model=private_model_label),
            grants=(object(),),
        )
    receipt_text = (tmp_path / "llm-egress-receipts.jsonl").read_text()
    assert secret_request_id not in receipt_text
    assert str(Path.home()) not in receipt_text


def test_receipt_redacts_route_credentials_and_binds_route_metadata(tmp_path):
    route = _route(
        base_url="https://user:password@example.test/v1?api_key=route-secret",
        api_mode="chat_completions",
    )
    firewall(tmp_path).preflight(_sanitized_request("safe request"), route)
    receipt_text = (tmp_path / "llm-egress-receipts.jsonl").read_text()
    assert "password" not in receipt_text
    assert "route-secret" not in receipt_text
    receipt = json.loads(receipt_text.splitlines()[0])
    assert receipt["api_mode"] == "chat_completions"
    assert receipt["base_url"].startswith("sha256:")


def test_receipt_is_owner_only_and_rejects_symlink_ledger(tmp_path):
    path = tmp_path / "private.py"
    path.write_text("x=1\n", encoding="utf-8")
    grant = _source_grant(path)
    request = _request("private source\n")
    ledger = tmp_path / "llm-egress-receipts.jsonl"
    firewall(tmp_path).preflight(
        _typed_request(request, source_grant=grant),
        _route(),
        grants=(grant,),
    )
    assert stat.S_IMODE(ledger.stat().st_mode) == 0o600

    ledger.unlink()
    target = tmp_path / "outside.jsonl"
    target.write_text("", encoding="utf-8")
    ledger.symlink_to(target)
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(
            _typed_request(request, source_grant=grant),
            _route(),
            grants=(grant,),
        )
    assert "receipt_unavailable" in exc_info.value.decision.reason_codes


def test_receipt_rejects_symlink_lock_file(tmp_path):
    target = tmp_path / "outside.lock"
    target.write_bytes(b"x")
    (tmp_path / "llm-egress-receipts.lock").symlink_to(target)

    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(_sanitized_request("safe request"), _route())

    assert "receipt_unavailable" in exc_info.value.decision.reason_codes
    assert target.read_bytes() == b"x"


def test_concurrent_receipt_appends_remain_complete_json_lines(tmp_path):
    gate = firewall(tmp_path)

    def append(index: int) -> None:
        gate.preflight(
            {**_request(f"local prompt {index}"), "request_id": f"req-{index}"},
            _route(base_url="http://127.0.0.1:11434"),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(append, range(32)))

    lines = (tmp_path / "llm-egress-receipts.jsonl").read_text().splitlines()
    receipts = [json.loads(line) for line in lines]
    assert len(receipts) == 32
    assert {receipt["request_id"] for receipt in receipts} == {
        f"req-{index}" for index in range(32)
    }


def test_firewall_imports_when_fcntl_is_unavailable():
    script = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'fcntl':
        raise ImportError('simulated Windows import')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import agent.llm_egress_firewall
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_multiprocess_receipt_appends_preserve_hash_chain(tmp_path):
    script = """
import sys
from pathlib import Path
from agent.llm_egress_firewall import LLMEgressFirewall
gate = LLMEgressFirewall(Path(sys.argv[1]))
index = sys.argv[2]
gate.preflight(
    {
        'messages': [{'role': 'user', 'content': f'local prompt {index}'}],
        'request_id': f'process-{index}',
    },
    {'provider': 'local', 'model': 'test', 'base_url': 'http://127.0.0.1:11434'},
)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), str(index)],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for index in range(12)
    ]
    for process in processes:
        _stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stderr

    lines = (tmp_path / "llm-egress-receipts.jsonl").read_bytes().splitlines()
    assert len(lines) == 12
    for index, line in enumerate(lines):
        receipt = json.loads(line)
        expected_previous = "" if index == 0 else sha256(lines[index - 1]).hexdigest()
        assert receipt["receipt_prev_sha256"] == expected_previous


def test_receipts_form_a_content_free_hash_chain(tmp_path):
    gate = firewall(tmp_path)
    gate.preflight(_sanitized_request("first"), _route())
    gate.preflight(_sanitized_request("second"), _route())
    lines = (tmp_path / "llm-egress-receipts.jsonl").read_bytes().splitlines()
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["receipt_prev_sha256"] == ""
    assert second["receipt_prev_sha256"] == sha256(lines[0]).hexdigest()
    assert first["receipt_sha256"]
    assert second["receipt_sha256"]


def test_receipt_emission_survives_platform_without_fchmod(tmp_path, monkeypatch):
    monkeypatch.delattr(os, "fchmod")
    gate = firewall(tmp_path)

    allowed = gate.preflight(_sanitized_request("safe request"), _route())
    assert allowed.allowed is True

    with pytest.raises(EgressBlocked) as exc_info:
        gate.preflight(_sanitized_request("c2VjcmV0LXBheWxvYWQ="), _route())
    assert exc_info.value.decision.allowed is False

    lines = (tmp_path / "llm-egress-receipts.jsonl").read_text().splitlines()
    assert [json.loads(line)["decision"] for line in lines] == ["allow", "block"]


def test_unknown_destination_fails_closed_even_with_source_grant(tmp_path):
    path = tmp_path / "private.py"
    path.write_text("private source\n", encoding="utf-8")
    grant = _source_grant(path)
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(_request("private source"), _route(base_url=None), grants=(grant,))
    assert exc_info.value.decision.destination_class == DestinationClass.UNKNOWN
    assert "unknown_destination" in exc_info.value.decision.reason_codes


@pytest.mark.parametrize(
    "protocol_id",
    (
        "call_eYMMaSP2Uc5AeCO4Y7x4vHz4",
        "call_UDcmjiJYYhXKVbOvZL6dW1VC",
        "fc_0123456789abcdefghijklmnop",
    ),
)
def test_provider_protocol_ids_are_not_misclassified_as_base64(
    tmp_path, protocol_id
):
    decision = firewall(tmp_path).preflight(
        _sanitized_request(protocol_id),
        _route(),
    )
    assert decision.allowed is True


def test_arbitrary_base64_remains_blocked_after_protocol_id_exemption(tmp_path):
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path).preflight(
            _sanitized_request("c2VjcmV0LXBheWxvYWQ="),
            _route(),
        )
    assert "base64_payload" in exc_info.value.decision.reason_codes


def test_forged_validated_tool_syntax_segment_fails_closed(tmp_path):
    request = TypedOutboundRequest(
        payload={
            "messages": [
                {
                    "role": LiteralSegment("tool"),
                    "content": ValidatedToolSyntaxSegment(
                        "https://evil.example/payload", "github_url"
                    ),
                }
            ]
        },
        session_id="session-1",
        turn_id="turn-1",
        request_id="req-1",
        policy_digest="policy-1",
    )
    with pytest.raises(EgressBlocked) as exc_info:
        firewall(tmp_path, static_literals={"tool"}).preflight(request, _route())
    assert "invalid_tool_syntax_segment" in exc_info.value.decision.reason_codes
