from __future__ import annotations

import json
from pathlib import Path

import pytest

from ares_runtime.collaboration import SpecialistDescriptorV1, specialist_descriptor_ref
from ares_runtime.specialist_dispatch import (
    DispatchError,
    _parse_capability_bindings,
    client_main,
    explicit_dispatch_decision,
    parse_dispatch_request,
    run_explicit_dispatch,
)


ROLE_ARTIFACTS = {"role.explorer": ["explorer_dissent"]}


def descriptor(
    *,
    enabled: bool = True,
    profile_id: str = "explorer",
    capability: str = "competing_design",
) -> SpecialistDescriptorV1:
    return SpecialistDescriptorV1.create(
        {
            "profile_id": profile_id,
            "semantic_role_id": "role.explorer",
            "enabled": enabled,
            "narrow_purpose": "Competing hypotheses only.",
            "capability_classes": [capability],
            "tool_classes": ["artifact_read"],
            "required_artifact_ids": ["explorer_dissent"],
            "input_evidence_classes": ["pinned_source"],
            "required_outputs": ["falsifier"],
            "explicit_exclusions": ["runtime_authority"],
            "mandatory_deferrals": ["statistician"],
            "handoff_rules": ["preserve_evidence"],
            "failure_and_abstention_behavior": {
                "on_insufficient_evidence": "blocked_or_unknown",
                "on_unavailable_or_disabled_contract": "blocked_or_unknown",
                "generic_fallback_label": "forbidden",
            },
            "activation_evidence_refs": ["evidence:activation"],
            "provenance": {
                "source_refs": ["evidence:source"],
                "semantic_registry_ref": "docs:role-contracts/role-contracts.json",
                "semantic_registry_digest": "sha256:" + "a" * 64,
            },
        },
        profile_exists=lambda candidate: candidate in {"explorer", "statistician"},
        semantic_role_artifacts=ROLE_ARTIFACTS,
    )


def raw_request(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": "AresExplicitSpecialistDispatchRequestV1",
        "run_id": "specialist-run-00000001",
        "profile_ids": ["explorer"],
        "requested_capabilities": {"explorer": "competing_design"},
        "workspace": "/tmp",
        "brief": "Produce one bounded falsification plan.",
    }
    body.update(overrides)
    from ares_runtime.collaboration import digest

    body["request_digest"] = digest(body)
    return body


def test_request_rejects_unknown_fields_duplicate_keys_and_bad_digest():
    malformed = json.dumps({**raw_request(), "unexpected": True}).encode("utf-8")
    with pytest.raises(DispatchError, match="UNKNOWN_FIELD"):
        parse_dispatch_request(malformed)

    duplicate = b'{"schema":"AresExplicitSpecialistDispatchRequestV1","schema":"wrong"}'
    with pytest.raises(DispatchError, match="DUPLICATE_JSON_KEY"):
        parse_dispatch_request(duplicate)

    bad = raw_request(request_digest="sha256:" + "0" * 64)
    with pytest.raises(DispatchError, match="REQUEST_DIGEST_MISMATCH"):
        parse_dispatch_request(json.dumps(bad).encode("utf-8"))

    incomplete = raw_request(
        profile_ids=["explorer", "statistician"],
        requested_capabilities={"explorer": "competing_design"},
    )
    from ares_runtime.collaboration import digest

    incomplete["request_digest"] = digest(
        {key: value for key, value in incomplete.items() if key != "request_digest"}
    )
    with pytest.raises(DispatchError, match="INVALID_CAPABILITY_SET"):
        parse_dispatch_request(json.dumps(incomplete).encode("utf-8"))


def test_cli_capability_bindings_are_exact_and_profile_scoped():
    assert _parse_capability_bindings(
        ["statistician:quantitative_comparison", "explorer:competing_design"],
        ["explorer", "statistician"],
    ) == {
        "explorer": "competing_design",
        "statistician": "quantitative_comparison",
    }
    with pytest.raises(DispatchError, match="INVALID_CAPABILITY_SET"):
        _parse_capability_bindings(["explorer:competing_design"], ["explorer", "statistician"])


def test_cli_quiesce_and_unquiesce_send_only_exact_typed_envelopes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ares_runtime import specialist_dispatch

    root = tmp_path / "ares"
    root.mkdir()
    (root / "specialist-dispatch.json").write_text(
        json.dumps(
            {
                "schema": "AresDesktopSpecialistDispatchEndpointV1",
                "host": "127.0.0.1",
                "port": 1,
                "token": "x" * 32,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    calls: list[dict[str, object]] = []
    responses = iter(
        [
            {"outcome": "quiesced", "leaseId": "specialist-quiesce-00000000-0000-4000-8000-000000000000", "profileIds": ["explorer", "public"]},
            {"outcome": "unquiesced", "leaseId": "specialist-quiesce-00000000-0000-4000-8000-000000000000", "profileIds": ["explorer", "public"]},
        ]
    )
    monkeypatch.setattr(
        specialist_dispatch,
        "_desktop_request",
        lambda _root, envelope: calls.append(dict(envelope)) or next(responses),
    )

    assert client_main(["quiesce", "--profile", "public", "--profile", "explorer"]) == 0
    assert client_main(["unquiesce", "--lease-id", "specialist-quiesce-00000000-0000-4000-8000-000000000000"]) == 0
    assert calls[0]["operation"] == "quiesce"
    assert calls[0]["profile_ids"] == ["explorer", "public"]
    assert calls[1] == {
        "schema": "AresDesktopSpecialistDispatchEnvelopeV1",
        "operation": "unquiesce",
        "token": "x" * 32,
        "lease_id": "specialist-quiesce-00000000-0000-4000-8000-000000000000",
    }


def test_explicit_dispatch_requires_enabled_exact_binding_and_matching_capability():
    candidate = descriptor(enabled=True)
    accepted = explicit_dispatch_decision(
        request=parse_dispatch_request(json.dumps(raw_request()).encode("utf-8")),
        candidates={"explorer": candidate},
        profile_binding_refs={"explorer": specialist_descriptor_ref(candidate)},
    )
    assert accepted["outcome"] == "eligible"
    assert accepted["selected_profile_ids"] == ["explorer"]
    assert accepted["capacity_authority"] == "electron_required"

    disabled = descriptor(enabled=False)
    rejected = explicit_dispatch_decision(
        request=parse_dispatch_request(json.dumps(raw_request()).encode("utf-8")),
        candidates={"explorer": disabled},
        profile_binding_refs={"explorer": specialist_descriptor_ref(disabled)},
    )
    assert rejected["outcome"] == "blocked"
    assert rejected["candidate_rejections"] == [
        {"profile_id": "explorer", "reason_code": "DESCRIPTOR_DISABLED"}
    ]

    mismatch = explicit_dispatch_decision(
        request=parse_dispatch_request(json.dumps(raw_request()).encode("utf-8")),
        candidates={"explorer": candidate},
        profile_binding_refs={"explorer": "specialist-descriptor:" + "0" * 64},
    )
    assert mismatch["outcome"] == "blocked"
    assert mismatch["selected_profile_ids"] == []
    assert mismatch["candidate_rejections"] == [
        {"profile_id": "explorer", "reason_code": "PROFILE_BINDING_MISMATCH"}
    ]


def test_explicit_multi_profile_dispatch_requires_one_capability_per_named_profile():
    request_body = raw_request(
        profile_ids=["explorer", "statistician"],
        requested_capabilities={
            "explorer": "competing_design",
            "statistician": "quantitative_comparison",
        },
    )
    from ares_runtime.collaboration import digest

    request_body["request_digest"] = digest(
        {key: value for key, value in request_body.items() if key != "request_digest"}
    )
    explorer = descriptor(profile_id="explorer", capability="competing_design")
    statistician = descriptor(
        profile_id="statistician", capability="quantitative_comparison"
    )

    decision = explicit_dispatch_decision(
        request=parse_dispatch_request(json.dumps(request_body).encode("utf-8")),
        candidates={"explorer": explorer, "statistician": statistician},
        profile_binding_refs={
            "explorer": specialist_descriptor_ref(explorer),
            "statistician": specialist_descriptor_ref(statistician),
        },
    )

    assert decision["outcome"] == "eligible"
    assert decision["selected_profile_ids"] == ["explorer", "statistician"]
    assert decision["requested_capabilities"] == {
        "explorer": "competing_design",
        "statistician": "quantitative_comparison",
    }


def test_runner_writes_receipt_and_never_calls_worker_after_rejection(tmp_path: Path):
    candidate = descriptor(enabled=False)
    called: list[str] = []

    receipt = run_explicit_dispatch(
        request=parse_dispatch_request(json.dumps(raw_request()).encode("utf-8")),
        candidates={"explorer": candidate},
        profile_binding_refs={"explorer": specialist_descriptor_ref(candidate)},
        receipt_root=tmp_path,
        worker=lambda _profile, _request: called.append("worker") or {"outcome": "returned", "exit_code": 0},
    )

    assert called == []
    assert receipt["terminal_state"] == "rejected"
    saved = json.loads((tmp_path / "specialist-run-00000001" / "receipt.json").read_text(encoding="utf-8"))
    assert saved["terminal_state"] == "rejected"
    assert "brief" not in json.dumps(saved)


def test_runner_executes_only_the_eligible_explicit_profile_and_records_terminal_receipt(tmp_path: Path):
    candidate = descriptor(enabled=True)
    calls: list[str] = []

    receipt = run_explicit_dispatch(
        request=parse_dispatch_request(json.dumps(raw_request()).encode("utf-8")),
        candidates={"explorer": candidate},
        profile_binding_refs={"explorer": specialist_descriptor_ref(candidate)},
        receipt_root=tmp_path,
        worker=lambda profile, _request: calls.append(profile) or {"outcome": "returned", "exit_code": 0},
    )

    assert calls == ["explorer"]
    assert receipt["terminal_state"] == "released"
    assert receipt["profiles"] == [{"profile_id": "explorer", "outcome": "returned", "exit_code": 0}]
