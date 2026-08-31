import json
import os
import socket
import struct
import threading
from pathlib import Path

import pytest

import ares_runtime.collaboration as collaboration
from ares_runtime.collaboration import (
    BlindWitness,
    ContractBindings,
    ClosureProjector,
    ContextCompiler,
    ContextPacketV1,
    ContractError,
    DaemonPermitReceiptAdapter,
    DesktopProductionApprovalEnvelope,
    DesktopProductionApprovalWitnessProvider,
    GatewayProductionApprovalWitnessProvider,
    PermitBridgeState,
    MissionContractV1,
    RoleContractV1,
    dispatcher_boundary,
    evaluation_ui_projection,
    freeze_replay_corpus,
    make_artifact,
    permit_adapter,
    replay_mutations,
    reset_permit_adapter,
    EvidenceItemV1,
    FindingV1,
    HandoffPacketV1,
    TestRequestV1 as TestRequestContract,
)


def role():
    return RoleContractV1.create({
        "role_id": "ares.principal", "role_kind": "permanent", "profile_ref": "profile:primary",
        "durable_ownership": ["mission reasoning"], "objective": "execute",
        "unique_questions": ["what is true"], "mandatory_triggers": [], "exclusions": ["authority"],
        "context_policy": {"allowed": ["source"], "withheld_until_commit": [], "forbidden": [], "max_context_class": "standard"},
        "capability_profile": {"discoverable_capabilities": [], "required_permit_classes": [], "forbidden_capability_classes": ["authority"], "secret_policy": "none"},
        "mutation_authority": "proposal_only", "output_schema_ref": "schema:finding:v1",
        "stop_conditions": ["evidence gap"], "typed_failures": ["BLOCKED"], "handoff_rules": ["artifact-only"],
        "model_eligibility": {"required_capabilities": [], "disqualifying_negative_capabilities": [], "preferred_capability_class": "general", "fallback_capability_class": "general"},
        "evaluation": {"corpus_ref": "corpus:v1", "promotion_gates": ["no authority violation"], "minimum_cases": 1},
    })


def mission():
    return MissionContractV1.create({
        "mission_id": "mission:1", "kanban_root_task_ref": "task:1", "objective": "verify",
        "source_freeze": [{"source_ref": "repo:main", "revision_or_digest": "abc", "state": "LIVE_VERIFIED"}],
        "closure_profile": "engineering", "risk_class": "low", "effect_class": "none",
        "boundaries": {"allowed": ["repo"], "forbidden": ["publish"], "source_owners": ["repo:main"]},
        "required_evidence": [{"gate_id": "test", "evidence_kinds": ["test_result"], "minimum_count": 1, "independent_witness_required": False}],
        "topology_policy": {"default_executor_count": 1, "allowed_patterns": ["single"], "max_concurrent_specialists": 1, "escalation_requires_evidence_gap": True},
        "stop_conditions": ["missing evidence"],
    })


def test_contracts_are_deterministic_and_bound_to_existing_ids():
    r = role(); m = mission()
    assert r.artifact_digest.startswith("sha256:")
    assert r.canonical_bytes() == RoleContractV1.parse(r.to_dict()).canonical_bytes()
    with pytest.raises(ContractError):
        MissionContractV1.create({**m.to_dict(), "kanban_root_task_ref": "task:new"}, task_exists=lambda x: False)


def test_context_is_byte_stable_and_sorted():
    c = ContextCompiler()
    a = c.compile("mission:1", "role:principal", [
        {"ref": "e:b", "digest": "sha256:" + "b" * 64, "purpose": "b"},
        {"ref": "e:a", "digest": "sha256:" + "a" * 64, "purpose": "a"},
    ])
    b = c.compile("mission:1", "role:principal", list(reversed(a.to_dict()["included_refs"])))
    assert a.canonical_bytes() == b.canonical_bytes()


def _production_canary_config(session_id="session:canary"):
    return {
        "ares": {
            "permit_daemon": {
                "mode": "production_per_call",
                "enabled": True,
                "canary_session_id": session_id,
                "socket_path": "/tmp/ares-production-permit-test.sock",
                "worktree_root": "/tmp",
                "timeout_seconds": 5,
            }
        }
    }


def test_strict_effect_schema_denies_coercion_and_missing_permit_for_configured_canary(monkeypatch):
    monkeypatch.setattr(collaboration, "_load_runtime_config", lambda: _production_canary_config(), raising=False)
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }
    assert dispatcher_boundary(
        "write_file", {"path": 3, "content": "exact"}, mission_ref="mission:1", schema=schema, session_id="session:canary"
    )[1] == "COERCION_REQUIRED"
    assert dispatcher_boundary(
        "write_file", {"path": "/tmp/ares-permit-test.txt", "content": "exact"}, mission_ref="mission:1", schema=schema, session_id="session:canary"
    )[1] == "OPERATOR_APPROVAL_WITNESS_MISSING"


def test_production_canary_is_config_only_and_exact_session_scoped(monkeypatch):
    monkeypatch.setenv("ARES_RUNTIME_PERMITS_V1", "1")
    config = _production_canary_config()
    monkeypatch.setattr(collaboration, "_load_runtime_config", lambda: config, raising=False)
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }

    # A legacy ambient environment flag must neither activate production mode nor
    # pull another session into the canary boundary.
    assert dispatcher_boundary(
        "write_file", {"path": "x", "content": "exact"}, mission_ref="mission:1", schema=schema, session_id="session:other"
    ) == (True, None, None)

    # The selected session gets strict validation before coercion and fails
    # closed before daemon contact when the Desktop witness owner is unavailable.
    assert dispatcher_boundary(
        "write_file", {"path": 3, "content": "exact"}, mission_ref="mission:1", schema=schema, session_id="session:canary"
    )[1] == "COERCION_REQUIRED"
    assert dispatcher_boundary(
        "write_file", {"path": "/tmp/ares-permit-test.txt", "content": "exact"}, mission_ref="mission:1", schema=schema, session_id="session:canary"
    )[1] == "OPERATOR_APPROVAL_WITNESS_MISSING"

    # Omitting the explicit session binding disables rather than broadens the
    # production boundary.
    config["ares"]["permit_daemon"].pop("canary_session_id")
    assert dispatcher_boundary(
        "write_file", {"path": "x", "content": "exact"}, mission_ref="mission:1", schema=schema, session_id="session:canary"
    ) == (True, None, None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canary_session_id", "   "),
        ("socket_path", "relative.sock"),
        ("socket_path", 7),
        ("worktree_root", "relative-root"),
        ("worktree_root", "/path/that/does/not/exist"),
        ("timeout_seconds", True),
        ("timeout_seconds", 0),
        ("timeout_seconds", 301),
    ],
)
def test_production_canary_requires_complete_well_typed_transport_before_activation(monkeypatch, field, value):
    config = _production_canary_config()
    config["ares"]["permit_daemon"][field] = value
    monkeypatch.setattr(collaboration, "_load_runtime_config", lambda: config, raising=False)

    assert collaboration.production_permit_canary_enabled(session_id="session:canary") is False
    assert dispatcher_boundary(
        "write_file",
        {"path": "/tmp/ares-permit-test.txt", "content": "exact"},
        mission_ref="mission:1",
        session_id="session:canary",
    ) == (True, None, None)


def event_exists(ref: str) -> bool:
    return ref.startswith("event:")


def test_blind_witness_commit_reveal_and_derived_closure():
    witness = BlindWitness()
    commitment = witness.commit("sha256:" + "a" * 64, executor_route_ref="route:executor", witness_route_ref="route:witness")
    assert commitment.startswith("sha256:")
    receipt = witness.reveal_and_record({"verdict": "pass"}, mission_ref="mission:1", test_request_ref="test:1", role_contract_ref="role:witness", context_digest="sha256:" + "a" * 64, executor_route_ref="route:executor", witness_route_ref="route:witness")
    assert receipt.to_dict()["independence"]["commitment"] == commitment
    projected = ClosureProjector().project("mission:1", "engineering", {"test": True}, source_event_refs=["event:1"], source_event_exists=event_exists)
    assert projected.to_dict()["state"] == "closed"


def test_mutation_harness_blocks_required_mutations():
    result = replay_mutations([{"case_id": "x", "gates": {"test": True}, "mutation": "skip_required_test"}, {"case_id": "y", "gates": {"test": True}}])
    assert result["critical_mutations_blocked"] is True


def test_dispatcher_permit_mode_cannot_disable_authorization():
    source = (os.path.join(os.path.dirname(__file__), "..", "model_tools.py"))
    with open(source, encoding="utf-8") as handle:
        implementation = handle.read()
    assert "authorize_permit=True" in implementation
    assert 'authorize_permit=os.getenv("ARES_RUNTIME_PERMITS_V1"' not in implementation


def test_typed_phase_two_artifacts_match_declared_shapes_and_are_replayable():
    evidence = EvidenceItemV1.create({
        "evidence_id": "evidence:1", "mission_ref": "mission:1", "kind": "test_result",
        "source_ref": "test:1", "recorded_at": "2026-08-23T00:00:00Z",
        "evidence_state": "LIVE_VERIFIED", "authority_class": "direct_observation",
        "taint": {"untrusted_data": False, "contains_instructions": False, "secret_class": "none", "allowed_sinks": []},
        "acquisition_receipt_ref": "receipt:1",
    })
    assert evidence.to_dict()["artifact_digest"].startswith("sha256:")
    assert EvidenceItemV1.parse(evidence.to_dict()).canonical_bytes() == evidence.canonical_bytes()
    handoff = HandoffPacketV1.create({
        "handoff_id": "handoff:1", "mission_ref": "mission:1", "from_role_ref": "role:a", "to_role_ref": "role:b",
        "owned_question": "is it true", "context_packet_ref": "ctx:1", "evidence_refs": ["evidence:1"],
        "unresolved_claim_refs": [], "withheld_classes": [], "permit_refs": [],
        "required_output_schema_ref": "schema:finding:v1", "stop_conditions": ["gap"],
    })
    assert HandoffPacketV1.parse(handoff.to_dict()).canonical_bytes() == handoff.canonical_bytes()
    request = TestRequestContract.create({
        "test_request_id": "test:1", "mission_ref": "mission:1", "question": "is it true",
        "oracle_class": "unit", "procedure": ["run test"], "expected_discriminators": ["pass"],
        "required_environment": [], "authority_requirements": [], "stop_conditions": ["missing"],
    })
    assert TestRequestContract.parse(request.to_dict()).artifact_digest == request.artifact_digest


def test_context_resolver_and_source_freeze_fail_closed():
    compiler = ContextCompiler()
    with pytest.raises(ContractError):
        compiler.compile("mission:1", "role:principal", [{"ref": "e:1", "digest": "sha256:" + "a" * 64, "purpose": "proof"}], resolve_ref=lambda *_: False)
    with pytest.raises(ContractError):
        compiler.compile("mission:1", "role:principal", [{"ref": "e:1", "digest": "sha256:" + "a" * 64, "purpose": "proof"}], source_revision="new", frozen_source_revision="old")


def test_skipped_required_check_is_not_closed():
    projected = ClosureProjector().project("mission:1", "engineering", {"test": False}, source_event_refs=["event:skipped-test"], source_event_exists=event_exists)
    assert projected.to_dict()["state"] == "evidence_pending"
    assert projected.to_dict()["unsatisfied_gate_ids"] == ["test"]


def test_missing_source_event_is_rejected():
    with pytest.raises(ContractError, match="MISSING_SOURCE_EVENT"):
        ClosureProjector().project("mission:1", "engineering", {"test": True}, source_event_refs=["event:missing"], source_event_exists=lambda _ref: False)


def test_ambiguous_effect_is_quarantined():
    projected = ClosureProjector().project("mission:1", "effectful_operation", {"effect_receipt": True}, source_event_refs=["event:effect"], source_event_exists=event_exists, flags=["AMBIGUOUS_EFFECT"])
    assert projected.to_dict()["state"] == "quarantined"
    assert projected.to_dict()["divergence_flags"] == ["AMBIGUOUS_EFFECT"]


def test_witness_route_non_independence_is_denied():
    with pytest.raises(ContractError, match="WITNESS_ROUTE_NOT_INDEPENDENT"):
        BlindWitness().commit("sha256:" + "a" * 64, executor_route_ref="route:shared", witness_route_ref="route:shared")


def test_reopen_requires_a_new_evidence_projection():
    projector = ClosureProjector()
    closed = projector.project("mission:1", "engineering", {"test": True}, source_event_refs=["event:close"], source_event_exists=event_exists)
    with pytest.raises(ContractError, match="REOPEN_REQUIRES_NEW_EVIDENCE"):
        projector.project("mission:1", "engineering", {"test": False}, source_event_refs=["event:close"], source_event_exists=event_exists, previous_projection=closed)
    reopened = projector.project("mission:1", "engineering", {"test": False}, source_event_refs=["event:close", "event:new-failure"], source_event_exists=event_exists, previous_projection=closed)
    assert reopened.to_dict()["state"] == "evidence_pending"
    assert not hasattr(projector, "close") and not hasattr(projector, "reopen")


def _serve_permit_responses(path, *response_factories):
    ready = threading.Event()

    def serve():
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            listener.listen(len(response_factories))
            ready.set()
            for response_factory in response_factories:
                with listener.accept()[0] as client:
                    size = struct.unpack(">I", client.recv(4))[0]
                    payload = bytearray()
                    while len(payload) < size:
                        payload.extend(client.recv(size - len(payload)))
                    request = json.loads(bytes(payload))
                    response = response_factory(request)
                    encoded = json.dumps(response).encode()
                    client.sendall(struct.pack(">I", len(encoded)) + encoded)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(2)
    return thread


class _WitnessProvider:
    def __init__(self, witness):
        self.witness = witness
        self.calls = []

    def issue_witness(self, *, mission_ref, target_ref, call):
        self.calls.append({"mission_ref": mission_ref, "target_ref": target_ref, "call": call})
        return self.witness


class _DesktopController:
    def __init__(self, witness=None):
        self.witness = witness
        self.envelopes = []

    def request_signed_witness(self, *, envelope):
        self.envelopes.append(envelope)
        return self.witness


def _test_witness(_text="fixture"):
    return {
        "operator_case": "approval:test:one",
        "request_digest": "a" * 64,
        "authenticator": "b" * 64,
    }


def _daemon_adapter(path, witness: object = "valid", text="fixture"):
    resolved_witness = _test_witness(text) if witness == "valid" else witness
    return DaemonPermitReceiptAdapter(
        {"socket_path": str(path), "mode": "test_only_echo"},
        approval_witness_provider=_WitnessProvider(resolved_witness),
    )


def _write_digest_helper(path: Path, output: str, captured: Path | None = None) -> None:
    capture = "" if captured is None else f"Path({str(captured)!r}).write_text(json.dumps({{'path': sys.argv[3], 'payload': json.loads(Path(sys.argv[3]).read_text())}}))"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "assert sys.argv[1:3] == ['canonical-digest', '--json']\n"
        f"{capture}\n"
        f"print({output!r})\n"
    )
    path.chmod(0o700)


def test_daemon_permit_digest_command_unavailable_fails_closed(tmp_path):
    adapter = DaemonPermitReceiptAdapter({"canonical_digest_command": str(tmp_path / "missing-ra-daemon")})
    with pytest.raises(ContractError, match="CANONICAL_DIGEST_UNAVAILABLE"):
        adapter.canonical_args_digest({"path": "x"})


def test_daemon_permit_digest_command_rejects_malformed_output(tmp_path):
    helper = tmp_path / "ra-daemon"
    _write_digest_helper(helper, "not-a-digest")
    adapter = DaemonPermitReceiptAdapter({"canonical_digest_command": str(helper)})
    with pytest.raises(ContractError, match="CANONICAL_DIGEST_MALFORMED"):
        adapter.canonical_args_digest({"path": "x"})


def test_daemon_permit_argument_digest_is_daemon_owned(monkeypatch, tmp_path):
    monkeypatch.setattr(collaboration, "_load_runtime_config", lambda: _production_canary_config(), raising=False)
    adapter = DaemonPermitReceiptAdapter({
        "socket_path": str(tmp_path / "unneeded.sock"),
    })
    token = permit_adapter(adapter)
    try:
        allowed, code, _permit = dispatcher_boundary(
            "write_file",
            {"path": "x", "content": "payload"},
            mission_ref="mission:1",
            session_id="session:canary",
        )
    finally:
        reset_permit_adapter(token)
    assert (allowed, code) == (False, "TEST_ONLY_ECHO_DISABLED")


def test_daemon_permit_call_is_exact_test_only_echo_sequence(tmp_path):
    path = tmp_path / "daemon.sock"

    def issued(request):
        issuance_payload = {"call": {"tool": "echo", "args": {"text": "payload"}, "frozen_clock": None}, "requested_validity_ms": 300000}
        assert request["request"] == {
            "kind": "permit_issue",
            "request": issuance_payload,
            "approval": _test_witness("payload"),
        }
        return {"request_id": request["request_id"], "permit_id": "permit:one", "binding": {"issued_binding": "opaque"}}

    def consumed(request):
        assert request["request"] == {
            "kind": "permit_consume", "permit_id": "permit:one", "binding": {"issued_binding": "opaque"},
            "call": {"tool": "echo", "args": {"text": "payload"}, "frozen_clock": None},
        }
        return {"request_id": request["request_id"], "permit_id": "permit:one", "evidence": {}, "preflight_artifact": {}, "receipt_artifact": {"digest": "receipt"}}

    _serve_permit_responses(path, issued, consumed)
    outcome = _daemon_adapter(path, text="payload").consume(mission_ref="mission:1", tool_name="echo", args={"text": "payload"}, target_ref="tool:echo")
    assert (outcome.state, outcome.code) == (PermitBridgeState.CONSUMED, "PERMIT_CONSUMED")
    assert outcome.facts is not None and outcome.facts["canonical_permit_ref"] == "permit:one"


def test_daemon_permit_bridge_unavailable_denies_test_only_echo(tmp_path):
    adapter = _daemon_adapter(tmp_path / "missing.sock")
    outcome = adapter.consume(mission_ref="mission:1", tool_name="echo", args={"text": "fixture"}, target_ref="tool:echo")
    assert (outcome.state, outcome.code) == (PermitBridgeState.UNAVAILABLE, "PERMIT_BRIDGE_UNAVAILABLE")


def test_desktop_production_provider_default_denies_without_controller(tmp_path):
    provider = DesktopProductionApprovalWitnessProvider(controller=None, worktree_root=tmp_path)
    assert provider.issue_witness(
        mission_ref="mission:1",
        target_ref="path:target",
        call={"tool": "write_file", "args": {"path": str(tmp_path / "allowed.txt"), "content": "exact"}, "frozen_clock": None},
    ) is None


def test_desktop_production_provider_does_not_treat_existing_approval_choice_as_witness(tmp_path):
    controller = _DesktopController("once")
    provider = DesktopProductionApprovalWitnessProvider(controller=controller, worktree_root=tmp_path)
    assert provider.issue_witness(
        mission_ref="mission:1",
        target_ref="path:target",
        call={"tool": "write_file", "args": {"path": str(tmp_path / "allowed.txt"), "content": "exact"}, "frozen_clock": None},
    ) is None
    assert len(controller.envelopes) == 1


def test_desktop_production_envelope_preserves_exact_write_file_display_and_constraints(tmp_path):
    controller = _DesktopController({"opaque": "daemon-verifies-this-later"})
    provider = DesktopProductionApprovalWitnessProvider(controller=controller, worktree_root=tmp_path)
    source_path = str(tmp_path / "nested" / "../allowed.txt")
    witness = provider.issue_witness(
        mission_ref="mission:1",
        target_ref="path:target",
        call={"tool": "write_file", "args": {"path": source_path, "content": "exact\ncontent"}, "frozen_clock": None},
    )
    assert witness == {"opaque": "daemon-verifies-this-later"}
    assert len(controller.envelopes) == 1
    envelope = controller.envelopes[0].to_dict()
    assert envelope["call"] == {"tool": "write_file", "args": {"path": source_path, "content": "exact\ncontent"}, "frozen_clock": None}
    assert envelope["constraints"] == {
        "validity_ms": 300000,
        "one_use": True,
        "retry_allowed": False,
        "network_allowed": False,
        "delegation_allowed": False,
        "allowed_write_root": str(tmp_path.resolve()),
        "ambiguous_outcome": "terminal_quarantine",
    }


def test_gateway_production_provider_requires_typed_witness(monkeypatch, tmp_path):
    import tools.approval as approval

    session_key = "gateway:production-test"
    monkeypatch.setattr(approval, "get_current_session_key", lambda default="": session_key)
    notified = []
    approval.register_gateway_notify(session_key, notified.append)
    provider = GatewayProductionApprovalWitnessProvider(worktree_root=tmp_path)
    call = {
        "tool": "write_file",
        "args": {"path": str(tmp_path / "allowed.txt"), "content": "exact"},
        "frozen_clock": None,
    }
    result = {}

    def run():
        result["witness"] = provider.issue_witness(
            mission_ref="mission:1", target_ref="path:target", call=call
        )

    thread = threading.Thread(target=run)
    thread.start()
    for _ in range(500):
        if notified:
            break
        threading.Event().wait(0.01)
    assert notified and notified[0]["production_permit"]["call"] == call
    request_id = notified[0]["request_id"]
    assert approval.resolve_gateway_approval(session_key, "once", request_id=request_id) == 1
    thread.join(timeout=2)
    assert result["witness"] is None

    notified.clear()
    thread = threading.Thread(target=run)
    thread.start()
    for _ in range(500):
        if notified:
            break
        threading.Event().wait(0.01)
    witness = {"signature": "opaque", "key_id": "desktop-key"}
    assert approval.resolve_gateway_approval(
        session_key, "once", request_id=notified[0]["request_id"], witness=witness
    ) == 1
    thread.join(timeout=2)
    assert result["witness"] == witness
    approval.unregister_gateway_notify(session_key)


@pytest.mark.parametrize(
    "call",
    [
        {"tool": "patch", "args": {"path": "x", "old_string": "a", "new_string": "b"}, "frozen_clock": None},
        {"tool": "write_file", "args": {"path": "x", "content": "x", "extra": True}, "frozen_clock": None},
        {"tool": "write_file", "args": {"path": "x", "content": "x"}, "frozen_clock": "retry"},
    ],
)
def test_desktop_production_envelope_denies_out_of_scope_or_non_exact_calls(tmp_path, call):
    with pytest.raises(ContractError, match="DESKTOP_APPROVAL_SCOPE_DENIED"):
        DesktopProductionApprovalEnvelope.for_call(
            mission_ref="mission:1", target_ref="path:target", call=call, worktree_root=tmp_path
        )


def test_desktop_production_envelope_denies_write_outside_worktree(tmp_path):
    with pytest.raises(ContractError, match="DESKTOP_APPROVAL_SCOPE_DENIED"):
        DesktopProductionApprovalEnvelope.for_call(
            mission_ref="mission:1",
            target_ref="path:target",
            call={"tool": "write_file", "args": {"path": str(tmp_path.parent / "outside.txt"), "content": "x"}, "frozen_clock": None},
            worktree_root=tmp_path,
        )


def test_daemon_production_per_call_adapter_forwards_opaque_witness_and_exact_write_file(tmp_path):
    path = tmp_path / "daemon.sock"
    witness = {"daemon_verifies": "opaque-signed-witness"}

    def issued(request):
        assert request["request"] == {
            "kind": "permit_issue_production",
            "witness": witness,
        }
        return {"request_id": request["request_id"], "permit_id": "permit:production", "binding": {"issued_binding": "opaque"}}

    def consumed(request):
        assert request["request"] == {
            "kind": "permit_consume",
            "permit_id": "permit:production",
            "binding": {"issued_binding": "opaque"},
            "call": {"tool": "write_file", "args": {"path": str(tmp_path / "allowed.txt"), "content": "exact"}, "frozen_clock": None},
        }
        return {"request_id": request["request_id"], "permit_id": "permit:production", "evidence": {}, "preflight_artifact": {}, "receipt_artifact": {"digest": "receipt"}}

    _serve_permit_responses(path, issued, consumed)
    adapter = DaemonPermitReceiptAdapter(
        {"socket_path": str(path), "mode": "production_per_call"},
        approval_witness_provider=_WitnessProvider(witness),
    )
    outcome = adapter.consume(
        mission_ref="mission:1",
        tool_name="write_file",
        args={"path": str(tmp_path / "allowed.txt"), "content": "exact"},
        target_ref="path:approved-result",
    )
    assert (outcome.state, outcome.code) == (PermitBridgeState.CONSUMED, "PERMIT_CONSUMED")


def test_daemon_production_mode_defaults_deny_without_witness_provider(tmp_path):
    adapter = DaemonPermitReceiptAdapter({"socket_path": str(tmp_path / "missing.sock"), "mode": "production_per_call"})
    outcome = adapter.consume(
        mission_ref="mission:1",
        tool_name="write_file",
        args={"path": str(tmp_path / "allowed.txt"), "content": "exact"},
        target_ref="path:approved-result",
    )
    assert (outcome.state, outcome.code) == (PermitBridgeState.UNAVAILABLE, "OPERATOR_APPROVAL_WITNESS_UNAVAILABLE")


def test_test_only_adapter_refuses_non_echo_and_default_config(tmp_path):
    adapter = _daemon_adapter(tmp_path / "must-not-connect.sock")
    assert (adapter.consume(mission_ref="mission:1", tool_name="write_file", args={"path": "x"}, target_ref="path:x").state, adapter.consume(mission_ref="mission:1", tool_name="write_file", args={"path": "x"}, target_ref="path:x").code) == (PermitBridgeState.DENIED, "TEST_ONLY_ECHO_REQUIRED")
    production = DaemonPermitReceiptAdapter({"socket_path": str(tmp_path / "production.sock")}, approval_witness_provider=_WitnessProvider(_test_witness()))
    assert production.consume(mission_ref="mission:1", tool_name="echo", args={"text": "fixture"}, target_ref="tool:echo").code == "TEST_ONLY_ECHO_DISABLED"


def test_daemon_permit_bridge_rejects_malformed_issuance_response(tmp_path):
    path = tmp_path / "daemon.sock"
    _serve_permit_responses(path, lambda _request: {"request_id": "wrong", "permit_id": "permit:one"})
    outcome = _daemon_adapter(path).consume(mission_ref="mission:1", tool_name="echo", args={"text": "fixture"}, target_ref="tool:echo")
    assert (outcome.state, outcome.code) == (PermitBridgeState.MALFORMED, "PERMIT_BRIDGE_MALFORMED")


@pytest.mark.parametrize("witness, code", [(None, "OPERATOR_APPROVAL_WITNESS_MISSING"), ("not-a-witness", "OPERATOR_APPROVAL_WITNESS_MALFORMED"), ({"tampered": True}, "OPERATOR_APPROVAL_WITNESS_MALFORMED")])
def test_daemon_adapter_missing_malformed_or_tampered_witness_fails_closed(tmp_path, witness, code):
    adapter = _daemon_adapter(tmp_path / "must-not-connect.sock", witness)
    outcome = adapter.consume(mission_ref="mission:1", tool_name="echo", args={"text": "fixture"}, target_ref="tool:echo")
    assert (outcome.state, outcome.code) == (PermitBridgeState.DENIED, code)


def test_daemon_adapter_forwards_bounded_and_ambiguous_quarantined_outcomes(tmp_path):
    path = tmp_path / "outcome.sock"

    def outcome(request):
        assert request["request"]["kind"] == "permit_outcome_record"
        assert request["request"]["reported"] == {"state": "outcome_ambiguous", "duration_ms": 7, "error_type": None}
        return {"request_id": request["request_id"], "permit_id": "permit:one", "outcome_artifact": {"receipt_digest": "b" * 64, "state": "terminal_quarantine"}}

    _serve_permit_responses(path, outcome)
    _daemon_adapter(path).record_receipt({"permit_ref": "permit:one", "preflight_receipt": {"receipt_digest": "a" * 64}, "state": "ambiguous", "duration_ms": 7, "error_type": None})


def test_phase_five_frozen_corpus_is_order_independent_and_covers_all_baselines():
    cases = [
        {"case_id": "case:ordinary", "gates": {"test": True}},
        {"case_id": "case:skipped", "gates": {"test": True}, "mutation": "skip_required_test"},
        {"case_id": "case:ambiguous", "gates": {"effect": True}, "mutation": "ambiguous_effect"},
        {"case_id": "case:witness", "gates": {"witness": True}, "mutation": "missing_witness"},
        {"case_id": "case:stale", "gates": {"source": True}, "mutation": "source_revision_mismatch"},
        {"case_id": "case:evidence", "gates": {"evidence": True}, "mutation": "missing_evidence_ref"},
        {"case_id": "case:false", "gates": {"test": True}, "expected_verified_closure": False},
    ]
    frozen = freeze_replay_corpus(cases)
    assert frozen.to_dict() == freeze_replay_corpus(list(reversed(cases))).to_dict()

    result = replay_mutations(frozen)
    assert set(result["baseline_results"]) == {"B0", "B1", "B2", "B3"}
    assert result["mutation_coverage"] == ["ambiguous_effect", "missing_evidence", "missing_witness", "skipped_test", "stale_source"]
    assert result["critical_mutations_blocked"] is True
    for baseline in result["baseline_results"].values():
        assert baseline["verified_closure_count"] == 1
        assert baseline["false_closure_count"] == 1
        assert baseline["authority_violation_count"] == 0
        assert {"verified_closure", "false_closure", "authority_violation"} <= set(baseline["cases"][0])


def test_phase_five_injected_authority_mutation_quarantines_case_not_candidate():
    result = replay_mutations([
        {"case_id": "case:valid", "gates": {"test": True}},
        {"case_id": "case:injected-authority", "gates": {"test": True}, "mutation": "critical_authority_violation"},
    ])
    injected = next(case for case in result["cases"] if case["case_id"] == "case:injected-authority")
    assert injected["quarantined"] is True
    assert injected["injected_mutation"] is True
    assert result["quarantined"] is False
    assert result["promotion_state"] == "NOT_PROMOTED"


def test_phase_five_critical_authority_violation_is_quarantined_and_ui_is_derived():
    result = replay_mutations([
        {"case_id": "case:authority", "gates": {"test": True}, "authority_violation": True, "critical_authority_violation": True},
    ])
    assert result["quarantined"] is True
    assert result["promotion_state"] == "QUARANTINED"
    assert result["baseline_results"]["B0"]["cases"][0]["closure_state"] == "quarantined"
    ui = evaluation_ui_projection(result, source_refs=["event:phase5"])
    assert ui["authoritative"] is False
    assert ui["promotion_state"] == "QUARANTINED"
    assert ui["source_refs"] == ["event:phase5"]


def test_phase_two_immutable_artifacts_are_typed_and_byte_stable():
    finding = FindingV1.create({
        "finding_id": "finding:one", "mission_ref": "mission:1", "role_contract_ref": "role:auditor",
        "severity": "high", "release_impact": "blocks_phase", "surface": "compiler",
        "evidence_refs": ["evidence:one"], "consequence": "wrong projection", "root_cause": "missing validation",
        "owner_preserving_fix": "reference canonical evidence", "acceptance_test": "focused test",
        "rollback_or_quarantine": "quarantine", "confidence_basis": {"confidence": 0.9, "basis": "test", "unknowns": []},
        "status": "open",
    })
    evidence = EvidenceItemV1.create({
        "evidence_id": "evidence:one", "mission_ref": "mission:1", "kind": "artifact", "source_ref": "repo:main",
        "evidence_state": "OBSERVED_SOURCE", "authority_class": "canonical_source",
        "taint": {"untrusted_data": False, "contains_instructions": False, "secret_class": "none", "allowed_sinks": []},
        "acquisition_receipt_ref": "receipt:one", "source_locator": "ares_runtime/collaboration.py",
    })
    request = TestRequestContract.create({
        "test_request_id": "test:typed", "mission_ref": "mission:1", "question": "does it reject bad input?",
        "oracle_class": "unit", "procedure": ["run focused test"], "expected_discriminators": ["typed error"],
        "required_environment": [], "authority_requirements": [], "input_refs": ["evidence:one"], "stop_conditions": ["missing evidence"],
    })
    for cls, artifact in ((FindingV1, finding), (EvidenceItemV1, evidence), (TestRequestContract, request)):
        assert cls.parse(artifact.to_dict()).canonical_bytes() == artifact.canonical_bytes()
    with pytest.raises(TypeError):
        finding.payload["status"] = "resolved"  # type: ignore[index]
    copied = evidence.to_dict(); copied["taint"]["secret_class"] = "restricted"
    assert evidence.to_dict()["taint"]["secret_class"] == "none"


def test_phase_two_context_compiler_truncates_deterministically_and_replays_idempotently():
    compiler = ContextCompiler()
    refs = [
        {"ref": "evidence:z", "digest": "sha256:" + "b" * 64, "purpose": "z"},
        {"ref": "evidence:a", "digest": "sha256:" + "a" * 64, "purpose": "a"},
        {"ref": "evidence:a", "digest": "sha256:" + "a" * 64, "purpose": "a"},
    ]
    first = compiler.compile("mission:1", "role:principal", refs, max_refs=1)
    replay = compiler.compile("mission:1", "role:principal", list(reversed(refs)), max_refs=1)
    assert isinstance(first, ContextPacketV1)
    assert first.canonical_bytes() == replay.canonical_bytes()
    assert first.to_dict()["included_refs"] == [{"ref": "evidence:a", "digest": "sha256:" + "a" * 64, "purpose": "a"}]
    assert first.to_dict()["omitted_classes"] == ["context_truncated"]
    with pytest.raises(ContractError, match="EVIDENCE_DEFICIT"):
        compiler.compile("mission:1", "role:principal", [], max_refs=1)


def test_phase_two_context_compiler_rejects_stale_or_missing_evidence_and_withholds_secrets():
    compiler = ContextCompiler()
    ref = {"ref": "evidence:public", "digest": "sha256:" + "a" * 64, "purpose": "proof"}
    with pytest.raises(ContractError, match="STALE_SOURCE_FREEZE"):
        compiler.compile("mission:1", "role:principal", [ref], source_revision="new", frozen_source_revision="old")
    with pytest.raises(ContractError, match="MISSING_EVIDENCE_REF"):
        compiler.compile("mission:1", "role:principal", [ref], resolve_ref=lambda *_: False)
    packet = compiler.compile("mission:1", "role:principal", [
        ref,
        {"ref": "evidence:secret", "digest": "sha256:" + "b" * 64, "purpose": "credential", "secret_class": "restricted"},
    ])
    rendered = packet.to_dict()
    assert rendered["included_refs"] == [ref]
    assert rendered["withheld_until_commit"] == ["restricted"]
    assert "credential" not in packet.canonical_bytes().decode()


def test_phase_two_handoff_is_artifact_only_and_replayable():
    values = {
        "handoff_id": "handoff:artifact-only", "mission_ref": "mission:1", "from_role_ref": "role:author", "to_role_ref": "role:reviewer",
        "owned_question": "is the finding supported?", "context_packet_ref": "ctx:one", "evidence_refs": ["evidence:one"],
        "unresolved_claim_refs": ["claim:one"], "withheld_classes": ["restricted"], "permit_refs": [],
        "required_output_schema_ref": "schema:finding:v1", "stop_conditions": ["evidence deficit"],
    }
    first = HandoffPacketV1.create(values)
    assert HandoffPacketV1.create(dict(values)).canonical_bytes() == first.canonical_bytes()
    assert HandoffPacketV1.parse(first.to_dict()).canonical_bytes() == first.canonical_bytes()
    with pytest.raises(ContractError, match="UNKNOWN_FIELD"):
        HandoffPacketV1.create({**values, "embedded_secret": "never transfer source content"})


def test_phase_one_bindings_deny_unknown_owners_and_preserve_supersession():
    writes = []
    bindings = ContractBindings(
        enabled=lambda: True,
        profile_exists=lambda ref: ref == "profile:primary",
        task_exists=lambda ref: ref == "task:1",
        set_profile_contract_ref=lambda *args: writes.append(("profile", args)),
        attach_task_contract_ref=lambda *args: writes.append(("task", args)),
        attach_goal_contract_ref=lambda *args: writes.append(("goal", args)),
    )
    first_role, first_mission = role(), mission()
    next_role = RoleContractV1.create({**first_role.to_dict(), "supersedes_contract_ref": "contract:" + first_role.artifact_digest[7:]})
    next_mission = MissionContractV1.create({**first_mission.to_dict(), "goal_ref": "goal:session-1", "supersedes_contract_ref": "contract:" + first_mission.artifact_digest[7:]})
    assert bindings.bind_role(next_role).startswith("contract:")
    assert bindings.bind_mission(next_mission).startswith("contract:")
    assert [kind for kind, _ in writes] == ["profile", "task", "goal"]
    with pytest.raises(ContractError, match="UNKNOWN_PROFILE"):
        bindings.bind_role(RoleContractV1.create({**first_role.to_dict(), "profile_ref": "profile:unknown"}))
    with pytest.raises(ContractError, match="UNKNOWN_OWNER_REFERENCE"):
        bindings.bind_mission(MissionContractV1.create({**first_mission.to_dict(), "kanban_root_task_ref": "task:unknown"}))


def test_phase_one_feature_off_is_noop_and_profile_secret_is_not_copied(tmp_path):
    writes = []
    secret = "do-not-copy-this-profile-secret"
    profile = tmp_path / "profile"; profile.mkdir()
    (profile / ".env").write_text("API_KEY=" + secret, encoding="utf-8")
    bindings = ContractBindings(
        enabled=lambda: False,
        profile_exists=lambda _: False,
        task_exists=lambda _: False,
        set_profile_contract_ref=lambda *args: writes.append(args),
        attach_task_contract_ref=lambda *args: writes.append(args),
    )
    assert bindings.bind_role(role()).startswith("contract:")
    assert bindings.bind_mission(mission()).startswith("contract:")
    assert writes == []
    assert secret not in role().canonical_bytes().decode("utf-8")
