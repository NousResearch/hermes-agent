from __future__ import annotations

from ares_runtime.collaboration import SpecialistDescriptorV1, specialist_descriptor_ref
from ares_runtime.specialist_routing import decide_specialist_nonactivation


ROLE_ARTIFACTS = {"role.explorer": ["explorer_dissent"]}


def descriptor(profile_id: str = "explorer") -> SpecialistDescriptorV1:
    return SpecialistDescriptorV1.create(
        {
            "profile_id": profile_id,
            "semantic_role_id": "role.explorer",
            "enabled": False,
            "narrow_purpose": "Competing hypotheses only.",
            "capability_classes": ["competing_design"],
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
        profile_exists=lambda candidate: candidate in {"explorer", "explorer-two"},
        semantic_role_artifacts=ROLE_ARTIFACTS,
    )


def test_no_material_gap_is_a_first_class_no_specialist_receipt():
    receipt = decide_specialist_nonactivation(
        request_id="request:1",
        request_digest="sha256:" + "a" * 64,
        requested_capability=None,
        candidates=[descriptor()],
        profile_binding_refs={"explorer": specialist_descriptor_ref(descriptor())},
        policy_version="specialist-policy-v1",
    ).to_dict()

    assert receipt["outcome"] == "no_specialist_needed"
    assert receipt["selected_profile_id"] is None
    assert receipt["dispatch_authorized"] is False


def test_matching_disabled_descriptor_is_blocked_not_selected():
    candidate = descriptor()
    receipt = decide_specialist_nonactivation(
        request_id="request:2",
        request_digest="sha256:" + "b" * 64,
        requested_capability="competing_design",
        candidates=[candidate],
        profile_binding_refs={"explorer": specialist_descriptor_ref(candidate)},
        policy_version="specialist-policy-v1",
    ).to_dict()

    assert receipt["outcome"] == "blocked"
    assert receipt["selected_profile_id"] is None
    assert receipt["candidate_rejections"] == [
        {"profile_id": "explorer", "reason_code": "DESCRIPTOR_DISABLED"}
    ]


def test_binding_mismatch_cannot_be_presented_as_specialist_selection():
    candidate = descriptor()
    receipt = decide_specialist_nonactivation(
        request_id="request:3",
        request_digest="sha256:" + "c" * 64,
        requested_capability="competing_design",
        candidates=[candidate],
        profile_binding_refs={"explorer": "specialist-descriptor:" + "0" * 64},
        policy_version="specialist-policy-v1",
    ).to_dict()

    assert receipt["selected_profile_id"] is None
    assert receipt["candidate_rejections"] == [
        {"profile_id": "explorer", "reason_code": "PROFILE_BINDING_MISMATCH"}
    ]