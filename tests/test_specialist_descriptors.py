from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from ares_runtime.collaboration import (
    ContractError,
    SpecialistDescriptorV1,
    digest,
    specialist_descriptor_ref,
    validate_specialist_descriptor_set,
)


ROLE_ARTIFACTS = {
    "role.explorer": ["explorer_dissent"],
    "role.data_evidence": ["evidence_lineage_record"],
}
PROFILES = ["explorer", "longmemeval-bench"]


def descriptor_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "profile_id": "explorer",
        "semantic_role_id": "role.explorer",
        "enabled": False,
        "narrow_purpose": "Generate competing hypotheses and a falsifier before a commitment.",
        "capability_classes": ["competing_design_generation"],
        "tool_classes": ["source_read"],
        "required_artifact_ids": ["explorer_dissent"],
        "input_evidence_classes": ["pinned_source"],
        "required_outputs": ["falsifier"],
        "explicit_exclusions": ["runtime_authority"],
        "mandatory_deferrals": ["statistician_for_quantitative_inference"],
        "handoff_rules": ["handoff_requires_preserved_artifact_reference"],
        "failure_and_abstention_behavior": {
            "on_insufficient_evidence": "blocked_or_unknown",
            "on_unavailable_or_disabled_contract": "blocked_or_unknown",
            "generic_fallback_label": "forbidden",
        },
        "activation_evidence_refs": ["evidence:activation-protocol-v1"],
        "provenance": {
            "source_refs": ["evidence:research-draft-v1"],
            "semantic_registry_ref": "docs:role-contracts/role-contracts.json",
            "semantic_registry_digest": "sha256:" + "a" * 64,
        },
    }
    values.update(overrides)
    return values


def create_descriptor(**overrides: object) -> SpecialistDescriptorV1:
    return SpecialistDescriptorV1.create(
        descriptor_values(**overrides),
        profile_exists=lambda profile_id: profile_id in PROFILES,
        semantic_role_artifacts=ROLE_ARTIFACTS,
    )


def test_descriptor_is_immutable_deterministic_and_unbound() -> None:
    first = create_descriptor()
    second = create_descriptor()

    assert first.artifact_digest.startswith("sha256:")
    assert first.canonical_bytes() == second.canonical_bytes()
    assert (
        SpecialistDescriptorV1.parse(
            first.to_dict(),
            profile_exists=lambda profile_id: profile_id in PROFILES,
            semantic_role_artifacts=ROLE_ARTIFACTS,
        ).canonical_bytes()
        == first.canonical_bytes()
    )
    assert specialist_descriptor_ref(
        first
    ) == "specialist-descriptor:" + first.artifact_digest.removeprefix("sha256:")
    with pytest.raises(TypeError):
        first.payload["enabled"] = True  # type: ignore[index]


def test_descriptor_rejects_unknown_dynamic_runtime_fields() -> None:
    for field in (
        "provider_health",
        "current_session_model",
        "credential",
        "gateway_state",
        "desktop_capacity",
        "capacity_reservation_id",
        "latency_measurement",
        "cost_measurement",
    ):
        with pytest.raises(ContractError, match="UNKNOWN_FIELD"):
            create_descriptor(**{field: "must-remain-runtime-evidence"})


def test_descriptor_rejects_unknown_profile_role_artifact_and_digest_tampering() -> (
    None
):
    with pytest.raises(ContractError, match="UNKNOWN_PROFILE"):
        create_descriptor(profile_id="not-a-profile")
    with pytest.raises(ContractError, match="UNKNOWN_SEMANTIC_ROLE"):
        create_descriptor(semantic_role_id="role.missing")
    with pytest.raises(ContractError, match="REQUIRED_ARTIFACT_MISMATCH"):
        create_descriptor(required_artifact_ids=["wrong_artifact"])

    raw = create_descriptor().to_dict()
    raw["descriptor_digest"] = "sha256:" + "b" * 64
    with pytest.raises(ContractError, match="DIGEST_MISMATCH"):
        SpecialistDescriptorV1.parse(raw)


def test_descriptor_requires_explicit_abstention_and_never_labels_a_generic_fallback_as_specialist() -> (
    None
):
    behavior = copy.deepcopy(descriptor_values()["failure_and_abstention_behavior"])
    assert isinstance(behavior, dict)
    behavior["generic_fallback_label"] = "allowed"
    with pytest.raises(ContractError, match="SCHEMA_INVALID"):
        create_descriptor(failure_and_abstention_behavior=behavior)


def test_descriptor_set_requires_exact_roster_and_disabled_contracts() -> None:
    explorer = create_descriptor().to_dict()
    data = create_descriptor(
        profile_id="longmemeval-bench",
        semantic_role_id="role.data_evidence",
        required_artifact_ids=["evidence_lineage_record"],
    ).to_dict()

    assert (
        validate_specialist_descriptor_set(
            [explorer, data],
            profile_ids=PROFILES,
            semantic_role_artifacts=ROLE_ARTIFACTS,
        )
        == []
    )

    enabled = create_descriptor(enabled=True).to_dict()
    assert (
        "descriptor must remain disabled: explorer"
        in validate_specialist_descriptor_set(
            [enabled, data],
            profile_ids=PROFILES,
            semantic_role_artifacts=ROLE_ARTIFACTS,
        )
    )
    assert (
        "missing descriptor profile_ids: longmemeval-bench"
        in validate_specialist_descriptor_set(
            [explorer],
            profile_ids=PROFILES,
            semantic_role_artifacts=ROLE_ARTIFACTS,
        )
    )


def _load_validator():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "validate_specialist_descriptors",
        root / "scripts/validate_specialist_descriptors.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_descriptor_cli_accepts_content_addressed_disabled_manifest(
    tmp_path, capsys
) -> None:
    validator = _load_validator()
    registry = {
        "roles": [
            {
                "role_id": "role.explorer",
                "required_artifacts": [{"artifact_id": "explorer_dissent"}],
            }
        ]
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    descriptor = create_descriptor(
        provenance={
            "source_refs": ["evidence:research-draft-v1"],
            "semantic_registry_ref": "docs:role-contracts/role-contracts.json",
            "semantic_registry_digest": "sha256:"
            + hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        }
    ).to_dict()
    descriptor_dir = tmp_path / "descriptors"
    descriptor_dir.mkdir()
    (descriptor_dir / "explorer.json").write_text(
        json.dumps(descriptor), encoding="utf-8"
    )
    manifest = {
        "schema": "AresSpecialistDescriptorManifestV1",
        "profile_ids": ["explorer"],
        "descriptors": [
            {
                "profile_id": "explorer",
                "path": "explorer.json",
                "descriptor_digest": descriptor["descriptor_digest"],
            }
        ],
    }
    manifest["manifest_digest"] = digest(manifest)
    (descriptor_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    assert (
        validator.main([
            "--registry",
            str(registry_path),
            "--descriptor-dir",
            str(descriptor_dir),
            "--profile",
            "explorer",
        ])
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["enabled_descriptor_count"] == 0

    descriptor["provider_health"] = "not-static"
    (descriptor_dir / "explorer.json").write_text(
        json.dumps(descriptor), encoding="utf-8"
    )
    assert (
        validator.main([
            "--registry",
            str(registry_path),
            "--descriptor-dir",
            str(descriptor_dir),
            "--profile",
            "explorer",
        ])
        == 1
    )
    assert "UNKNOWN_FIELD: provider_health" in capsys.readouterr().out


def test_descriptor_cli_rejects_manifest_consistent_registry_provenance_drift(
    tmp_path, capsys
) -> None:
    validator = _load_validator()
    registry = {
        "roles": [
            {
                "role_id": "role.explorer",
                "required_artifacts": [{"artifact_id": "explorer_dissent"}],
            }
        ]
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    registry_digest = "sha256:" + hashlib.sha256(registry_path.read_bytes()).hexdigest()
    descriptor = create_descriptor(
        provenance={
            "source_refs": ["evidence:research-draft-v1"],
            "semantic_registry_ref": "docs:role-contracts/role-contracts.json",
            "semantic_registry_digest": registry_digest,
        }
    ).to_dict()
    descriptor_dir = tmp_path / "descriptors"
    descriptor_dir.mkdir()

    drifted = copy.deepcopy(descriptor)
    drifted["provenance"]["semantic_registry_digest"] = "sha256:" + "0" * 64
    drifted["descriptor_digest"] = digest(
        {key: value for key, value in drifted.items() if key != "descriptor_digest"}
    )
    (descriptor_dir / "explorer.json").write_text(json.dumps(drifted), encoding="utf-8")
    manifest = {
        "schema": "AresSpecialistDescriptorManifestV1",
        "profile_ids": ["explorer"],
        "descriptors": [
            {
                "profile_id": "explorer",
                "path": "explorer.json",
                "descriptor_digest": drifted["descriptor_digest"],
            }
        ],
    }
    manifest["manifest_digest"] = digest(manifest)
    (descriptor_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert (
        validator.main(
            [
                "--registry",
                str(registry_path),
                "--descriptor-dir",
                str(descriptor_dir),
                "--profile",
                "explorer",
            ]
        )
        == 1
    )
    assert "SEMANTIC_REGISTRY_DIGEST_MISMATCH" in capsys.readouterr().out
