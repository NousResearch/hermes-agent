from __future__ import annotations

import json

import pytest

from ares_runtime import (
    ActivationGrant,
    AresRuntimeError,
    AresRuntimeLayout,
    InstalledRuntimePointer,
    ReleaseReference,
    RuntimeIdentity,
)


def _digest(char: str) -> str:
    return char * 64


def _reference(char: str = "a") -> ReleaseReference:
    return ReleaseReference(
        "sealed_candidate", _digest(char), _digest("b"), _digest("c")
    )


def _pointer(root, generation: int = 1) -> InstalledRuntimePointer:
    return InstalledRuntimePointer(
        generation=generation,
        current=_reference("a"),
        previous=_reference("c") if generation > 1 else None,
        committed_transaction_id=_digest("d"),
        state_root=str(root.parent),
    )


def test_pointer_round_trip_is_canonical_and_atomic(tmp_path):
    layout = AresRuntimeLayout(tmp_path / "hermes" / "ares")
    pointer = _pointer(layout.root)

    layout.write_pointer_atomic(pointer)

    assert layout.read_pointer() == pointer
    assert layout.pointer_path.read_bytes() == pointer.canonical_bytes()
    assert not list(layout.root.glob(".release-state-*.tmp"))


def test_pointer_rejects_symlink_and_duplicate_json(tmp_path):
    layout = AresRuntimeLayout(tmp_path / "hermes" / "ares")
    layout.initialize()
    layout.pointer_path.write_text(
        '{"schema":"AresInstalledRuntimePointerV1","schema":"bad"}\n'
    )
    layout.pointer_path.chmod(0o600)

    with pytest.raises(AresRuntimeError, match="DUPLICATE_JSON_KEY"):
        layout.read_pointer()


def test_release_reference_rejects_non_content_addressed_identity():
    with pytest.raises(AresRuntimeError, match="INVALID_IDENTITY"):
        ReleaseReference("sealed_candidate", "candidate-v1", _digest("a"), _digest("b"))


def test_pointer_cannot_select_the_same_current_and_previous(tmp_path):
    current = _reference()

    with pytest.raises(AresRuntimeError, match="current equals previous"):
        InstalledRuntimePointer(
            generation=1,
            current=current,
            previous=current,
            committed_transaction_id=_digest("c"),
            state_root=str(tmp_path),
        )


def test_activation_grant_is_bound_to_all_identity_fields(tmp_path):
    grant = ActivationGrant(
        candidate_id=_digest("a"),
        certification_set_id=_digest("b"),
        sealed_candidate_id=_digest("c"),
        audit_subject_id=_digest("d"),
        audit_subject_sha256=_digest("e"),
        audit_result_sha256=_digest("f"),
        archive_sha256=_digest("1"),
        candidate_core_sha256=_digest("2"),
        sealed_manifest_sha256=_digest("3"),
        release_manifest_sha256=_digest("4"),
        runtime_tree_sha256=_digest("5"),
        custody_event_sequence=4,
        target_platform="linux-x86_64",
        target_release_root=str(tmp_path / "ares" / "releases"),
        materializer_contract="AresReleaseMaterializerV1",
        activator_contract="AresReleaseActivatorV1",
        resolver_contract="AresRuntimeResolverV1",
    ).with_grant_id()

    parsed = ActivationGrant.parse(grant.canonical_bytes())
    assert parsed == grant
    changed = json.loads(grant.canonical_bytes())
    changed["archive_sha256"] = _digest("6")
    changed_raw = (
        json.dumps(changed, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )

    with pytest.raises(AresRuntimeError, match="grant_id"):
        ActivationGrant.parse(changed_raw)


def test_runtime_identity_requires_exact_runtime_digests():
    identity = RuntimeIdentity(
        sealed_candidate_id=_digest("a"),
        release_manifest_sha256=_digest("b"),
        runtime_tree_sha256=_digest("c"),
        resolver_sha256=_digest("d"),
        role="gateway",
        generation=1,
    )

    assert identity.to_dict()["role"] == "gateway"
