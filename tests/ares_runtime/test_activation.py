from __future__ import annotations

from pathlib import Path

import pytest

from ares_runtime import (
    ActivationGrant,
    ActivationState,
    AresReleaseActivator,
    AresRuntimeError,
    AresRuntimeLayout,
    InstalledRuntimePointer,
    ReleaseReference,
    RuntimeIdentity,
)
from ares_runtime.materializer import MaterializedRelease


def _digest(char: str) -> str:
    return char * 64


def _grant(layout: AresRuntimeLayout) -> ActivationGrant:
    return ActivationGrant(
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
        custody_event_sequence=5,
        target_platform="linux-x86_64",
        target_release_root=str(layout.releases_dir),
        materializer_contract="AresReleaseMaterializerV1",
        activator_contract="AresReleaseActivatorV1",
        resolver_contract="AresRuntimeResolverV1",
    ).with_grant_id()


class FakeStore:
    def __init__(self, grant: ActivationGrant):
        self.grant = grant
        self.events: list[str] = []

    def verify(self, sealed_candidate_id: str):
        assert sealed_candidate_id == self.grant.sealed_candidate_id
        return {
            "lifecycle_state": "AWAITING_ACTIVATION",
            "audit_state": "AUDIT_PASSED",
            "activation_authorization_state": "AUTHORIZED",
        }

    def read_activation_grant(self, sealed_candidate_id: str) -> ActivationGrant:
        assert sealed_candidate_id == self.grant.sealed_candidate_id
        return self.grant

    def record_activation_success(
        self, sealed_candidate_id: str, *, grant_id: str, reason: str
    ):
        assert sealed_candidate_id == self.grant.sealed_candidate_id
        assert grant_id == self.grant.grant_id
        self.events.append("active")

    def record_rollback_required(
        self, sealed_candidate_id: str, *, grant_id: str, reason: str
    ):
        assert grant_id == self.grant.grant_id
        self.events.append("rollback-required")

    def record_rollback_success(
        self, sealed_candidate_id: str, *, grant_id: str, reason: str
    ):
        assert grant_id == self.grant.grant_id
        self.events.append("rolled-back")


class FakeSupervisor:
    def __init__(self, grant: ActivationGrant, *, unhealthy_new: bool = False):
        self.grant = grant
        self.unhealthy_new = unhealthy_new
        self.calls: list[tuple[str, str]] = []

    def quiesce(self, transaction_id: str) -> None:
        self.calls.append(("quiesce", transaction_id))

    def start(self, release: ReleaseReference) -> None:
        self.calls.append(("start", release.release_id))

    def health(self, release: ReleaseReference, generation: int) -> RuntimeIdentity:
        self.calls.append(("health", release.release_id))
        if self.unhealthy_new and release.release_id == self.grant.sealed_candidate_id:
            return RuntimeIdentity(
                sealed_candidate_id=_digest("0"),
                release_manifest_sha256=self.grant.release_manifest_sha256,
                runtime_tree_sha256=self.grant.runtime_tree_sha256,
                resolver_sha256=_digest("6"),
                role="gateway",
                generation=generation,
            )
        return RuntimeIdentity(
            sealed_candidate_id=release.release_id,
            release_manifest_sha256=release.release_manifest_sha256,
            runtime_tree_sha256=release.runtime_tree_sha256,
            resolver_sha256=_digest("6"),
            role="gateway",
            generation=generation,
        )


def _old_reference() -> ReleaseReference:
    return ReleaseReference(
        "sealed_candidate", _digest("7"), _digest("8"), _digest("a")
    )


def _install_previous(layout: AresRuntimeLayout) -> None:
    layout.write_pointer_atomic(
        InstalledRuntimePointer(
            generation=1,
            current=_old_reference(),
            previous=None,
            committed_transaction_id=_digest("9"),
            state_root=str(layout.root.parent),
        )
    )


def _materializer(
    _store, layout: AresRuntimeLayout, sealed_candidate_id: str, grant: ActivationGrant
):
    root = layout.release_dir(sealed_candidate_id)
    root.mkdir(parents=True)
    (root / "release-manifest.json").write_text("{}\n")
    return MaterializedRelease(
        sealed_candidate_id, root, {"archive_sha256": grant.archive_sha256}
    )


def test_activation_commits_exact_release_after_health(tmp_path: Path):
    layout = AresRuntimeLayout(tmp_path / "hermes" / "ares")
    _install_previous(layout)
    grant = _grant(layout)
    store = FakeStore(grant)
    supervisor = FakeSupervisor(grant)

    result = AresReleaseActivator(
        store=store, layout=layout, supervisor=supervisor, materializer=_materializer
    ).activate(grant.sealed_candidate_id)

    assert result.state == ActivationState.ACTIVATED
    assert layout.read_pointer().current.release_id == grant.sealed_candidate_id
    assert layout.read_pointer().previous == _old_reference()
    assert store.events == ["active"]
    assert [name for name, _value in supervisor.calls] == [
        "quiesce",
        "start",
        "health",
    ]


def test_failed_postcommit_health_restores_previous_release(tmp_path: Path):
    layout = AresRuntimeLayout(tmp_path / "hermes" / "ares")
    _install_previous(layout)
    grant = _grant(layout)
    store = FakeStore(grant)
    supervisor = FakeSupervisor(grant, unhealthy_new=True)

    result = AresReleaseActivator(
        store=store, layout=layout, supervisor=supervisor, materializer=_materializer
    ).activate(grant.sealed_candidate_id)

    assert result.state == ActivationState.ROLLED_BACK
    assert layout.read_pointer().current == _old_reference()
    assert store.events == ["rollback-required", "rolled-back"]


def test_activation_without_verified_rollback_target_fails_precommit(tmp_path: Path):
    layout = AresRuntimeLayout(tmp_path / "hermes" / "ares")
    grant = _grant(layout)

    with pytest.raises(AresRuntimeError, match="ROLLBACK_TARGET_MISSING"):
        AresReleaseActivator(
            store=FakeStore(grant),
            layout=layout,
            supervisor=FakeSupervisor(grant),
            materializer=_materializer,
        ).activate(grant.sealed_candidate_id)
