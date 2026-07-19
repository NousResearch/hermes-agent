from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import direct_iam_identity_author as author
from scripts.canary import direct_iam_identity_authority as authority_schema
from scripts.canary import full_canary_owner_launcher as owner_launcher
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import source_artifact_publication as source_publication


REVISION = "a" * 40
OWNER_REAUTH_SHA256 = "8" * 64
PRE_FOUNDATION_SHA256 = "6" * 64
FOUNDATION_APPLY_SHA256 = "7" * 64
NOW = 1_800_000_000
ORGANIZATION = "organizations/123456789012"
OWNER_MEMBER = (
    "serviceAccount:"
    f"{authority_schema.OWNER_GATE_SERVICE_ACCOUNT_EMAIL}"
)
EXTERNAL_OWNER_MEMBER = "user:owner@example.test"


class _TrustedOwnerTokenProvider(owner_launcher.GcloudOwnerAccessToken):
    def __init__(
        self,
        token: str = "owner-token-that-is-not-printed",
        *,
        account: str = author.OWNER_ACCOUNT,
        failure: BaseException | None = None,
    ) -> None:
        self._test_token = token
        self._test_account = account
        self._test_failure = failure
        self.calls = 0
        self.stability_checks = 0

    @property
    def approved_account(self) -> str:
        return self._test_account

    def __call__(self) -> str:
        self.calls += 1
        if self._test_failure is not None:
            raise self._test_failure
        return self._test_token

    def require_stable(self) -> None:
        self.stability_checks += 1


def _role(
    name: str,
    *,
    title: str,
    description: str,
    permissions: list[str],
) -> Mapping[str, Any]:
    return {
        "name": name,
        "title": title,
        "description": description,
        "includedPermissions": permissions,
        "stage": "GA",
        "deleted": False,
        "etag": "role-etag",
    }


def _live_facts() -> Mapping[str, Any]:
    ancestor_role = (
        f"{ORGANIZATION}/roles/"
        f"{foundation.ANCESTOR_READ_ONLY_IAM_ROLE_ID}"
    )
    external_role = "roles/owner"
    project_policy = {
        "version": 3,
        "etag": "project-policy-etag",
        "bindings": [
            {
                "role": author.PROJECT_READ_ROLE,
                "members": [OWNER_MEMBER],
            },
            {
                "role": external_role,
                "members": [EXTERNAL_OWNER_MEMBER],
            },
        ],
        "auditConfigs": [],
    }
    organization_policy = {
        "version": 3,
        "etag": "organization-policy-etag",
        "bindings": [{
            "role": ancestor_role,
            "members": [OWNER_MEMBER],
        }],
        "auditConfigs": [],
    }
    return {
        "instance": {
            "name": foundation.VM_NAME,
            "id": "1234567890123456789",
            "status": "RUNNING",
            "zone": (
                "https://www.googleapis.com/compute/v1/projects/"
                f"{foundation.PROJECT}/zones/{foundation.ZONE}"
            ),
            "serviceAccounts": [{
                "email": authority_schema.OWNER_GATE_SERVICE_ACCOUNT_EMAIL,
                "scopes": list(reversed(foundation.OWNER_GATE_OAUTH_SCOPES)),
            }],
        },
        "resources": [
            {
                "name": author.PROJECT_RESOURCE,
                "projectId": foundation.PROJECT,
                "parent": ORGANIZATION,
                "state": "ACTIVE",
                "etag": "project-etag",
            },
            {
                "name": ORGANIZATION,
                "state": "ACTIVE",
                "etag": "organization-etag",
            },
        ],
        "resource_names": [author.PROJECT_RESOURCE, ORGANIZATION],
        "policies": [project_policy, organization_policy],
        "roles": {
            author.PROJECT_READ_ROLE: _role(
                author.PROJECT_READ_ROLE,
                title=foundation.PROJECT_READ_ROLE_TITLE,
                description=foundation.PROJECT_READ_ROLE_DESCRIPTION,
                permissions=list(reversed(foundation.READ_ONLY_IAM_PERMISSIONS)),
            ),
            ancestor_role: _role(
                ancestor_role,
                title=foundation.ANCESTOR_READ_ROLE_TITLE,
                description=foundation.ANCESTOR_READ_ROLE_DESCRIPTION,
                permissions=list(
                    reversed(foundation.DIRECT_IAM_ANCESTOR_PERMISSIONS)
                ),
            ),
            author.MUTATION_ROLE: _role(
                author.MUTATION_ROLE,
                title=foundation.MUTATION_ROLE_TITLE,
                description=foundation.MUTATION_ROLE_DESCRIPTION,
                permissions=list(reversed(foundation.MUTATION_PERMISSIONS)),
            ),
            external_role: _role(
                external_role,
                title="Owner",
                description="Full project authority",
                permissions=["resourcemanager.projects.setIamPolicy"],
            ),
        },
        "owner_gate_service_account": {
            "name": author.OWNER_GATE_SERVICE_ACCOUNT_PATH.removeprefix("/v1/"),
            "projectId": foundation.PROJECT,
            "email": authority_schema.OWNER_GATE_SERVICE_ACCOUNT_EMAIL,
            "uniqueId": "123456789012345678901",
            "disabled": False,
        },
        "target_service_account": {
            "name": author.TARGET_SERVICE_ACCOUNT_PATH.removeprefix("/v1/"),
            "projectId": foundation.PROJECT,
            "email": authority_schema.TARGET_SERVICE_ACCOUNT_EMAIL,
            "uniqueId": "223456789012345678901",
            "disabled": False,
        },
        "owner_gate_service_account_policy": {},
        "owner_gate_user_managed_keys": {},
    }


def _build(facts: Mapping[str, Any] | None = None) -> bytes:
    return author._build_authority_bytes(
        live_facts=_live_facts() if facts is None else facts,
        release_revision=REVISION,
        owner_reauthentication_receipt_sha256=OWNER_REAUTH_SHA256,
        pre_foundation_authority_sha256=PRE_FOUNDATION_SHA256,
        foundation_apply_receipt_sha256=FOUNDATION_APPLY_SHA256,
        collected_at_unix=NOW,
    )


def _validated_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Mapping[str, bytes], Any, Any]:
    from scripts.canary import owner_gate_foundation_apply as foundation_apply
    from scripts.canary import owner_gate_trust as trust
    from tests.scripts.canary import test_owner_gate_pre_foundation as fixture

    monkeypatch.setattr(
        fixture,
        "PROJECT_NUMBER",
        authority_schema.PROJECT_NUMBER,
    )
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        fixture.RELEASE_KEY_ID,
    )
    signed_authority, plan, _evidence = fixture._authority()
    raws = {
        "pre": foundation.canonical_json_bytes(signed_authority),
        "reauth": foundation.canonical_json_bytes(
            fixture._owner_reauth_receipt()
        ),
        "network": foundation.canonical_json_bytes(
            fixture._signed_network_evidence()
        ),
        "ancestry": fixture._signed_ancestry_raw(),
        "apply": foundation.canonical_json_bytes(
            fixture._apply_receipt(signed_authority, plan)
        ),
    }
    foundation_a = foundation_apply.decode_validated_foundation_a_chain(
        pre_foundation_authority_raw=raws["pre"],
        owner_reauthentication_receipt_raw=raws["reauth"],
        network_evidence_raw=raws["network"],
        project_ancestry_evidence_raw=raws["ancestry"],
        release_public_key=fixture.RELEASE_KEY.public_key(),
        network_collector_public_key=fixture.NETWORK_KEY.public_key(),
        project_ancestry_collector_public_key=fixture.NETWORK_KEY.public_key(),
        now_unix=fixture.NOW + 1,
    )
    chain = foundation_apply._decode_validated_foundation_apply_chain(
        foundation_a=foundation_a,
        apply_receipt_raw=raws["apply"],
        now_unix=fixture.NOW + 21,
    )

    def load_fixed_journal(candidate: Any) -> Any:
        expected = chain.foundation_a
        if (
            type(candidate) is not foundation_apply.ValidatedFoundationAChain
            or candidate.pre_foundation_authority_raw
            != expected.pre_foundation_authority_raw
            or candidate.owner_reauthentication_receipt_raw
            != expected.owner_reauthentication_receipt_raw
            or candidate.network_evidence_raw != expected.network_evidence_raw
            or candidate.ancestry_evidence_raw != expected.ancestry_evidence_raw
        ):
            raise foundation_apply.OwnerGateFoundationApplyError(
                "owner_gate_foundation_success_journal_invalid"
            )
        return foundation_apply.ValidatedFoundationApplyChain._create(
            foundation_a=candidate,
            apply_receipt=chain.apply_receipt,
            apply_receipt_raw=chain.apply_receipt_raw,
        )

    monkeypatch.setattr(
        foundation_apply,
        "load_validated_foundation_apply_chain",
        load_fixed_journal,
    )
    return chain, raws, fixture.RELEASE_KEY, fixture.NETWORK_KEY


def _facts_for_chain(chain: Any) -> Mapping[str, Any]:
    from tests.scripts.canary import test_owner_gate_pre_foundation as fixture

    facts = copy.deepcopy(_live_facts())
    facts["instance"]["id"] = chain.owner_gate_vm_identity["numeric_id"]
    facts["owner_gate_service_account"]["uniqueId"] = (
        chain.service_account_identity["unique_id"]
    )
    folder = f"folders/{fixture.FOLDER_ID}"
    organization = f"organizations/{fixture.ORGANIZATION_ID}"
    facts["resources"][0]["parent"] = folder
    facts["resources"].insert(1, {
        "name": folder,
        "parent": organization,
        "state": "ACTIVE",
        "etag": "folder-etag",
    })
    facts["resource_names"] = [author.PROJECT_RESOURCE, folder, organization]
    facts["policies"].insert(1, {
        "version": 3,
        "etag": "folder-policy-etag",
        "bindings": [],
        "auditConfigs": [],
    })
    return facts


def _authority_for_chain(chain: Any, *, facts: Mapping[str, Any] | None = None) -> bytes:
    from tests.scripts.canary import test_owner_gate_pre_foundation as fixture

    return author._build_authority_bytes(
        live_facts=_facts_for_chain(chain) if facts is None else facts,
        release_revision=chain.foundation_source_revision,
        owner_reauthentication_receipt_sha256=(
            chain.owner_reauthentication_receipt_sha256
        ),
        pre_foundation_authority_sha256=(
            chain.pre_foundation_authority_sha256
        ),
        foundation_apply_receipt_sha256=(
            chain.foundation_apply_receipt_sha256
        ),
        collected_at_unix=fixture.NOW + 21,
    )


def test_author_builds_only_schema_canonical_bytes_and_binds_chain() -> None:
    raw = _build()
    value = authority_schema.decode_canonical(raw, release_revision=REVISION)

    assert raw == foundation.canonical_json_bytes(value)
    assert value["pre_foundation_authority_sha256"] == PRE_FOUNDATION_SHA256
    assert value["foundation_apply_receipt_sha256"] == FOUNDATION_APPLY_SHA256
    assert value["owner_reauthentication_receipt_sha256"] == OWNER_REAUTH_SHA256
    root = value["external_gcp_admin_trust_root"]
    assert root["structural_partition_complete"] is True
    assert root["resource_policy_generations"] == [
        {
            "resource": f"projects/{foundation.PROJECT}",
            "version": 3,
            "etag": "project-policy-etag",
            "audit_configs": [],
        },
        {
            "resource": ORGANIZATION,
            "version": 3,
            "etag": "organization-policy-etag",
            "audit_configs": [],
        },
    ]
    assert root["allowed_residual_bindings"] == [{
        "resource": f"projects/{foundation.PROJECT}",
        "role": "roles/owner",
        "members": [EXTERNAL_OWNER_MEMBER],
        "condition": None,
    }]
    assert root["allowed_residual_role_definitions"] == [{
        "name": "roles/owner",
        "title": "Owner",
        "description": "Full project authority",
        "included_permissions": ["resourcemanager.projects.setIamPolicy"],
        "stage": "GA",
        "deleted": False,
        "etag": "role-etag",
    }]
    unsigned = {key: item for key, item in value.items() if key != "authority_sha256"}
    assert value["authority_sha256"] == foundation.sha256_json(unsigned)


def test_large_residual_role_inventory_fits_bounded_canonical_authority(
) -> None:
    facts = copy.deepcopy(_live_facts())
    facts["policies"][0]["bindings"].append({
        "role": "roles/editor",
        "members": ["user:editor@example.test"],
    })
    large_permissions = [
        f"service.resource.permission{index:05d}"
        for index in range(12_000)
    ]
    facts["roles"]["roles/owner"] = _role(
        "roles/owner",
        title="Owner",
        description="Full project authority",
        permissions=large_permissions,
    )
    facts["roles"]["roles/editor"] = _role(
        "roles/editor",
        title="Editor",
        description="Project editor authority",
        permissions=large_permissions,
    )

    raw = _build(facts)

    assert 256 * 1024 < len(raw) <= authority_schema.MAX_BYTES
    decoded = authority_schema.decode_canonical(raw, release_revision=REVISION)
    definitions = decoded["external_gcp_admin_trust_root"][
        "allowed_residual_role_definitions"
    ]
    assert {item["name"] for item in definitions} == {
        "roles/editor",
        "roles/owner",
    }
    with pytest.raises(
        authority_schema.DirectIamIdentityAuthorityError,
        match="^direct_iam_identity_authority_invalid$",
    ):
        authority_schema.decode_canonical(
            b"x" * (authority_schema.MAX_BYTES + 1),
            release_revision=REVISION,
        )


def test_canonical_authority_revalidates_real_signed_apply_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.scripts.canary import test_owner_gate_pre_foundation as fixture

    chain, _raws, _release_key, _network_key = _validated_chain(monkeypatch)
    raw = _authority_for_chain(chain)

    canonical = author.decode_canonical_authority_for_validated_chain(
        raw,
        foundation_chain=chain,
        now_unix=fixture.NOW + 21,
    )

    assert type(canonical) is author.CanonicalDirectIamAuthority
    assert canonical.raw == raw
    assert canonical.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert canonical.value["release_revision"] == chain.foundation_source_revision
    assert canonical.value["owner_gate_vm_numeric_id"] == (
        chain.owner_gate_vm_identity["numeric_id"]
    )
    assert canonical.value["owner_gate_service_account_unique_id"] == (
        chain.service_account_identity["unique_id"]
    )


def test_forged_apply_chain_marker_is_redecoded_and_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.canary import owner_gate_foundation_apply as foundation_apply
    from tests.scripts.canary import test_owner_gate_pre_foundation as fixture

    chain, _raws, _release_key, _network_key = _validated_chain(monkeypatch)
    forged = object.__new__(foundation_apply.ValidatedFoundationApplyChain)
    object.__setattr__(forged, "foundation_a", chain.foundation_a)
    tampered_receipt = copy.deepcopy(chain.apply_receipt)
    tampered_receipt["completed_at_unix"] += 1
    object.__setattr__(forged, "apply_receipt", tampered_receipt)
    object.__setattr__(forged, "apply_receipt_raw", chain.apply_receipt_raw)
    object.__setattr__(forged, "_marker", foundation_apply._CHAIN_MARKER)

    with pytest.raises(
        author.DirectIamIdentityAuthorError,
        match="direct_iam_identity_author_foundation_chain_invalid",
    ):
        author.decode_canonical_authority_for_validated_chain(
            _authority_for_chain(chain),
            foundation_chain=forged,
            now_unix=fixture.NOW + 21,
        )


@pytest.mark.parametrize(
    "substitution",
    ["release", "owner_reauth", "pre_foundation", "apply", "vm", "service_account"],
)
def test_canonical_authority_rejects_a_b_hash_or_resource_id_substitution(
    monkeypatch: pytest.MonkeyPatch,
    substitution: str,
) -> None:
    from tests.scripts.canary import test_owner_gate_pre_foundation as fixture

    chain, _raws, _release_key, _network_key = _validated_chain(monkeypatch)
    facts = _facts_for_chain(chain)
    release_revision = chain.foundation_source_revision
    owner_sha = chain.owner_reauthentication_receipt_sha256
    pre_sha = chain.pre_foundation_authority_sha256
    apply_sha = chain.foundation_apply_receipt_sha256
    if substitution == "release":
        release_revision = "b" * 40
    elif substitution == "owner_reauth":
        owner_sha = "8" * 64
    elif substitution == "pre_foundation":
        pre_sha = "6" * 64
    elif substitution == "apply":
        apply_sha = "7" * 64
    elif substitution == "vm":
        facts["instance"]["id"] = "7777777777777777777"
    else:
        facts["owner_gate_service_account"]["uniqueId"] = (
            "777777777777777777777"
        )
    raw = author._build_authority_bytes(
        live_facts=facts,
        release_revision=release_revision,
        owner_reauthentication_receipt_sha256=owner_sha,
        pre_foundation_authority_sha256=pre_sha,
        foundation_apply_receipt_sha256=apply_sha,
        collected_at_unix=fixture.NOW + 21,
    )

    with pytest.raises(author.DirectIamIdentityAuthorError):
        author.decode_canonical_authority_for_validated_chain(
            raw,
            foundation_chain=chain,
            now_unix=fixture.NOW + 21,
        )


def test_public_live_author_rejects_conflicting_journal_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.canary import owner_gate_foundation_apply as foundation_apply

    chain, raws, release_key, network_key = _validated_chain(monkeypatch)
    del chain
    monkeypatch.setattr(
        foundation_apply,
        "load_validated_foundation_apply_chain",
        lambda _foundation_a: (_ for _ in ()).throw(
            foundation_apply.OwnerGateFoundationApplyError(
                "owner_gate_foundation_success_journal_invalid"
            )
        ),
    )
    monkeypatch.setattr(
        owner_launcher.TrustedGcloudExecutable,
        "__init__",
        lambda *_args, **_kwargs: pytest.fail("runtime must not be created"),
    )

    with pytest.raises(
        author.DirectIamIdentityAuthorError,
        match="direct_iam_identity_author_foundation_chain_invalid",
    ):
        author.collect_and_publish_canonical_authority(
            pre_foundation_authority_raw=raws["pre"],
            owner_reauthentication_receipt_raw=raws["reauth"],
            network_evidence_raw=raws["network"],
            project_ancestry_evidence_raw=raws["ancestry"],
            release_public_key=release_key.public_key(),
            network_collector_public_key=network_key.public_key(),
            project_ancestry_collector_public_key=network_key.public_key(),
        )


def test_live_capabilities_are_constructed_only_inside_publication_collector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.scripts.canary import test_owner_gate_pre_foundation as fixture

    _chain_value, raws, release_key, network_key = _validated_chain(monkeypatch)
    monkeypatch.setattr(author.time, "time", lambda: float(fixture.NOW + 21))
    monkeypatch.setattr(
        owner_launcher,
        "_canonical_owner_home",
        lambda: str(tmp_path),
    )
    inside_publication = False

    class CapabilityReached(BaseException):
        pass

    class BoundaryComplete(BaseException):
        pass

    def runtime_init(*_args: Any, **_kwargs: Any) -> None:
        assert inside_publication is True
        raise CapabilityReached

    def run_publication(**kwargs: Any) -> Any:
        nonlocal inside_publication
        assert inside_publication is False
        assert kwargs["_recovery_only"] is False
        assert "output_path" not in kwargs
        assert "journal_path" not in kwargs
        inside_publication = True
        with pytest.raises(CapabilityReached):
            kwargs["collector"]()
        raise BoundaryComplete

    monkeypatch.setattr(
        owner_launcher.TrustedGcloudExecutable,
        "__init__",
        runtime_init,
    )
    monkeypatch.setattr(
        source_publication,
        "_run_direct_iam",
        run_publication,
    )

    with pytest.raises(BoundaryComplete):
        author.collect_and_publish_canonical_authority(
            pre_foundation_authority_raw=raws["pre"],
            owner_reauthentication_receipt_raw=raws["reauth"],
            network_evidence_raw=raws["network"],
            project_ancestry_evidence_raw=raws["ancestry"],
            release_public_key=release_key.public_key(),
            network_collector_public_key=network_key.public_key(),
            project_ancestry_collector_public_key=network_key.public_key(),
        )


@pytest.mark.parametrize(
    ("source_state", "resumes"),
    [
        ("candidate", True),
        ("final", True),
        ("intent", False),
        ("empty", False),
    ],
)
def test_expired_foundation_chain_only_resumes_complete_source_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_state: str,
    resumes: bool,
) -> None:
    chain, raws, release_key, network_key = _validated_chain(monkeypatch)
    projection = author._projection_from_validated_apply_chain(
        chain,
        now_unix=int(chain.apply_receipt["completed_at_unix"]),
        require_fresh_owner=False,
    )
    publication_chain = author._direct_publication_chain(projection)
    raw = _authority_for_chain(chain)
    owner_home = tmp_path / "owner-home"
    trusted = owner_home / ".hermes" / "trusted"
    trusted.mkdir(parents=True, mode=0o700)
    os.chmod(trusted, 0o700)
    os.chown(trusted, -1, os.getegid())

    def validate(value: bytes) -> source_publication._ValidatedArtifact:
        canonical = author._decode_canonical_authority_for_recovery_chain(
            value,
            foundation_chain=chain,
        )
        return source_publication._ValidatedArtifact(
            value=canonical,
            logical_sha256=str(canonical.value["authority_sha256"]),
        )

    class StopSeed(BaseException):
        pass

    checkpoint = {
        "candidate": "after_candidate",
        "final": "after_final_link",
        "intent": "after_intent",
    }.get(source_state)

    def stop(name: str) -> None:
        if name == checkpoint:
            raise StopSeed

    if checkpoint is not None:
        with pytest.raises(StopSeed):
            source_publication._run_direct_iam(
                owner_home=owner_home,
                chain=publication_chain,
                maximum=authority_schema.MAX_BYTES,
                validator=validate,
                collector=lambda: raw,
                _checkpoint=stop,
            )

    def expired(**_kwargs: Any) -> Any:
        raise author.DirectIamIdentityAuthorError(
            "direct_iam_identity_author_owner_reauth_expired"
        )

    monkeypatch.setattr(
        author,
        "_validated_foundation_apply_chain_from_raw",
        expired,
    )
    monkeypatch.setattr(
        author,
        "_recovery_foundation_apply_chain_from_raw",
        lambda **_kwargs: (chain, projection),
    )
    monkeypatch.setattr(
        owner_launcher,
        "_canonical_owner_home",
        lambda: str(owner_home),
    )
    monkeypatch.setattr(
        owner_launcher.TrustedGcloudExecutable,
        "__init__",
        lambda *_args, **_kwargs: pytest.fail(
            "expired-chain recovery must not create live capabilities"
        ),
    )
    def invoke() -> Mapping[str, Any]:
        return author.collect_and_publish_canonical_authority(
            pre_foundation_authority_raw=raws["pre"],
            owner_reauthentication_receipt_raw=raws["reauth"],
            network_evidence_raw=raws["network"],
            project_ancestry_evidence_raw=raws["ancestry"],
            release_public_key=release_key.public_key(),
            network_collector_public_key=network_key.public_key(),
            project_ancestry_collector_public_key=network_key.public_key(),
        )

    if resumes:
        result = invoke()
        authority = result["authority"]
        assert authority.raw == raw
        assert authority.value["collected_at_unix"] == json.loads(raw)[
            "collected_at_unix"
        ]
        assert Path(result["publication"]["path"]).read_bytes() == raw
    else:
        with pytest.raises(
            author.DirectIamIdentityAuthorError,
            match="direct_iam_identity_author_publication_failed",
        ):
            invoke()
        assert not (owner_home / author.DIRECT_IAM_AUTHORITY_RELATIVE).exists()


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda facts: facts["policies"][0]["bindings"].append({
                "role": author.MUTATION_ROLE,
                "members": [OWNER_MEMBER],
                "condition": dict(author.MUTATION_CONDITION),
            }),
            "direct_iam_identity_author_binding_drift",
        ),
        (
            lambda facts: facts["owner_gate_service_account_policy"].update({
                "bindings": [{
                    "role": "roles/iam.serviceAccountTokenCreator",
                    "members": ["user:attacker@example.test"],
                }]
            }),
            "direct_iam_identity_author_impersonation_invalid",
        ),
        (
            lambda facts: facts["owner_gate_user_managed_keys"].update({
                "keys": [{"name": "projects/p/serviceAccounts/s/keys/k"}]
            }),
            "direct_iam_identity_author_user_keys_invalid",
        ),
        (
            lambda facts: facts["roles"][author.MUTATION_ROLE].update({
                "includedPermissions": ["compute.instances.delete"]
            }),
            "direct_iam_identity_author_role_drift",
        ),
    ],
)
def test_author_rejects_privilege_and_identity_drift(
    mutate: Any,
    code: str,
) -> None:
    facts = _live_facts()
    mutate(facts)
    with pytest.raises(author.DirectIamIdentityAuthorError, match=code):
        _build(facts)


class _Response:
    def __init__(
        self,
        value: Mapping[str, Any],
        *,
        status: int = 200,
        content_type: str = "application/json; charset=utf-8",
        location: str | None = None,
    ) -> None:
        self._raw = foundation.canonical_json_bytes(value)
        self.status = status
        self._headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(self._raw)),
            "Location": location,
        }

    def getheader(self, name: str) -> str | None:
        return self._headers.get(name)

    def read(self, maximum: int) -> bytes:
        return self._raw[:maximum]


def test_http_body_bound_accepts_large_predefined_role_and_rejects_oversize(
) -> None:
    current_large_role = _Response({
        "name": "roles/owner",
        "description": "x" * (600 * 1024),
    })
    current_large_role._headers.pop("Content-Length")
    assert 512 * 1024 < len(current_large_role._raw)
    assert len(current_large_role._raw) <= author.MAX_HTTP_BODY_BYTES
    assert author._bounded_http_body(current_large_role) == current_large_role._raw

    oversized = _Response({
        "name": "roles/owner",
        "description": "x" * author.MAX_HTTP_BODY_BYTES,
    })
    assert len(oversized._raw) > author.MAX_HTTP_BODY_BYTES
    with pytest.raises(
        author.DirectIamIdentityAuthorError,
        match="^direct_iam_identity_author_cloud_http_invalid$",
    ):
        author._bounded_http_body(oversized)


class _Router:
    def __init__(self, routes: Mapping[tuple[str, str], Mapping[str, Any]]) -> None:
        self.routes = dict(routes)
        self.calls: list[tuple[str, str, bytes | None, Mapping[str, str]]] = []

    def factory(self) -> "_Connection":
        return _Connection(self)


class _Connection:
    def __init__(self, router: _Router) -> None:
        self.router = router
        self.key: tuple[str, str] | None = None

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        headers: Mapping[str, str],
    ) -> None:
        self.key = (method, path)
        self.router.calls.append((method, path, body, dict(headers)))

    def getresponse(self) -> _Response:
        assert self.key is not None
        return _Response(self.router.routes[self.key])

    def close(self) -> None:
        return None


def test_reader_uses_only_exact_read_rest_inventory_and_sa_local_post(
) -> None:
    facts = _live_facts()
    role_routes = {
        ("GET", f"/v1/{name}"): value for name, value in facts["roles"].items()
    }
    compute = _Router({
        ("GET", author.OWNER_GATE_INSTANCE_PATH): facts["instance"],
    })
    resource = _Router({
        ("GET", f"/v3/{author.PROJECT_RESOURCE}"): facts["resources"][0],
        ("GET", f"/v3/{ORGANIZATION}"): facts["resources"][1],
        (
            "POST",
            f"/v3/{author.PROJECT_RESOURCE}:getIamPolicy",
        ): facts["policies"][0],
        (
            "POST",
            f"/v3/{ORGANIZATION}:getIamPolicy",
        ): facts["policies"][1],
    })
    iam = _Router({
        **role_routes,
        ("GET", author.OWNER_GATE_SERVICE_ACCOUNT_PATH): facts[
            "owner_gate_service_account"
        ],
        ("GET", author.TARGET_SERVICE_ACCOUNT_PATH): facts[
            "target_service_account"
        ],
        ("POST", author.OWNER_GATE_SERVICE_ACCOUNT_POLICY_PATH): {},
        ("GET", author.OWNER_GATE_USER_KEYS_PATH): {},
    })
    provider = _TrustedOwnerTokenProvider()
    token = author._acquire_access_token_with_provider(provider)
    reader = author.DirectIamOwnerFactsReader(
        token=token,
        compute_connection_factory=compute.factory,
        resource_manager_connection_factory=resource.factory,
        iam_connection_factory=iam.factory,
    )

    collected = reader.collect()

    assert collected == facts
    all_calls = [*compute.calls, *resource.calls, *iam.calls]
    assert {method for method, _, _, _ in all_calls} <= {"GET", "POST"}
    assert all(
        headers["Authorization"] == "Bearer owner-token-that-is-not-printed"
        for _, _, _, headers in all_calls
    )
    policy_calls = [
        call for call in resource.calls if call[0] == "POST"
    ]
    assert all(
        body
        == foundation.canonical_json_bytes({
            "options": {"requestedPolicyVersion": 3}
        })
        for _, _, body, _ in policy_calls
    )
    sa_policy = next(
        call
        for call in iam.calls
        if call[1] == author.OWNER_GATE_SERVICE_ACCOUNT_POLICY_PATH
    )
    assert sa_policy[0] == "POST"
    assert sa_policy[2] is None
    assert iam.calls[-1][1] == author.OWNER_GATE_USER_KEYS_PATH
    call_keys = [(method, path) for method, path, _, _ in all_calls]
    assert all(call_keys.count(key) == 2 for key in set(call_keys))
    assert provider.calls == 1
    assert provider.stability_checks == 1


def test_reader_rejects_mixed_generation_two_pass_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = author._acquire_access_token_with_provider(
        _TrustedOwnerTokenProvider()
    )
    reader = author.DirectIamOwnerFactsReader(token=token)
    first = _live_facts()
    second = _live_facts()
    second["policies"][0]["etag"] = "new-generation-etag"
    snapshots = iter((first, second))
    monkeypatch.setattr(reader, "_collect_once", lambda: next(snapshots))

    with pytest.raises(
        author.DirectIamIdentityAuthorError,
        match="^direct_iam_identity_author_facts_unstable$",
    ):
        reader.collect()
    author.wipe_access_token(token)


def _exact_live_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    chain: Any,
    *,
    drift: str | None = None,
) -> tuple[Any, Any, Any, Mapping[str, Any]]:
    runtime = object.__new__(owner_launcher.TrustedGcloudExecutable)
    configuration = object.__new__(owner_launcher.PinnedGcloudConfiguration)
    owner = object.__new__(owner_launcher.GcloudOwnerAccessToken)
    owner._gcloud_configuration = configuration
    owner._gcloud_executable = runtime
    checks: dict[str, Any] = {
        "runtime": 0,
        "configuration": 0,
        "account": 0,
        "owner": 0,
        "wipe": 0,
        "wiped": False,
    }

    def sealed(_self: Any, *, expected_release_sha: str) -> Mapping[str, Any]:
        assert expected_release_sha == chain.foundation_source_revision
        checks["runtime"] += 1
        digest = (
            "2" * 64
            if drift == "runtime" and checks["runtime"] >= 2
            else "1" * 64
        )
        return {"identity_sha256": digest}

    def stable_configuration(_self: Any) -> None:
        checks["configuration"] += 1
        if drift == "configuration" and checks["configuration"] >= 2:
            raise owner_launcher.OwnerLauncherError("test_configuration_changed")

    def account(_self: Any) -> str:
        checks["account"] += 1
        return (
            "attacker@example.test"
            if drift == "account" and checks["account"] >= 2
            else author.OWNER_ACCOUNT
        )

    def bind(subject: Any, expected_sha256: str) -> None:
        assert expected_sha256 == hashlib.sha256(
            author.OWNER_ACCOUNT.encode("ascii")
        ).hexdigest()
        subject._approved = True
        subject._pinned_account = author.OWNER_ACCOUNT
        subject.owner_subject_sha256 = expected_sha256

    def stable_owner(_self: Any) -> None:
        checks["owner"] += 1
        if drift == "owner" and checks["owner"] >= 3:
            raise owner_launcher.OwnerLauncherError("test_owner_changed")

    original_wipe = author.wipe_access_token

    def tracked_wipe(token: Any) -> None:
        checks["wipe"] += 1
        original_wipe(token)
        checks["wiped"] = not any(token._raw)

    monkeypatch.setattr(
        owner_launcher.TrustedGcloudExecutable,
        "sealed_runtime_identity",
        sealed,
    )
    monkeypatch.setattr(
        owner_launcher.PinnedGcloudConfiguration,
        "assert_stable",
        stable_configuration,
    )
    monkeypatch.setattr(
        owner_launcher.PinnedGcloudConfiguration,
        "account",
        property(account),
    )
    monkeypatch.setattr(
        owner_launcher.GcloudOwnerAccessToken,
        "bind_approved_subject",
        bind,
    )
    monkeypatch.setattr(
        owner_launcher.GcloudOwnerAccessToken,
        "approved_account",
        property(lambda _self: author.OWNER_ACCOUNT),
    )
    monkeypatch.setattr(
        owner_launcher.GcloudOwnerAccessToken,
        "require_stable",
        stable_owner,
    )
    monkeypatch.setattr(
        owner_launcher.GcloudOwnerAccessToken,
        "__call__",
        lambda _self: "owner-token-that-is-not-printed",
    )
    monkeypatch.setattr(author, "wipe_access_token", tracked_wipe)
    monkeypatch.setattr(
        author.DirectIamOwnerFactsReader,
        "collect",
        lambda _self: _facts_for_chain(chain),
    )
    return runtime, configuration, owner, checks


def test_live_capabilities_recheck_runtime_config_owner_and_wipe_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.scripts.canary import test_owner_gate_pre_foundation as fixture

    chain, _raws, _release_key, _network_key = _validated_chain(monkeypatch)
    runtime, configuration, owner, checks = _exact_live_capabilities(
        monkeypatch,
        chain,
    )

    canonical = author._collect_canonical_authority_with_capabilities(
        foundation_chain=chain,
        runtime=runtime,
        configuration=configuration,
        owner_identity=owner,
        facts_reader_factory=author.DirectIamOwnerFactsReader,
        clock=lambda: float(fixture.NOW + 21),
    )

    assert canonical.value["owner_gate_vm_numeric_id"] == (
        chain.owner_gate_vm_identity["numeric_id"]
    )
    assert checks["runtime"] == 2
    assert checks["configuration"] >= 2
    assert checks["account"] == 2
    assert checks["owner"] == 3
    assert checks["wipe"] == 1
    assert checks["wiped"] is True


@pytest.mark.parametrize("drift", ["runtime", "configuration", "account", "owner"])
def test_live_capabilities_fail_closed_on_final_identity_drift_after_reads(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    from tests.scripts.canary import test_owner_gate_pre_foundation as fixture

    chain, _raws, _release_key, _network_key = _validated_chain(monkeypatch)
    runtime, configuration, owner, checks = _exact_live_capabilities(
        monkeypatch,
        chain,
        drift=drift,
    )

    with pytest.raises(
        author.DirectIamIdentityAuthorError,
        match="direct_iam_identity_author_capability_changed",
    ):
        author._collect_canonical_authority_with_capabilities(
            foundation_chain=chain,
            runtime=runtime,
            configuration=configuration,
            owner_identity=owner,
            facts_reader_factory=author.DirectIamOwnerFactsReader,
            clock=lambda: float(fixture.NOW + 21),
        )
    assert checks["wipe"] == 1
    assert checks["wiped"] is True


def test_gcloud_token_reuses_owner_launcher_trust_and_failures_are_secret_free() -> None:
    provider = _TrustedOwnerTokenProvider()
    token = author._acquire_access_token_with_provider(provider)
    assert provider.calls == 1
    assert provider.stability_checks == 1
    assert token._raw == bytearray(b"owner-token-that-is-not-printed")
    author.wipe_access_token(token)
    assert token._raw == bytearray(len(token._raw))

    secret = "credential-must-never-appear"

    with pytest.raises(
        author.DirectIamIdentityAuthorError,
        match="^direct_iam_identity_author_gcloud_unavailable$",
    ) as failure:
        author._acquire_access_token_with_provider(
            _TrustedOwnerTokenProvider(failure=RuntimeError(secret)),
        )
    assert secret not in str(failure.value)

    with pytest.raises(
        author.DirectIamIdentityAuthorError,
        match="^direct_iam_identity_author_gcloud_provider_untrusted$",
    ):
        author.acquire_access_token(token_provider=lambda: secret)

    with pytest.raises(
        author.DirectIamIdentityAuthorError,
        match="^direct_iam_identity_author_gcloud_provider_untrusted$",
    ):
        author.acquire_access_token(
            token_provider=_TrustedOwnerTokenProvider()
        )

    with pytest.raises(
        author.DirectIamIdentityAuthorError,
        match="^direct_iam_identity_author_owner_identity_invalid$",
    ):
        author._acquire_access_token_with_provider(
            _TrustedOwnerTokenProvider(account="other-owner@example.test")
        )


@pytest.mark.parametrize(
    "name",
    [
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "SSLKEYLOGFILE",
        "OPENSSL_CONF",
        "OPENSSL_MODULES",
    ],
)
def test_direct_https_rejects_custom_tls_environment(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    for variable in author._FORBIDDEN_NETWORK_ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv(name, "/tmp/untrusted-custom-tls")
    with pytest.raises(
        author.DirectIamIdentityAuthorError,
        match="^direct_iam_identity_author_cloud_tls_invalid$",
    ):
        author._default_connection(author.COMPUTE_HOST)


def test_direct_https_rejects_proxy_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in author._FORBIDDEN_NETWORK_ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    with pytest.raises(
        author.DirectIamIdentityAuthorError,
        match="^direct_iam_identity_author_cloud_tls_invalid$",
    ):
        author._default_connection(author.COMPUTE_HOST)


def test_direct_https_uses_only_owner_launcher_pinned_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in author._FORBIDDEN_NETWORK_ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)
    context = object()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        owner_launcher,
        "_pinned_system_tls_context",
        lambda: context,
    )

    def connection(host: str, port: int, **kwargs: Any) -> object:
        captured.update({"host": host, "port": port, **kwargs})
        return object()

    monkeypatch.setattr(author.http.client, "HTTPSConnection", connection)

    assert author._default_connection(author.COMPUTE_HOST) is not None
    assert captured == {
        "host": author.COMPUTE_HOST,
        "port": 443,
        "timeout": author.HTTP_TIMEOUT_SECONDS,
        "context": context,
    }

    with pytest.raises(
        author.DirectIamIdentityAuthorError,
        match="direct_iam_identity_author_cloud_resource_forbidden",
    ):
        author._default_connection("attacker.invalid")


def _owner_cli_inputs(
    tmp_path: Path,
    *,
    release_key: Ed25519PrivateKey,
    collector_key: Ed25519PrivateKey,
) -> tuple[list[str], Mapping[str, bytes]]:
    artifacts = {
        "pre-foundation-authority": b'{"artifact":"pre"}',
        "owner-reauth-receipt": b'{"artifact":"reauth"}',
        "network-evidence": b'{"artifact":"network"}',
        "project-ancestry-evidence": b'{"artifact":"ancestry"}',
    }
    argv: list[str] = []
    for name, raw in artifacts.items():
        path = tmp_path / f"{name}.json"
        path.write_bytes(raw)
        os.chmod(path, 0o444)
        argv.extend((f"--{name}", str(path)))
    release_path = tmp_path / "release.pub"
    release_path.write_bytes(release_key.public_key().public_bytes_raw())
    os.chmod(release_path, 0o444)
    collector_path = tmp_path / "collector.pub"
    collector_path.write_bytes(collector_key.public_key().public_bytes_raw())
    os.chmod(collector_path, 0o444)
    argv.extend(("--release-trust-public-key", str(release_path)))
    argv.extend(("--network-collector-public-key", str(collector_path)))
    argv.extend(
        ("--project-ancestry-collector-public-key", str(collector_path))
    )
    return argv, artifacts


def test_owner_cli_uses_fixed_output_and_prints_only_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.canary import owner_gate_trust as trust

    release_key = Ed25519PrivateKey.generate()
    collector_key = Ed25519PrivateKey.generate()
    argv, artifacts = _owner_cli_inputs(
        tmp_path,
        release_key=release_key,
        collector_key=collector_key,
    )
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        hashlib.sha256(release_key.public_key().public_bytes_raw()).hexdigest(),
    )
    owner_home = tmp_path / "owner-home"
    owner_home.mkdir(mode=0o700)
    monkeypatch.setattr(
        owner_launcher,
        "_canonical_owner_home",
        lambda: str(owner_home),
    )
    expected_output = str(owner_home / author.DIRECT_IAM_AUTHORITY_RELATIVE)
    called = False

    def collect(**kwargs: Any) -> Mapping[str, Any]:
        nonlocal called
        called = True
        assert kwargs["pre_foundation_authority_raw"] == artifacts[
            "pre-foundation-authority"
        ]
        assert kwargs["owner_reauthentication_receipt_raw"] == artifacts[
            "owner-reauth-receipt"
        ]
        assert kwargs["network_evidence_raw"] == artifacts["network-evidence"]
        assert kwargs["project_ancestry_evidence_raw"] == artifacts[
            "project-ancestry-evidence"
        ]
        assert "foundation_apply_receipt_raw" not in kwargs
        return {
            "authority": {"must_not_be_printed": "secret-marker"},
            "publication": {
                "path": expected_output,
                "authority_sha256": "1" * 64,
                "authority_file_sha256": "2" * 64,
            },
        }

    monkeypatch.setattr(
        author,
        "collect_and_publish_canonical_authority",
        collect,
    )
    assert author.main(argv) == 0
    assert called is True
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "schema": "muncho-owner-gate-direct-iam-authority-publication.v1",
        "authority_published": True,
        "path": expected_output,
        "authority_sha256": "1" * 64,
        "authority_file_sha256": "2" * 64,
    }
    assert "secret-marker" not in json.dumps(report)


@pytest.mark.parametrize("path_attack", ["writable", "symlink"])
def test_owner_cli_rejects_mutable_or_symlink_input_before_live_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_attack: str,
) -> None:
    from scripts.canary import owner_gate_trust as trust

    release_key = Ed25519PrivateKey.generate()
    collector_key = Ed25519PrivateKey.generate()
    argv, _artifacts = _owner_cli_inputs(
        tmp_path,
        release_key=release_key,
        collector_key=collector_key,
    )
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        hashlib.sha256(release_key.public_key().public_bytes_raw()).hexdigest(),
    )
    index = argv.index("--pre-foundation-authority") + 1
    authority_path = Path(argv[index])
    if path_attack == "writable":
        os.chmod(authority_path, 0o600)
    else:
        real = tmp_path / "pre-authority-real.json"
        real.write_bytes(authority_path.read_bytes())
        os.chmod(real, 0o444)
        authority_path.unlink()
        authority_path.symlink_to(real)
    monkeypatch.setattr(
        author,
        "collect_and_publish_canonical_authority",
        lambda **_kwargs: pytest.fail("live collection must not start"),
    )

    with pytest.raises(
        author.DirectIamIdentityAuthorError,
        match="direct_iam_identity_author_owner_input_invalid",
    ):
        author.main(argv)


def test_owner_cli_has_no_output_path_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.canary import owner_gate_trust as trust

    release_key = Ed25519PrivateKey.generate()
    collector_key = Ed25519PrivateKey.generate()
    argv, _artifacts = _owner_cli_inputs(
        tmp_path,
        release_key=release_key,
        collector_key=collector_key,
    )
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        hashlib.sha256(release_key.public_key().public_bytes_raw()).hexdigest(),
    )
    with pytest.raises(SystemExit):
        author.main([*argv, "--output", str(tmp_path / "attacker.json")])


def test_public_live_surface_has_no_raw_apply_receipt_input() -> None:
    parameters = inspect.signature(
        author.collect_and_publish_canonical_authority
    ).parameters

    assert "foundation_apply_receipt_raw" not in parameters
    assert "foundation_apply_receipt_path" not in parameters
    assert "output_path" not in parameters
    assert "journal_path" not in parameters


def test_live_builder_and_caller_authored_chain_values_are_not_exported() -> None:
    assert "build_authority_bytes" not in author.__all__
    assert "collect_and_build_authority" not in author.__all__
    assert not hasattr(author, "build_authority_bytes")
    assert not hasattr(author, "collect_and_build_authority")


def test_publication_helper_is_not_exported_as_caller_surface() -> None:
    assert "_publish_canonical_authority_exclusive" not in author.__all__
    assert not hasattr(author, "_publish_canonical_authority_exclusive")
