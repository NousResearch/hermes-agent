from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_owner_reauth as reauth
from scripts.canary import owner_gate_project_ancestry as ancestry
from scripts.canary import owner_gate_trust as trust
from scripts.canary import owner_gate_trust_author as release_author
from scripts.canary import trusted_signer_author as signer_author


REVISION = "a" * 40
NOW = 2_000_000_000
PROJECT_NUMBER = "123456789012"
FOLDER_ID = "234567890123"
ORGANIZATION_ID = "345678901234"
RELEASE_KEY = Ed25519PrivateKey.generate()
RELEASE_KEY_ID = hashlib.sha256(
    RELEASE_KEY.public_key().public_bytes_raw()
).hexdigest()


@pytest.fixture(autouse=True)
def _isolated_authority_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "authority-parent"
    parent.mkdir(mode=0o700)
    key_directory = parent / "owner-gate-release-authority"
    monkeypatch.setattr(release_author, "AUTHORITY_PARENT", parent)
    monkeypatch.setattr(release_author, "KEY_DIRECTORY", key_directory)
    monkeypatch.setattr(
        release_author,
        "MANIFEST_DIRECTORY",
        key_directory / "manifests",
    )
    monkeypatch.setattr(
        signer_author,
        "OBSERVATION_ROOT",
        key_directory / "observation-signers",
    )
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        RELEASE_KEY_ID,
    )
    release_author.initialize_keypair()
    signer_author.initialize_observation_keys(
        REVISION,
        mode="network-bootstrap",
    )


def _owner_reauth_receipt(*, expires_at_unix: int = NOW + 600) -> dict:
    body = {
        "schema": reauth.RECEIPT_SCHEMA,
        "purpose": reauth.RECEIPT_PURPOSE,
        "trusted_runtime_identity": {
            "release_revision": REVISION,
            "sealed_runtime_identity_sha256": "0" * 64,
            "command_prefix_sha256": "1" * 64,
            "python_executable_sha256": "2" * 64,
            "gcloud_module_sha256": "3" * 64,
            "sdk_root": (
                "/sealed/google-cloud-sdk-"
                f"{reauth.launcher._GCLOUD_SDK_VERSION}"
            ),
            "sdk_python_config_identity_sha256": "4" * 64,
            "closed_environment_sha256": "5" * 64,
            "configuration": reauth.GCLOUD_CONFIGURATION,
            "account": reauth.OWNER_ACCOUNT,
            "project": foundation.PROJECT,
            "zone": foundation.ZONE,
        },
        "interactive_reauthentication": {
            "method": "gcloud_auth_login_force_interactive",
            "started_at_unix": NOW - 2,
            "completed_at_unix": NOW - 1,
            "command_sha256": "6" * 64,
            "interactive_tty_verified": True,
            "access_token_requested": False,
            "credential_material_captured": False,
        },
        "authenticated_probe": {
            "command_sha256": "7" * 64,
            "output_sha256": "8" * 64,
            "project_id": foundation.PROJECT,
            "project_number": PROJECT_NUMBER,
        },
        "issued_at_unix": NOW,
        "expires_at_unix": expires_at_unix,
        "signer_key_id": RELEASE_KEY_ID,
    }
    return dict(
        reauth._sign_owner_reauth_receipt(body, private_key=RELEASE_KEY)
    )


def _resources() -> dict[str, Mapping[str, Any]]:
    return {
        f"projects/{PROJECT_NUMBER}": {
            "name": f"projects/{PROJECT_NUMBER}",
            "projectId": foundation.PROJECT,
            "displayName": "Adventico AI Platform",
            "state": "ACTIVE",
            "etag": "etag-project-1",
            "parent": f"folders/{FOLDER_ID}",
        },
        f"folders/{FOLDER_ID}": {
            "name": f"folders/{FOLDER_ID}",
            "displayName": "Adventico Production",
            "state": "ACTIVE",
            "etag": "etag-folder-1",
            "parent": f"organizations/{ORGANIZATION_ID}",
        },
        f"organizations/{ORGANIZATION_ID}": {
            "name": f"organizations/{ORGANIZATION_ID}",
            "displayName": "Adventico",
            "state": "ACTIVE",
            "etag": "etag-organization-1",
        },
    }


class _Reader:
    def __init__(self, resources: Mapping[str, Mapping[str, Any]]) -> None:
        self.resources = resources
        self.calls: list[str] = []

    def __call__(self, resource_name: str) -> Mapping[str, Any]:
        self.calls.append(resource_name)
        return copy.deepcopy(self.resources[resource_name])


def _signed_evidence() -> tuple[dict, _Reader]:
    reader = _Reader(_resources())
    result = ancestry._collect_and_author_with_reader(
        release_revision=REVISION,
        read_resource=reader,
        collected_at_unix=NOW + 1,
        owner_reauthentication_receipt=_owner_reauth_receipt(),
        owner_reauthentication_public_key=RELEASE_KEY.public_key(),
    )
    return dict(result), reader


def _collector_public_key():
    return Ed25519PrivateKey.from_private_bytes(
        signer_author._private_path(REVISION, "network").read_bytes()
    ).public_key()


def test_evidence_binds_two_stable_complete_reads_and_owner_reauth() -> None:
    signed, reader = _signed_evidence()
    raw = foundation.canonical_json_bytes(signed)
    checked = ancestry.decode_canonical_project_ancestry_evidence(
        raw,
        collector_public_key=_collector_public_key(),
        owner_reauthentication_receipt=_owner_reauth_receipt(),
        owner_reauthentication_public_key=RELEASE_KEY.public_key(),
        expected_release_revision=REVISION,
        now_unix=NOW + 2,
    )

    assert checked.project_number == PROJECT_NUMBER
    assert checked.organization_id == ORGANIZATION_ID
    assert checked.signed_evidence_sha256 == hashlib.sha256(raw).hexdigest()
    assert [item["resource_type"] for item in checked.ordered_chain] == [
        "project",
        "folder",
        "organization",
    ]
    assert signed["stable_reads"][0]["chain_sha256"] == signed[
        "stable_reads"
    ][1]["chain_sha256"]
    assert signed["stable_reads"][0][
        "provider_consistency_token_sha256"
    ] == signed["stable_reads"][1]["provider_consistency_token_sha256"]
    assert reader.calls == [
        f"projects/{PROJECT_NUMBER}",
        f"folders/{FOLDER_ID}",
        f"organizations/{ORGANIZATION_ID}",
    ] * 2


def test_evidence_rejects_second_read_etag_drift() -> None:
    resources = _resources()
    calls = 0

    def reader(resource_name: str) -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        result = copy.deepcopy(resources[resource_name])
        if calls > 3 and resource_name == f"folders/{FOLDER_ID}":
            result["etag"] = "etag-folder-2"
        return result

    with pytest.raises(
        ancestry.OwnerGateProjectAncestryError,
        match="owner_gate_project_ancestry_unstable",
    ):
        ancestry._collect_and_author_with_reader(
            release_revision=REVISION,
            read_resource=reader,
            collected_at_unix=NOW + 1,
            owner_reauthentication_receipt=_owner_reauth_receipt(),
            owner_reauthentication_public_key=RELEASE_KEY.public_key(),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda resources: resources[f"projects/{PROJECT_NUMBER}"].pop("etag"),
        lambda resources: resources[f"folders/{FOLDER_ID}"].update(
            {"state": "DELETE_REQUESTED"}
        ),
        lambda resources: resources[f"folders/{FOLDER_ID}"].update(
            {"parent": f"folders/{FOLDER_ID}"}
        ),
        lambda resources: resources[f"organizations/{ORGANIZATION_ID}"].update(
            {"parent": f"folders/{FOLDER_ID}"}
        ),
    ],
)
def test_evidence_rejects_incomplete_or_cyclic_chain(mutation) -> None:
    resources = _resources()
    mutation(resources)
    with pytest.raises(ancestry.OwnerGateProjectAncestryError):
        ancestry._collect_and_author_with_reader(
            release_revision=REVISION,
            read_resource=_Reader(resources),
            collected_at_unix=NOW + 1,
            owner_reauthentication_receipt=_owner_reauth_receipt(),
            owner_reauthentication_public_key=RELEASE_KEY.public_key(),
        )


def test_evidence_rejects_tamper_noncanonical_and_wrong_reauth() -> None:
    signed, _reader = _signed_evidence()
    tampered = copy.deepcopy(signed)
    tampered["organization_id"] = "456789012345"
    with pytest.raises(ancestry.OwnerGateProjectAncestryError):
        ancestry.validate_project_ancestry_evidence(
            tampered,
            collector_public_key=_collector_public_key(),
            owner_reauthentication_receipt=_owner_reauth_receipt(),
            owner_reauthentication_public_key=RELEASE_KEY.public_key(),
            expected_release_revision=REVISION,
            now_unix=NOW + 2,
        )
    raw = foundation.canonical_json_bytes(signed)
    with pytest.raises(
        ancestry.OwnerGateProjectAncestryError,
        match="owner_gate_project_ancestry_canonical_invalid",
    ):
        ancestry.decode_canonical_project_ancestry_evidence(
            b" " + raw,
            collector_public_key=_collector_public_key(),
            owner_reauthentication_receipt=_owner_reauth_receipt(),
            owner_reauthentication_public_key=RELEASE_KEY.public_key(),
            expected_release_revision=REVISION,
            now_unix=NOW + 2,
        )
    with pytest.raises(
        ancestry.OwnerGateProjectAncestryError,
        match="owner_gate_project_ancestry_evidence_invalid",
    ):
        ancestry.validate_project_ancestry_evidence(
            signed,
            collector_public_key=_collector_public_key(),
            owner_reauthentication_receipt=_owner_reauth_receipt(
                expires_at_unix=NOW + 500
            ),
            owner_reauthentication_public_key=RELEASE_KEY.public_key(),
            expected_release_revision=REVISION,
            now_unix=NOW + 2,
        )


def test_public_collector_rejects_structural_runtime_fakes() -> None:
    with pytest.raises(
        ancestry.OwnerGateProjectAncestryError,
        match="owner_gate_project_ancestry_runtime_invalid",
    ):
        ancestry.collect_and_author_project_ancestry(
            release_revision=REVISION,
            collected_at_unix=NOW + 1,
            owner_reauthentication_receipt=_owner_reauth_receipt(),
            owner_reauthentication_public_key=RELEASE_KEY.public_key(),
            gcloud_executable=object(),  # type: ignore[arg-type]
            gcloud_configuration=object(),  # type: ignore[arg-type]
        )


def test_exact_live_chain_comparison_has_no_partial_match() -> None:
    signed, _reader = _signed_evidence()
    evidence = ancestry.validate_project_ancestry_evidence(
        signed,
        collector_public_key=_collector_public_key(),
        owner_reauthentication_receipt=_owner_reauth_receipt(),
        owner_reauthentication_public_key=RELEASE_KEY.public_key(),
        expected_release_revision=REVISION,
        now_unix=NOW + 2,
    )
    exact = [dict(item) for item in evidence.ordered_chain]
    assert ancestry.live_chain_equals_evidence(evidence, exact)
    exact[1]["etag"] = "drift"
    assert not ancestry.live_chain_equals_evidence(evidence, exact)


def test_live_collection_uses_one_token_for_both_complete_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RotatingTokenProvider:
        def __init__(self) -> None:
            self.calls = 0
            self.stability_checks = 0

        def __call__(self) -> str:
            self.calls += 1
            return f"bearer-{self.calls}"

        def require_stable(self) -> None:
            self.stability_checks += 1

    provider = RotatingTokenProvider()
    observed_tokens: list[str] = []
    resources = _resources()

    def resource_get(token: str, resource_name: str) -> Mapping[str, Any]:
        observed_tokens.append(token)
        return copy.deepcopy(resources[resource_name])

    monkeypatch.setattr(ancestry, "_resource_manager_get", resource_get)
    ancestry._collect_and_author_with_token_provider(
        release_revision=REVISION,
        collected_at_unix=NOW + 1,
        owner_reauthentication_receipt=_owner_reauth_receipt(),
        owner_reauthentication_public_key=RELEASE_KEY.public_key(),
        token_provider=provider,
    )

    assert provider.calls == 1
    assert observed_tokens == ["bearer-1"] * 6
    assert provider.stability_checks >= 3


class _RestResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        url: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status = status
        self._body = body
        self._url = url
        self.headers = dict(headers or {})

    def __enter__(self) -> "_RestResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def geturl(self) -> str:
        assert self._url is not None
        return self._url

    def read(self, maximum: int) -> bytes:
        return self._body[:maximum]


class _RestOpener:
    def __init__(self, response: _RestResponse) -> None:
        self.response = response
        self.requests: list[Any] = []

    def open(self, request: Any, *, timeout: float) -> _RestResponse:
        assert timeout == 30.0
        self.requests.append(request)
        return self.response


def _clear_network_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ancestry._FORBIDDEN_NETWORK_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def test_resource_manager_get_uses_fixed_tls_no_proxy_no_redirect_and_strict_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_network_environment(monkeypatch)
    resource_name = f"projects/{PROJECT_NUMBER}"
    url = ancestry._RESOURCE_MANAGER_BASE + resource_name
    body = b'{"etag":"stable","name":"projects/123456789012"}'
    response = _RestResponse(
        body,
        url=url,
        headers={
            "Content-Type": "application/json; charset=UTF-8",
            "Content-Length": str(len(body)),
        },
    )
    opener = _RestOpener(response)
    handlers: list[Any] = []
    pinned_context = object()

    monkeypatch.setattr(
        ancestry.launcher,
        "_pinned_system_tls_context",
        lambda: pinned_context,
    )

    def build_opener(*values: Any) -> _RestOpener:
        handlers.extend(values)
        return opener

    monkeypatch.setattr(ancestry.urllib.request, "build_opener", build_opener)

    value = ancestry._resource_manager_get("opaque-token", resource_name)

    assert value == {"etag": "stable", "name": resource_name}
    assert len(opener.requests) == 1
    assert opener.requests[0].full_url == url
    assert any(isinstance(item, ancestry._NoRedirectHandler) for item in handlers)
    assert any(
        isinstance(item, ancestry.urllib.request.ProxyHandler)
        for item in handlers
    )
    https = next(
        item
        for item in handlers
        if isinstance(item, ancestry.urllib.request.HTTPSHandler)
    )
    assert https._context is pinned_context


@pytest.mark.parametrize(
    ("body", "status", "final_url", "headers"),
    [
        (
            b'{"name":"projects/123456789012"}',
            302,
            "https://attacker.invalid/redirect",
            {"Content-Type": "application/json"},
        ),
        (
            b'{"name":"projects/123456789012"}',
            200,
            "https://attacker.invalid/substitution",
            {"Content-Type": "application/json"},
        ),
        (
            b'<html>not json</html>',
            200,
            None,
            {"Content-Type": "text/html"},
        ),
        (
            b'{"name":"projects/123456789012"}',
            200,
            None,
            {"Content-Type": "application/json", "Content-Length": "bogus"},
        ),
        (
            b'{"name":"projects/123456789012"}',
            200,
            None,
            {"Content-Type": "application/json", "Content-Length": "1"},
        ),
        (
            b'{"name":"one","name":"two"}',
            200,
            None,
            {"Content-Type": "application/json"},
        ),
        (
            b'{"value":NaN}',
            200,
            None,
            {"Content-Type": "application/json"},
        ),
        (
            b'[{"name":"projects/123456789012"}]',
            200,
            None,
            {"Content-Type": "application/json"},
        ),
        (
            b"{" + b"x" * ancestry.MAX_JSON_BYTES + b"}",
            200,
            None,
            {"Content-Type": "application/json"},
        ),
    ],
)
def test_resource_manager_get_rejects_redirect_shape_and_json_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
    status: int,
    final_url: str | None,
    headers: Mapping[str, str],
) -> None:
    _clear_network_environment(monkeypatch)
    resource_name = f"projects/{PROJECT_NUMBER}"
    url = ancestry._RESOURCE_MANAGER_BASE + resource_name
    response = _RestResponse(
        body,
        status=status,
        url=final_url or url,
        headers=headers,
    )
    monkeypatch.setattr(
        ancestry.launcher,
        "_pinned_system_tls_context",
        lambda: object(),
    )
    monkeypatch.setattr(
        ancestry.urllib.request,
        "build_opener",
        lambda *_handlers: _RestOpener(response),
    )

    with pytest.raises(
        ancestry.OwnerGateProjectAncestryError,
        match="owner_gate_project_ancestry_rest_invalid",
    ):
        ancestry._resource_manager_get("opaque-token", resource_name)


@pytest.mark.parametrize(
    "name",
    [
        "HTTPS_PROXY",
        "SSL_CERT_DIR",
        "SSLKEYLOGFILE",
        "OPENSSL_CONF",
        "OPENSSL_MODULES",
    ],
)
def test_resource_manager_get_rejects_ambient_network_influence(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    _clear_network_environment(monkeypatch)
    monkeypatch.setenv(name, "/tmp/attacker-controlled")
    monkeypatch.setattr(
        ancestry.urllib.request,
        "build_opener",
        lambda *_handlers: pytest.fail("network must not start"),
    )

    with pytest.raises(
        ancestry.OwnerGateProjectAncestryError,
        match="owner_gate_project_ancestry_rest_invalid",
    ):
        ancestry._resource_manager_get(
            "opaque-token",
            f"projects/{PROJECT_NUMBER}",
        )


@pytest.mark.parametrize("token", ["token\nleak", "token\tcontrol", "tökën"])
def test_resource_manager_get_rejects_non_header_safe_token(
    monkeypatch: pytest.MonkeyPatch,
    token: str,
) -> None:
    _clear_network_environment(monkeypatch)
    with pytest.raises(
        ancestry.OwnerGateProjectAncestryError,
        match="owner_gate_project_ancestry_rest_invalid",
    ):
        ancestry._resource_manager_get(token, f"projects/{PROJECT_NUMBER}")
