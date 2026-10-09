"""Behaviour of the Phase 6 contracts; no pool/storage migration in this slice."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import subprocess
import sys

import pytest

from auth.context import AuthContext, AuthSettings, CredentialScope
from auth.credentials import (
    CredentialIdentity,
    CredentialLease,
    CredentialMaterial,
    CredentialRequest,
    CredentialSelection,
    EndpointScope,
)
from auth.status import CredentialFailure, CredentialStatus


def make_request(home, *, provider="openai", endpoint="https://api.example/v1",
                 custom_id=None, model="model-a", execution=None):
    scope = CredentialScope(home, execution_id=execution)
    context = AuthContext(scope, AuthSettings(False, "fill_first"), lambda scope, name: None)
    return CredentialRequest(provider, EndpointScope(endpoint, custom_id), context, model)


def identity_for(request, credential_id="row-1"):
    return CredentialIdentity(request.provider_id, request.endpoint, request.scope, credential_id)


def test_scope_is_explicit_and_stable_across_ambient_profile_changes(tmp_path, monkeypatch):
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()
    request_a = make_request(home_a)
    request_b = make_request(home_b)
    equivalent_a = make_request(home_a / ".." / "a")
    original = identity_for(request_a)
    for home in (home_a, home_b, home_a):
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert original.matches(request_a)
        assert original.matches(equivalent_a)
        assert not original.matches(request_b)
        assert request_a.scope.profile_home == home_a.resolve()
    with pytest.raises(ValueError, match="absolute"):
        CredentialScope(Path("relative-home"))


@pytest.mark.parametrize("change", [
    {"provider_id": "anthropic"},
    {"endpoint": EndpointScope("https://other.example/v1")},
    {"endpoint": EndpointScope("https://api.example/v1", "another-custom-route")},
    {"model_id": "other-model"},
])
def test_selection_captures_request_but_identity_only_names_credential_scope(tmp_path, change):
    request = make_request(tmp_path)
    identity = identity_for(request)
    material = CredentialMaterial(api_key="private-key")
    selection = CredentialSelection(request, identity, material, "manual")
    changed = replace(request, **change)
    if "model_id" in change:
        # Cooldown reporting retains the selected model even when a caller changes routes.
        assert identity.matches(changed)
        assert selection.request.model_id == request.model_id
    else:
        assert not identity.matches(changed)
        with pytest.raises(ValueError, match="identity"):
            CredentialSelection(changed, identity, material, "manual")


def test_custom_routes_with_shared_urls_never_share_selection_identity(tmp_path):
    first = make_request(tmp_path, provider="custom", custom_id="team-one")
    second = make_request(tmp_path, provider="custom", custom_id="team-two")
    identity = identity_for(first)
    assert first.endpoint.endpoint == second.endpoint.endpoint
    assert not identity.matches(second)
    with pytest.raises(ValueError, match="custom_provider_id"):
        make_request(tmp_path, provider="custom")
    with pytest.raises(ValueError, match="custom_provider_id"):
        make_request(tmp_path, provider="custom:team-one")


def test_leases_are_bound_to_the_selected_credential_and_execution(tmp_path):
    request = make_request(tmp_path, execution="turn-one")
    identity = identity_for(request)
    lease = CredentialLease(identity, identity.credential_id)
    selected = CredentialSelection(request, identity, CredentialMaterial(), "sdk", lease)
    assert selected.lease.identity == identity
    assert selected.status is CredentialStatus.SELECTED
    wrong_leases = (
        CredentialLease(identity_for(request, "row-2"), "row-2"),
        CredentialLease(identity_for(make_request(tmp_path, execution="turn-two")), "row-1"),
        CredentialLease(identity_for(make_request(tmp_path / "other")), "row-1"),
    )
    for wrong in wrong_leases:
        with pytest.raises(ValueError, match="lease"):
            replace(selected, lease=wrong)
    with pytest.raises(FrozenInstanceError):
        selected.identity = wrong_leases[0].identity


@pytest.mark.parametrize("status", [
    CredentialStatus.NOT_CONFIGURED,
    CredentialStatus.EXHAUSTED,
    CredentialStatus.INVALID,
    CredentialStatus.REFRESH_FAILED,
    CredentialStatus.PERSISTENCE_FAILED,
])
def test_failures_keep_their_distinct_outcome_and_never_carry_material(tmp_path, status):
    request = make_request(tmp_path)
    failure = CredentialFailure(
        request, status,
        identity=identity_for(request) if status is not CredentialStatus.NOT_CONFIGURED else None,
        code="provider-error",
        relogin_required=status is CredentialStatus.INVALID,
        retry_at=1234.5 if status is CredentialStatus.EXHAUSTED else None,
        detail="untrusted diagnostic private-token",
    )
    assert failure.status is status
    assert failure.request is request
    assert not hasattr(failure, "material")
    assert "private-token" not in repr(failure)
    assert failure.relogin_required is (status is CredentialStatus.INVALID)


@pytest.mark.parametrize("status", [CredentialStatus.SELECTED, "not_configured", "unknown"])
def test_failure_cannot_misrepresent_success_or_an_untyped_status(tmp_path, status):
    with pytest.raises(ValueError, match="non-selected"):
        CredentialFailure(make_request(tmp_path), status)


@pytest.mark.parametrize("retry_at", [float("nan"), float("inf"), -1])
def test_failure_rejects_invalid_recovery_timestamps(tmp_path, retry_at):
    with pytest.raises(ValueError, match="finite epoch"):
        CredentialFailure(make_request(tmp_path), CredentialStatus.EXHAUSTED, retry_at=retry_at)


def test_failure_cannot_report_a_different_profiles_credential(tmp_path):
    first = make_request(tmp_path / "a")
    second = make_request(tmp_path / "b")
    with pytest.raises(ValueError, match="identity"):
        CredentialFailure(second, CredentialStatus.PERSISTENCE_FAILED, identity=identity_for(first))


def test_material_is_redacted_and_copies_runtime_options_without_rewriting_them(tmp_path):
    headers = {"Authorization": "Bearer private-header"}
    options = {"token_provider": lambda: "private-callback"}
    material = CredentialMaterial("private-key", headers, options)
    selected = CredentialSelection(make_request(tmp_path), identity_for(make_request(tmp_path)),
                                   material, "manual")
    headers["Authorization"] = "changed"
    options["another"] = "new"
    assert material.api_key == "private-key"
    assert material.headers["Authorization"] == "Bearer private-header"
    assert material.client_options["token_provider"]() == "private-callback"
    assert "another" not in material.client_options
    for value in ("private-key", "private-header", "private-callback"):
        assert value not in repr(material)
        assert value not in repr(selected)
    with pytest.raises(TypeError):
        material.headers["Authorization"] = "changed"


def test_application_context_passes_the_owning_scope_and_updated_policy(tmp_path):
    scope_a, scope_b = CredentialScope(tmp_path / "a"), CredentialScope(tmp_path / "b")
    secrets = {scope_a.home_key: {"KEY": "a-secret"}, scope_b.home_key: {}}
    reader = lambda scope, name: secrets[scope.home_key].get(name)
    first = AuthContext(scope_a, AuthSettings(True, "fill_first"), reader)
    second = AuthContext(scope_b, AuthSettings(False, "round_robin"), reader)
    for context, expected in ((first, "a-secret"), (second, None), (first, "a-secret")):
        assert context.read_secret(context.scope, "KEY") == expected
    changed = replace(first, settings=AuthSettings(False, "round_robin"))
    assert changed.settings != first.settings
    assert changed.scope == first.scope


@pytest.mark.parametrize("provider", ["auto", "", " openai "])
def test_requests_require_an_already_selected_explicit_provider(tmp_path, provider):
    with pytest.raises(ValueError):
        make_request(tmp_path, provider=provider)


@pytest.mark.parametrize("endpoint", ["", " ", " https://api.example/v1"])
def test_endpoint_scope_cannot_be_implicit_or_empty(endpoint):
    with pytest.raises(ValueError, match="endpoint"):
        EndpointScope(endpoint)


def test_contract_imports_and_construction_do_not_discover_providers_or_import_cli(tmp_path):
    code = """
import importlib.abc
import sys
from pathlib import Path

class NoRuntimeDiscovery(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'hermes_cli', 'nous_cli', 'providers'}:
            raise AssertionError('unexpected runtime dependency: ' + fullname)

sys.meta_path.insert(0, NoRuntimeDiscovery())
from auth.context import AuthContext, AuthSettings, CredentialScope
from auth.credentials import CredentialRequest, EndpointScope
from auth.status import CredentialFailure, CredentialStatus
scope = CredentialScope(Path(sys.argv[1]))
context = AuthContext(scope, AuthSettings(False, 'fill_first'), lambda scope, name: None)
request = CredentialRequest('late-plugin', EndpointScope('process:late-plugin'), context)
assert request.provider_id == 'late-plugin'
CredentialFailure(request, CredentialStatus.NOT_CONFIGURED)
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True,
                            text=True, timeout=30, cwd=Path(__file__).resolve().parents[2])
    assert result.returncode == 0, result.stderr
