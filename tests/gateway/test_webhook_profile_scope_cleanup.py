"""Every acquired webhook profile binding is released, including failed entry."""

import hashlib
import json
import os
from types import SimpleNamespace

import pytest

import gateway.webhook_config as webhook_config
from agent.secret_scope import current_secret_scope, reset_secret_scope, set_secret_scope
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


def _environment_digest():
    # Pytest changes this runner-owned key between fixture setup and test call.
    # Hash the remaining environment so failed assertions cannot dump secrets.
    values = {key: value for key, value in os.environ.items() if key != "PYTEST_CURRENT_TEST"}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


@pytest.fixture(params=[False, True], ids=["unscoped-caller", "scoped-caller"])
def scope_pair(tmp_path, monkeypatch, request):
    import hermes_cli.profiles as profiles

    owner = tmp_path / "owner"
    worker = tmp_path / "worker"
    nested = tmp_path / "nested"
    for home in (owner, worker, nested):
        home.mkdir()
    homes = {"worker": worker, "nested": nested}
    # Isolate lookup from the user's real profile registry. Scope mutation and
    # restoration use the actual Hermes ContextVar APIs, not mock tokens.
    monkeypatch.setattr(profiles, "get_profile_dir", homes.__getitem__)
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in homes)
    home_token = set_hermes_home_override(owner)
    secret_token = set_secret_scope({"CALLER": "owner-only"} if request.param else None)
    state = SimpleNamespace(
        owner=owner, worker=worker, nested=nested,
        secrets=current_secret_scope(), environment=_environment_digest(),
    )
    try:
        yield state
    finally:
        # The baseline's failing cases must not contaminate subsequent tests.
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def assert_caller_restored(state):
    assert get_hermes_home() == state.owner
    assert current_secret_scope() is state.secrets
    assert _environment_digest() == state.environment


@pytest.mark.parametrize("stage", ["build", "install"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_public_resolver_restores_home_when_scope_entry_fails(
    scope_pair, monkeypatch, stage, error_type,
):
    error = error_type("injected scope-entry failure")
    observed = []

    def build(home):
        observed.append((home, get_hermes_home(), current_secret_scope()))
        if stage == "build":
            raise error
        return {"WORKER": "worker-only"}

    def install(_mapping):
        raise error

    monkeypatch.setattr(webhook_config, "build_profile_secret_scope", build)
    if stage == "install":
        monkeypatch.setattr(webhook_config, "set_secret_scope", install)

    with pytest.raises(error_type) as caught:
        webhook_config.resolve_effective_webhook_config("worker")

    assert caught.value is error
    assert observed == [(scope_pair.worker, scope_pair.worker, scope_pair.secrets)]
    assert_caller_restored(scope_pair)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_body_failure_restores_both_bindings(scope_pair, monkeypatch, error_type):
    worker_secrets = {"WORKER": "worker-only"}
    monkeypatch.setattr(webhook_config, "build_profile_secret_scope", lambda _home: worker_secrets)
    error = error_type("injected body failure")

    with pytest.raises(error_type) as caught:
        with webhook_config._profile_config_scope("worker"):
            assert get_hermes_home() == scope_pair.worker
            assert current_secret_scope() is worker_secrets
            raise error

    assert caught.value is error
    assert_caller_restored(scope_pair)


def test_nested_success_restores_immediate_caller_then_original(scope_pair, monkeypatch):
    secrets = {
        scope_pair.worker: {"WORKER": "worker-only"},
        scope_pair.nested: {"NESTED": "nested-only"},
    }
    monkeypatch.setattr(webhook_config, "build_profile_secret_scope", secrets.__getitem__)

    with webhook_config._profile_config_scope("worker"):
        assert get_hermes_home() == scope_pair.worker
        assert current_secret_scope() is secrets[scope_pair.worker]
        with webhook_config._profile_config_scope("nested"):
            assert get_hermes_home() == scope_pair.nested
            assert current_secret_scope() is secrets[scope_pair.nested]
        assert get_hermes_home() == scope_pair.worker
        assert current_secret_scope() is secrets[scope_pair.worker]

    assert_caller_restored(scope_pair)


def test_home_cleanup_runs_even_when_secret_cleanup_raises(scope_pair, monkeypatch):
    monkeypatch.setattr(webhook_config, "build_profile_secret_scope", lambda _home: {})
    error = RuntimeError("injected secret-cleanup failure")

    def reset_then_fail(token):
        reset_secret_scope(token)
        raise error

    monkeypatch.setattr(webhook_config, "reset_secret_scope", reset_then_fail)
    with pytest.raises(RuntimeError) as caught:
        with webhook_config._profile_config_scope("worker"):
            assert get_hermes_home() == scope_pair.worker

    assert caught.value is error
    assert_caller_restored(scope_pair)
