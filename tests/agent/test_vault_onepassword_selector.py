"""Metadata-only selector regressions; no real password manager is invoked."""
import json
from unittest.mock import Mock, patch

import pytest

from agent.vault_backends.onepassword import OnePasswordLoginBackend


def item(item_id="item-a", vault: object = "vault-a"):
    return {"id": item_id, "title": "Example", "vault": {"id": vault},
            "urls": [{"href": "https://example.com/login"}]}


@pytest.fixture
def backend():
    with patch("agent.secret_scope.get_secret", return_value="dummy-service-token"):
        backend = OnePasswordLoginBackend()
    backend._run = Mock()
    return backend


@pytest.mark.parametrize("method,flags,value", [
    ("resolve_password", ("--fields", "label=password", "--reveal"), "dummy-password"),
    ("resolve_otp", ("--otp",), "123456"),
])
def test_legacy_handles_resolve_each_items_vault(backend, method, flags, value):
    records = [item(), item("item-b", "vault-b")]
    backend._run.return_value = json.dumps(records)
    assert [m.id for m in backend.list_items()] == ["op:item-a", "op:item-b"]
    assert backend.get_meta("op:item-b").origin == "https://example.com"
    for item_id, vault_id in [("item-a", "vault-a"), ("item-b", "vault-b")]:
        backend._run.reset_mock()
        backend._run.side_effect = [json.dumps(records), value + "\r\n"]
        assert getattr(backend, method)("op:" + item_id) == value
        assert backend._run.call_args_list[0].args == (
            "item", "list", "--categories", "Login", "--format", "json")
        assert backend._run.call_args_list[1].args == (
            "item", "get", item_id, "--vault", vault_id, *flags)


@pytest.mark.parametrize("records", [
    [], [item("other")], [item(vault="")], [item(vault=None)],
    [dict(item(), vault=None)], [dict(item(), vault="not-a-record")],
    [item(), item(vault="vault-b")], {"error": "not-a-list"},
])
@pytest.mark.parametrize("method", ["resolve_password", "resolve_otp"])
def test_missing_or_ambiguous_metadata_never_reads_secret(backend, records, method):
    backend._run.return_value = json.dumps(records)
    if method == "resolve_password":
        with pytest.raises(RuntimeError):
            backend.resolve_password("op:item-a")
    else:
        assert backend.resolve_otp("op:item-a") is None
    assert all(c.args[:2] != ("item", "get") for c in backend._run.call_args_list)


@pytest.mark.parametrize("method", ["resolve_password", "resolve_otp"])
@pytest.mark.parametrize("failure", [RuntimeError("metadata unavailable"), "not-json", "[]"])
def test_refresh_failure_does_not_reuse_old_vault(backend, method, failure):
    backend._run.return_value = json.dumps([item()])
    backend.list_items()
    backend._run.reset_mock()
    backend._run.side_effect = [failure]
    if method == "resolve_password":
        with pytest.raises((RuntimeError, ValueError)):
            backend.resolve_password("op:item-a")
    else:
        assert backend.resolve_otp("op:item-a") is None
    assert all(c.args[:2] != ("item", "get") for c in backend._run.call_args_list)


@pytest.mark.parametrize("handle", ["bw:item-a", "op:", "op:--help", "op:vault-a:item-a"])
def test_invalid_handles_fail_before_cli(backend, handle):
    with pytest.raises(ValueError):
        backend.resolve_password(handle)
    assert backend.resolve_otp(handle) is None
    backend._run.assert_not_called()


def test_personal_session_without_vault_keeps_unscoped_read(backend):
    backend._service_token = ""
    backend._run.side_effect = [json.dumps([item(vault="")]), "dummy-password\n"]
    assert backend.resolve_password("op:item-a") == "dummy-password"
    assert backend._run.call_args.args == (
        "item", "get", "item-a", "--fields", "label=password", "--reveal")


def test_fresh_backend_does_not_depend_on_listing_cache(backend):
    backend._run.side_effect = [json.dumps([item(vault="vault-new")]), "dummy-password\n"]
    assert backend.resolve_password("op:item-a") == "dummy-password"
    assert backend._run.call_args.args[3:5] == ("--vault", "vault-new")
