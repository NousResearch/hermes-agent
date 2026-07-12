"""Auth fail-closed + path ACL tests."""

from __future__ import annotations

import pytest

from gateway.brain_rpc.auth import path_allowed, verify_auth
from gateway.brain_rpc.errors import FORBIDDEN, UNAUTHENTICATED, BrainRpcError
from tests.gateway.brain_rpc.conftest import make_auth, make_request


@pytest.mark.asyncio
async def test_missing_auth(dispatcher):
    req = make_request("brain.ping", auth=None)
    req["auth"] = None
    res = await dispatcher.handle(req)
    assert res["ok"] is False
    assert res["error"]["code"] == UNAUTHENTICATED


@pytest.mark.asyncio
async def test_expired_auth(dispatcher):
    res = await dispatcher.handle(
        make_request("brain.ping", auth=make_auth(expired=True))
    )
    assert res["ok"] is False
    assert res["error"]["code"] == UNAUTHENTICATED
    assert "expired" in res["error"]["message"].lower()


@pytest.mark.asyncio
async def test_instance_mismatch(dispatcher):
    res = await dispatcher.handle(
        make_request("brain.ping", auth=make_auth(instance_id="inst_other"))
    )
    assert res["ok"] is False
    assert res["error"]["code"] == FORBIDDEN


@pytest.mark.asyncio
async def test_tenant_mismatch(dispatcher):
    res = await dispatcher.handle(
        make_request("brain.ping", auth=make_auth(tenant_id="ten_other"))
    )
    assert res["ok"] is False
    assert res["error"]["code"] == FORBIDDEN


@pytest.mark.asyncio
async def test_unknown_profile_forbidden(dispatcher):
    res = await dispatcher.handle(
        make_request("brain.ping", auth=make_auth(profile="nosuchprofile"))
    )
    assert res["ok"] is False
    assert res["error"]["code"] == FORBIDDEN


@pytest.mark.asyncio
async def test_path_outside_prefixes_forbidden(dispatcher):
    res = await dispatcher.handle(
        make_request(
            "vault.stat",
            params={"path": "/Secrets/keys.txt"},
            auth=make_auth(path_prefixes=["/Projects/Legal"]),
        )
    )
    assert res["ok"] is False
    assert res["error"]["code"] == FORBIDDEN


@pytest.mark.asyncio
async def test_path_escape_dotdot(dispatcher):
    res = await dispatcher.handle(
        make_request(
            "vault.stat",
            params={"path": "/Projects/Legal/../../Secrets/keys.txt"},
            auth=make_auth(path_prefixes=["/Projects/Legal"]),
        )
    )
    assert res["ok"] is False
    assert res["error"]["code"] in {FORBIDDEN, "invalid_argument"}


def test_path_allowed_helpers(host_config):
    auth = verify_auth(
        make_auth(path_prefixes=["/Projects/Legal"]),
        method="vault.list",
        host=host_config,
    )
    assert path_allowed("/Projects/Legal", auth)
    assert path_allowed("/Projects/Legal/memo.md", auth)
    assert not path_allowed("/Secrets", auth)
    assert not path_allowed("/Projects", auth)


def test_admin_vault_full_empty_prefixes(host_config):
    auth = verify_auth(
        make_auth(profile="admin", path_prefixes=[]),
        method="vault.list",
        host=host_config,
    )
    assert path_allowed("/Secrets/keys.txt", auth)


def test_contributor_empty_prefixes_deny(host_config):
    auth = verify_auth(
        make_auth(profile="contributor", path_prefixes=[]),
        method="vault.list",
        host=host_config,
    )
    assert not path_allowed("/Projects/Legal", auth)


def test_missing_subject_raises(host_config):
    bad = make_auth()
    del bad["subject"]
    with pytest.raises(BrainRpcError) as ei:
        verify_auth(bad, method="brain.ping", host=host_config)
    assert ei.value.code == UNAUTHENTICATED
