"""Shared fixtures for brain RPC tests — tmp vault + profile seeds, no network."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gateway.brain_rpc.config import BrainRpcHostConfig
from gateway.brain_rpc.dispatcher import BrainRpcDispatcher, reset_default_dispatcher


@pytest.fixture(autouse=True)
def _reset_dispatcher():
    reset_default_dispatcher()
    yield
    reset_default_dispatcher()


@pytest.fixture
def vault_root(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    legal = root / "Projects" / "Legal"
    legal.mkdir(parents=True)
    (legal / "memo.md").write_text("# Memo\nsecret body never logged\n", encoding="utf-8")
    (legal / "exhibits").mkdir()
    (legal / "exhibits" / "a.txt").write_text("exhibit-a", encoding="utf-8")
    secrets = root / "Secrets"
    secrets.mkdir()
    (secrets / "keys.txt").write_text("do-not-read", encoding="utf-8")
    return root


@pytest.fixture
def profiles_dir(tmp_path: Path) -> Path:
    d = tmp_path / "profiles"
    d.mkdir()
    (d / "admin.json").write_text(
        json.dumps(
            {
                "name": "admin",
                "capabilities": [
                    "vault_full",
                    "vault_read",
                    "vault_write",
                    "chat",
                    "profile_manage",
                    "system_ops",
                ],
            }
        ),
        encoding="utf-8",
    )
    (d / "contributor.json").write_text(
        json.dumps(
            {
                "name": "contributor",
                "capabilities": ["vault_read", "vault_write", "chat"],
            }
        ),
        encoding="utf-8",
    )
    return d


@pytest.fixture
def host_config(vault_root: Path, profiles_dir: Path) -> BrainRpcHostConfig:
    return BrainRpcHostConfig(
        vault_root=vault_root,
        instance_id="inst_test",
        tenant_id="ten_test",
        profiles_dir=profiles_dir,
    )


@pytest.fixture
def dispatcher(host_config: BrainRpcHostConfig) -> BrainRpcDispatcher:
    return BrainRpcDispatcher(host=host_config)


def _expires_at(*, minutes: int = 5) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(minutes=minutes)
    ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def make_auth(
    *,
    profile: str = "contributor",
    path_prefixes: list | None = None,
    tenant_id: str = "ten_test",
    instance_id: str = "inst_test",
    expired: bool = False,
    roles: list | None = None,
) -> dict:
    if path_prefixes is None:
        path_prefixes = ["/Projects/Legal"]
    if roles is None:
        roles = [profile]
    return {
        "tenant_id": tenant_id,
        "instance_id": instance_id,
        "subject": {
            "portal_user_id": "usr_test",
            "hermes_profile": profile,
            "roles": roles,
            "path_prefixes": path_prefixes,
        },
        "session_id": "sess_test",
        "issued_at": _expires_at(minutes=-1),
        "expires_at": _expires_at(minutes=-5 if expired else 5),
    }


def make_request(
    method: str,
    *,
    params: dict | None = None,
    auth: dict | None = None,
    request_id: str = "req_1",
    contract_version: int = 1,
    timeout_ms: int = 10_000,
) -> dict:
    return {
        "type": "brain_rpc_request",
        "contract_version": contract_version,
        "request_id": request_id,
        "method": method,
        "timeout_ms": timeout_ms,
        "auth": auth if auth is not None else make_auth(),
        "params": params if params is not None else {},
        "idempotency_key": None,
    }
