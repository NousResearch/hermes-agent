"""Yuanbao authorization-scope regression tests (#93522).

``dm_policy``/``group_policy``/allowlist reads and the ``AccessPolicy``
allow-all opt-in must honor the active profile secret scope under
multiplexing: a secondary profile's own scope is authoritative and must
not inherit the default profile's process-env authorization config.
"""

import pytest

from agent import secret_scope
from gateway.config import PlatformConfig
from gateway.platforms.yuanbao import AccessPolicy, YuanbaoAdapter


@pytest.fixture()
def multiplex_on():
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        yield
    finally:
        secret_scope.set_multiplex_active(previous)


class TestYuanbaoAdapterAuthzScope:
    def test_scoped_construction_reads_authz_from_scope_not_environ(self, multiplex_on, monkeypatch):
        monkeypatch.setenv("YUANBAO_DM_POLICY", "pairing")
        monkeypatch.setenv("YUANBAO_DM_ALLOW_FROM", "default-user")
        token = secret_scope.set_secret_scope(
            {"YUANBAO_DM_POLICY": "allowlist", "YUANBAO_DM_ALLOW_FROM": "scoped-user"}
        )
        try:
            adapter = YuanbaoAdapter(PlatformConfig(enabled=True))
        finally:
            secret_scope.reset_secret_scope(token)
        assert adapter._access_policy._dm_policy == "allowlist"
        assert adapter._access_policy._dm_allow_from == ["scoped-user"]

    def test_scoped_miss_does_not_admit_default_profiles_allowlist(self, multiplex_on, monkeypatch):
        monkeypatch.setenv("YUANBAO_DM_POLICY", "allowlist")
        monkeypatch.setenv("YUANBAO_DM_ALLOW_FROM", "default-user")
        token = secret_scope.set_secret_scope({"SOMETHING_ELSE": "x"})
        try:
            adapter = YuanbaoAdapter(PlatformConfig(enabled=True))
        finally:
            secret_scope.reset_secret_scope(token)
        assert adapter._access_policy._dm_policy == "pairing"
        assert adapter._access_policy._dm_allow_from == []


class TestYuanbaoAccessPolicyOpenDmOptIn:
    def test_scoped_allow_all_admits(self, multiplex_on, monkeypatch):
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("YUANBAO_ALLOW_ALL_USERS", raising=False)
        policy = AccessPolicy(dm_policy="open", dm_allow_from=[], group_policy="pairing", group_allow_from=[])
        token = secret_scope.set_secret_scope({"YUANBAO_ALLOW_ALL_USERS": "true"})
        try:
            assert policy._open_dm_opted_in() is True
        finally:
            secret_scope.reset_secret_scope(token)

    def test_default_profiles_allow_all_does_not_leak_into_scoped_miss(self, multiplex_on, monkeypatch):
        """The default profile's env-only GATEWAY_ALLOW_ALL_USERS must not
        admit a secondary profile that never opted in."""
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        policy = AccessPolicy(dm_policy="open", dm_allow_from=[], group_policy="pairing", group_allow_from=[])
        token = secret_scope.set_secret_scope({"SOMETHING_ELSE": "x"})
        try:
            assert policy._open_dm_opted_in() is False
        finally:
            secret_scope.reset_secret_scope(token)
