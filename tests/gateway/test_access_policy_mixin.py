"""Every own-policy adapter must reach the same allow/deny verdict for the same policy inputs.

The predicates used to be five hand-copied bodies (weixin, wecom, qqbot, whatsapp, yuanbao) that
drifted on blank principals and scoped env reads; ``OwnAccessPolicyMixin`` is the single rule.
"""

from __future__ import annotations

import contextlib
import itertools

import pytest

from agent.secret_scope import reset_secret_scope, set_secret_scope
from gateway.platforms.access_policy_mixin import OwnAccessPolicyMixin

POLICIES = ("open", "allowlist", "disabled", "pairing", "typo")
SENDERS = ("alice", "stranger", "", "   ", None)


@contextlib.contextmanager
def _scope(secrets):
    token = set_secret_scope(secrets)
    try:
        yield
    finally:
        reset_secret_scope(token)


def _hosts():
    """One bare instance per own-policy class, attributes set exactly as the adapters do."""
    from gateway.platforms.qqbot.adapter import QQAdapter
    from gateway.platforms.weixin import WeixinAdapter
    from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin
    from gateway.platforms.yuanbao import AccessPolicy
    from plugins.platforms.wecom.adapter import WeComAdapter

    hosts = {
        "weixin": object.__new__(WeixinAdapter),
        "wecom": object.__new__(WeComAdapter),
        "qqbot": object.__new__(QQAdapter),
        "whatsapp": WhatsAppBehaviorMixin(),
        "yuanbao": AccessPolicy("open", [], "open", []),
    }
    hosts["whatsapp"]._dm_allowlist_source = "config"
    hosts["wecom"]._groups = {}
    return hosts


def _verdicts(host, name, dm_policy, group_policy):
    host._dm_policy, host._group_policy = dm_policy, group_policy
    host._allow_from, host._group_allow_from = ["alice"], ["room-1"]
    out = {}
    for sender in SENDERS:
        out[("dm", sender)] = host._is_dm_allowed(sender or "")
        out[("intake", sender)] = host._is_dm_intake_allowed(sender)
    for group in ("room-1", "room-2"):
        args = (group, "alice") if name in ("wecom", "qqbot") else (group,)
        out[("group", group)] = host._is_group_allowed(*args)
    return out


@pytest.mark.parametrize("opt_in", [{}, {"GATEWAY_ALLOW_ALL_USERS": "true"}])
def test_all_own_policy_adapters_agree(monkeypatch, opt_in):
    for var in ("GATEWAY_ALLOW_ALL_USERS", "WEIXIN_ALLOW_ALL_USERS", "WECOM_ALLOW_ALL_USERS",
                "QQ_ALLOW_ALL_USERS", "WHATSAPP_ALLOW_ALL_USERS", "YUANBAO_ALLOW_ALL_USERS"):
        monkeypatch.delenv(var, raising=False)
    hosts = _hosts()
    with _scope(opt_in):
        for dm_policy, group_policy in itertools.product(POLICIES, POLICIES):
            table = {name: _verdicts(host, name, dm_policy, group_policy) for name, host in hosts.items()}
            for name, verdicts in table.items():
                assert verdicts == table["weixin"], (name, dm_policy, group_policy, opt_in)
            expected_open = bool(opt_in) and dm_policy == "open"
            assert table["weixin"][("dm", "stranger")] is expected_open
            assert table["weixin"][("intake", "   ")] is False  # blank principal never admitted
            assert table["weixin"][("intake", "stranger")] is (dm_policy == "pairing" or expected_open)


def test_platform_prefix_env_name_is_scoped_and_fail_closed(monkeypatch):
    class Host(OwnAccessPolicyMixin):
        ALLOW_ALL_ENV_PREFIX = "DEMO"
        _dm_policy, _group_policy, _allow_from, _group_allow_from = "open", "open", [], []

    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    monkeypatch.setenv("DEMO_ALLOW_ALL_USERS", "true")  # the DEFAULT profile's opt-in
    assert Host()._allow_all_env_names() == ("GATEWAY_ALLOW_ALL_USERS", "DEMO_ALLOW_ALL_USERS")
    with _scope({}):  # secondary profile: scope installed, key absent
        assert Host()._is_dm_allowed("anyone") is False
    with _scope({"DEMO_ALLOW_ALL_USERS": "yes"}):
        assert Host()._is_dm_allowed("anyone") is True
