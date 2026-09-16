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
    from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter
    from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin
    from gateway.platforms.yuanbao import AccessPolicy
    from plugins.platforms.wecom.adapter import WeComAdapter

    hosts = {
        "weixin": object.__new__(WeixinAdapter),
        "wecom": object.__new__(WeComAdapter),
        "qqbot": object.__new__(QQAdapter),
        "whatsapp": WhatsAppBehaviorMixin(),
        "whatsapp_cloud": object.__new__(WhatsAppCloudAdapter),
        "yuanbao": AccessPolicy("open", [], "open", []),
    }
    hosts["whatsapp"]._dm_allowlist_source = "config"
    hosts["whatsapp_cloud"]._dm_allowlist_source = "config"
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


# Per-host opt-in var (the one ``hermes gateway setup`` writes). Setting only GATEWAY_ALLOW_ALL_USERS
# would pass with a host whose prefix is missing — that is exactly how WeCom regressed once.
PLATFORM_OPT_IN = {"weixin": "WEIXIN_ALLOW_ALL_USERS", "wecom": "WECOM_ALLOW_ALL_USERS",
                   "qqbot": "QQ_ALLOW_ALL_USERS", "whatsapp": "WHATSAPP_ALLOW_ALL_USERS",
                   "whatsapp_cloud": "WHATSAPP_CLOUD_ALLOW_ALL_USERS", "yuanbao": "YUANBAO_ALLOW_ALL_USERS"}
# Hosts whose allowlists document ``*`` (weixin/yuanbao match literally, as before).
WILDCARD_HOSTS = ("wecom", "qqbot", "whatsapp", "whatsapp_cloud")


def _all_agree(hosts, opt_in_for, label):
    for dm_policy, group_policy in itertools.product(POLICIES, POLICIES):
        table = {}
        for name, host in hosts.items():
            with _scope(opt_in_for(name)):
                table[name] = _verdicts(host, name, dm_policy, group_policy)
        for name, verdicts in table.items():
            assert verdicts == table["weixin"], (name, dm_policy, group_policy, label)
        expected_open = bool(opt_in_for("weixin")) and dm_policy == "open"
        assert table["weixin"][("dm", "stranger")] is expected_open
        assert table["weixin"][("intake", "   ")] is False  # blank principal never admitted
        assert table["weixin"][("intake", "stranger")] is (dm_policy == "pairing" or expected_open)


@pytest.mark.parametrize("mode", ["none", "gateway", "platform"])
def test_all_own_policy_adapters_agree(monkeypatch, mode):
    for var in ("GATEWAY_ALLOW_ALL_USERS", *PLATFORM_OPT_IN.values()):
        monkeypatch.delenv(var, raising=False)
    opt_in_for = {
        "none": lambda name: {},
        "gateway": lambda name: {"GATEWAY_ALLOW_ALL_USERS": "true"},
        "platform": lambda name: {PLATFORM_OPT_IN[name]: "true"},
    }[mode]
    _all_agree(_hosts(), opt_in_for, mode)


@pytest.mark.parametrize("name", WILDCARD_HOSTS)
def test_wildcard_allowlist_admits_strangers_on_every_path(name):
    """``*`` must open strict DM auth, DM intake AND group intake alike — whatsapp_cloud once
    honoured it on intake only, then on neither."""
    host = _hosts()[name]
    host._dm_policy = host._group_policy = "allowlist"
    host._allow_from, host._group_allow_from = ["*"], ["*"]
    group_args = ("room-2", "stranger") if name in ("wecom", "qqbot") else ("room-2",)
    assert host._is_dm_allowed("stranger") is True
    assert host._is_dm_intake_allowed("stranger") is True
    assert host._is_group_allowed(*group_args) is True
    assert host._is_dm_intake_allowed("   ") is False


def test_mixin_host_without_prefix_is_rejected_at_class_creation():
    with pytest.raises(TypeError, match="ALLOW_ALL_ENV_PREFIX"):
        class Host(OwnAccessPolicyMixin):  # noqa: F841
            _dm_policy = "open"


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
