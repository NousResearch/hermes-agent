"""Exact Feishu group rules grant chat-scoped gateway access (PR #62340)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from tests.gateway.feishu_helpers import make_message, make_sender


@pytest.fixture
def rig(monkeypatch):
    from gateway.run import GatewayRunner
    from plugins.platforms.feishu.adapter import FeishuAdapter

    for key in (
        "FEISHU_ALLOWED_USERS", "FEISHU_ALLOW_ALL_USERS", "FEISHU_ALLOW_BOTS",
        "FEISHU_GROUP_POLICY", "FEISHU_ALLOWED_GROUP_USERS",
        "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_args: False)
    runner._primary_profile_name = "default"
    runner.adapters = {}
    runner._profile_adapters = {}

    def install(rules, *, profile=None, **extra):
        cfg = PlatformConfig(enabled=True, extra={
            "require_mention": False, "group_rules": rules, **extra,
        })
        adapter = FeishuAdapter(cfg)
        adapter.set_owner_profile(profile)
        adapters = runner._profile_adapters.setdefault(profile, {}) if profile else runner.adapters
        adapters[Platform.FEISHU] = adapter
        return adapter

    return SimpleNamespace(runner=runner, install=install)


@pytest.mark.parametrize("env_allowlist", [None, "ou_dm_owner"])
def test_named_open_group_grant_is_scoped_and_delegation_can_be_disabled(rig, monkeypatch, env_allowlist):
    if env_allowlist:
        monkeypatch.setenv("FEISHU_ALLOWED_USERS", env_allowlist)
    adapter = rig.install({"oc_named": {"policy": "open"}}, dm_policy="allowlist")
    source = adapter.build_source(chat_id="oc_named", chat_type="group", user_id="ou_guest")
    assert adapter._admit(make_sender(open_id="ou_guest"), make_message(chat_type="group", chat_id="oc_named")) is None
    assert rig.runner._is_user_authorized(source) is True
    assert rig.runner._is_user_authorized(source, allow_adapter_delegation=False) is False
    for chat_type, chat_id in (("group", "oc_other"), ("dm", "oc_named")):
        other = adapter.build_source(chat_id=chat_id, chat_type=chat_type, user_id="ou_guest")
        assert rig.runner._is_user_authorized(other) is False


@pytest.mark.parametrize(
    "rule,admins,user,admitted",
    [
        ({}, [], "ou_guest", True),
        ({"policy": "allowlist", "allowlist": ["ou_member"]}, [], "ou_member", True),
        ({"policy": "allowlist", "allowlist": ["u_member"]}, [], "ou_member", True),
        ({"policy": "allowlist", "allowlist": ["ou_member"]}, [], "ou_guest", False),
        ({"policy": "blacklist", "blacklist": ["ou_blocked"]}, [], "ou_guest", True),
        ({"policy": "blacklist", "blacklist": ["ou_blocked"]}, [], "ou_blocked", False),
        ({"policy": "admin_only"}, ["ou_admin"], "ou_admin", True),
        ({"policy": "admin_only"}, ["ou_admin"], "ou_guest", False),
        ({"policy": "disabled"}, [], "ou_guest", False),
        ({"policy": "unknown"}, [], "ou_guest", False),
    ],
)
@pytest.mark.asyncio
async def test_real_intake_and_gateway_respect_named_group_policy(rig, rule, admins, user, admitted):
    adapter = rig.install({"oc_named": rule}, admins=admins)
    adapter.get_chat_info = AsyncMock(return_value={"name": "Named group", "type": "group"})
    adapter._resolve_sender_name_from_api = AsyncMock(return_value="Guest")
    decisions = []

    async def receive(event):
        decisions.append(rig.runner._is_user_authorized(event.source))

    adapter._dispatch_inbound_event = receive
    message = make_message(chat_type="group", chat_id="oc_named")
    message.content = '{"text":"hello"}'
    await adapter._handle_message_event_data(SimpleNamespace(
        event=SimpleNamespace(sender=make_sender(open_id=user, user_id=user.replace("ou_", "u_")), message=message),
    ))
    assert decisions == ([True] if admitted else [])


@pytest.mark.parametrize("rules", [{}, {"*": {"policy": "open"}}, {"oc_named": {"policy": "disabled"}}])
def test_default_open_and_inactive_rules_do_not_grant_access(rig, rules):
    adapter = rig.install(rules, default_group_policy="open")
    source = adapter.build_source(chat_id="oc_named", chat_type="group", user_id="ou_guest")
    assert rig.runner._is_user_authorized(source) is False


def test_only_a_live_adapter_declaring_rule_enforcement_can_grant_access(rig):
    adapter = rig.install({"oc_named": {"policy": "open"}})
    source = adapter.build_source(chat_id="oc_named", chat_type="group", user_id="ou_guest")
    adapter._enforces_group_rules = False
    assert rig.runner._is_user_authorized(source) is False
    adapter._enforces_group_rules = "true"
    assert rig.runner._is_user_authorized(source) is False
    rig.runner.config.platforms[Platform.FEISHU] = adapter.config
    rig.runner.adapters.clear()
    assert rig.runner._is_user_authorized(source) is False


@pytest.mark.parametrize("transport_profile,runtime_profile", [(None, "beta"), ("beta", "default")])
def test_routed_and_restored_sources_use_receiving_bot_rules(rig, tmp_path, transport_profile, runtime_profile):
    from gateway.session_identity import RoutingIdentity, restore_identity

    rig.runner.config.multiplex_profiles = True
    default = rig.install({"oc_named": {"policy": "open"}})
    beta = rig.install({}, profile="beta")
    receiver = beta if transport_profile else default
    source = receiver.build_source(chat_id="oc_named", chat_type="group", user_id="ou_guest")
    source.profile = runtime_profile
    source._identity = RoutingIdentity(
        transport_profile=transport_profile or "default", runtime_profile=runtime_profile,
        authorization_home=tmp_path, runtime_home=tmp_path,
    )
    assert rig.runner._is_user_authorized(source) is (transport_profile is None)

    restored = SessionSource.from_dict(source.to_dict())
    restore_identity(restored, runner=rig.runner, transport_profile=transport_profile or "default")
    assert rig.runner._is_user_authorized(restored) is (transport_profile is None)

    # An explicitly identified receiving bot going offline cannot borrow the runtime bot's rule.
    rig.runner._profile_adapters["beta"].clear()
    if transport_profile == "beta":
        assert rig.runner._is_user_authorized(restored) is False


@pytest.mark.parametrize("policy", ["allowlist", "blacklist", "admin_only"])
@pytest.mark.parametrize("known_transport", [False, True])
def test_restored_source_does_not_inherit_sender_restricted_rule(rig, policy, known_transport):
    from gateway.session_identity import restore_identity

    rig.runner.config.multiplex_profiles = True
    adapter = rig.install({"oc_named": {"policy": policy, "allowlist": ["ou_member"]}}, admins=["ou_member"])
    source = adapter.build_source(chat_id="oc_named", chat_type="group", user_id="ou_member")
    restored = SessionSource.from_dict(source.to_dict())
    if known_transport:
        restore_identity(restored, runner=rig.runner, transport_profile="default")
    assert rig.runner._is_user_authorized(restored) is False


@pytest.mark.parametrize(
    "policy,user,allowed",
    [
        ("allowlist", "ou_member", True), ("allowlist", "ou_guest", False),
        ("blacklist", "ou_guest", True), ("blacklist", "ou_blocked", False),
        ("admin_only", "ou_admin", True), ("admin_only", "ou_guest", False),
        ("disabled", "ou_guest", False),
    ],
)
@pytest.mark.asyncio
async def test_synthetic_group_events_enforce_sender_policy_before_gateway(rig, policy, user, allowed):
    from gateway.platforms.event import MessageType

    adapter = rig.install({"oc_named": {
        "policy": policy, "allowlist": ["ou_member"], "blacklist": ["ou_blocked"],
    }}, admins=["ou_admin"])
    adapter.get_chat_info = AsyncMock(return_value={"name": "Named group", "type": "group"})
    adapter._resolve_sender_name_from_api = AsyncMock(return_value="Guest")
    decisions = []

    async def receive(event):
        decisions.append(rig.runner._is_user_authorized(event.source))

    adapter._handle_message_with_guards = receive
    await adapter._dispatch_synthetic_event(
        text="clicked", message_type=MessageType.COMMAND, chat_id="oc_named",
        sender_id=make_sender(open_id=user).sender_id, event_chat_type="group",
        raw_message=None, message_id="om_click",
    )
    assert decisions == ([True] if allowed else [])


@pytest.mark.parametrize("event_chat_type", ["p2p", "group"])
@pytest.mark.asyncio
async def test_synthetic_events_without_exact_rule_preserve_independent_user_access(rig, monkeypatch, event_chat_type):
    from gateway.platforms.event import MessageType

    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_guest")
    adapter = rig.install({}, default_group_policy="disabled")
    adapter.get_chat_info = AsyncMock(return_value={"name": "Direct chat", "type": "p2p"})
    adapter._resolve_sender_name_from_api = AsyncMock(return_value="Guest")
    decisions = []

    async def receive(event):
        decisions.append(rig.runner._is_user_authorized(event.source))

    adapter._handle_message_with_guards = receive
    await adapter._dispatch_synthetic_event(
        text="clicked", message_type=MessageType.COMMAND, chat_id="oc_other",
        sender_id=make_sender(open_id="ou_guest").sender_id, event_chat_type=event_chat_type,
        raw_message=None, message_id="om_click",
    )
    assert decisions == [True]
