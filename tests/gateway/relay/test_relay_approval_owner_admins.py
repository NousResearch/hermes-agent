"""A relayed approval answer is checked against the admins of the bot-owning (transport) profile.

Prompt answers are consumed before ``handle_message``, so before canonicalization the admin lookup fell
back to the routed runtime profile: its admin could answer another member's approval and the bot
owner's admin was refused, unlike the same event's ``/approve``.
"""

import json

import pytest

from hermes_agent.gateway.config import GatewayConfig, Platform, PlatformConfig


@pytest.fixture
def relay(tmp_path, monkeypatch):
    from hermes_agent.gateway.relay.adapter import RelayAdapter
    from hermes_agent.gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
    from hermes_agent.gateway.run import GatewayRunner
    from hermes_cli.profiles import get_profile_dir
    from hermes_constants import get_hermes_home
    from tools import approval

    (get_hermes_home() / "config.yaml").write_text("approvals:\n  approvers: []\n")
    secondary = get_profile_dir("satellite")
    secondary.mkdir(parents=True, exist_ok=True)
    (secondary / "config.yaml").write_text("approvals:\n  approvers: []\n")
    owner = {"group_allow_admin_from": ["owner-admin"]}
    runner = GatewayRunner(config=GatewayConfig(multiplex_profiles=True, platforms={
        Platform.TELEGRAM: PlatformConfig(extra=owner), Platform.DISCORD: PlatformConfig(extra=owner)}))
    runner._primary_profile_name = "default"
    satellite = {"group_allow_admin_from": ["satellite-admin"]}
    runner._profile_configs["satellite"] = GatewayConfig(platforms={
        Platform.TELEGRAM: PlatformConfig(extra=satellite), Platform.DISCORD: PlatformConfig(extra=satellite)})
    descriptor = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION, platform="telegram", label="t", max_message_length=4096,
        supports_draft_streaming=False, supports_edit=True, supports_threads=True, markdown_dialect="markdown_v2",
        len_unit="utf16", supported_ops=("send", "prompt"))
    adapter = RelayAdapter(PlatformConfig(), descriptor)
    adapter.gateway_runner = runner
    runner.adapters[Platform.RELAY] = adapter
    yield adapter
    approval._gateway_queues.clear()
    runner.close_all_session_db_handles()


def _pending(adapter, platform, key):
    from hermes_agent.gateway.session_context import clear_session_vars, set_session_vars
    from tools import approval
    from tools.approval_gateway_wait import _ApprovalEntry

    tokens = set_session_vars(user_id="alice", user_name="Alice", session_key=key, platform=platform, chat_type="group")
    try:
        entry = _ApprovalEntry({"command": "rm -rf build"})
    finally:
        clear_session_vars(tokens)
    approval._gateway_queues[key] = [entry]
    prompt_id = adapter._mint_prompt("exec_approval", {"session_key": key, "request_id": entry.data["request_id"],
                                                       "chat_id": "c1"})
    return entry, prompt_id


async def _answer(adapter, platform, actor, prompt_id):
    from hermes_agent.gateway.relay.ws_transport import PassthroughForward, _event_from_wire

    if platform == "telegram":
        await adapter._on_inbound(_event_from_wire({
            "text": "/once", "message_type": "command",
            "source": {"platform": "telegram", "user_id": actor, "chat_id": "c1", "chat_type": "group",
                       "profile": "satellite"},
            "prompt_response": {"prompt_id": prompt_id, "option_id": "once"}}))
    else:
        body = json.dumps({"type": 3, "guild_id": "guild", "channel_id": "c1", "member": {"user": {"id": actor}},
                           "data": {"custom_id": f"hp1:{prompt_id}:once"}}).encode()
        await adapter._on_passthrough(PassthroughForward(platform="discord", bot_id="bot", method="POST", path="/",
                                                         headers=[], body=body, profile="satellite"))


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["telegram", "discord"])
@pytest.mark.parametrize(("actor", "answers"), [("owner-admin", True), ("satellite-admin", False), ("alice", True)])
async def test_relayed_answer_uses_the_bot_owning_profiles_admins(relay, platform, actor, answers):
    entry, prompt_id = _pending(relay, platform, f"{platform}-{actor}")
    await _answer(relay, platform, actor, prompt_id)
    assert (entry.result == "once") is answers
