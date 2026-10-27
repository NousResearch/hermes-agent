"""Owner-write permission in a group chat: evaluated as a real DM by the receiving bot, never
borrowed from a group-only grant, and never granted where other members share the session."""

import contextvars

import pytest

import gateway.run as gateway_run
from gateway.channel_memory import bind_channel_memory, channel_memory_binding
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource
from tools import memory_tool as mt
from tools.memory_tool_channel import CHANNEL_MEMORY, ChannelMemoryBinding


class _Runner:
    """Admits a DM only under the receiving bot's profile home, as the gateway does when a
    shared bot routes a group into another profile (the runtime profile has no allowlist)."""

    config = GatewayConfig(platforms={})  # threads shared, groups per user: the defaults

    def __init__(self, bot_home="/homes/receiving-bot"):
        self.bot_home = bot_home
        self.active_home = None

    def _under_authorization_profile(self, source, check):
        previous, self.active_home = self.active_home, getattr(source, "_authorization_profile_home", None)
        try:
            return check()
        finally:
            self.active_home = previous

    def _principal_authorized(self, source, *, allow_adapter_delegation):
        if allow_adapter_delegation and source.role_authorized is True:
            return True  # what gateway/authz_mixin.py::_principal_authorized does with the flag
        return source.chat_type == "dm" and source.user_id == "owner" and self.active_home == self.bot_home


def _group(user_id, **extra):
    source = SessionSource(platform=Platform.DISCORD, chat_id="c1", chat_type="group", user_id=user_id, **extra)
    source._authorization_profile_home = "/homes/receiving-bot"
    return source


def test_owner_is_authorized_by_the_receiving_bot_not_the_runtime_profile():
    assert channel_memory_binding(_Runner(), _group("owner"), {}).owner_writes is True


def test_group_role_grant_does_not_count_as_dm_access():
    # A member admitted to the guild channel only by DISCORD_ALLOWED_ROLES cannot DM the bot by
    # default, so it must not write the owner's files either.
    assert channel_memory_binding(_Runner(), _group("member", role_authorized=True), {}).owner_writes is False


def test_shared_session_keeps_even_the_owner_on_channel_memory(tmp_path, monkeypatch):
    # Other members steer a shared thread's running turn and its background review sees their
    # messages: a turn-start grant would let their text reach MEMORY.md / USER.md.
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    def bound(source):
        bind_channel_memory(_Runner(), source)
        return CHANNEL_MEMORY.get()

    in_thread = contextvars.copy_context().run(bound, _group("owner", thread_id="t1", parent_chat_id="c1"))
    in_group = contextvars.copy_context().run(bound, _group("owner"))
    assert in_thread.scope == in_group.scope == "discord/c1"
    assert in_thread.owner_writes is False
    assert in_group.owner_writes is True


@pytest.mark.parametrize("settings, expected", [
    ("{scope: workspace, owner_writes_from_groups: false, char_limit: 40}",
     ChannelMemoryBinding(scope="discord/G1", owner_writes=False, char_limit=40)),
    ("{scope: chat, char_limit: 40}", ChannelMemoryBinding(scope="discord/G1_c1", owner_writes=True, char_limit=40)),
    ("{enabled: false}", None),
])
def test_config_yaml_reaches_the_turn_binding_and_its_file(tmp_path, monkeypatch, settings, expected):
    (tmp_path / "config.yaml").write_text(f"memory:\n  channel_memory: {settings}\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(mt, "get_memory_dir", lambda: tmp_path / "memories")
    source = _group("owner", scope_id="G1")

    def turn():
        bind_channel_memory(_Runner(), source)
        binding = CHANNEL_MEMORY.get()
        store = mt.MemoryStore()
        store.load_from_disk()
        short = mt.memory_tool(action="add", target="channel", content="Standup 9am", store=store)
        long = mt.memory_tool(action="add", target="channel", content="x" * 60, store=store)
        return binding, short, long

    binding, short, long = contextvars.copy_context().run(turn)
    assert binding == expected
    if expected is None:
        assert '"success": false' in short  # no binding: 'channel' is not a target outside a group turn
        return
    platform, _, key = expected.scope.partition("/")
    assert "Standup 9am" in (tmp_path / "memories" / "channels" / platform / f"{key}.md").read_text(encoding="utf-8-sig")
    assert '"success": false' in long  # char_limit from config.yaml, not the 1500 default
