"""Focused contracts for persistent per-channel Buzz controls."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from gateway.config import Platform, PlatformConfig, load_gateway_config
from gateway.profile_routing import ProfileRoute
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.plugins import PluginCommandAccessContext
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


_buzz = load_plugin_adapter("buzz")

BuzzAdapter = _buzz.BuzzAdapter
CHANNEL = "ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd"
OTHER_CHANNEL = "6468cc16-a114-4f23-8b8c-02c1655cbf6b"
ADMIN_HEX = "9fd5c7ba6d3ef224da78f541e0fcb9c50f72cc63edb19aae76ac6a0474dfa860"
ADMIN_NPUB = "npub1nl2u0wnd8mezfknc74q7pl9ec58h9nrrakce4tnk434qgaxl4psqe5twr6"


def _adapter(*, extra=None, reply_to_mode="first"):
    return BuzzAdapter(
        PlatformConfig(
            enabled=True,
            reply_to_mode=reply_to_mode,
            extra={"relay_url": "https://test.relay", **(extra or {})},
        )
    )


@pytest.mark.parametrize("args", ["status", "listen status", "replies status"])
def test_only_exact_status_forms_receive_user_access(args):
    assert _buzz._buzz_command_access(args) == "user"


@pytest.mark.parametrize(
    "args",
    ["", "listen", "replies", "listen always", "replies hybrid", "status now", "wat"],
)
def test_mutations_and_malformed_forms_fail_closed_to_admin(args):
    assert _buzz._buzz_command_access(args) == "admin"


def test_valid_dm_mutation_reaches_noop_while_group_mutation_stays_admin():
    def context(chat_type):
        return PluginCommandAccessContext(
            platform="buzz",
            channel_id=CHANNEL,
            thread_id=None,
            chat_type=chat_type,
            scope_id=None,
            source_identity_candidates=("ordinary",),
            routed_profile="default",
        )

    assert _buzz._buzz_command_access("listen always", context("dm")) == "user"
    assert _buzz._buzz_command_access("listen always", context("group")) == "admin"
    assert _buzz._buzz_command_access("listen sometimes", context("dm")) == "admin"


def test_gateway_access_allows_valid_dm_noop_for_non_admin(monkeypatch):
    adapter = _adapter()
    runner = _runner_for_profiles(adapter)
    runner.config = SimpleNamespace(
        platforms={
            Platform("buzz"): PlatformConfig(
                enabled=True,
                extra={"allow_admin_from": [ADMIN_NPUB]},
            )
        }
    )
    source = SessionSource(
        platform=Platform("buzz"),
        user_id="ordinary",
        chat_id=CHANNEL,
        chat_type="dm",
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_command",
        lambda _name: {"access": _buzz._buzz_command_access, "with_context": True},
    )

    assert runner._check_slash_access(source, "buzz", "listen always") is None


def test_adapter_hydrates_strict_sparse_modes_and_reports_inheritance():
    adapter = _adapter(
        extra={
            "channel_modes": {
                CHANNEL: {"listen": "always", "replies": "hybrid"},
                OTHER_CHANNEL: {"listen": "mentions"},
            }
        }
    )

    assert adapter.channel_policy_status(CHANNEL) == {
        "applicable": True,
        "listen": {"effective": "always", "source": "explicit"},
        "replies": {"effective": "hybrid", "source": "explicit"},
    }
    assert adapter.channel_policy_status("third") == {
        "applicable": True,
        "listen": {"effective": "mentions", "source": "inherited"},
        "replies": {"effective": "threaded", "source": "inherited"},
    }


def test_shared_transport_hydrates_routed_inherited_defaults_and_reset():
    adapter = _adapter(extra={"require_mention": True}, reply_to_mode="first")
    adapter.hydrate_routed_profile_config(
        "team-b",
        PlatformConfig(
            enabled=False,
            reply_to_mode="first",
            extra={"require_mention": False, "reply_in_thread": False},
        ),
    )

    assert adapter.channel_policy_status(CHANNEL) == {
        "applicable": True,
        "listen": {"effective": "mentions", "source": "inherited"},
        "replies": {"effective": "threaded", "source": "inherited"},
    }
    assert adapter.channel_policy_status(CHANNEL, routed_profile="team-b") == {
        "applicable": True,
        "listen": {"effective": "always", "source": "inherited"},
        "replies": {"effective": "flat", "source": "inherited"},
    }

    adapter.apply_channel_policy(
        CHANNEL, "listen", "mentions", routed_profile="team-b"
    )
    adapter.apply_channel_policy(
        CHANNEL, "replies", "threaded", routed_profile="team-b"
    )
    adapter.apply_channel_policy(CHANNEL, "listen", None, routed_profile="team-b")
    adapter.apply_channel_policy(CHANNEL, "replies", None, routed_profile="team-b")

    assert adapter.channel_policy_status(CHANNEL, routed_profile="team-b") == {
        "applicable": True,
        "listen": {"effective": "always", "source": "inherited"},
        "replies": {"effective": "flat", "source": "inherited"},
    }


@pytest.mark.parametrize(
    "channel_modes",
    [
        [],
        {CHANNEL: "always"},
        {CHANNEL: {"listen": "sometimes"}},
        {CHANNEL: {"replies": "nested"}},
        {CHANNEL: {"listen": "always", "unknown": True}},
    ],
)
def test_adapter_rejects_malformed_persisted_channel_modes(channel_modes):
    with pytest.raises(ValueError, match="channel_modes"):
        _adapter(extra={"channel_modes": channel_modes})


def test_live_setter_is_no_io_sparse_and_status_handles_dm():
    adapter = _adapter(reply_to_mode="off")

    adapter.apply_channel_policy(CHANNEL, "listen", "always")
    adapter.apply_channel_policy(CHANNEL, "replies", "hybrid")
    adapter.apply_channel_policy(CHANNEL, "listen", None)

    assert adapter._channel_modes == {CHANNEL: {"replies": "hybrid"}}
    assert adapter.channel_policy_status(CHANNEL, chat_type="dm") == {
        "applicable": False,
        "listen": {"effective": "always", "source": "direct-message"},
        "replies": {"effective": "flat", "source": "inherited"},
    }


def test_buzz_identity_candidates_include_equivalent_hex_and_npub():
    adapter = _adapter()
    source = SimpleNamespace(user_id=ADMIN_HEX.upper(), user_id_alt=ADMIN_NPUB.upper())

    assert adapter.normalize_source_identity_candidates(source) == (
        ADMIN_HEX,
        ADMIN_NPUB,
    )


def test_command_status_and_mutation_use_source_bound_actions():
    actions = SimpleNamespace(
        get_channel_policy_status=AsyncMock(
            return_value={
                "ok": True,
                "status": {
                    "applicable": True,
                    "listen": {"effective": "mentions", "source": "inherited"},
                    "replies": {"effective": "threaded", "source": "inherited"},
                },
            }
        ),
        set_channel_policy=AsyncMock(
            return_value={
                "ok": True,
                "status": {
                    "applicable": True,
                    "listen": {"effective": "always", "source": "explicit"},
                    "replies": {"effective": "threaded", "source": "inherited"},
                },
                "live_applied": True,
            }
        ),
    )
    invocation = SimpleNamespace(
        platform="buzz",
        chat_type="channel",
        platform_actions=actions,
    )

    status_text = asyncio.run(_buzz._handle_buzz_command("status", invocation))
    mutation_text = asyncio.run(
        _buzz._handle_buzz_command("listen always", invocation)
    )

    assert "Listening: mentions (inherited)" in status_text
    assert "Replies: threaded (inherited)" in status_text
    assert "Listening: always (explicit)" in mutation_text
    assert "active now" in mutation_text
    actions.get_channel_policy_status.assert_awaited_once_with()
    actions.set_channel_policy.assert_awaited_once_with("listen", "always")


def test_command_reports_persisted_restart_requirement_without_status():
    actions = SimpleNamespace(
        set_channel_policy=AsyncMock(
            return_value={
                "ok": True,
                "persisted": True,
                "live_applied": False,
                "restart_required": True,
            }
        )
    )
    invocation = SimpleNamespace(
        platform="buzz",
        chat_type="channel",
        platform_actions=actions,
    )

    text = asyncio.run(_buzz._handle_buzz_command("listen always", invocation))

    assert text == "Saved, but the live adapter could not be updated; restart required."
    actions.set_channel_policy.assert_awaited_once_with("listen", "always")


def test_command_dm_mutation_is_a_noop_and_malformed_returns_usage():
    actions = SimpleNamespace(
        get_channel_policy_status=AsyncMock(),
        set_channel_policy=AsyncMock(),
    )
    invocation = SimpleNamespace(
        platform="buzz",
        chat_type="dm",
        platform_actions=actions,
    )

    text = asyncio.run(_buzz._handle_buzz_command("listen always", invocation))
    usage = asyncio.run(_buzz._handle_buzz_command("listen sometimes", invocation))

    assert "do not apply to direct messages" in text
    assert "Usage:" in usage
    actions.set_channel_policy.assert_not_awaited()


def test_register_declares_contextual_mixed_reject_while_busy_command():
    ctx = MagicMock()

    _buzz.register(ctx)

    _args, kwargs = ctx.register_command.call_args
    assert _args[:2] == ("buzz", _buzz._handle_buzz_command)
    assert kwargs["argument_mode"] == "mixed"
    assert kwargs["with_context"] is True
    assert kwargs["access"] is _buzz._buzz_command_access
    assert kwargs["busy_policy"] == "reject"


@pytest.mark.asyncio
async def test_exact_buzz_token_bypasses_mentions_but_embedded_prose_does_not():
    adapter = _adapter()
    adapter._running = True
    adapter._self_pubkey = "b" * 64
    adapter._self_npub = _buzz.hex_to_npub(adapter._self_pubkey) or ""
    adapter._display_name = "Hermes"
    adapter.set_authorization_check(lambda *_args: True)
    handler = AsyncMock(return_value=None)
    adapter.set_message_handler(handler)
    state = adapter._new_channel_state("group")

    def event(event_id, content):
        return {
            "id": event_id,
            "pubkey": ADMIN_HEX,
            "content": content,
            "created_at": 1,
            "kind": 9,
            "tags": [["h", CHANNEL]],
        }

    await adapter._handle_event(CHANNEL, state, event("command", "/buzz status"))
    await adapter._handle_event(
        CHANNEL, state, event("prose", "please explain /buzz status")
    )

    assert handler.await_count == 1
    assert handler.await_args.args[0].text == "/buzz status"


def _raw_profile_config(admin=ADMIN_NPUB):
    return {
        "unrelated": {"keep": True},
        "plugins": {
            "entries": {
                "buzz-platform": {
                    "granted_capabilities": ["gateway.platform_actions"]
                }
            }
        },
        "gateway": {
            "platforms": {
                "buzz": {
                    "extra": {
                        "group_allow_admin_from": [admin] if admin else [],
                    }
                }
            }
        },
    }


def _runner_for_profiles(default_adapter, profile_adapters=None):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform("buzz"): default_adapter} if default_adapter else {}
    runner._profile_adapters = profile_adapters or {}
    runner._primary_profile_name = "default"
    return runner


def _restart_adapter(home, monkeypatch):
    monkeypatch.setattr("gateway.config.get_hermes_home", lambda: home)
    config = load_gateway_config()
    return BuzzAdapter(config.platforms[Platform("buzz")])


@pytest.mark.asyncio
async def test_real_status_service_returns_effective_buzz_policy(tmp_path, monkeypatch):
    home = tmp_path / "default"
    home.mkdir()
    raw = _raw_profile_config()
    raw["gateway"]["platforms"]["buzz"]["extra"]["channel_modes"] = {
        CHANNEL: {"listen": "always", "replies": "hybrid"}
    }
    (home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    adapter = _adapter(
        extra={"channel_modes": raw["gateway"]["platforms"]["buzz"]["extra"]["channel_modes"]}
    )
    adapter._running = True
    runner = _runner_for_profiles(adapter)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: home)

    result = await runner._get_plugin_channel_policy_status_action(
        plugin_id="buzz-platform",
        platform="buzz",
        routed_profile="default",
        channel_id=CHANNEL,
        thread_id=None,
        chat_type="group",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
    )

    assert result == {
        "ok": True,
        "status": {
            "applicable": True,
            "listen": {"effective": "always", "source": "explicit"},
            "replies": {"effective": "hybrid", "source": "explicit"},
        },
    }


@pytest.mark.asyncio
async def test_real_status_service_denies_routed_profile_without_capability(
    tmp_path, monkeypatch
):
    default_home = tmp_path / "default"
    team_home = tmp_path / "profiles" / "team-b"
    default_home.mkdir()
    team_home.mkdir(parents=True)
    (default_home / "config.yaml").write_text(yaml.safe_dump(_raw_profile_config()), encoding="utf-8")
    team_raw = _raw_profile_config()
    team_raw["plugins"]["entries"]["buzz-platform"]["granted_capabilities"] = []
    (team_home / "config.yaml").write_text(yaml.safe_dump(team_raw), encoding="utf-8")
    default_adapter = _adapter()
    team_adapter = _adapter()
    default_adapter._running = team_adapter._running = True
    runner = _runner_for_profiles(
        default_adapter,
        {"team-b": {Platform("buzz"): team_adapter}},
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: team_home if name == "team-b" else default_home,
    )

    result = await runner._get_plugin_channel_policy_status_action(
        plugin_id="buzz-platform",
        platform="buzz",
        routed_profile="team-b",
        channel_id=CHANNEL,
        thread_id=None,
        chat_type="group",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
    )

    assert result["ok"] is False
    assert result["error"] == "capability_not_granted"


@pytest.mark.asyncio
async def test_shared_primary_transport_uses_routed_authority_and_live_policy(
    tmp_path, monkeypatch
):
    default_home = tmp_path / "default"
    team_home = tmp_path / "profiles" / "team-b"
    default_home.mkdir()
    team_home.mkdir(parents=True)
    (default_home / "config.yaml").write_text(yaml.safe_dump(_raw_profile_config()), encoding="utf-8")
    team_modes = {CHANNEL: {"listen": "always"}}
    team_raw = _raw_profile_config()
    team_raw["gateway"]["platforms"]["buzz"]["extra"]["channel_modes"] = team_modes
    (team_home / "config.yaml").write_text(yaml.safe_dump(team_raw), encoding="utf-8")

    transport_adapter = _adapter()
    transport_adapter._running = True
    transport_adapter.hydrate_routed_profile_config(
        "team-b",
        PlatformConfig(enabled=False, extra={"channel_modes": team_modes}),
    )
    runner = _runner_for_profiles(transport_adapter)
    runner.config = SimpleNamespace(
        multiplex_profiles=True,
        profile_routes=[
            ProfileRoute(
                name="buzz-team",
                platform="buzz",
                profile="team-b",
                chat_id=CHANNEL,
            )
        ],
    )
    transport_adapter.gateway_runner = runner
    monkeypatch.setattr(
        "gateway.run._multiplex_profile_homes",
        lambda _config: [("default", default_home), ("team-b", team_home)],
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: team_home if name == "team-b" else default_home,
    )

    source = transport_adapter.build_source(
        chat_id=CHANNEL,
        chat_type="group",
        user_id=ADMIN_HEX.upper(),
        message_id="event-1",
    )
    assert source.profile == "team-b"
    assert runner._plugin_transport_profile(source) == "default"
    assert runner._plugin_source_identity_candidates(source) == (
        ADMIN_HEX,
        ADMIN_NPUB,
    )

    before = await runner._get_plugin_channel_policy_status_action(
        plugin_id="buzz-platform",
        platform="buzz",
        routed_profile="team-b",
        transport_profile="default",
        channel_id=CHANNEL,
        thread_id=None,
        chat_type="group",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
    )
    changed = await runner._apply_plugin_channel_policy_action(
        plugin_id="buzz-platform",
        platform="buzz",
        routed_profile="team-b",
        transport_profile="default",
        channel_id=CHANNEL,
        thread_id=None,
        chat_type="group",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
        policy="replies",
        value="hybrid",
    )

    assert before["status"]["listen"] == {
        "effective": "always",
        "source": "explicit",
    }
    assert changed["ok"] is True
    assert changed["live_applied"] is True
    persisted = yaml.safe_load((team_home / "config.yaml").read_text(encoding="utf-8"))
    assert persisted["gateway"]["platforms"]["buzz"]["extra"]["channel_modes"] == {
        CHANNEL: {"listen": "always", "replies": "hybrid"}
    }
    assert transport_adapter._channel_modes == {}
    assert transport_adapter._routed_channel_modes["team-b"] == {
        CHANNEL: {"listen": "always", "replies": "hybrid"}
    }
    assert (
        transport_adapter._resolve_outbound_reply_anchor(
            CHANNEL,
            "event-1",
            {
                "hermes_profile": "team-b",
                "buzz_trigger_placement": "top_level",
                "reply_to_message_id": "event-1",
            },
        )
        is None
    )


@pytest.mark.asyncio
async def test_shared_transport_restart_hydrates_routed_policy_before_intake(
    tmp_path, monkeypatch
):
    default_home = tmp_path / "default"
    team_home = tmp_path / "profiles" / "team-b"
    default_home.mkdir()
    team_home.mkdir(parents=True)
    (default_home / "config.yaml").write_text(yaml.safe_dump(_raw_profile_config()), encoding="utf-8")
    team_raw = _raw_profile_config()
    team_raw["gateway"]["platforms"]["buzz"]["enabled"] = True
    team_raw["gateway"]["platforms"]["buzz"]["extra"]["channel_modes"] = {
        CHANNEL: {"listen": "always"}
    }
    (team_home / "config.yaml").write_text(yaml.safe_dump(team_raw), encoding="utf-8")

    restarted_transport = _adapter()
    runner = _runner_for_profiles(restarted_transport)
    runner.config = SimpleNamespace(
        multiplex_profiles=True,
        profile_routes=[
            ProfileRoute(
                name="buzz-team",
                platform="buzz",
                profile="team-b",
                chat_id=CHANNEL,
            )
        ],
    )
    restarted_transport.gateway_runner = runner
    monkeypatch.setattr(
        "gateway.run._multiplex_profile_homes",
        lambda _config: [("default", default_home), ("team-b", team_home)],
    )

    runner._hydrate_shared_adapter_routed_profile_configs(
        restarted_transport,
        Platform("buzz"),
    )

    assert restarted_transport.channel_policy_status(
        CHANNEL,
        routed_profile="team-b",
    )["listen"] == {"effective": "always", "source": "explicit"}
    assert restarted_transport.is_connected is False

    handler = AsyncMock(return_value=None)
    restarted_transport._running = True
    restarted_transport.set_message_handler(handler)
    restarted_transport.set_authorization_check(lambda *_args: True)
    restarted_transport._self_pubkey = "b" * 64
    restarted_transport._self_npub = _buzz.hex_to_npub("b" * 64) or ""
    restarted_transport._display_name = "Hermes"
    monkeypatch.setattr(
        restarted_transport,
        "_resolve_user_name",
        AsyncMock(return_value="Admin"),
    )
    monkeypatch.setattr(
        restarted_transport,
        "send_reaction",
        AsyncMock(return_value=True),
    )
    state = restarted_transport._new_channel_state("group")

    await restarted_transport._handle_event(
        CHANNEL,
        state,
        {
            "id": "ambient-after-restart",
            "pubkey": ADMIN_HEX,
            "content": "ambient routed message",
            "created_at": 1,
            "kind": 9,
            "tags": [["h", CHANNEL]],
        },
    )
    await asyncio.gather(*list(restarted_transport._session_tasks.values()))

    handler.assert_awaited_once()
    assert handler.await_args.args[0].source.profile == "team-b"


@pytest.mark.asyncio
async def test_mutation_persists_sparse_routed_profile_then_updates_live_adapter(
    tmp_path, monkeypatch
):
    default_home = tmp_path / "default"
    team_home = tmp_path / "profiles" / "team-b"
    default_home.mkdir()
    team_home.mkdir(parents=True)
    default_config = _raw_profile_config()
    team_config = _raw_profile_config()
    (default_home / "config.yaml").write_text(yaml.safe_dump(default_config), encoding="utf-8")
    (team_home / "config.yaml").write_text(yaml.safe_dump(team_config), encoding="utf-8")
    default_before = (default_home / "config.yaml").read_bytes()

    default_adapter = _adapter()
    team_adapter = _adapter()
    default_adapter._running = team_adapter._running = True
    runner = _runner_for_profiles(
        default_adapter,
        {"team-b": {Platform("buzz"): team_adapter}},
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: team_home if name == "team-b" else default_home,
    )

    result = await runner._apply_plugin_channel_policy_action(
        plugin_id="buzz-platform",
        platform="buzz",
        routed_profile="team-b",
        channel_id=CHANNEL,
        thread_id=None,
        chat_type="group",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
        policy="listen",
        value="always",
    )

    assert result["ok"] is True
    assert result["live_applied"] is True
    persisted = yaml.safe_load((team_home / "config.yaml").read_text(encoding="utf-8"))
    assert persisted["gateway"]["platforms"]["buzz"]["extra"]["channel_modes"] == {
        CHANNEL: {"listen": "always"}
    }
    assert persisted["unrelated"] == {"keep": True}
    assert team_adapter._channel_modes == {CHANNEL: {"listen": "always"}}
    assert default_adapter._channel_modes == {}
    assert (default_home / "config.yaml").read_bytes() == default_before


@pytest.mark.asyncio
async def test_top_level_platform_modes_migrate_on_set_and_stay_reset_after_restart(
    tmp_path, monkeypatch
):
    home = tmp_path / "default"
    home.mkdir()
    raw = _raw_profile_config()
    raw["gateway"]["platforms"]["buzz"]["enabled"] = True
    top_modes = {CHANNEL: {"listen": "mentions", "replies": "hybrid"}}
    raw["platforms"] = {
        "buzz": {
            "extra": {
                "channel_modes": top_modes,
                "top_level_setting": "preserved",
            }
        }
    }
    config_path = home / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    adapter = _adapter(extra={"channel_modes": top_modes})
    adapter._running = True
    runner = _runner_for_profiles(adapter)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: home)

    set_result = await runner._apply_plugin_channel_policy_action(
        plugin_id="buzz-platform",
        platform="buzz",
        routed_profile="default",
        channel_id=CHANNEL,
        thread_id=None,
        chat_type="group",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
        policy="listen",
        value="always",
    )

    assert set_result["ok"] is True
    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert persisted["gateway"]["platforms"]["buzz"]["extra"]["channel_modes"] == {
        CHANNEL: {"listen": "always", "replies": "hybrid"}
    }
    assert "channel_modes" not in persisted["platforms"]["buzz"]["extra"]
    assert (
        persisted["platforms"]["buzz"]["extra"]["top_level_setting"]
        == "preserved"
    )
    restarted = _restart_adapter(home, monkeypatch)
    assert restarted.channel_policy_status(CHANNEL)["listen"] == {
        "effective": "always",
        "source": "explicit",
    }

    reset_result = await runner._apply_plugin_channel_policy_action(
        plugin_id="buzz-platform",
        platform="buzz",
        routed_profile="default",
        channel_id=CHANNEL,
        thread_id=None,
        chat_type="group",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
        policy="listen",
        value="reset",
    )

    assert reset_result["ok"] is True
    restarted_after_reset = _restart_adapter(home, monkeypatch)
    assert restarted_after_reset.channel_policy_status(CHANNEL) == {
        "applicable": True,
        "listen": {"effective": "mentions", "source": "inherited"},
        "replies": {"effective": "hybrid", "source": "explicit"},
    }


@pytest.mark.asyncio
async def test_duplicate_mode_nodes_migrate_to_canonical_and_reset_without_shadow(
    tmp_path, monkeypatch
):
    home = tmp_path / "default"
    home.mkdir()
    raw = _raw_profile_config()
    raw["gateway"]["platforms"]["buzz"]["enabled"] = True
    raw["gateway"]["platforms"]["buzz"]["extra"]["channel_modes"] = {
        CHANNEL: {"listen": "mentions"}
    }
    raw["platforms"] = {
        "buzz": {
            "extra": {
                "channel_modes": {CHANNEL: {"listen": "mentions", "replies": "threaded"}},
                "top_level_setting": "preserved",
            }
        }
    }
    raw["gateway"]["buzz"] = {
        "extra": {
            "channel_modes": {CHANNEL: {"listen": "always"}},
            "direct_setting": "preserved",
        }
    }
    config_path = home / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    adapter = _adapter(extra={"channel_modes": {CHANNEL: {"listen": "always"}}})
    adapter._running = True
    runner = _runner_for_profiles(adapter)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: home)

    set_result = await runner._apply_plugin_channel_policy_action(
        plugin_id="buzz-platform",
        platform="buzz",
        routed_profile="default",
        channel_id=CHANNEL,
        thread_id=None,
        chat_type="group",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
        policy="replies",
        value="hybrid",
    )

    assert set_result["ok"] is True
    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert persisted["gateway"]["platforms"]["buzz"]["extra"]["channel_modes"] == {
        CHANNEL: {"listen": "always", "replies": "hybrid"}
    }
    assert "channel_modes" not in persisted["platforms"]["buzz"]["extra"]
    assert "channel_modes" not in persisted["gateway"]["buzz"]["extra"]
    assert (
        persisted["platforms"]["buzz"]["extra"]["top_level_setting"]
        == "preserved"
    )
    assert persisted["gateway"]["buzz"]["extra"]["direct_setting"] == "preserved"
    restarted = _restart_adapter(home, monkeypatch)
    assert restarted.channel_policy_status(CHANNEL) == {
        "applicable": True,
        "listen": {"effective": "always", "source": "explicit"},
        "replies": {"effective": "hybrid", "source": "explicit"},
    }

    reset_result = await runner._apply_plugin_channel_policy_action(
        plugin_id="buzz-platform",
        platform="buzz",
        routed_profile="default",
        channel_id=CHANNEL,
        thread_id=None,
        chat_type="group",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
        policy="listen",
        value="reset",
    )

    assert reset_result["ok"] is True
    persisted_after_reset = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert persisted_after_reset["gateway"]["platforms"]["buzz"]["extra"][
        "channel_modes"
    ] == {CHANNEL: {"replies": "hybrid"}}
    restarted_after_reset = _restart_adapter(home, monkeypatch)
    assert restarted_after_reset.channel_policy_status(CHANNEL) == {
        "applicable": True,
        "listen": {"effective": "mentions", "source": "inherited"},
        "replies": {"effective": "hybrid", "source": "explicit"},
    }


@pytest.mark.asyncio
async def test_non_admin_or_write_failure_never_changes_live_state(tmp_path, monkeypatch):
    home = tmp_path / "default"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_text(yaml.safe_dump(_raw_profile_config()), encoding="utf-8")
    adapter = _adapter()
    adapter._running = True
    runner = _runner_for_profiles(adapter)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: home)

    denied = await runner._apply_plugin_channel_policy_action(
        plugin_id="buzz-platform",
        platform="buzz",
        routed_profile="default",
        channel_id=CHANNEL,
        thread_id=None,
        chat_type="group",
        source_identity_candidates=("a" * 64,),
        policy="listen",
        value="always",
    )
    assert denied["ok"] is False
    assert denied["error"] == "explicit_admin_required"
    assert adapter._channel_modes == {}

    before = config_path.read_bytes()
    monkeypatch.setattr(
        "hermes_cli.config.atomic_config_write",
        MagicMock(side_effect=OSError("disk full")),
    )
    failed = await runner._apply_plugin_channel_policy_action(
        plugin_id="buzz-platform",
        platform="buzz",
        routed_profile="default",
        channel_id=CHANNEL,
        thread_id=None,
        chat_type="group",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
        policy="listen",
        value="always",
    )
    assert failed["ok"] is False
    assert failed["error"] == "persistence_failed"
    assert adapter._channel_modes == {}
    assert config_path.read_bytes() == before


@pytest.mark.asyncio
async def test_host_rejects_dm_scope_before_persistence(tmp_path, monkeypatch):
    home = tmp_path / "default"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_text(yaml.safe_dump(_raw_profile_config()), encoding="utf-8")
    before = config_path.read_bytes()
    adapter = _adapter()
    adapter._running = True
    runner = _runner_for_profiles(adapter)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: home)

    result = await runner._apply_plugin_channel_policy_action(
        plugin_id="buzz-platform",
        platform="buzz",
        routed_profile="default",
        channel_id=CHANNEL,
        thread_id=None,
        chat_type="dm",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
        policy="listen",
        value="always",
    )

    assert result["ok"] is False
    assert result["error"] == "unsupported_context"
    assert config_path.read_bytes() == before
    assert adapter._channel_modes == {}


@pytest.mark.asyncio
async def test_active_custom_profile_writes_current_hermes_home(tmp_path, monkeypatch):
    custom_home = tmp_path / "custom-home"
    custom_home.mkdir()
    (custom_home / "config.yaml").write_text(yaml.safe_dump(_raw_profile_config()), encoding="utf-8")
    adapter = _adapter()
    adapter._running = True
    runner = _runner_for_profiles(adapter)
    runner._primary_profile_name = "custom"
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "custom")
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: custom_home)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        MagicMock(side_effect=AssertionError("must not resolve active custom home as a named profile")),
    )

    result = await runner._apply_plugin_channel_policy_action(
        plugin_id="buzz-platform",
        platform="buzz",
        routed_profile="custom",
        channel_id=CHANNEL,
        thread_id=None,
        chat_type="group",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
        policy="replies",
        value="hybrid",
    )

    assert result["ok"] is True
    persisted = yaml.safe_load((custom_home / "config.yaml").read_text(encoding="utf-8"))
    assert persisted["gateway"]["platforms"]["buzz"]["extra"]["channel_modes"] == {
        CHANNEL: {"replies": "hybrid"}
    }
    assert adapter._channel_modes == {CHANNEL: {"replies": "hybrid"}}


@pytest.mark.asyncio
async def test_reset_prunes_only_selected_property_and_empty_channel(tmp_path, monkeypatch):
    home = tmp_path / "default"
    home.mkdir()
    raw = _raw_profile_config()
    raw["gateway"]["platforms"]["buzz"]["extra"]["channel_modes"] = {
        CHANNEL: {"listen": "always", "replies": "hybrid"},
        OTHER_CHANNEL: {"listen": "mentions"},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    adapter = _adapter(extra={"channel_modes": raw["gateway"]["platforms"]["buzz"]["extra"]["channel_modes"]})
    adapter._running = True
    runner = _runner_for_profiles(adapter)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: home)

    first = await runner._apply_plugin_channel_policy_action(
        plugin_id="buzz-platform", platform="buzz", routed_profile="default",
        channel_id=CHANNEL, thread_id=None, chat_type="group",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
        policy="listen", value="reset",
    )
    second = await runner._apply_plugin_channel_policy_action(
        plugin_id="buzz-platform", platform="buzz", routed_profile="default",
        channel_id=CHANNEL, thread_id=None, chat_type="group",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
        policy="replies", value="reset",
    )

    assert first["ok"] is second["ok"] is True
    persisted = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert persisted["gateway"]["platforms"]["buzz"]["extra"]["channel_modes"] == {
        OTHER_CHANNEL: {"listen": "mentions"}
    }
    assert adapter._channel_modes == {OTHER_CHANNEL: {"listen": "mentions"}}


@pytest.mark.asyncio
async def test_adapter_replacement_after_persist_receives_live_update(tmp_path, monkeypatch):
    home = tmp_path / "default"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump(_raw_profile_config()), encoding="utf-8")
    original_adapter = _adapter()
    replacement_adapter = _adapter()
    original_adapter._running = replacement_adapter._running = True
    runner = _runner_for_profiles(original_adapter)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: home)
    persist = runner._persist_plugin_channel_policy

    def persist_then_replace(**kwargs):
        result = persist(**kwargs)
        runner.adapters[Platform("buzz")] = replacement_adapter
        return result

    monkeypatch.setattr(runner, "_persist_plugin_channel_policy", persist_then_replace)

    result = await runner._apply_plugin_channel_policy_action(
        plugin_id="buzz-platform", platform="buzz", routed_profile="default",
        channel_id=CHANNEL, thread_id=None, chat_type="group",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
        policy="replies", value="hybrid",
    )

    assert result["live_applied"] is True
    assert original_adapter._channel_modes == {}
    assert replacement_adapter._channel_modes == {CHANNEL: {"replies": "hybrid"}}


@pytest.mark.asyncio
async def test_live_setter_failure_reports_persisted_restart_required(tmp_path, monkeypatch):
    home = tmp_path / "default"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump(_raw_profile_config()), encoding="utf-8")
    adapter = _adapter()
    adapter._running = True
    runner = _runner_for_profiles(adapter)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: home)
    monkeypatch.setattr(
        adapter,
        "apply_channel_policy",
        MagicMock(side_effect=RuntimeError("live setter failed")),
    )

    result = await runner._apply_plugin_channel_policy_action(
        plugin_id="buzz-platform", platform="buzz", routed_profile="default",
        channel_id=CHANNEL, thread_id=None, chat_type="group",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
        policy="listen", value="always",
    )

    assert result["ok"] is True
    assert result["persisted"] is True
    assert result["live_applied"] is False
    assert result["restart_required"] is True
    persisted = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert persisted["gateway"]["platforms"]["buzz"]["extra"]["channel_modes"] == {
        CHANNEL: {"listen": "always"}
    }


@pytest.mark.asyncio
async def test_reasserting_canonical_policy_skips_write_but_repairs_live_state(
    tmp_path, monkeypatch
):
    home = tmp_path / "default"
    home.mkdir()
    raw = _raw_profile_config()
    raw["gateway"]["platforms"]["buzz"]["extra"]["channel_modes"] = {
        CHANNEL: {"listen": "always"}
    }
    config_path = home / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    before = config_path.read_bytes()
    adapter = _adapter()
    adapter._running = True
    runner = _runner_for_profiles(adapter)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: home)
    write = MagicMock()
    monkeypatch.setattr("hermes_cli.config.atomic_config_write", write)

    result = await runner._apply_plugin_channel_policy_action(
        plugin_id="buzz-platform",
        platform="buzz",
        routed_profile="default",
        channel_id=CHANNEL,
        thread_id=None,
        chat_type="group",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
        policy="listen",
        value="always",
    )

    assert result["ok"] is True
    assert result["live_applied"] is True
    write.assert_not_called()
    assert config_path.read_bytes() == before
    assert adapter._channel_modes == {CHANNEL: {"listen": "always"}}


@pytest.mark.asyncio
async def test_routed_capability_recheck_reports_capability_error(tmp_path, monkeypatch):
    home = tmp_path / "default"
    home.mkdir()
    raw = _raw_profile_config()
    raw["plugins"]["entries"]["buzz-platform"]["granted_capabilities"] = []
    config_path = home / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    before = config_path.read_bytes()
    adapter = _adapter()
    adapter._running = True
    runner = _runner_for_profiles(adapter)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: home)

    result = await runner._apply_plugin_channel_policy_action(
        plugin_id="buzz-platform", platform="buzz", routed_profile="default",
        channel_id=CHANNEL, thread_id=None, chat_type="group",
        source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
        policy="listen", value="always",
    )

    assert result["ok"] is False
    assert result["error"] == "capability_not_granted"
    assert config_path.read_bytes() == before
    assert adapter._channel_modes == {}


@pytest.mark.asyncio
async def test_concurrent_mutations_merge_without_lost_updates(tmp_path, monkeypatch):
    home = tmp_path / "default"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump(_raw_profile_config()), encoding="utf-8")
    adapter = _adapter()
    adapter._running = True
    runner = _runner_for_profiles(adapter)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: home)

    async def mutate(channel_id, policy, value):
        return await runner._apply_plugin_channel_policy_action(
            plugin_id="buzz-platform", platform="buzz", routed_profile="default",
            channel_id=channel_id, thread_id=None, chat_type="group",
            source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
            policy=policy, value=value,
        )

    listen, replies = await asyncio.gather(
        mutate(CHANNEL, "listen", "always"),
        mutate(OTHER_CHANNEL, "replies", "hybrid"),
    )

    assert listen["ok"] is replies["ok"] is True
    persisted = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert persisted["gateway"]["platforms"]["buzz"]["extra"]["channel_modes"] == {
        CHANNEL: {"listen": "always"},
        OTHER_CHANNEL: {"replies": "hybrid"},
    }
    assert adapter._channel_modes == {
        CHANNEL: {"listen": "always"},
        OTHER_CHANNEL: {"replies": "hybrid"},
    }


@pytest.mark.asyncio
async def test_same_property_operations_serialize_persistence_through_live_apply(
    tmp_path, monkeypatch
):
    home = tmp_path / "default"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_text(yaml.safe_dump(_raw_profile_config()), encoding="utf-8")
    adapter = _adapter()
    adapter._running = True
    runner = _runner_for_profiles(adapter)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: home)
    persist = runner._persist_plugin_channel_policy
    first_persisted = threading.Event()
    release_first = threading.Event()
    persisted_values: list[str] = []

    def pause_first_after_persistence(**kwargs):
        result = persist(**kwargs)
        persisted_values.append(kwargs["value"])
        if kwargs["value"] == "always":
            first_persisted.set()
            if not release_first.wait(timeout=5):
                raise TimeoutError("test did not release the first live apply")
        return result

    monkeypatch.setattr(
        runner, "_persist_plugin_channel_policy", pause_first_after_persistence
    )

    async def mutate(value):
        return await runner._apply_plugin_channel_policy_action(
            plugin_id="buzz-platform",
            platform="buzz",
            routed_profile="default",
            channel_id=CHANNEL,
            thread_id=None,
            chat_type="group",
            source_identity_candidates=(ADMIN_HEX, ADMIN_NPUB),
            policy="listen",
            value=value,
        )

    first = asyncio.create_task(mutate("always"))
    assert await asyncio.to_thread(first_persisted.wait, 5)
    second = asyncio.create_task(mutate("mentions"))
    try:
        await asyncio.sleep(0.05)
        assert persisted_values == ["always"]
        assert not second.done()
    finally:
        release_first.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert first_result["ok"] is second_result["ok"] is True
    assert persisted_values == ["always", "mentions"]
    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert persisted["gateway"]["platforms"]["buzz"]["extra"]["channel_modes"] == {
        CHANNEL: {"listen": "mentions"}
    }
    assert adapter._channel_modes == {CHANNEL: {"listen": "mentions"}}
