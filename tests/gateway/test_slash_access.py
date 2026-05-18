"""Unit tests for gateway.slash_access — per-platform slash command access control.

Tests the pure policy resolver (no gateway plumbing). Integration tests that
exercise the dispatch site live in test_slash_access_dispatch.py.
"""
from __future__ import annotations

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from gateway.slash_access import (
    SlashAccessPolicy,
    policy_for_source,
    policy_from_extra,
)


# ---------------------------------------------------------------------------
# policy_from_extra — input normalization + scope resolution
# ---------------------------------------------------------------------------


class TestPolicyFromExtra:
    def test_empty_extra_is_disabled(self):
        p = policy_from_extra({}, "dm")
        assert p.enabled is False
        assert p.admin_user_ids == frozenset()
        assert p.user_allowed_commands == frozenset()

    def test_disabled_policy_treats_anyone_as_admin(self):
        # When gating is off, downstream code uses is_admin/can_run uniformly.
        # Both must short-circuit to True so existing behavior is preserved.
        p = policy_from_extra({}, "dm")
        assert p.is_admin("anyone") is True
        assert p.can_run("anyone", "stop") is True

    def test_dm_admin_list_only(self):
        p = policy_from_extra({"allow_admin_from": ["111", "222"]}, "dm")
        assert p.enabled is True
        assert p.admin_user_ids == frozenset({"111", "222"})
        assert p.user_allowed_commands == frozenset()

    def test_admin_runs_anything(self):
        p = policy_from_extra(
            {"allow_admin_from": [111], "user_allowed_commands": ["help"]},
            "dm",
        )
        assert p.is_admin("111") is True
        assert p.can_run("111", "stop") is True
        assert p.can_run("111", "kanban") is True

    def test_non_admin_runs_only_listed_commands(self):
        p = policy_from_extra(
            {
                "allow_admin_from": ["111"],
                "user_allowed_commands": ["status", "model"],
            },
            "dm",
        )
        assert p.is_admin("999") is False
        assert p.can_run("999", "status") is True
        assert p.can_run("999", "model") is True
        assert p.can_run("999", "stop") is False
        assert p.can_run("999", "kanban") is False

    def test_always_allowed_floor_for_non_admin(self):
        # /help and /whoami always reachable so users can see what they can do.
        p = policy_from_extra(
            {"allow_admin_from": ["111"], "user_allowed_commands": []},
            "dm",
        )
        assert p.can_run("999", "help") is True
        assert p.can_run("999", "whoami") is True
        assert p.can_run("999", "stop") is False

    def test_unknown_user_id_blocked(self):
        # Empty/None user_id → no admin status, no command access (except floor).
        p = policy_from_extra(
            {"allow_admin_from": ["111"], "user_allowed_commands": ["status"]},
            "dm",
        )
        assert p.is_admin(None) is False
        assert p.can_run(None, "status") is True  # listed command works
        assert p.can_run(None, "stop") is False
        assert p.can_run("", "stop") is False

    def test_id_coercion_ints_become_strings(self):
        # YAML often loads numeric IDs as ints; we stringify on ingest.
        p = policy_from_extra({"allow_admin_from": [12345, 67890]}, "dm")
        assert p.admin_user_ids == frozenset({"12345", "67890"})
        assert p.is_admin("12345") is True
        assert p.is_admin(12345) is True  # is_admin also stringifies

    def test_id_coercion_csv_string(self):
        p = policy_from_extra({"allow_admin_from": "111, 222 ,333"}, "dm")
        assert p.admin_user_ids == frozenset({"111", "222", "333"})

    def test_command_coercion_strips_leading_slash_and_lowercases(self):
        p = policy_from_extra(
            {
                "allow_admin_from": ["111"],
                "user_allowed_commands": ["/Status", "MODEL", "/help"],
            },
            "dm",
        )
        assert p.user_allowed_commands == frozenset({"status", "model", "help"})

    def test_command_coercion_csv_string(self):
        p = policy_from_extra(
            {
                "allow_admin_from": ["111"],
                "user_allowed_commands": "status, model , /help",
            },
            "dm",
        )
        assert p.user_allowed_commands == frozenset({"status", "model", "help"})

    def test_group_scope_uses_group_keys(self):
        extra = {
            "allow_admin_from": ["111"],          # DM admins
            "user_allowed_commands": ["status"],  # DM commands
            "group_allow_admin_from": ["222"],
            "group_user_allowed_commands": ["help"],
        }
        dm = policy_from_extra(extra, "dm")
        gp = policy_from_extra(extra, "group")
        assert dm.admin_user_ids == frozenset({"111"})
        assert gp.admin_user_ids == frozenset({"222"})
        assert dm.user_allowed_commands == frozenset({"status"})
        # group's user_allowed_commands does not leak into DM's allowed list
        # except via the explicit fallback rule (only when DM list is unset).
        assert "help" in gp.user_allowed_commands

    def test_dm_falls_back_to_group_user_commands_when_dm_unset(self):
        # Common case: operator wants the same command set DM and group;
        # they should only have to list it once on the group keys.
        extra = {
            "allow_admin_from": ["111"],
            "group_user_allowed_commands": ["status", "model"],
        }
        dm = policy_from_extra(extra, "dm")
        assert dm.user_allowed_commands == frozenset({"status", "model"})

    def test_dm_admin_does_not_imply_group_admin(self):
        # Admin lists are scope-specific. DM admin must not auto-promote in groups.
        extra = {"allow_admin_from": ["111"]}
        dm = policy_from_extra(extra, "dm")
        gp = policy_from_extra(extra, "group")
        assert dm.is_admin("111") is True
        # Group has no admin list set → gating disabled in groups → "111"
        # gets unrestricted access, but that's the backward-compat fallback,
        # not implicit admin promotion. The distinction matters when the
        # group DOES have an admin list set:
        extra2 = {
            "allow_admin_from": ["111"],
            "group_allow_admin_from": ["222"],
        }
        gp2 = policy_from_extra(extra2, "group")
        assert gp2.is_admin("111") is False
        assert gp2.is_admin("222") is True


# ---------------------------------------------------------------------------
# policy_for_source — wires GatewayConfig + SessionSource together
# ---------------------------------------------------------------------------


class TestPolicyForSource:
    def test_no_config_returns_disabled(self):
        p = policy_for_source(None, None)
        assert p.enabled is False
        assert p.is_admin("anyone") is True

    def test_no_platform_config_returns_disabled(self):
        cfg = GatewayConfig(platforms={})
        src = SessionSource(
            platform=Platform.DISCORD, chat_id="42", chat_type="dm", user_id="7"
        )
        p = policy_for_source(cfg, src)
        assert p.enabled is False

    def test_dm_chat_type_resolves_to_dm_scope(self):
        cfg = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    extra={
                        "allow_admin_from": ["111"],
                        "user_allowed_commands": ["status"],
                        "group_allow_admin_from": ["222"],
                        "group_user_allowed_commands": ["help"],
                    },
                )
            }
        )
        dm_src = SessionSource(
            platform=Platform.DISCORD, chat_id="A", chat_type="dm", user_id="111"
        )
        p = policy_for_source(cfg, dm_src)
        assert p.is_admin("111") is True
        assert p.can_run("999", "status") is True
        assert p.can_run("999", "help") is True  # always-allowed floor
        assert p.can_run("999", "kanban") is False

    def test_group_chat_type_resolves_to_group_scope(self):
        cfg = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    extra={
                        "allow_admin_from": ["111"],
                        "user_allowed_commands": ["status"],
                        "group_allow_admin_from": ["222"],
                        "group_user_allowed_commands": ["help"],
                    },
                )
            }
        )
        grp_src = SessionSource(
            platform=Platform.DISCORD, chat_id="G", chat_type="group", user_id="222"
        )
        p = policy_for_source(cfg, grp_src)
        assert p.is_admin("222") is True
        assert p.is_admin("111") is False  # DM admin, not group admin
        # In group scope, the only listed user command is "help"; "status"
        # is not in the group list and should be denied for non-admins.
        assert p.can_run("999", "help") is True
        assert p.can_run("999", "status") is False

    def test_channel_thread_chat_types_treated_as_group_scope(self):
        # Discord channels and threads are group-scoped, not DM-scoped.
        cfg = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    extra={
                        "allow_admin_from": ["111"],
                        "group_allow_admin_from": ["222"],
                    },
                )
            }
        )
        for ct in ("group", "channel", "thread", "supergroup"):
            src = SessionSource(
                platform=Platform.DISCORD, chat_id="X", chat_type=ct, user_id="222"
            )
            p = policy_for_source(cfg, src)
            assert p.is_admin("222") is True, f"chat_type={ct} should map to group scope"
            assert p.is_admin("111") is False, f"chat_type={ct} should not see DM admins"

    def test_no_admin_list_for_dm_means_unrestricted_in_dm(self):
        # Group has admin list, DM does not → DM gating disabled, group active.
        cfg = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    extra={"group_allow_admin_from": ["222"]},
                )
            }
        )
        dm_src = SessionSource(
            platform=Platform.DISCORD, chat_id="A", chat_type="dm", user_id="999"
        )
        grp_src = SessionSource(
            platform=Platform.DISCORD, chat_id="G", chat_type="group", user_id="999"
        )
        dm_p = policy_for_source(cfg, dm_src)
        grp_p = policy_for_source(cfg, grp_src)
        assert dm_p.enabled is False
        assert dm_p.can_run("999", "stop") is True  # backward compat
        assert grp_p.enabled is True
        assert grp_p.can_run("999", "stop") is False  # gated

    def test_per_platform_isolation(self):
        # Discord has gating, Telegram doesn't → Telegram is unaffected.
        cfg = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    extra={"allow_admin_from": ["111"]},
                ),
                Platform.TELEGRAM: PlatformConfig(enabled=True, extra={}),
            }
        )
        tg_src = SessionSource(
            platform=Platform.TELEGRAM, chat_id="T", chat_type="dm", user_id="999"
        )
        p = policy_for_source(cfg, tg_src)
        assert p.enabled is False
        assert p.can_run("999", "stop") is True
