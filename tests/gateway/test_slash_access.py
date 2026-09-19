"""Unit tests for gateway.slash_access — per-platform slash command access control.

Tests the pure policy resolver (no gateway plumbing). Integration tests that
exercise the dispatch site live in test_slash_access_dispatch.py.
"""
from __future__ import annotations

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from gateway.slash_access import (
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


    def test_id_coercion_ints_become_strings(self):
        # YAML often loads numeric IDs as ints; we stringify on ingest.
        p = policy_from_extra({"allow_admin_from": [12345, 67890]}, "dm")
        assert p.admin_user_ids == frozenset({"12345", "67890"})
        assert p.is_admin("12345") is True
        assert p.is_admin(12345) is True  # is_admin also stringifies


    def test_command_coercion_strips_leading_slash_and_lowercases(self):
        p = policy_from_extra(
            {
                "allow_admin_from": ["111"],
                "user_allowed_commands": ["/Status", "MODEL", "/help"],
            },
            "dm",
        )
        assert p.user_allowed_commands == frozenset({"status", "model", "help"})


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


class TestBlankChatType:
    """Blank/None chat_type (relay frames, restored rows) must not land in an
    ungated scope: resolve to whichever scope is gated, group on tie."""

    def _cfg(self, extra):
        return GatewayConfig(
            platforms={Platform.DISCORD: PlatformConfig(enabled=True, extra=extra)}
        )

    def _blank_src(self, chat_type):
        return SessionSource(
            platform=Platform.DISCORD, chat_id="A", chat_type=chat_type, user_id="999"
        )

    def test_blank_falls_into_gated_dm_scope(self):
        # DM-gated install: blank previously resolved to group, whose unset
        # admin list disabled gating entirely.
        cfg = self._cfg({"allow_admin_from": ["111"], "user_allowed_commands": ["status"]})
        for chat_type in ("", None, "   "):
            p = policy_for_source(cfg, self._blank_src(chat_type))
            assert p.enabled is True, repr(chat_type)
            assert p.is_admin("111") is True
            assert p.can_run("999", "stop") is False
            assert p.can_run("999", "status") is True

    def test_blank_keeps_group_scope_when_only_group_gated(self):
        cfg = self._cfg({"group_allow_admin_from": ["222"]})
        p = policy_for_source(cfg, self._blank_src(""))
        assert p.enabled is True
        assert p.is_admin("222") is True
        assert p.can_run("999", "stop") is False

    def test_blank_keeps_group_scope_on_tie(self):
        # Both scopes gated: historical blank -> group resolution is preserved.
        cfg = self._cfg({
            "allow_admin_from": ["111"],
            "group_allow_admin_from": ["222"],
        })
        p = policy_for_source(cfg, self._blank_src(None))
        assert p.enabled is True
        assert p.is_admin("222") is True
        assert p.is_admin("111") is False

    def test_blank_disabled_when_neither_scope_gated(self):
        cfg = self._cfg({})
        p = policy_for_source(cfg, self._blank_src(""))
        assert p.enabled is False
        assert p.can_run("999", "stop") is True

    def test_relay_wire_blank_chat_type_is_gated(self):
        """E2e through the real producer: _event_from_wire honors a blank
        chat_type from the wire verbatim (src.get default only covers a MISSING
        key), so the policy must gate the blank source it produces."""
        from gateway.relay.ws_transport import _event_from_wire

        cfg = self._cfg({"allow_admin_from": ["111"]})
        for blank in ("", None):
            evt = _event_from_wire({
                "text": "/stop",
                "source": {"platform": "discord", "chat_id": "c1",
                           "chat_type": blank, "user_id": "999"},
            })
            assert not evt.source.chat_type  # producer really did emit blank
            p = policy_for_source(cfg, evt.source)
            assert p.enabled is True, repr(blank)
            assert p.can_run("999", "stop") is False

