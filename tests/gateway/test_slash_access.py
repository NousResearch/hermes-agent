"""Unit tests for gateway.slash_access — per-platform slash command access control.

Tests the pure policy resolver (no gateway plumbing). Integration tests that
exercise the dispatch site live in test_slash_access_dispatch.py.
"""
from __future__ import annotations

import json

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from gateway.slash_access import (
    identity_candidates,
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



# ---------------------------------------------------------------------------
# identity_candidates — a WhatsApp group sender arrives as a LID, but the
# operator's admin list holds phone numbers (live regression, 2026-08-14).
# ---------------------------------------------------------------------------


class TestIdentityCandidates:
    @staticmethod
    def _paired_home(tmp_path, monkeypatch):
        """A gateway home carrying the bridge's LID->phone mapping."""
        home = tmp_path / "hermes-home"
        session_dir = home / "platforms" / "whatsapp" / "session"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "lid-mapping-160868067209361.json").write_text(
            json.dumps("972547401660@s.whatsapp.net"), encoding="utf-8"
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        return home

    def test_whatsapp_group_lid_expands_to_the_configured_phone_admin(
        self, tmp_path, monkeypatch
    ):
        """The exact failure: a configured admin refused in their own group.

        WhatsApp hands a group message's sender over as ``<lid>@lid``. The
        operator wrote a phone number into ``group_allow_admin_from``, so the
        raw comparison missed and /new came back
        "admin-only here ... ask an admin to add you to allow_admin_from" --
        addressed to the admin.
        """
        self._paired_home(tmp_path, monkeypatch)
        cfg = GatewayConfig(
            platforms={
                Platform.WHATSAPP: PlatformConfig(
                    enabled=True,
                    extra={"group_allow_admin_from": ["972547401660"]},
                )
            }
        )
        source = SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="120363428948689789@g.us",
            chat_type="group",
            user_id="160868067209361@lid",
        )
        policy = policy_for_source(cfg, source)
        assert policy.enabled is True
        # The raw transport id is NOT in the admin list...
        assert policy.is_admin(source.user_id) is False
        # ...but one of the sender's identities is, and that is what gates read.
        assert any(policy.is_admin(uid) for uid in identity_candidates(source))
        assert any(
            policy.can_run(uid, "new") for uid in identity_candidates(source)
        )

    def test_a_stranger_in_the_same_group_is_still_denied(
        self, tmp_path, monkeypatch
    ):
        """Alias expansion must widen recognition, never authority."""
        self._paired_home(tmp_path, monkeypatch)
        cfg = GatewayConfig(
            platforms={
                Platform.WHATSAPP: PlatformConfig(
                    enabled=True,
                    extra={"group_allow_admin_from": ["972547401660"]},
                )
            }
        )
        source = SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="120363428948689789@g.us",
            chat_type="group",
            user_id="112790773731504@lid",  # unmapped: some other member
        )
        policy = policy_for_source(cfg, source)
        assert not any(policy.is_admin(uid) for uid in identity_candidates(source))
        assert not any(
            policy.can_run(uid, "new") for uid in identity_candidates(source)
        )

    def test_non_whatsapp_platforms_are_untouched(self):
        source = SessionSource(
            platform=Platform.DISCORD, chat_id="G", chat_type="group", user_id="111"
        )
        assert identity_candidates(source) == ("111",)

    def test_a_sourceless_or_anonymous_sender_yields_nothing(self):
        source = SessionSource(
            platform=Platform.WHATSAPP, chat_id="G", chat_type="group", user_id=""
        )
        assert identity_candidates(source) == ()
