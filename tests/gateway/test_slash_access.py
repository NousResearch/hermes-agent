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


    def test_command_coercion_folds_underscores_to_hyphens(self):
        # Telegram's command menu surfaces a hyphenated command as /my_cmd and
        # sends that form back, so operators reasonably write the underscored
        # spelling in their allowlist. Both spellings name the same command.
        p = policy_from_extra(
            {
                "allow_admin_from": ["111"],
                "user_allowed_commands": ["my_cmd", "/Reload_MCP"],
            },
            "dm",
        )
        assert p.user_allowed_commands == frozenset({"my-cmd", "reload-mcp"})


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
# can_run — underscore/hyphen equivalence
# ---------------------------------------------------------------------------


class TestCanRunSeparatorFolding:
    """``_`` and ``-`` are the same character to slash dispatch, so they must
    be the same character to the allowlist.

    Plugin dispatch normalizes the typed name (``my_cmd``) to the registered
    one (``my-cmd``) before looking a handler up, and the gate runs on that
    normalized name. Built-ins go further and register both spellings as
    aliases of one command (``reload-mcp`` / ``reload_mcp``). An operator
    cannot be expected to know which spelling the gate will see, so the
    policy folds both sides of the comparison.
    """

    @staticmethod
    def _policy(listed):
        return policy_from_extra(
            {"allow_admin_from": ["111"], "user_allowed_commands": listed}, "dm"
        )


    def test_underscored_listing_admits_the_hyphenated_command(self):
        # The operator wrote the form Telegram's menu shows.
        p = self._policy(["my_cmd"])
        assert p.can_run("999", "my-cmd") is True


    def test_hyphenated_listing_admits_the_underscored_command(self):
        # The mirror image — a quick command or plugin registered with an
        # underscore, gated on the raw typed name.
        p = self._policy(["my-cmd"])
        assert p.can_run("999", "my_cmd") is True


    def test_both_spellings_keep_working_for_the_exact_form(self):
        for listed in (["my_cmd"], ["my-cmd"]):
            p = self._policy(listed)
            assert p.can_run("999", listed[0]) is True


    def test_builtin_alias_pair_is_allowed_under_either_spelling(self):
        # ``reload-mcp`` and ``reload_mcp`` are aliases of one built-in, but
        # the gate only ever sees the canonical name.
        for listed in ("reload_mcp", "reload-mcp"):
            p = self._policy([listed])
            assert p.can_run("999", "reload-mcp") is True
            assert p.can_run("999", "reload_mcp") is True


    def test_folding_does_not_admit_unrelated_commands(self):
        p = self._policy(["my_cmd"])
        assert p.can_run("999", "mycmd") is False
        assert p.can_run("999", "my cmd") is False
        assert p.can_run("999", "other-cmd") is False
        assert p.can_run("999", "my-cmd-2") is False


    def test_non_allowlisted_user_stays_denied_for_either_spelling(self):
        # Nothing here is a way in: an empty allowlist denies both forms.
        p = self._policy([])
        assert p.can_run("999", "my-cmd") is False
        assert p.can_run("999", "my_cmd") is False


    def test_admin_and_disabled_gating_are_unaffected(self):
        p = self._policy([])
        assert p.can_run("111", "my_cmd") is True  # admin
        off = policy_from_extra({}, "dm")
        assert off.can_run("999", "my_cmd") is True  # gating not configured


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

