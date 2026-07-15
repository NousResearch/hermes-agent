"""Per-chat toolset isolation via ``channel_overrides`` (WhatsApp principal ACL).

Requirements driving these tests:

- Group chats can be pinned to run with NO tools via a ``group:*`` wildcard
  override — even when the sender is an admin/allowlisted user (the lookup is
  chat-scoped, never sender-scoped).
- A specific DM can be restricted to conversation-only (``enabled_toolsets: []``)
  while another DM inherits the full platform toolsets.
- WhatsApp config keys accept bare phone numbers, ``+``-prefixed numbers, full
  JIDs and LIDs interchangeably — the lookup resolves aliases through the
  bridge's ``lid-mapping-*.json`` files, mirroring session-key canonicalisation.
- Fail-closed: if override resolution blows up, the effective toolsets are
  empty (no tools) rather than falling back to the full platform default.
- Compatibility: configs without the new field behave exactly as before.

The policy is resolved from chat identity BEFORE the agent is created/reused
(``enabled_toolsets`` is part of the agent-cache signature and sessions are
per-chat), so a session never flips toolsets mid-conversation and prompt
caching is preserved.
"""

import json

import pytest

from gateway.config import (
    ChannelOverride,
    GatewayConfig,
    Platform,
    PlatformConfig,
)
from gateway.run import _get_channel_override, _resolve_channel_toolsets
from gateway.session import SessionSource


ADMIN_PHONE = "5519992283338"
ADMIN_LID = "239019292106976"
CLINIC_PHONE = "5519981153040"
GROUP_JID = "120363999999999999@g.us"

FULL_TOOLSETS = ["hermes-whatsapp"]


def _whatsapp_config(overrides):
    return GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(
                enabled=True,
                channel_overrides=overrides,
            ),
        },
    )


def _source(chat_id, *, chat_type="dm", user_id="someone"):
    return SessionSource(
        platform=Platform.WHATSAPP,
        chat_id=chat_id,
        user_id=user_id,
        user_name="tester",
        chat_type=chat_type,
    )


def _write_lid_mappings(tmp_path, monkeypatch):
    """Bridge session dir mapping ADMIN_PHONE <-> ADMIN_LID, both directions."""
    tmp_home = tmp_path / "hermes-home"
    mapping_dir = tmp_home / "platforms" / "whatsapp" / "session"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    (mapping_dir / f"lid-mapping-{ADMIN_PHONE}.json").write_text(
        json.dumps(f"{ADMIN_LID}@lid"), encoding="utf-8"
    )
    (mapping_dir / f"lid-mapping-{ADMIN_LID}.json").write_text(
        json.dumps(f"{ADMIN_PHONE}@s.whatsapp.net"), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))


# ---------------------------------------------------------------------------
# ChannelOverride config surface
# ---------------------------------------------------------------------------


class TestChannelOverrideToolsetsConfig:
    def test_default_is_inherit(self):
        """No ``enabled_toolsets`` key → None → inherit platform default."""
        assert ChannelOverride().enabled_toolsets is None
        assert ChannelOverride.from_dict({"model": "m"}).enabled_toolsets is None

    def test_from_dict_parses_list(self):
        ov = ChannelOverride.from_dict({"enabled_toolsets": ["memory", "web"]})
        assert ov.enabled_toolsets == ["memory", "web"]

    def test_from_dict_empty_list_is_explicit_no_tools(self):
        """``enabled_toolsets: []`` must stay [] (no tools), not collapse to None."""
        ov = ChannelOverride.from_dict({"enabled_toolsets": []})
        assert ov.enabled_toolsets == []

    def test_from_dict_string_becomes_single_item_list(self):
        ov = ChannelOverride.from_dict({"enabled_toolsets": "memory"})
        assert ov.enabled_toolsets == ["memory"]

    def test_from_dict_malformed_value_fails_closed(self):
        """A truthy non-list/str value is unusable → no tools, never full access."""
        ov = ChannelOverride.from_dict({"enabled_toolsets": {"oops": True}})
        assert ov.enabled_toolsets == []

    def test_to_dict_round_trip(self):
        ov = ChannelOverride(enabled_toolsets=[])
        assert ChannelOverride.from_dict(ov.to_dict()).enabled_toolsets == []
        ov2 = ChannelOverride(enabled_toolsets=["memory"])
        assert ChannelOverride.from_dict(ov2.to_dict()).enabled_toolsets == ["memory"]

    def test_to_dict_omits_unset_field_for_compat(self):
        """Legacy overrides (model/prompt only) serialize byte-identically."""
        ov = ChannelOverride(model="m", system_prompt="p")
        assert "enabled_toolsets" not in ov.to_dict()

    def test_platform_config_yaml_round_trip(self):
        plat = PlatformConfig.from_dict(
            {
                "enabled": True,
                "channel_overrides": {
                    CLINIC_PHONE: {"enabled_toolsets": [], "system_prompt": "clinica"},
                    "group:*": {"enabled_toolsets": []},
                },
            }
        )
        assert plat.channel_overrides[CLINIC_PHONE].enabled_toolsets == []
        assert plat.channel_overrides[CLINIC_PHONE].system_prompt == "clinica"
        assert plat.channel_overrides["group:*"].enabled_toolsets == []


# ---------------------------------------------------------------------------
# Wildcard lookup (generic, all platforms)
# ---------------------------------------------------------------------------


class TestWildcardChannelOverrideLookup:
    def test_chat_type_wildcard_matches_group(self):
        config = _whatsapp_config({"group:*": ChannelOverride(enabled_toolsets=[])})
        ov = _get_channel_override(
            config, Platform.WHATSAPP, GROUP_JID, chat_type="group"
        )
        assert ov is not None
        assert ov.enabled_toolsets == []

    def test_chat_type_wildcard_does_not_match_dm(self):
        config = _whatsapp_config({"group:*": ChannelOverride(enabled_toolsets=[])})
        assert (
            _get_channel_override(
                config, Platform.WHATSAPP, f"{CLINIC_PHONE}@s.whatsapp.net",
                chat_type="dm",
            )
            is None
        )

    def test_global_wildcard_matches_any_chat(self):
        config = _whatsapp_config({"*": ChannelOverride(enabled_toolsets=[])})
        ov = _get_channel_override(
            config, Platform.WHATSAPP, GROUP_JID, chat_type="group"
        )
        assert ov is not None and ov.enabled_toolsets == []

    def test_specific_chat_beats_chat_type_wildcard(self):
        config = _whatsapp_config(
            {
                GROUP_JID: ChannelOverride(enabled_toolsets=["memory"]),
                "group:*": ChannelOverride(enabled_toolsets=[]),
            }
        )
        ov = _get_channel_override(
            config, Platform.WHATSAPP, GROUP_JID, chat_type="group"
        )
        assert ov.enabled_toolsets == ["memory"]

    def test_chat_type_wildcard_beats_global_wildcard(self):
        config = _whatsapp_config(
            {
                "group:*": ChannelOverride(enabled_toolsets=["memory"]),
                "*": ChannelOverride(enabled_toolsets=[]),
            }
        )
        ov = _get_channel_override(
            config, Platform.WHATSAPP, GROUP_JID, chat_type="group"
        )
        assert ov.enabled_toolsets == ["memory"]

    def test_no_wildcards_configured_is_unchanged(self):
        """Compat: exact-key-only configs miss exactly as before."""
        config = _whatsapp_config({"999": ChannelOverride(model="m")})
        assert (
            _get_channel_override(config, Platform.WHATSAPP, "123", chat_type="dm")
            is None
        )

    def test_wildcards_apply_on_other_platforms_too(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={"*": ChannelOverride(enabled_toolsets=[])},
                ),
            },
        )
        ov = _get_channel_override(config, Platform.DISCORD, "42", chat_type="channel")
        assert ov is not None and ov.enabled_toolsets == []


# ---------------------------------------------------------------------------
# WhatsApp phone/LID/JID alias matching
# ---------------------------------------------------------------------------


class TestWhatsAppAliasChannelOverrideLookup:
    def test_bare_phone_key_matches_full_jid_chat_id(self):
        config = _whatsapp_config({CLINIC_PHONE: ChannelOverride(enabled_toolsets=[])})
        ov = _get_channel_override(
            config, Platform.WHATSAPP, f"{CLINIC_PHONE}@s.whatsapp.net", chat_type="dm"
        )
        assert ov is not None and ov.enabled_toolsets == []

    def test_plus_prefixed_config_key_matches_jid_chat_id(self):
        config = _whatsapp_config(
            {f"+{CLINIC_PHONE}": ChannelOverride(enabled_toolsets=[])}
        )
        ov = _get_channel_override(
            config, Platform.WHATSAPP, f"{CLINIC_PHONE}@s.whatsapp.net", chat_type="dm"
        )
        assert ov is not None and ov.enabled_toolsets == []

    def test_lid_chat_id_matches_phone_config_key(self, tmp_path, monkeypatch):
        """Bridge flips the admin DM to LID form → still matches the phone key."""
        _write_lid_mappings(tmp_path, monkeypatch)
        config = _whatsapp_config({ADMIN_PHONE: ChannelOverride(model="admin-model")})
        ov = _get_channel_override(
            config, Platform.WHATSAPP, f"{ADMIN_LID}@lid", chat_type="dm"
        )
        assert ov is not None and ov.model == "admin-model"

    def test_phone_jid_chat_id_matches_lid_config_key(self, tmp_path, monkeypatch):
        _write_lid_mappings(tmp_path, monkeypatch)
        config = _whatsapp_config({ADMIN_LID: ChannelOverride(model="admin-model")})
        ov = _get_channel_override(
            config,
            Platform.WHATSAPP,
            f"{ADMIN_PHONE}@s.whatsapp.net",
            chat_type="dm",
        )
        assert ov is not None and ov.model == "admin-model"

    def test_alias_expansion_does_not_leak_to_other_platforms(self):
        """Discord keys are matched literally — no phone normalization."""
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={"+123": ChannelOverride(model="m")},
                ),
            },
        )
        assert (
            _get_channel_override(config, Platform.DISCORD, "123", chat_type="dm")
            is None
        )


# ---------------------------------------------------------------------------
# Effective toolset resolution (the piece _run_agent_step consumes)
# ---------------------------------------------------------------------------


class TestResolveChannelToolsets:
    def test_no_override_inherits_platform_default(self):
        config = _whatsapp_config({})
        source = _source(f"{ADMIN_PHONE}@s.whatsapp.net")
        assert (
            _resolve_channel_toolsets(config, source, FULL_TOOLSETS) == FULL_TOOLSETS
        )

    def test_clinic_dm_has_no_tools(self):
        config = _whatsapp_config(
            {CLINIC_PHONE: ChannelOverride(enabled_toolsets=[])}
        )
        source = _source(f"{CLINIC_PHONE}@s.whatsapp.net")
        assert _resolve_channel_toolsets(config, source, FULL_TOOLSETS) == []

    def test_admin_dm_keeps_full_toolsets_alongside_restricted_peers(self):
        config = _whatsapp_config(
            {
                CLINIC_PHONE: ChannelOverride(enabled_toolsets=[]),
                "group:*": ChannelOverride(enabled_toolsets=[]),
            }
        )
        source = _source(f"{ADMIN_PHONE}@s.whatsapp.net")
        assert (
            _resolve_channel_toolsets(config, source, FULL_TOOLSETS) == FULL_TOOLSETS
        )

    def test_admin_lid_dm_keeps_full_toolsets(self, tmp_path, monkeypatch):
        """Admin DM arriving under the LID alias resolves the same policy.

        With a restrictive ``dm:*`` wildcard in place, the admin grant must be
        EXPLICIT (list the toolsets) — an empty entry falls through to the
        wildcard by design (fail-closed).
        """
        _write_lid_mappings(tmp_path, monkeypatch)
        config = _whatsapp_config(
            {
                ADMIN_PHONE: ChannelOverride(enabled_toolsets=list(FULL_TOOLSETS)),
                "dm:*": ChannelOverride(enabled_toolsets=[]),
            }
        )
        source = _source(f"{ADMIN_LID}@lid")
        assert (
            _resolve_channel_toolsets(config, source, FULL_TOOLSETS) == FULL_TOOLSETS
        )

    def test_group_has_no_tools_even_when_sender_is_admin(self):
        """Chat-scoped policy: an admin speaking in a group gets NO tools."""
        config = _whatsapp_config(
            {
                ADMIN_PHONE: ChannelOverride(),  # admin DM inherits full
                "group:*": ChannelOverride(enabled_toolsets=[]),
            }
        )
        source = _source(GROUP_JID, chat_type="group", user_id=ADMIN_PHONE)
        assert _resolve_channel_toolsets(config, source, FULL_TOOLSETS) == []

    def test_explicit_toolsets_replace_default(self):
        config = _whatsapp_config(
            {CLINIC_PHONE: ChannelOverride(enabled_toolsets=["memory"])}
        )
        source = _source(f"{CLINIC_PHONE}@s.whatsapp.net")
        assert _resolve_channel_toolsets(config, source, FULL_TOOLSETS) == ["memory"]

    def test_inherit_entry_falls_through_to_wildcard_toolsets(self):
        """First match without the field keeps walking — a prompt-only entry
        must not shadow a restrictive wildcard into full access (fail-closed)."""
        config = _whatsapp_config(
            {
                CLINIC_PHONE: ChannelOverride(system_prompt="clinica"),
                "dm:*": ChannelOverride(enabled_toolsets=[]),
            }
        )
        source = _source(f"{CLINIC_PHONE}@s.whatsapp.net")
        assert _resolve_channel_toolsets(config, source, FULL_TOOLSETS) == []

    def test_lookup_error_fails_closed_to_no_tools(self):
        class _BoomConfig:
            @property
            def platforms(self):
                raise RuntimeError("corrupted config")

        source = _source(f"{ADMIN_PHONE}@s.whatsapp.net")
        assert _resolve_channel_toolsets(_BoomConfig(), source, FULL_TOOLSETS) == []

    def test_missing_config_inherits_default(self):
        """No gateway config object at all (unit contexts) → platform default."""
        source = _source(f"{ADMIN_PHONE}@s.whatsapp.net")
        assert (
            _resolve_channel_toolsets(None, source, FULL_TOOLSETS) == FULL_TOOLSETS
        )


# ---------------------------------------------------------------------------
# Gateway wiring: policy resolved before agent creation/reuse in both paths
# ---------------------------------------------------------------------------


class TestEffectiveEnabledToolsets:
    def _runner(self, overrides):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.config = _whatsapp_config(overrides)
        return runner

    def test_channel_override_restricts_platform_default(self, monkeypatch):
        import hermes_cli.tools_config as tools_config

        monkeypatch.setattr(
            tools_config, "_get_platform_tools", lambda *_a, **_k: {"hermes-whatsapp"}
        )
        runner = self._runner({CLINIC_PHONE: ChannelOverride(enabled_toolsets=[])})
        source = _source(f"{CLINIC_PHONE}@s.whatsapp.net")
        assert runner._effective_enabled_toolsets({}, "whatsapp", source) == []

    def test_no_override_returns_sorted_platform_default(self, monkeypatch):
        import hermes_cli.tools_config as tools_config

        monkeypatch.setattr(
            tools_config,
            "_get_platform_tools",
            lambda *_a, **_k: {"web", "memory"},
        )
        runner = self._runner({})
        source = _source(f"{ADMIN_PHONE}@s.whatsapp.net")
        assert runner._effective_enabled_toolsets({}, "whatsapp", source) == [
            "memory",
            "web",
        ]


class TestAgentCreationPathsUseChannelPolicy:
    """The two agent-construction paths must resolve toolsets through the
    per-channel policy helper BEFORE building/reusing the AIAgent (source
    inspection — cheap pin that the wiring isn't dropped in a refactor)."""

    def test_run_agent_path_uses_helper(self):
        import inspect

        from gateway.run import GatewayRunner

        assert "_effective_enabled_toolsets" in inspect.getsource(
            GatewayRunner._run_agent_inner
        )

    def test_background_task_path_uses_helper(self):
        import inspect

        from gateway.run import GatewayRunner

        assert "_effective_enabled_toolsets" in inspect.getsource(
            GatewayRunner._run_background_task
        )
