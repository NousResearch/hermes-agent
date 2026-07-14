"""Tests for the Discord collaborative-workspace identity & permission policy.

Covers identity resolution, the per-message speaker prefix, the system-prompt
policy/visibility block, and the wiring through build_session_context /
build_session_context_prompt — including that unconfigured deployments are
unchanged.
"""

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.discord_identity import (
    COLLABORATOR_ROLE,
    OWNER_ROLE,
    DiscordIdentity,
    build_collab_policy_block,
    discord_speaker_prefix,
    identities_roster,
    load_discord_identities,
    resolve_discord_identity,
    visibility_label,
)
from gateway.session import (
    SessionSource,
    build_session_context,
    build_session_context_prompt,
)


OWNER_ID = "111111111111111111"
COLLAB_ID = "222222222222222222"


def _collab_config() -> GatewayConfig:
    """A gateway config with Jayden (owner) + Peter (collaborator) on Discord."""
    return GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                extra={
                    "identities": [
                        {"id": COLLAB_ID, "name": "Peter", "role": "collaborator"},
                        {"id": OWNER_ID, "name": "Jayden", "role": "owner"},
                    ]
                },
            )
        }
    )


def _discord_source(user_id=OWNER_ID, user_name="Jayden", chat_type="channel"):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="900",
        chat_name="workspace",
        chat_type=chat_type,
        user_id=user_id,
        user_name=user_name,
    )


# --------------------------------------------------------------------------- #
# Identity resolution
# --------------------------------------------------------------------------- #

class TestIdentityResolution:
    def test_load_maps_by_id(self):
        ids = load_discord_identities(_collab_config())
        assert set(ids) == {OWNER_ID, COLLAB_ID}
        assert ids[OWNER_ID] == DiscordIdentity(OWNER_ID, "Jayden", OWNER_ROLE)
        assert ids[COLLAB_ID].role == COLLABORATOR_ROLE

    def test_resolve_known_and_unknown(self):
        cfg = _collab_config()
        assert resolve_discord_identity(cfg, OWNER_ID).is_owner is True
        assert resolve_discord_identity(cfg, COLLAB_ID).display_name == "Peter"
        assert resolve_discord_identity(cfg, "999") is None
        assert resolve_discord_identity(cfg, None) is None

    def test_roster_orders_owner_first(self):
        roster = identities_roster(_collab_config())
        assert [i.display_name for i in roster] == ["Jayden", "Peter"]

    def test_custom_role_normalized_lowercase(self):
        # Roles are normalised to lowercase (stripped); a custom role is kept
        # in that lowercase form and treated as a non-owner.
        cfg = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    extra={"identities": [{"id": "8", "name": "Rae", "role": "  Reviewer "}]}
                )
            }
        )
        ident = resolve_discord_identity(cfg, "8")
        assert ident.role == "reviewer"
        assert ident.is_owner is False

    def test_user_id_and_display_name_aliases(self):
        cfg = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    extra={"identities": [{"user_id": "5", "display_name": "Sam"}]}
                )
            }
        )
        ident = resolve_discord_identity(cfg, "5")
        assert ident.display_name == "Sam"
        assert ident.role == COLLABORATOR_ROLE  # default

    def test_absent_config_resolves_to_empty(self):
        cfg = GatewayConfig()
        assert load_discord_identities(cfg) == {}
        assert resolve_discord_identity(cfg, OWNER_ID) is None

    def test_malformed_entries_are_skipped(self):
        cfg = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    extra={"identities": ["nope", {"name": "no-id"}, {"id": "7", "name": "Ok"}]}
                )
            }
        )
        ids = load_discord_identities(cfg)
        assert set(ids) == {"7"}


# --------------------------------------------------------------------------- #
# Speaker prefix + visibility
# --------------------------------------------------------------------------- #

class TestSpeakerPrefix:
    def test_visibility_label(self):
        assert visibility_label(_discord_source(chat_type="dm")) == "dm"
        assert visibility_label(_discord_source(chat_type="channel")) == "shared_channel"
        assert visibility_label(_discord_source(chat_type="thread")) == "shared_channel"

    def test_prefix_shared_channel(self):
        cfg = _collab_config()
        prefix = discord_speaker_prefix(_discord_source(COLLAB_ID, "Peter"), cfg)
        assert prefix == f"[Peter · collaborator · id:{COLLAB_ID} · shared_channel]"

    def test_prefix_dm(self):
        cfg = _collab_config()
        src = _discord_source(OWNER_ID, "Jayden", chat_type="dm")
        prefix = discord_speaker_prefix(src, cfg)
        assert prefix == f"[Jayden · owner · id:{OWNER_ID} · dm]"

    def test_prefix_none_when_unconfigured(self):
        # No identities configured → None so callers keep prior behavior.
        assert discord_speaker_prefix(_discord_source(), GatewayConfig()) is None

    def test_prefix_none_for_unknown_user(self):
        assert discord_speaker_prefix(_discord_source("999", "Stranger"), _collab_config()) is None


def _apply_inbound_prefix(message_text, source, config, *, is_shared_multi_user):
    """Mirror of the inbound prefix selection in ``GatewayRunner`` (run.py).

    Kept in lockstep with that block: a configured Discord sender gets the rich
    speaker prefix; otherwise we fall back to the plain ``[name]`` multi-user
    prefix.  Testing the selection here avoids standing up the full runner.
    """
    _speaker_prefix = None
    if source.platform == Platform.DISCORD:
        _speaker_prefix = discord_speaker_prefix(source, config)
    if _speaker_prefix:
        return f"{_speaker_prefix} {message_text}"
    elif is_shared_multi_user and source.user_name:
        return f"[{source.user_name}] {message_text}"
    return message_text


class TestInboundPrefixWiring:
    def test_known_sender_gets_rich_prefix(self):
        out = _apply_inbound_prefix(
            "hi", _discord_source(COLLAB_ID, "Peter"), _collab_config(),
            is_shared_multi_user=True,
        )
        assert out == f"[Peter · collaborator · id:{COLLAB_ID} · shared_channel] hi"

    def test_unknown_sender_falls_back_to_plain_prefix(self):
        out = _apply_inbound_prefix(
            "hi", _discord_source("999", "Stranger"), _collab_config(),
            is_shared_multi_user=True,
        )
        assert out == "[Stranger] hi"

    def test_unconfigured_discord_falls_back_to_plain_prefix(self):
        out = _apply_inbound_prefix(
            "hi", _discord_source(OWNER_ID, "Jayden"), GatewayConfig(),
            is_shared_multi_user=True,
        )
        assert out == "[Jayden] hi"


# --------------------------------------------------------------------------- #
# Policy block rendering
# --------------------------------------------------------------------------- #

class TestPolicyBlock:
    def test_empty_without_identities(self):
        assert build_collab_policy_block([], "shared_channel") == ""

    def test_roster_and_owner_named(self):
        block = build_collab_policy_block(identities_roster(_collab_config()), "shared_channel")
        assert "Jayden — owner" in block
        assert "Peter — collaborator" in block
        assert "Jayden" in block  # owner named in the permission policy line

    def test_shared_vs_dm_visibility_text(self):
        roster = identities_roster(_collab_config())
        shared = build_collab_policy_block(roster, "shared_channel")
        dm = build_collab_policy_block(roster, "dm")
        assert "shared channel" in shared.lower()
        assert "every configured collaborator can" in shared.lower()
        assert "private dm" in dm.lower()
        assert "not visible to other collaborators" in dm.lower()

    def test_permission_policy_lists_sensitive_categories(self):
        block = build_collab_policy_block(identities_roster(_collab_config()), "shared_channel")
        low = block.lower()
        assert "require" in low and "approval" in low
        for cat in ("paid", "destructive", "security", "production", "customer", "memory"):
            assert cat in low, cat


# --------------------------------------------------------------------------- #
# Wiring through session context
# --------------------------------------------------------------------------- #

class TestSessionContextWiring:
    def test_context_populates_roster_for_discord(self):
        ctx = build_session_context(_discord_source(), _collab_config())
        assert [i.display_name for i in ctx.discord_identities] == ["Jayden", "Peter"]
        # Serialization round-trips the roster.
        assert ctx.to_dict()["discord_identities"][0]["role"] == "owner"

    def test_context_empty_for_non_discord(self):
        src = SessionSource(platform=Platform.LOCAL, chat_id="local")
        ctx = build_session_context(src, _collab_config())
        assert ctx.discord_identities == []

    def test_context_empty_for_unknown_discord_sender(self):
        # Configured identities exist, but the sender is not one of them.
        # The roster/policy must NOT leak to an unknown/unconfigured sender.
        ctx = build_session_context(
            _discord_source(user_id="999", user_name="Stranger"), _collab_config()
        )
        assert ctx.discord_identities == []

    def test_prompt_excludes_policy_for_unknown_discord_sender(self):
        ctx = build_session_context(
            _discord_source(user_id="999", user_name="Stranger"), _collab_config()
        )
        prompt = build_session_context_prompt(ctx)
        assert "Discord collaborative workspace" not in prompt

    def test_prompt_includes_policy_for_shared_channel(self):
        ctx = build_session_context(_discord_source(chat_type="channel"), _collab_config())
        prompt = build_session_context_prompt(ctx)
        assert "Discord collaborative workspace" in prompt
        assert "Jayden — owner" in prompt
        assert "shared channel" in prompt.lower()
        assert "require" in prompt.lower() and "approval" in prompt.lower()

    def test_prompt_dm_visibility_text(self):
        ctx = build_session_context(_discord_source(chat_type="dm"), _collab_config())
        prompt = build_session_context_prompt(ctx)
        assert "private DM" in prompt
        assert "not visible to other collaborators" in prompt.lower()

    def test_prompt_unchanged_when_no_identities(self):
        # Discord source, but no identities configured → no policy block.
        cfg = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True)})
        ctx = build_session_context(_discord_source(), cfg)
        prompt = build_session_context_prompt(ctx)
        assert "Discord collaborative workspace" not in prompt
