"""Tests for the uniform Channel Capabilities block injected into the
gateway's session-context prompt.

Issue: NousResearch/hermes-agent#45122 (Phase 1).

Contracts covered (behavior, not snapshot):

1. Every registered platform value produces a non-empty ``ChannelCapabilities``
   block whose rendered first line is exactly ``**Channel Capabilities:**``.
2. Every registry entry's ``mode`` starts with one of the three canonical
   tokens ``sync``, ``async``, ``autonomous`` (a short parenthetical suffix is
   allowed; the dataclass field itself must START with one of those three).
3. ``channel_capabilities_block("definitely_not_a_platform", "dm")`` returns
   an empty string (no entry -> no block, not a crash).
4. ``build_session_context_prompt`` for a Telegram DM source contains
   ``**Channel Capabilities:**`` positioned AFTER ``**Source:**`` and BEFORE
   the platform-specific notes (``**Channel Topic:**`` etc.).
5. The rendered prompt is byte-identical across two calls with the same
   SessionContext — the pin-safe invariant already asserted by
   ``test_prompt_tail_freeze.py`` also holds here.
6. sms capabilities mention ``1600`` so it stays consistent with
   ``PLATFORM_HINTS["sms"]``'s "limited to ~1600 characters" (relationship,
   not snapshot).
7. A ``Platform.LOCAL`` source renders a block whose ``mode`` starts with
   ``sync``.
8. Registry keys all correspond to real ``Platform`` members or bundled
   plugin platform names (parametrize over ``CHANNEL_CAPABILITIES`` keys
   and assert ``Platform(key)`` does not raise).
"""

from __future__ import annotations

import pytest

from gateway.channel_capabilities import (
    CHANNEL_CAPABILITIES,
    ChannelCapabilities,
    channel_capabilities_block,
)
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import (
    SessionSource,
    _STATIC_PLATFORM_NOTES,
    build_session_context,
    build_session_context_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _telegram_dm_ctx(chat_topic: str | None = None):
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token"),
        },
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="111",
        chat_name="Home",
        chat_type="dm",
        chat_topic=chat_topic,
    )
    return build_session_context(source, config)


def _local_ctx():
    config = GatewayConfig()
    source = SessionSource(
        platform=Platform.LOCAL,
        chat_id="cli",
        chat_name="CLI terminal",
        chat_type="dm",
    )
    return build_session_context(source, config)


# ---------------------------------------------------------------------------
# 1. Every platform renders a block; first line is exact.
# ---------------------------------------------------------------------------


class TestEveryPlatformRendersABlock:
    def test_every_registry_key_renders_non_empty_block(self):
        for platform_value in CHANNEL_CAPABILITIES:
            rendered = channel_capabilities_block(platform_value, "dm")
            assert rendered, (
                f"Platform {platform_value!r} returned an empty block "
                "— registry should cover every key it owns."
            )
            assert rendered.startswith("**Channel Capabilities:**"), (
                f"Platform {platform_value!r} block does not start with the "
                f"expected heading; got:\n{rendered!r}"
            )

    def test_block_is_first_line_aware_under_chat_type_variation(self):
        """The heading must remain exact regardless of chat_type."""
        for platform_value in list(CHANNEL_CAPABILITIES)[:5]:
            rendered = channel_capabilities_block(platform_value, "group")
            if rendered:  # every current entry emits identically for all chat types
                first_line = rendered.split("\n", 1)[0]
                assert first_line == "**Channel Capabilities:**"


# ---------------------------------------------------------------------------
# 2. mode is one of {sync, async, autonomous}.
# ---------------------------------------------------------------------------


class TestModeCanonical:
    def test_every_entry_has_canonical_mode(self):
        canonical = ("sync", "async", "autonomous")
        for platform_value, caps in CHANNEL_CAPABILITIES.items():
            assert isinstance(caps, ChannelCapabilities), (
                f"{platform_value!r} entry is not a ChannelCapabilities instance"
            )
            assert caps.mode, f"{platform_value!r} has empty mode"
            assert caps.mode.startswith(canonical), (
                f"{platform_value!r} mode {caps.mode!r} does not start with one of {canonical}"
            )


# ---------------------------------------------------------------------------
# 3. Unknown platform returns "" (no crash, no block).
# ---------------------------------------------------------------------------


class TestUnknownPlatform:
    def test_unknown_platform_returns_empty(self):
        assert channel_capabilities_block("definitely_not_a_platform", "dm") == ""

    def test_unknown_platform_with_weird_chat_type_returns_empty(self):
        assert channel_capabilities_block("not_real", "this-is-garbage") == ""


# ---------------------------------------------------------------------------
# 4. build_session_context_prompt ordering for a Telegram DM.
# ---------------------------------------------------------------------------


class TestSessionPromptIntegration:
    def test_capabilities_block_appears_after_source_before_topic(self):
        ctx = _telegram_dm_ctx(chat_topic="Home General")
        prompt = build_session_context_prompt(ctx)

        assert "**Source:**" in prompt
        assert "**Channel Capabilities:**" in prompt
        assert "**Channel Topic:**" in prompt

        source_idx = prompt.index("**Source:**")
        caps_idx = prompt.index("**Channel Capabilities:**")
        topic_idx = prompt.index("**Channel Topic:**")
        assert source_idx < caps_idx < topic_idx, (
            "Channel Capabilities block must appear AFTER **Source:** and "
            f"BEFORE **Channel Topic:**; source at {source_idx}, caps at {caps_idx}, "
            f"topic at {topic_idx}"
        )

        # Connected Platforms is later in the prompt (after capabilities); this
        # is the natural ordering contract — capabilities belong with the
        # source/identity block, not with the delivery options.
        connected_idx = prompt.index("**Connected Platforms:**")
        assert caps_idx < connected_idx

    def test_local_prompt_contains_capabilities_block(self):
        prompt = build_session_context_prompt(_local_ctx())
        assert "**Channel Capabilities:**" in prompt

    def test_matrix_prompt_does_not_break_with_capabilities(self):
        """Matrix gets a room-boundary block after Source; ensure the
        capabilities block still renders in the right slot."""
        config = GatewayConfig(
            platforms={
                Platform.MATRIX: PlatformConfig(enabled=True, token="fake"),
            },
        )
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:example.org",
            chat_name="ops",
            chat_type="channel",
            chat_topic="ops chatter",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)
        assert "**Channel Capabilities:**" in prompt
        # The Matrix room boundary appears AFTER capabilities.
        assert prompt.index("**Source:**") < prompt.index("**Channel Capabilities:**")
        assert prompt.index("**Channel Capabilities:**") < prompt.index("**Matrix Room:**")


# ---------------------------------------------------------------------------
# 5. Pin-safety sanity: same SessionContext renders byte-identical bytes.
# ---------------------------------------------------------------------------


class TestPinSafety:
    def test_same_context_renders_byte_identical(self):
        ctx = _telegram_dm_ctx()
        a = build_session_context_prompt(ctx)
        b = build_session_context_prompt(ctx)
        assert a == b
        assert a.encode("utf-8") == b.encode("utf-8")


# ---------------------------------------------------------------------------
# 6. SMS capabilities stay consistent with PLATFORM_HINTS.
# ---------------------------------------------------------------------------


class TestSmsConsistencyWithPlatformHints:
    def test_sms_mentions_1600(self):
        caps = CHANNEL_CAPABILITIES["sms"]
        rendered = channel_capabilities_block("sms", "dm")
        # The relationship contract: the rendered block must mention the
        # SMS ~1600 char limit (also stated in PLATFORM_HINTS["sms"]).
        assert "1600" in rendered, (
            "SMS channel capabilities must mention the 1600-char limit to stay "
            "consistent with PLATFORM_HINTS['sms']."
        )
        # And it must be plain text / no markdown formatting (matches PLATFORM_HINTS).
        assert "plain text" in rendered.lower() or "no markdown" in rendered.lower() or "markdown" not in rendered.lower()
        # Also verify the dataclass itself carries limits text.
        assert "1600" in caps.limits


# ---------------------------------------------------------------------------
# 7. Local source renders sync mode.
# ---------------------------------------------------------------------------


class TestLocalSource:
    def test_local_mode_starts_with_sync(self):
        caps = CHANNEL_CAPABILITIES["local"]
        assert caps.mode.startswith("sync")
        rendered = channel_capabilities_block("local", "dm")
        # The rendered - mode: line starts with "- mode: sync"
        assert "- mode: sync" in rendered


# ---------------------------------------------------------------------------
# 8. Registry keys correspond to real Platform members or bundled plugin names.
# ---------------------------------------------------------------------------


class TestRegistryKeysAreRealPlatforms:
    @pytest.mark.parametrize("platform_value", list(CHANNEL_CAPABILITIES.keys()))
    def test_key_resolves_to_real_platform(self, platform_value):
        # Platform._missing_ accepts bundled-plugin names AND any name
        # already in the registry; CHANNEL_CAPABILITIES keys must be either
        # built-in members or one of those.
        try:
            member = Platform(platform_value)
        except ValueError as e:
            pytest.fail(
                f"CHANNEL_CAPABILITIES key {platform_value!r} does not resolve "
                f"to a real Platform member: {e}"
            )
        assert member.value == platform_value


# ---------------------------------------------------------------------------
# Renderer shape contract: omit empty fields, every line starts with '- '.
# ---------------------------------------------------------------------------


class TestRendererShape:
    @pytest.mark.parametrize("platform_value", ["telegram", "sms", "discord", "local"])
    def test_only_dash_prefixed_lines_below_heading(self, platform_value):
        rendered = channel_capabilities_block(platform_value, "dm")
        assert rendered
        lines = rendered.split("\n")
        assert lines[0] == "**Channel Capabilities:**"
        for line in lines[1:]:
            assert line.startswith("- "), (
                f"{platform_value!r}: rendered line {line!r} missing '- ' prefix"
            )

    def test_empty_fields_are_omitted(self):
        """The renderer contract: empty fields emit no line (sms/homeassistant
        carry actions=(); a missing entry field must not render an empty label)."""
        for platform_value in ("sms", "homeassistant"):
            rendered = channel_capabilities_block(platform_value, "dm")
            assert rendered
            assert "- actions:" not in rendered, (
                f"{platform_value!r} has actions=(); the renderer must omit the line"
            )
            assert ": \n" not in rendered and "::" not in rendered.replace("Channel Capabilities:", "")


# ---------------------------------------------------------------------------
# Registry consistency: cross-source drift tripwires (registry vs
# PLATFORM_HINTS / _STATIC_PLATFORM_NOTES) and capability-claim integrity.
# ---------------------------------------------------------------------------


class TestRegistryConsistency:
    def test_registry_stays_consistent_with_platform_hints(self):
        """#104685 review (Enough1122): the registry derives from
        PLATFORM_HINTS / _STATIC_PLATFORM_NOTES; both sources evolve, so the
        shared claims need a drift tripwire. Asserts relationships, not
        snapshots. When one side changes intentionally, update BOTH and this
        test together."""
        from agent.prompt_builder import PLATFORM_HINTS

        # SMS: the ~1600-char limit is stated by BOTH sources — if either
        # drops or changes it, the other must be updated in the same commit.
        assert "1600" in PLATFORM_HINTS["sms"]
        assert "1600" in CHANNEL_CAPABILITIES["sms"].limits

        # Discord: both sources must agree markdown-tables are unsupported.
        assert "tables" in PLATFORM_HINTS["discord"].lower()
        assert "no tables" in CHANNEL_CAPABILITIES["discord"].style

        # BlueBubbles: _STATIC_PLATFORM_NOTES says "texts, not essays"; the
        # registry must not contradict it.
        assert "texts, not essays" in _STATIC_PLATFORM_NOTES[Platform.BLUEBUBBLES]
        assert "essays" in CHANNEL_CAPABILITIES["bluebubbles"].style

    def test_reactions_claims_are_agent_facing_only(self):
        """#104685 review (Enough1122): reaction/tool-gated claims are the
        highest drift risk. Only photon implements the agent-facing
        send_message(action="react") path (adapter.add_reaction); no other
        registry entry may claim reactions, and photon's entry must keep the
        qualified wording."""
        claiming = {
            key for key, caps in CHANNEL_CAPABILITIES.items()
            if any("reaction" in action for action in caps.actions)
        }
        assert claiming == {"photon"}, f"expected reaction claims {{'photon'}}, got {claiming}"
        assert any("action='react'" in action for action in CHANNEL_CAPABILITIES["photon"].actions)
