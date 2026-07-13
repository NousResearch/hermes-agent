"""Hermetic tests for the turbohaul provider's build_extra_body role classification.

Tests the 4-role system (main / sub_agent / curator / compression),
thread_id derivation, session_id reuse semantics, and the
persist_disabled-before-parent_session_id precedence rule.

No network or engine required - pure unit tests against the profile object.
"""

from providers import get_provider_profile


class TestTurbohaulRoleClassification:
    """build_extra_body must classify the current turn into exactly one role
    and derive the correct thread_id / client_meta from it."""

    def setup_method(self):
        self.profile = get_provider_profile("turbohaul")
        assert self.profile is not None

    # - Role: main (plain context, no flags) -

    def test_plain_context_is_main(self):
        """Plain call -> main agent, thread hermes-main-<sid>."""
        body = self.profile.build_extra_body(session_id="main-abc123")
        assert body["client_meta"]["is_main"] is True
        assert body["client_meta"]["is_sub_agent"] is False
        assert body["client_meta"]["is_curator"] is False
        assert body["client_meta"]["is_compression"] is False
        assert body["thread_id"] == "hermes-main-main-abc123"

    def test_main_role_reuses_session_id(self):
        """Main agent thread_id embeds the provided session_id verbatim."""
        body = self.profile.build_extra_body(session_id="session-42")
        assert body["client_meta"]["session_id"] == "session-42"
        assert body["thread_id"] == "hermes-main-session-42"

    # - Role: curator (persist_disabled=True) -

    def test_persist_disabled_is_curator(self):
        """persist_disabled=True -> curator, thread hermes-main-<sid>."""
        body = self.profile.build_extra_body(
            session_id="main-abc123",
            persist_disabled=True,
        )
        assert body["client_meta"]["is_curator"] is True
        assert body["client_meta"]["is_main"] is False
        assert body["client_meta"]["is_sub_agent"] is False
        assert body["client_meta"]["is_compression"] is False

    def test_curator_reuses_main_session_id(self):
        """Curator thread_id is hermes-main-<sid>, same as main agent."""
        body = self.profile.build_extra_body(
            session_id="main-abc123",
            persist_disabled=True,
        )
        assert body["thread_id"] == "hermes-main-main-abc123"

    # - Role: sub_agent (parent_session_id set) -

    def test_parent_session_id_is_sub_agent(self):
        """parent_session_id present -> sub_agent, thread hermes-sub-<sid>."""
        body = self.profile.build_extra_body(
            session_id="sub-def456",
            parent_session_id="main-abc123",
        )
        assert body["client_meta"]["is_sub_agent"] is True
        assert body["client_meta"]["is_main"] is False
        assert body["client_meta"]["is_curator"] is False
        assert body["client_meta"]["is_compression"] is False
        assert body["thread_id"] == "hermes-sub-sub-def456"

    def test_sub_agent_fresh_session_id(self):
        """Sub-agent thread_id uses its own session_id (not the parent's)."""
        body = self.profile.build_extra_body(
            session_id="sub-xyz789",
            parent_session_id="main-abc123",
        )
        assert body["client_meta"]["session_id"] == "sub-xyz789"
        assert body["thread_id"] == "hermes-sub-sub-xyz789"

    # - Role: compression (is_compression=True) -

    def test_compression_is_compression(self):
        """is_compression=True -> compression role, thread hermes-main-<sid>."""
        body = self.profile.build_extra_body(
            session_id="main-abc123",
            is_compression=True,
        )
        assert body["client_meta"]["is_compression"] is True
        assert body["client_meta"]["is_main"] is False
        assert body["client_meta"]["is_sub_agent"] is False
        assert body["client_meta"]["is_curator"] is False
        assert body["thread_id"] == "hermes-main-main-abc123"

    def test_compression_reuses_main_thread_id(self):
        """Compression shares main's thread_id pattern (hermes-main-<sid>)."""
        body = self.profile.build_extra_body(
            session_id="main-abc123",
            is_compression=True,
        )
        assert body["thread_id"] == "hermes-main-main-abc123"

    # - Mutual exclusivity -

    def test_exactly_one_role_flag_true(self):
        """Across all four role scenarios, exactly one flag is True per call."""
        scenarios = [
            {"session_id": "s1"},
            {"session_id": "s1", "persist_disabled": True},
            {"session_id": "s1", "parent_session_id": "s0"},
            {"session_id": "s1", "is_compression": True},
        ]
        for ctx in scenarios:
            body = self.profile.build_extra_body(**ctx)
            flags = [
                body["client_meta"]["is_main"],
                body["client_meta"]["is_sub_agent"],
                body["client_meta"]["is_curator"],
                body["client_meta"]["is_compression"],
            ]
            assert sum(flags) == 1, f"Expected exactly one role flag true for {ctx}"

    # - Precedence: persist_disabled beats parent_session_id -

    def test_persist_disabled_beats_parent_session_id(self):
        """When BOTH persist_disabled and parent_session_id are present,
        persist_disabled wins -> curator (NOT sub_agent)."""
        body = self.profile.build_extra_body(
            session_id="main-abc123",
            persist_disabled=True,
            parent_session_id="main-abc123",
        )
        assert body["client_meta"]["is_curator"] is True
        assert body["client_meta"]["is_sub_agent"] is False

    def test_persist_disabled_thread_id_ignores_parent_session_id(self):
        """Even with parent_session_id, curator uses main thread_id."""
        body = self.profile.build_extra_body(
            session_id="main-abc123",
            persist_disabled=True,
            parent_session_id="main-abc123",
        )
        assert body["thread_id"] == "hermes-main-main-abc123"

    # - Contradiction guard: parent_session_id + is_compression -

    def test_contradiction_downgrades_to_sub_agent(self):
        """parent_session_id + is_compression is contradictory ->
        downgraded to sub_agent (not compression)."""
        body = self.profile.build_extra_body(
            session_id="sub-xyz789",
            parent_session_id="main-abc123",
            is_compression=True,
        )
        assert body["client_meta"]["is_sub_agent"] is True
        assert body["client_meta"]["is_compression"] is False

