"""Tests for ``agent.conversation_loop._restore_or_build_system_prompt``.

Validates the gateway DB-roundtrip path that keeps the system prompt
byte-stable across turns (fresh AIAgent → must restore from session DB
instead of rebuilding).  Covers:

  * Successful restore from a stored prompt (present row).
  * Legitimate first-turn build (no history).
  * Silent-failure recovery paths:
      - DB read raises → WARNING + fresh build
      - Row has system_prompt=NULL → WARNING + fresh build
      - Row has system_prompt="" → WARNING + fresh build
      - DB write fails → WARNING (subsequent turns will miss cache)
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from agent.conversation_loop import _restore_or_build_system_prompt


def _make_agent(session_db=None, prebuilt_prompt: str = "BUILT_PROMPT"):
    """Construct the minimal agent fake the helper needs."""
    agent = MagicMock()
    agent._cached_system_prompt = None
    agent.session_id = "test-session-id"
    agent.model = "test-model"
    agent.provider = "openrouter"
    agent.platform = "cli"
    agent._session_db = session_db
    # MagicMock attributes are truthy by default; the static-prefix
    # reconstruction is gated on _use_prompt_caching, so default it off
    # for the legacy restore tests (the reconstruction tests enable it).
    agent._use_prompt_caching = False
    agent._bot_mode_gateway_session = False
    agent._build_system_prompt = MagicMock(return_value=prebuilt_prompt)
    return agent


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestStoredPromptReuse:
    def test_present_row_is_reused_verbatim(self, caplog):
        """Continuing session with a stored prompt → reuse byte-for-byte."""
        stored = "Stored prompt from turn 1 — byte-identical reuse"
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)

        with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        assert agent._cached_system_prompt == stored
        agent._build_system_prompt.assert_not_called()
        db.update_system_prompt.assert_not_called()
        # No warnings on the happy path
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_present_row_with_unicode_preserved(self):
        """Non-ASCII bytes in the stored prompt are not mangled."""
        stored = "Stored prompt with unicode: ☤ ⚗ ◆ — and emoji 🦊"
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)

        _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])
        assert agent._cached_system_prompt == stored

    def test_present_row_with_stale_runtime_identity_rebuilds(self, caplog):
        """Stored prompts are cache gold unless their runtime identity is stale.

        A live /model switch updates the agent and DB model_config immediately.
        If the old system_prompt snapshot still says the previous model,
        blindly restoring it makes the next turn call the new model while the
        model reads old `Model:` metadata ("what model are you?" lies).
        """
        stored = (
            "You are Hermes Agent.\n\n"
            "Conversation started: Tuesday, June 16, 2026\n"
            "Session ID: test-session-id\n"
            "Model: anthropic/claude-opus-4.8-fast\n"
            "Provider: openrouter"
        )
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(
            session_db=db,
            prebuilt_prompt=(
                "You are Hermes Agent.\n\n"
                "Conversation started: Tuesday, June 16, 2026\n"
                "Session ID: test-session-id\n"
                "Model: openai/gpt-5.5\n"
                "Provider: openrouter"
            ),
        )
        agent.model = "openai/gpt-5.5"

        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        assert agent._cached_system_prompt.endswith(
            "Model: openai/gpt-5.5\nProvider: openrouter"
        )
        agent._build_system_prompt.assert_called_once_with(None)
        db.update_system_prompt.assert_called_once_with(
            agent.session_id, agent._cached_system_prompt
        )
        assert any("stale runtime identity" in r.getMessage() for r in caplog.records)

    def test_legacy_routed_topic_prompt_rebuilds_without_renaming_session(self, tmp_path):
        """A pre-protocol routed topic adopts Bot Mode on its next turn."""
        home = tmp_path / ".hermes"
        managed = home / "profiles" / "builder"
        managed.mkdir(parents=True)
        (managed / "profile.yaml").write_text(
            "ui_meta:\n  hermes-bots:\n    title: Builder\n", encoding="utf-8"
        )
        (home / "config.yaml").write_text(
            "gateway:\n"
            "  multiplex_profiles: true\n"
            "  profile_routes:\n"
            "    - name: hermes-topic\n"
            "      platform: telegram\n"
            "      chat_id: '42'\n"
            "      thread_id: '100'\n"
            "      profile: default\n",
            encoding="utf-8",
        )

        db = MagicMock()
        db.db_path = home / "state.db"
        db.get_session.return_value = {
            "system_prompt": "Stored prompt from before routed Bot Mode support"
        }
        db.get_session_title.return_value = "Hermes topic"
        agent = _make_agent(session_db=db, prebuilt_prompt="UPGRADED_PROMPT")
        agent.platform = "telegram"
        agent._bot_mode_protocol = True
        agent._bot_mode_gateway_session = True
        agent._session_title_hint = ""
        agent._gateway_session_key = "agent:main:telegram:group:42:100"

        from tools.bot_mode_probe import _reset_cache_for_tests

        _reset_cache_for_tests()
        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "continue"}]
        )

        assert agent._cached_system_prompt == "UPGRADED_PROMPT"
        assert agent._bot_capability_refreshed is True
        assert agent._session_title_hint == ""
        agent._build_system_prompt.assert_called_once_with(None)
        db.update_system_prompt.assert_called_once_with(
            agent.session_id, "UPGRADED_PROMPT"
        )

    @pytest.mark.parametrize(
        ("gateway_state", "expect_rebuild"),
        [(False, True), (None, False)],
        ids=["route-removed", "route-probe-indeterminate"],
    )
    def test_stamped_topic_prompt_respects_route_state(
        self, tmp_path, gateway_state, expect_rebuild
    ):
        """Remove on a known route change; preserve on an unknown probe state."""
        home = tmp_path / ".hermes"
        managed = home / "profiles" / "builder"
        managed.mkdir(parents=True)
        (managed / "profile.yaml").write_text(
            "ui_meta:\n  hermes-bots:\n    title: Builder\n", encoding="utf-8"
        )
        (home / "config.yaml").write_text(
            "gateway:\n  multiplex_profiles: true\n  profile_routes: []\n",
            encoding="utf-8",
        )

        from tools.bot_mode_probe import _reset_cache_for_tests, epoch_line

        _reset_cache_for_tests()
        stored = (
            "Stored routed topic prompt\n\n"
            "## Messaging other agents\nold injected protocol\n\n"
            f"{epoch_line(home)}"
        )
        db = MagicMock()
        db.db_path = home / "state.db"
        db.get_session.return_value = {"system_prompt": stored}
        db.get_session_title.return_value = "Former routed topic"
        agent = _make_agent(session_db=db, prebuilt_prompt="PROMPT_WITHOUT_PROTOCOL")
        agent.platform = "telegram"
        agent._bot_mode_protocol = True
        agent._bot_mode_gateway_session = gateway_state
        agent._session_title_hint = ""
        agent._gateway_session_key = "agent:main:telegram:group:42:100"

        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "continue"}]
        )

        if expect_rebuild:
            assert agent._cached_system_prompt == "PROMPT_WITHOUT_PROTOCOL"
            agent._build_system_prompt.assert_called_once_with(None)
            db.update_system_prompt.assert_called_once_with(
                agent.session_id, "PROMPT_WITHOUT_PROTOCOL"
            )
        else:
            assert agent._cached_system_prompt == stored
            agent._build_system_prompt.assert_not_called()
            db.update_system_prompt.assert_not_called()

    def test_stamped_canonical_bot_chat_keeps_section_while_protocol_on(self, tmp_path):
        """Stamped canonical Bot Chat + protocol ON → keep (no rebuild)."""
        home = tmp_path / ".hermes"
        home.mkdir()
        from tools.bot_mode_probe import _reset_cache_for_tests, epoch_line

        _reset_cache_for_tests()
        stored = (
            "Canonical prompt\n\n## Messaging other agents\nprotocol\n\n"
            f"{epoch_line(home)}"
        )
        db = MagicMock()
        db.db_path = home / "state.db"
        db.get_session.return_value = {"system_prompt": stored}
        db.get_session_title.return_value = "Bot Chat"
        agent = _make_agent(session_db=db, prebuilt_prompt="REBUILT")
        agent.platform = "telegram"
        agent._bot_mode_protocol = True
        agent._bot_mode_gateway_session = False
        agent._session_title_hint = "Bot Chat"

        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "continue"}]
        )

        assert agent._cached_system_prompt == stored
        agent._build_system_prompt.assert_not_called()

    @pytest.mark.parametrize(
        ("title", "gateway_state"),
        [("Bot Chat", False), ("Former routed topic", True)],
        ids=["canonical-bot-chat", "routed-topic"],
    )
    def test_stamped_prompts_strip_when_protocol_turned_off(
        self, tmp_path, title, gateway_state
    ):
        """Reviewer #2 gap: stamped Bot Mode prompts rebuild exactly once to
        drop the injected section when `bot_mode_protocol` is disabled — for
        BOTH entry-point kinds."""
        home = tmp_path / ".hermes"
        home.mkdir()
        from tools.bot_mode_probe import _reset_cache_for_tests, epoch_line

        _reset_cache_for_tests()
        stored = (
            "Prompt with protocol\n\n## Messaging other agents\nsection\n\n"
            f"{epoch_line(home)}"
        )
        db = MagicMock()
        db.db_path = home / "state.db"
        db.get_session.return_value = {"system_prompt": stored}
        db.get_session_title.return_value = title
        agent = _make_agent(session_db=db, prebuilt_prompt="STRIPPED_PROMPT")
        agent.platform = "telegram"
        agent._bot_mode_protocol = False
        agent._bot_mode_gateway_session = gateway_state
        agent._session_title_hint = title if title != "Former routed topic" else ""

        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "continue"}]
        )

        assert agent._cached_system_prompt == "STRIPPED_PROMPT"
        agent._build_system_prompt.assert_called_once_with(None)
        db.update_system_prompt.assert_called_once_with(
            agent.session_id, "STRIPPED_PROMPT"
        )


# ---------------------------------------------------------------------------
# Legitimate fresh-build paths (no history, no DB)
# ---------------------------------------------------------------------------


class TestLegitimateFreshBuild:
    def test_no_history_skips_db_and_builds_fresh(self, caplog):
        """First turn with empty history → build fresh, don't touch the DB."""
        db = MagicMock()
        agent = _make_agent(session_db=db)

        with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(agent, None, [])

        # No history → DB read skipped entirely
        db.get_session.assert_not_called()
        agent._build_system_prompt.assert_called_once_with(None)
        assert agent._cached_system_prompt == "BUILT_PROMPT"
        # Persisted to DB
        db.update_system_prompt.assert_called_once_with(agent.session_id, "BUILT_PROMPT")
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_no_db_skips_persistence(self):
        """When session DB is None, build and skip persistence silently."""
        agent = _make_agent(session_db=None)
        _restore_or_build_system_prompt(agent, None, [])
        agent._build_system_prompt.assert_called_once()
        assert agent._cached_system_prompt == "BUILT_PROMPT"


# ---------------------------------------------------------------------------
# Silent-failure recovery — these are the new A/B logging paths
# ---------------------------------------------------------------------------


class TestSilentFailureWarnings:



    def test_db_write_failure_warns_loudly(self, caplog):
        """update_system_prompt raising → WARNING (was DEBUG before)."""
        db = MagicMock()
        # No prior row (first turn)
        db.get_session.return_value = None
        db.update_system_prompt.side_effect = RuntimeError("database is locked")
        agent = _make_agent(session_db=db)

        with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(agent, None, [])

        # Built and assigned the cache anyway
        agent._build_system_prompt.assert_called_once()
        assert agent._cached_system_prompt == "BUILT_PROMPT"
        # Warning surfaced
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "update_system_prompt failed" in m and "database is locked" in m
            for m in warnings
        ), f"Expected write-failure warning, got: {warnings}"

    def test_no_history_with_null_row_does_not_warn(self, caplog):
        """First turn (no history) hitting a null row is not surprising — no warn."""
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": None}
        agent = _make_agent(session_db=db)

        with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
            # Empty history → DB read is skipped entirely
            _restore_or_build_system_prompt(agent, None, [])

        db.get_session.assert_not_called()
        # No "rebuilding from scratch" warning because history is empty
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("rebuilding" in m for m in warnings)


# ---------------------------------------------------------------------------
# Byte-stability invariant
# ---------------------------------------------------------------------------


class TestPromptStabilityInvariant:
    def test_restored_prompt_is_byte_identical_to_stored(self):
        """The restored prompt must equal the stored bytes exactly — no
        normalization, trimming, or concat that could shift the prefix.

        This is the core invariant: any byte-level change at this point
        invalidates KV cache on every prefix-cache backend.
        """
        stored = (
            "You are Hermes Agent.\n"
            "\n"
            "Conversation started: Sunday, May 17, 2026\n"
            "Session ID: 20260517_153500_abc123\n"
        )
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)

        _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        # Identity check — must be the same object reference for maximum
        # confidence we're not slicing/copying/normalizing.
        assert agent._cached_system_prompt == stored
        # Byte-level check
        assert agent._cached_system_prompt.encode("utf-8") == stored.encode("utf-8")


# ---------------------------------------------------------------------------
# Cross-session static prefix reconstruction (issue #68191 follow-up)
# ---------------------------------------------------------------------------


class TestStaticPrefixReconstructionOnRestore:
    """The two-block cache layout must survive session restore.

    Gateway surfaces construct a fresh AIAgent per turn and restore the
    persisted prompt from the session DB; the cross-session-stable prefix
    (``_cached_system_prompt_static``) is only set on fresh builds, so
    without reconstruction the wire layout silently degrades to the legacy
    single-breakpoint layout after turn 1 (flagged on PR #68258 review).
    """

    def test_restore_reconstructs_static_prefix_when_it_matches(self):
        stable = "STATIC IDENTITY AND GUIDANCE"
        stored = stable + "\n\nper-session context\n\nvolatile tail"
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)
        agent._use_prompt_caching = True
        agent._cached_system_prompt_static = None

        from unittest.mock import patch as _patch

        with _patch(
            "agent.system_prompt.build_system_prompt_parts",
            return_value={"stable": stable, "context": "", "volatile": ""},
        ):
            _restore_or_build_system_prompt(
                agent, None, [{"role": "user", "content": "hi"}]
            )

        # Restored prompt bytes untouched; static prefix reconstructed.
        assert agent._cached_system_prompt == stored
        assert agent._cached_system_prompt_static == stable

    def test_restore_leaves_static_unset_on_prefix_mismatch(self):
        """Stable-tier drift (skills edited since persist) → no static prefix,
        legacy layout, restored bytes still authoritative."""
        stored = "OLD STATIC HEAD\n\nper-session context"
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)
        agent._use_prompt_caching = True
        agent._cached_system_prompt_static = None

        from unittest.mock import patch as _patch

        with _patch(
            "agent.system_prompt.build_system_prompt_parts",
            return_value={"stable": "NEW STATIC HEAD", "context": "", "volatile": ""},
        ):
            _restore_or_build_system_prompt(
                agent, None, [{"role": "user", "content": "hi"}]
            )

        assert agent._cached_system_prompt == stored
        assert agent._cached_system_prompt_static is None

    def test_restore_survives_parts_builder_exception(self):
        """Prefix reconstruction is fail-open: a parts-builder crash must not
        break the byte-identical restore."""
        stored = "Stored prompt — must survive"
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)
        agent._use_prompt_caching = True
        agent._cached_system_prompt_static = None

        from unittest.mock import patch as _patch

        with _patch(
            "agent.system_prompt.build_system_prompt_parts",
            side_effect=RuntimeError("boom"),
        ):
            _restore_or_build_system_prompt(
                agent, None, [{"role": "user", "content": "hi"}]
            )

        assert agent._cached_system_prompt == stored
        assert agent._cached_system_prompt_static is None


class TestReconstructStaticPrefixMemoization:
    """A failed static rebuild must not re-run the parts builder every call.

    ``reconstruct_static_prefix`` sits on the retry-loop hot path via the
    failover redecoration chokepoint (#72626); ``build_system_prompt_parts``
    does real file I/O (SOUL.md, context files, memory), so a persistent
    stable-tier mismatch must be checked once per stored prompt, not on
    every attempt of every API call.
    """

    def _agent(self, stored):
        agent = _make_agent()
        agent._use_prompt_caching = True
        agent._cached_system_prompt = stored
        agent._cached_system_prompt_static = None
        return agent

    def test_failed_rebuild_is_memoized_per_stored_prompt(self):
        from unittest.mock import patch as _patch

        from agent.system_prompt import reconstruct_static_prefix

        stored = "STORED PROMPT\n\ntail"
        agent = self._agent(stored)
        with _patch(
            "agent.system_prompt.build_system_prompt_parts",
            return_value={"stable": "MISMATCH", "context": "", "volatile": ""},
        ) as build:
            reconstruct_static_prefix(agent)
            reconstruct_static_prefix(agent)
            reconstruct_static_prefix(agent)
        assert build.call_count == 1
        assert agent._cached_system_prompt_static is None

    def test_changed_stored_prompt_retries_once(self):
        from unittest.mock import patch as _patch

        from agent.system_prompt import reconstruct_static_prefix

        agent = self._agent("OLD STORED")
        with _patch(
            "agent.system_prompt.build_system_prompt_parts",
            return_value={"stable": "MISMATCH", "context": "", "volatile": ""},
        ) as build:
            reconstruct_static_prefix(agent)
            # A new stored prompt (e.g. after compression) invalidates the
            # failure memo and gets exactly one fresh attempt.
            agent._cached_system_prompt = "NEW STORED"
            reconstruct_static_prefix(agent)
            reconstruct_static_prefix(agent)
        assert build.call_count == 2

    def test_success_clears_failure_memo_and_early_returns(self):
        from unittest.mock import patch as _patch

        from agent.system_prompt import reconstruct_static_prefix

        stable = "STATIC HEAD"
        stored = stable + "\n\nvolatile"
        agent = self._agent(stored)
        with _patch(
            "agent.system_prompt.build_system_prompt_parts",
            return_value={"stable": stable, "context": "", "volatile": ""},
        ) as build:
            reconstruct_static_prefix(agent)
            reconstruct_static_prefix(agent)
        # Second call early-returns on the already-valid static prefix.
        assert build.call_count == 1
        assert agent._cached_system_prompt_static == stable
        assert getattr(agent, "_static_rebuild_failed_for", None) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
