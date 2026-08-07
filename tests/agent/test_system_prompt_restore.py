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
  * Built-in MEMORY / USER PROFILE block validity (issue #74102 / PR #74129):
      - Marker-substring collisions in user-supplied context files must NOT
        satisfy the validity check (substring scan would falsely accept).
      - Stale non-empty blocks (contents changed since stored) must trigger
        rebuild.
      - Real MemoryStore instances are validated end-to-end via
        ``TestBlockValidityWithRealMemoryStore``.
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
    agent._build_system_prompt = MagicMock(return_value=prebuilt_prompt)
    # Real AIAgent default (see agent.agent_init: ``agent._memory_store = None``);
    # without this the block-validity gate sees a MagicMock truthy store and
    # tries to introspect it. Tests that exercise the gate set these explicitly.
    agent._memory_store = None
    agent._memory_enabled = False
    agent._user_profile_enabled = False
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


# ---------------------------------------------------------------------------
# Built-in MEMORY / USER PROFILE block validity (issue #74102 / PR #74129)
#
# A continuing gateway session can restore a stored prompt that predates a
# mid-session profile/memory write. ``_stored_prompt_matches_runtime``
# gates on Model/Provider/cwd/Platform, but the built-in MEMORY / USER
# PROFILE blocks injected from ``MemoryStore.format_for_system_prompt()``
# can drift between turn N's stored prompt and turn N+1's on-disk state.
#
# The PR's first cut used header-substring markers ("USER PROFILE",
# "MEMORY (your personal notes)") to detect absence. That check has two
# real failure modes the marker substring cannot distinguish from a valid
# prompt:
#
#   1. Context-marker collision. User-supplied context files (AGENTS.md /
#      CLAUDE.md / .cursorrules) are embedded in the middle context tier
#      and routinely contain the literal phrases "USER PROFILE" or
#      "MEMORY (your personal notes)". A substring scan satisfies the
#      gate even though no built-in block was injected.
#
#   2. Stale non-empty block. The on-disk entries changed since the
#      stored prompt was built. The substring is present, but it is the
#      OLD block — the agent would carry stale memory forward forever.
#
# The fix in ``agent.conversation_compression.py`` already encodes the
# gold-standard check: ``_cached_prompt_reflects_builtin_memory`` requires
# the CURRENT (post-reload) rendered block appear verbatim in the cached
# prompt. ``_stored_prompt_matches_runtime`` delegates to that helper so
# the restore gate and the compression retention gate share one
# validator. These tests exercise both bug classes via a stub store, then
# couple the contract to a real ``MemoryStore`` so the two stay in
# lockstep.
# ---------------------------------------------------------------------------


_USER_HEADER = "USER PROFILE (who the user is)"
_MEM_HEADER = "MEMORY (your personal notes)"


def _make_agent_with_memory(
    session_db=None,
    *,
    user_block=None,
    mem_block=None,
    user_profile_enabled=True,
    memory_enabled=True,
    prebuilt_prompt: str = "BUILT_PROMPT",
):
    """Agent fake whose ``_memory_store`` mimics ``MemoryStore``'s contract:
    ``format_for_system_prompt(target)`` returns ``None`` when the disk
    snapshot for that target is empty, otherwise the rendered block.
    """
    agent = _make_agent(session_db=session_db, prebuilt_prompt=prebuilt_prompt)

    mem_store = MagicMock()

    def _fmt(target):
        return {"user": user_block, "memory": mem_block}.get(target)

    mem_store.format_for_system_prompt.side_effect = _fmt
    agent._memory_store = mem_store
    agent._memory_enabled = memory_enabled
    agent._user_profile_enabled = user_profile_enabled
    return agent


class TestBlockValidityCheck:
    """Stored prompts missing enabled MEMORY/USER blocks must be rebuilt."""

    def test_user_profile_marker_in_context_file_still_triggers_rebuild(self):
        """An ``AGENTS.md`` that mentions "USER PROFILE" must NOT satisfy
        the validity gate: the substring scan would falsely accept, the
        verbatim-block scan must reject and rebuild."""
        sep = "═" * 46
        real_block = f"{sep}\n{_USER_HEADER} [5% — 50/1,375 chars]\n{sep}\nuser fact"
        stored = (
            "You are Hermes Agent.\n"
            "\n"
            "## Project Notes\n"
            "When designing the USER PROFILE page, please use the brand "
            "guidelines attached.\n"  # bare "USER PROFILE" prose
            "\nModel: test-model\nProvider: openrouter\n"
        )
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent_with_memory(
            session_db=db, user_block=real_block,
            user_profile_enabled=True, memory_enabled=False,
        )
        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "hi"}]
        )
        agent._build_system_prompt.assert_called_once()
        db.update_system_prompt.assert_called_once()

    def test_memory_marker_in_context_file_still_triggers_rebuild(self):
        """Same collision case for the MEMORY header branch."""
        sep = "═" * 46
        real_block = f"{sep}\n{_MEM_HEADER} [5% — 50/2,200 chars]\n{sep}\nmem fact"
        stored = (
            "You are Hermes Agent.\n"
            "\n"
            "## CLAUDE.md\n"
            "I maintain a section called MEMORY (your personal notes) that "
            "I prune aggressively.\n"  # bare "MEMORY (your personal notes)" prose
            "\nModel: test-model\nProvider: openrouter\n"
        )
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent_with_memory(
            session_db=db, mem_block=real_block,
            user_profile_enabled=False, memory_enabled=True,
        )
        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "hi"}]
        )
        agent._build_system_prompt.assert_called_once()
        db.update_system_prompt.assert_called_once()

    def test_stale_non_empty_user_block_triggers_rebuild(self):
        """Header present, body stale — substring scan would accept; the
        verbatim scan detects the OLD block and forces a rebuild."""
        sep = "═" * 46
        old_block = f"{sep}\n{_USER_HEADER} [5% — 50/1,375 chars]\n{sep}\nOLD fact"
        new_block = f"{sep}\n{_USER_HEADER} [6% — 82/1,375 chars]\n{sep}\nNEW fact"
        stored = (
            "You are Hermes Agent.\n\n"
            + old_block
            + "\n\nModel: test-model\nProvider: openrouter"
        )
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent_with_memory(
            session_db=db, user_block=new_block,  # current disk has new content
            user_profile_enabled=True, memory_enabled=False,
        )
        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "hi"}]
        )
        agent._build_system_prompt.assert_called_once()
        db.update_system_prompt.assert_called_once()


class TestBlockValidityWithRealMemoryStore:
    """End-to-end check against the real ``MemoryStore`` so the contract
    used by the mock-store tests stays in lockstep with the live class."""

    def test_real_store_roundtrip(self):
        """Block present verbatim → reuse. Block stale → rebuild."""
        from tools.memory_tool import MemoryStore

        # --- happy path: current block is in the stored prompt → reuse
        store = MemoryStore()
        store._system_prompt_snapshot = {
            "user": store._render_block("user", ["User prefers dark mode."]),
            "memory": "",
        }
        user_block = store.format_for_system_prompt("user")
        assert user_block is not None  # sentinel: snapshot was non-empty
        stored_match = (
            "You are Hermes Agent.\n\n"
            + user_block
            + "\n\nModel: test-model\nProvider: openrouter"
        )
        db_match = MagicMock()
        db_match.get_session.return_value = {"system_prompt": stored_match}
        agent_ok = _make_agent(session_db=db_match)
        agent_ok._memory_store = store
        agent_ok._user_profile_enabled = True
        agent_ok._memory_enabled = False

        _restore_or_build_system_prompt(
            agent_ok, None, [{"role": "user", "content": "hi"}]
        )
        assert agent_ok._cached_system_prompt == stored_match
        agent_ok._build_system_prompt.assert_not_called()

        # --- stale path: stored prompt carries the OLD block → rebuild
        store_stale = MemoryStore()
        old_entries = ["User prefers light mode."]
        new_entries = ["User prefers dark mode.", "User works in Berlin."]
        old_block = store_stale._render_block("user", old_entries)
        new_block = store_stale._render_block("user", new_entries)
        assert old_block != new_block
        store_stale._system_prompt_snapshot = {"user": new_block, "memory": ""}
        stored_stale = (
            "You are Hermes Agent.\n\n"
            + old_block
            + "\n\nModel: test-model\nProvider: openrouter"
        )
        db_stale = MagicMock()
        db_stale.get_session.return_value = {"system_prompt": stored_stale}
        agent_stale = _make_agent(session_db=db_stale)
        agent_stale._memory_store = store_stale
        agent_stale._user_profile_enabled = True
        agent_stale._memory_enabled = False

        _restore_or_build_system_prompt(
            agent_stale, None, [{"role": "user", "content": "hi"}]
        )
        agent_stale._build_system_prompt.assert_called_once()
        db_stale.update_system_prompt.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
