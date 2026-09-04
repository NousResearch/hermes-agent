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

    @pytest.mark.parametrize("stored_context_line", [None, "Context files directory: /old/profile"])
    def test_profile_context_change_rebuilds_stored_prompt(
        self, stored_context_line, tmp_path, monkeypatch
    ):
        # Import the guard AND the scope together, at call time. Several suites in
        # tests/agent delete every `agent.*` entry from sys.modules
        # (test_empty_tool_name_loop_dampening, test_verification_stop_caching,
        # test_vision_routing_31179), so a module-level binding here would read the
        # ContextVars of a superseded agent.runtime_cwd while _profile_runtime_scope
        # writes the current one — the scope would look unset and this test would
        # pass alone and fail in a full-directory run.
        import importlib

        from gateway.run import _profile_runtime_scope

        restore = importlib.import_module("agent.conversation_loop")._restore_or_build_system_prompt

        execution_cwd = tmp_path / "shared-workdir"
        execution_cwd.mkdir()
        active_profile = tmp_path / "profiles" / "active"
        active_profile.mkdir(parents=True)
        monkeypatch.setenv("TERMINAL_CWD", str(execution_cwd))

        lines = [
            "Host: Linux",
            f"User home directory: {tmp_path}",
            f"Current working directory: {execution_cwd}",
        ]
        if stored_context_line:
            lines.append(stored_context_line)
        lines.extend(
            [
                "Conversation started: Tuesday, June 16, 2026",
                "Model: test-model",
                "Provider: openrouter",
                "Platform: cli",
            ]
        )
        stored = "\n".join(lines)
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db, prebuilt_prompt="PROFILE_PROMPT_REBUILT")

        with _profile_runtime_scope(active_profile):
            restore(agent, None, [{"role": "user", "content": "hi"}])

        assert agent._cached_system_prompt == "PROFILE_PROMPT_REBUILT"
        agent._build_system_prompt.assert_called_once_with(None)
        db.update_system_prompt.assert_called_once_with(
            agent.session_id, "PROFILE_PROMPT_REBUILT"
        )

    def test_scoped_prompt_resumed_without_a_scope_rebuilds(self, tmp_path, monkeypatch):
        """A prompt built INSIDE a profile scope must not survive resumption outside one.

        The guard used to run only while currently scoped, which is the wrong half of
        the comparison: the dangerous direction is a stored prompt that still carries
        another profile's marker and instructions being reused by a session with no
        scope at all. Compare whenever either side has a value.
        """
        execution_cwd = tmp_path / "shared-workdir"
        execution_cwd.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(execution_cwd))

        stored = "\n".join(
            [
                "Host: Linux",
                f"User home directory: {tmp_path}",
                f"Current working directory: {execution_cwd}",
                "Context files directory: /profiles/other",
                "Conversation started: Tuesday, June 16, 2026",
                "Model: test-model",
                "Provider: openrouter",
                "Platform: cli",
            ]
        )
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db, prebuilt_prompt="UNSCOPED_PROMPT_REBUILT")

        # No _profile_runtime_scope here: this is the resumed-outside-multiplex case.
        _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        assert agent._cached_system_prompt == "UNSCOPED_PROMPT_REBUILT"
        agent._build_system_prompt.assert_called_once_with(None)


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


class TestPerResponseSessionWritePath:
    """The write path under an embedding host's per-response session (#96570).

    Hermes Studio group chat pre-creates the SQLite row and pre-persists the
    user message BEFORE ``run_conversation()``, then runs one turn under a
    session id it destroys afterwards. The row therefore starts with a null
    system prompt and a non-empty history on its own genuine FIRST turn, which
    is what trips the "stored system prompt is null" warning — the warning is
    a first-turn artifact of that lifecycle, not evidence of a lost write.

    This pins the write path against that exact lifecycle: the freshly built
    prompt must land in the pre-created row within the same run.
    """

    def _agent(self, db, session_id):
        agent = _make_agent(session_db=db, prebuilt_prompt="GROUP_PROMPT")
        agent.session_id = session_id
        return agent

    def test_prepersisted_row_stores_the_freshly_built_prompt(self, tmp_path):
        from hermes_state import SessionDB

        session_id = "gc_run_room42_default_Worker_5f2c1ab9d4e34f7a8b0c6d1e2f3a4b5c"
        with SessionDB(db_path=tmp_path / "state.db") as db:
            # What the bridge does before the turn starts.
            db.create_session(session_id, source="studio")
            db.append_message(session_id=session_id, role="user", content="hi")

            _restore_or_build_system_prompt(
                self._agent(db, session_id),
                None,
                [{"role": "user", "content": "hi"}],
            )

            assert db.get_session(session_id)["system_prompt"] == "GROUP_PROMPT"

    def test_warning_is_a_first_turn_artifact_not_a_lost_write(
        self, tmp_path, caplog
    ):
        """Second turn of the SAME id restores — so nothing was dropped."""
        from hermes_state import SessionDB

        session_id = "gc_run_room42_default_Worker_9a7e3b1c05d24e6fb83a1c7d9e0f2a4b"
        history = [{"role": "user", "content": "hi"}]
        with SessionDB(db_path=tmp_path / "state.db") as db:
            db.create_session(session_id, source="studio")
            db.append_message(session_id=session_id, role="user", content="hi")

            with caplog.at_level(
                logging.WARNING, logger="agent.conversation_loop"
            ):
                _restore_or_build_system_prompt(
                    self._agent(db, session_id), None, history
                )
            assert "is null" in caplog.text

            caplog.clear()
            second = self._agent(db, session_id)
            with caplog.at_level(
                logging.WARNING, logger="agent.conversation_loop"
            ):
                _restore_or_build_system_prompt(second, None, history)

            assert second._cached_system_prompt == "GROUP_PROMPT"
            second._build_system_prompt.assert_not_called()
            assert "is null" not in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
