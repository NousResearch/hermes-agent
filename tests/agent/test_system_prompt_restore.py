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
  * Managed-context-input fingerprint gate (issue #68563):
      - stale fingerprint → INFO + fresh build, even with matching runtime
      - NULL/legacy fingerprint → treated as a mismatch, self-heals
      - fingerprint compute failure → fails open (runtime check only)

Real filesystem-based fingerprint computation is patched out via the
``_FP`` sentinel + the ``fixed_fingerprint`` autouse fixture below, so
these tests stay hermetic and don't depend on this checkout's actual
SOUL.md/AGENTS.md contents.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

import agent.conversation_loop as conversation_loop
from agent.conversation_loop import _restore_or_build_system_prompt

# Sentinel current-fingerprint value used across tests unless a test
# overrides ``conversation_loop.compute_current_context_fingerprint``
# itself (e.g. to simulate drift or a compute failure).
_FP = "fingerprint-current"

# Captured before the autouse fixture below ever patches the module
# attribute, so tests that exercise the REAL helper (failure → fail-open;
# the end-to-end SOUL.md drift test) can restore it deliberately.
_real_compute_current_context_fingerprint = (
    conversation_loop.compute_current_context_fingerprint
)


@pytest.fixture(autouse=True)
def fixed_fingerprint(monkeypatch):
    """Make ``compute_current_context_fingerprint`` deterministic.

    Without this, the helper would hit the real filesystem (SOUL.md,
    AGENTS.md, ...) via ``agent.prompt_builder.compute_context_fingerprint``,
    making these tests depend on this checkout's actual files.
    """
    monkeypatch.setattr(
        conversation_loop, "compute_current_context_fingerprint", lambda agent: _FP
    )


def _make_agent(session_db=None, prebuilt_prompt: str = "BUILT_PROMPT"):
    """Construct the minimal agent fake the helper needs."""
    agent = MagicMock()
    agent._cached_system_prompt = None
    agent.session_id = "test-session-id"
    agent.model = "test-model"
    agent.provider = "openrouter"
    agent.platform = "cli"
    agent._session_db = session_db
    agent._build_system_prompt = MagicMock(return_value=prebuilt_prompt)
    # Explicit (a bare MagicMock() attribute is truthy by default, which
    # would silently disable SOUL.md/project-context inclusion in the REAL
    # compute_current_context_fingerprint — only exercised directly by a
    # couple of tests below, but worth pinning down for all of them).
    agent.skip_context_files = False
    agent.load_soul_identity = False
    return agent


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestStoredPromptReuse:
    def test_present_row_is_reused_verbatim(self, caplog):
        """Continuing session with a stored prompt → reuse byte-for-byte."""
        stored = "Stored prompt from turn 1 — byte-identical reuse"
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored, "system_prompt_fingerprint": _FP}
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
        db.get_session.return_value = {"system_prompt": stored, "system_prompt_fingerprint": _FP}
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
            agent.session_id, agent._cached_system_prompt, _FP
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
        # Persisted to DB, alongside the current input fingerprint
        db.update_system_prompt.assert_called_once_with(agent.session_id, "BUILT_PROMPT", _FP)
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
    def test_db_read_exception_warns_and_rebuilds(self, caplog):
        """DB read raising → WARNING + fall through to fresh build."""
        db = MagicMock()
        db.get_session.side_effect = RuntimeError("disk full")
        agent = _make_agent(session_db=db)

        with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        # Built fresh
        agent._build_system_prompt.assert_called_once()
        assert agent._cached_system_prompt == "BUILT_PROMPT"
        # Loud warning about the read failure
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("get_session failed" in r.getMessage() for r in warnings), \
            f"Expected a get_session warning, got: {[r.getMessage() for r in warnings]}"
        assert any("disk full" in r.getMessage() for r in warnings)

    def test_null_system_prompt_warns_about_unusable_stored_state(self, caplog):
        """Row exists but system_prompt is NULL → WARNING + fresh build."""
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": None}
        agent = _make_agent(session_db=db)

        with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        agent._build_system_prompt.assert_called_once()
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("is null" in m and "rebuilding" in m for m in warnings), \
            f"Expected null-stored-prompt warning, got: {warnings}"

    def test_empty_system_prompt_warns_about_silent_persistence_bug(self, caplog):
        """Row exists but system_prompt is '' → WARNING about silent write bug."""
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": ""}
        agent = _make_agent(session_db=db)

        with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        agent._build_system_prompt.assert_called_once()
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("is empty" in m and "rebuilding" in m for m in warnings), \
            f"Expected empty-stored-prompt warning, got: {warnings}"

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
        db.get_session.return_value = {"system_prompt": stored, "system_prompt_fingerprint": _FP}
        agent = _make_agent(session_db=db)

        _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        # Identity check — must be the same object reference for maximum
        # confidence we're not slicing/copying/normalizing.
        assert agent._cached_system_prompt == stored
        # Byte-level check
        assert agent._cached_system_prompt.encode("utf-8") == stored.encode("utf-8")


# ---------------------------------------------------------------------------
# Managed-context-input fingerprint gate (issue #68563)
# ---------------------------------------------------------------------------


class TestFingerprintGate:
    def test_stale_fingerprint_rebuilds_despite_matching_runtime(self, caplog, monkeypatch):
        """SOUL.md (etc.) changed since this durable session's prompt was
        cached — the Model/Provider lines still match, but the stored
        prompt no longer reflects the current managed context inputs, so it
        must NOT be reused verbatim (the core bug in #68563)."""
        stored = "You are Hermes Agent.\n\nModel: test-model\nProvider: openrouter"
        db = MagicMock()
        db.get_session.return_value = {
            "system_prompt": stored,
            "system_prompt_fingerprint": "fingerprint-OLD",
        }
        agent = _make_agent(session_db=db)
        monkeypatch.setattr(
            conversation_loop, "compute_current_context_fingerprint", lambda a: "fingerprint-NEW"
        )

        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        agent._build_system_prompt.assert_called_once_with(None)
        assert agent._cached_system_prompt == "BUILT_PROMPT"
        db.update_system_prompt.assert_called_once_with(
            agent.session_id, "BUILT_PROMPT", "fingerprint-NEW"
        )
        assert any("stale" in r.getMessage().lower() for r in caplog.records)

    def test_null_stored_fingerprint_is_treated_as_mismatch_and_self_heals(self, caplog):
        """Legacy session predating this column (system_prompt_fingerprint
        is NULL) → treated as stale even though the prompt itself and the
        runtime identity both look fine. Rebuilding persists the
        fingerprint, so the NEXT restore on this session is a clean hit."""
        stored = "You are Hermes Agent.\n\nModel: test-model\nProvider: openrouter"
        db = MagicMock()
        db.get_session.return_value = {
            "system_prompt": stored,
            "system_prompt_fingerprint": None,
        }
        agent = _make_agent(session_db=db)

        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        agent._build_system_prompt.assert_called_once_with(None)
        db.update_system_prompt.assert_called_once_with(agent.session_id, "BUILT_PROMPT", _FP)

    def test_fingerprint_compute_failure_fails_open_and_reuses_stored_prompt(
        self, caplog, monkeypatch
    ):
        """A bug/IO error while computing the CURRENT fingerprint must never
        break the session — fall back to the pre-existing runtime-identity
        check alone, exactly like before this feature existed."""
        stored = "You are Hermes Agent.\n\nModel: test-model\nProvider: openrouter"
        db = MagicMock()
        db.get_session.return_value = {
            "system_prompt": stored,
            "system_prompt_fingerprint": "fingerprint-OLD",
        }
        agent = _make_agent(session_db=db)
        monkeypatch.setattr(
            conversation_loop, "compute_current_context_fingerprint", lambda a: None
        )

        _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        assert agent._cached_system_prompt == stored
        agent._build_system_prompt.assert_not_called()
        db.update_system_prompt.assert_not_called()

    def test_compute_current_context_fingerprint_failure_logs_warning_and_returns_none(
        self, monkeypatch, caplog
    ):
        """Unit-level check of the helper itself: an exception from
        ``compute_context_fingerprint`` must be caught, logged at WARNING,
        and surfaced to the caller as ``None`` (fail open)."""

        monkeypatch.setattr(
            conversation_loop,
            "compute_current_context_fingerprint",
            _real_compute_current_context_fingerprint,
        )

        def _boom(*a, **kw):
            raise OSError("disk full")

        # Patch the sys.modules-cached module object directly rather than
        # via monkeypatch's dotted-string form: the latter resolves through
        # the ``agent`` package's own ``prompt_builder`` attribute (see
        # ``_pytest.monkeypatch.resolve``), which can go stale relative to
        # sys.modules if an earlier test reloads the submodule in place
        # (e.g. TestPromptBuilderImports in test_prompt_builder.py) without
        # updating the parent package's attribute to match. The lazy
        # ``from agent.prompt_builder import ...`` inside the real helper
        # resolves via sys.modules, so that's what must be patched.
        import importlib as _importlib

        prompt_builder_module = _importlib.import_module("agent.prompt_builder")
        monkeypatch.setattr(prompt_builder_module, "compute_context_fingerprint", _boom)
        agent = _make_agent()

        with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
            result = conversation_loop.compute_current_context_fingerprint(agent)

        assert result is None
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("Failed to compute" in m and "disk full" in m for m in warnings)


# ---------------------------------------------------------------------------
# Prompt/fingerprint TOCTOU guard (issue #68563 follow-up review finding #1)
# ---------------------------------------------------------------------------


class TestBuildPromptWithFingerprint:
    def test_returns_none_fingerprint_when_pre_post_digests_disagree(self, monkeypatch):
        """An edit landing between the pre-build and post-build fingerprint
        reads must not produce a fingerprint that gets attached to the (now
        stale) prompt — persist NULL instead, which the restore guard
        safely treats as legacy/stale and rebuilds once next turn."""
        values = iter(["fingerprint-before", "fingerprint-after"])
        monkeypatch.setattr(
            conversation_loop,
            "compute_current_context_fingerprint",
            lambda a: next(values),
        )
        agent = _make_agent()

        prompt, fingerprint = conversation_loop.build_prompt_with_fingerprint(agent, None)

        assert prompt == "BUILT_PROMPT"
        assert fingerprint is None

    def test_returns_the_fingerprint_when_pre_post_digests_agree(self, monkeypatch):
        """Nothing changed mid-build → the (single, stable) digest is
        returned and safe to persist."""
        monkeypatch.setattr(
            conversation_loop,
            "compute_current_context_fingerprint",
            lambda a: "fingerprint-stable",
        )
        agent = _make_agent()

        prompt, fingerprint = conversation_loop.build_prompt_with_fingerprint(agent, None)

        assert prompt == "BUILT_PROMPT"
        assert fingerprint == "fingerprint-stable"


# ---------------------------------------------------------------------------
# End-to-end: real SOUL.md drift through the real code path (issue #68563
# follow-up review finding #4)
# ---------------------------------------------------------------------------


class TestEndToEndSoulMdDrift:
    def test_soul_md_edit_forces_one_rebuild_then_reuses_until_next_edit(
        self, tmp_path, monkeypatch
    ):
        """No fingerprint mocking here — real SOUL.md reads, real SessionDB.

        1. First turn (no history): fresh build, persists prompt + fingerprint.
        2. Restore with no edits: reused verbatim (byte-identical, no rebuild).
        3. Edit SOUL.md on disk, restore again: MUST rebuild and persist a
           new prompt + a DIFFERENT fingerprint.
        4. Restore once more with no further edits: reused verbatim again.
        """
        monkeypatch.setattr(
            conversation_loop,
            "compute_current_context_fingerprint",
            _real_compute_current_context_fingerprint,
        )

        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        (hermes_home / "SOUL.md").write_text("Be concise.", encoding="utf-8")
        monkeypatch.chdir(tmp_path)  # no AGENTS.md/etc here — only SOUL.md varies

        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session(session_id="e2e-session", source="cli")

            build_count = {"n": 0}

            def _fake_build(system_message):
                build_count["n"] += 1
                return f"PROMPT v{build_count['n']}"

            agent = MagicMock()
            agent._cached_system_prompt = None
            agent.session_id = "e2e-session"
            agent.model = "test-model"
            agent.provider = "openrouter"
            agent.platform = "cli"
            agent.skip_context_files = False
            agent.load_soul_identity = False
            agent.context_compressor = None
            agent._session_db = db
            agent._build_system_prompt = _fake_build

            # 1. First turn — legitimate first build.
            _restore_or_build_system_prompt(agent, None, [])
            assert agent._cached_system_prompt == "PROMPT v1"
            assert build_count["n"] == 1
            row = db.get_session("e2e-session")
            assert row["system_prompt"] == "PROMPT v1"
            fp1 = row["system_prompt_fingerprint"]
            assert fp1 is not None

            # 2. Fresh "AIAgent" (gateway pattern) restores with no edits.
            agent._cached_system_prompt = None
            _restore_or_build_system_prompt(
                agent, None, [{"role": "user", "content": "hi"}]
            )
            assert agent._cached_system_prompt == "PROMPT v1"
            assert build_count["n"] == 1  # reused — no rebuild

            # 3. Edit SOUL.md — the fingerprint drifts, forcing a rebuild.
            (hermes_home / "SOUL.md").write_text("Be verbose.", encoding="utf-8")
            agent._cached_system_prompt = None
            _restore_or_build_system_prompt(
                agent, None, [{"role": "user", "content": "hi"}]
            )
            assert agent._cached_system_prompt == "PROMPT v2"
            assert build_count["n"] == 2
            row2 = db.get_session("e2e-session")
            assert row2["system_prompt"] == "PROMPT v2"
            fp2 = row2["system_prompt_fingerprint"]
            assert fp2 is not None
            assert fp2 != fp1

            # 4. No further edits — reuses the NEW pair verbatim.
            agent._cached_system_prompt = None
            _restore_or_build_system_prompt(
                agent, None, [{"role": "user", "content": "hi"}]
            )
            assert agent._cached_system_prompt == "PROMPT v2"
            assert build_count["n"] == 2  # still just the one rebuild
        finally:
            db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
