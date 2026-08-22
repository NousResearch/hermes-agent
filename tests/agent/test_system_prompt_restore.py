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
from agent.system_prompt import stored_identity_is_stale


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
# PR #72253 redesign (v6) — restore-path contract pins (§6-C). Restore never
# rewrites the stored prompt on identity drift: AGENTS.md:19-23 forbids
# mid-conversation system-prompt rewrites outside the explicit compression
# exception, and restore is not that exception. agent/conversation_loop.py
# was reverted to main's behavior, so _restore_or_build_system_prompt no
# longer calls stored_identity_is_stale at all; identity drift is instead
# handled at the compaction keep-prompt gate (TestCompactionIdentityDriftGate).
# See .claude/archive/72253-redesign-v6.md §6-C for the design rationale.
# ---------------------------------------------------------------------------


class TestV6RedesignFailBeforePins:
    def test_soul_drift_does_not_rewrite_stored_prompt(self, monkeypatch, caplog):
        """v6 §1/§6-C-1: SOUL.md drift on restore must NOT rebuild or persist —
        AGENTS.md:19-23 forbids mid-conversation system-prompt rewrites outside
        the explicit compression exception. Restore is not that exception.

        _restore_or_build_system_prompt no longer imports or calls
        stored_identity_is_stale at all (agent/conversation_loop.py was fully
        reverted to main's behavior), so this reuse holds unconditionally.
        """
        stored = _stored_with_identity("OLD SOUL IDENTITY")
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)
        _patch_identity(monkeypatch, "NEW SOUL IDENTITY")

        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "hi"}]
        )

        assert agent._cached_system_prompt == stored
        agent._build_system_prompt.assert_not_called()
        db.update_system_prompt.assert_not_called()

    def test_restore_does_not_read_soul_md(self, monkeypatch):
        """v6 §6-A point 2: the restore hot path must not re-read SOUL.md on
        every reuse turn (no memoization exists for it, unlike
        reconstruct_static_prefix's retry-loop guard at system_prompt.py:610-614).

        Holds because _restore_or_build_system_prompt no longer calls
        stored_identity_is_stale (which is what previously chained into
        resolve_identity_block -> run_agent.load_soul_md on every turn); the
        real resolver is wired in below to prove that chain is genuinely
        unreached, not just mocked away.
        """
        identity = "CURRENT SOUL IDENTITY"
        stored = _stored_with_identity(identity)
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)
        agent.load_soul_identity = True
        agent.skip_context_files = False

        # Undo the file's autouse neutral-resolver patch for this test only,
        # so the real resolver (and therefore the real load_soul_md call) is
        # actually exercised — a fully-mocked resolver would make this pin
        # vacuous (it would pass regardless of what the restore path does).
        monkeypatch.setattr(
            "agent.system_prompt.resolve_identity_block", _REAL_RESOLVE_IDENTITY
        )
        soul_reader = MagicMock(return_value=identity)
        monkeypatch.setattr("run_agent.load_soul_md", soul_reader)

        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "hi"}]
        )

        soul_reader.assert_not_called()


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
# Identity (SOUL.md) staleness on restore — issue #68563
# ---------------------------------------------------------------------------

from agent.prompt_builder import HERMES_AGENT_HELP_GUIDANCE as _HELP
# Captured at import time, BEFORE the neutral autouse fixture patches the
# module attribute — the classification tests exercise the real function.
from agent.system_prompt import resolve_identity_block as _REAL_RESOLVE_IDENTITY


@pytest.fixture(autouse=True)
def _neutral_identity_resolver(monkeypatch):
    """Restore now consults the identity resolver on every reuse (#68563);
    running the real one against these MagicMock agents would read the test
    HERMES_HOME and randomly flip the decision. Default it to "no basis to
    judge" (empty text -> check skipped, reuse); the staleness tests below
    override it with explicit values."""
    monkeypatch.setattr(
        "agent.system_prompt.resolve_identity_block",
        lambda agent: {"text": "", "from_soul": False, "checkable": True},
    )


def _stored_with_identity(identity: str, tail: str = "per-session context") -> str:
    """Assemble a stored prompt the way the builder joins the stable tier."""
    return identity.strip() + "\n\n" + _HELP.strip() + "\n\n" + tail


def _patch_identity(monkeypatch, text, checkable=True, from_soul=True):
    monkeypatch.setattr(
        "agent.system_prompt.resolve_identity_block",
        lambda agent: {"text": text, "from_soul": from_soul, "checkable": checkable},
    )


class TestIdentityStalenessRebuild:
    """SOUL.md edits do NOT rebuild the restored prompt (PR #72253 v6
    redesign). ``AGENTS.md:19-23`` only exempts context compression from the
    "never rebuild mid-conversation" rule, and restore is not that
    exception, so drift here must fall through to verbatim reuse — even
    though ``_stored_prompt_matches_runtime`` already rejects Model/
    Provider/cwd/Platform drift on its own, unrelated axis. Identity drift
    reaching a running session is handled at the Hermes-native compaction
    keep-prompt gate instead (see ``TestCompactionIdentityDriftGate``)."""

    def test_edited_soul_reuses_verbatim(self, monkeypatch, caplog):
        stored = _stored_with_identity("OLD SOUL IDENTITY")
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)
        _patch_identity(monkeypatch, "NEW SOUL IDENTITY")

        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(
                agent, None, [{"role": "user", "content": "hi"}]
            )

        assert agent._cached_system_prompt == stored
        agent._build_system_prompt.assert_not_called()
        db.update_system_prompt.assert_not_called()

    def test_trailing_deletion_from_soul_reuses_verbatim(self, monkeypatch):
        """Deleting the TAIL of SOUL.md leaves the new block a prefix of the
        old one — restore doesn't compare them at all anymore, so the
        stored prompt is kept regardless."""
        stored = _stored_with_identity("You are concise.\nNever disclose secrets.")
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)
        _patch_identity(monkeypatch, "You are concise.")

        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "hi"}]
        )

        assert agent._cached_system_prompt == stored
        agent._build_system_prompt.assert_not_called()

    def test_deleted_soul_reuses_verbatim(self, monkeypatch):
        """SOUL.md removed → the resolver would return the hardcoded
        default, but restore never consults it, so the stored (SOUL-built)
        prompt is kept as-is."""
        stored = _stored_with_identity("CUSTOM SOUL IDENTITY")
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)
        _patch_identity(
            monkeypatch, "DEFAULT HARDCODED IDENTITY", from_soul=False
        )

        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "hi"}]
        )

        assert agent._cached_system_prompt == stored
        agent._build_system_prompt.assert_not_called()

    def test_matching_identity_reuses_verbatim(self, monkeypatch):
        identity = "CURRENT SOUL IDENTITY"
        stored = _stored_with_identity(identity)
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)
        _patch_identity(monkeypatch, identity)

        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "hi"}]
        )

        assert agent._cached_system_prompt == stored
        agent._build_system_prompt.assert_not_called()
        db.update_system_prompt.assert_not_called()

    def test_unreadable_soul_fails_open_to_reuse(self, monkeypatch):
        """checkable=False = SOUL.md exists but could not be read. Declaring
        staleness would persist a default-identity downgrade over a healthy
        custom identity, so the check must fail open to reuse."""
        stored = _stored_with_identity("CUSTOM SOUL IDENTITY")
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)
        _patch_identity(
            monkeypatch, "DEFAULT HARDCODED IDENTITY",
            checkable=False, from_soul=False,
        )

        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "hi"}]
        )

        assert agent._cached_system_prompt == stored
        agent._build_system_prompt.assert_not_called()

    def test_resolver_exception_fails_open_to_reuse(self, monkeypatch):
        stored = _stored_with_identity("CUSTOM SOUL IDENTITY")
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)

        def _boom(agent):
            raise RuntimeError("resolver crashed")

        monkeypatch.setattr("agent.system_prompt.resolve_identity_block", _boom)

        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "hi"}]
        )

        assert agent._cached_system_prompt == stored
        agent._build_system_prompt.assert_not_called()


class TestStoredIdentityIsStale:
    """Direct unit tests of ``stored_identity_is_stale()`` itself (v6 §6-F).

    Its only caller is now the compaction keep-prompt gate
    (``TestCompactionIdentityDriftGate``), not restore — the indirect
    coverage above (via ``_restore_or_build_system_prompt``) no longer
    exercises this function at all, since restore was reverted to main's
    behavior and never calls it. These pins keep the anchored-detection and
    fail-open properties covered independent of any caller."""

    def test_trailing_deletion_is_detected_as_stale(self, monkeypatch):
        """Deleting the TAIL of SOUL.md leaves the new block a prefix of the
        old one — a bare containment check would still "match". The anchor
        (help guidance immediately after the identity) catches it."""
        stored = _stored_with_identity("You are concise.\nNever disclose secrets.")
        _patch_identity(monkeypatch, "You are concise.")

        assert stored_identity_is_stale(MagicMock(), stored) is True

    def test_undetermined_identity_fails_open(self, monkeypatch):
        """checkable=True but text="" (no basis to judge, e.g. the file's
        own autouse neutral-resolver default) must not be called stale."""
        stored = _stored_with_identity("CUSTOM SOUL IDENTITY")
        _patch_identity(monkeypatch, "", checkable=True, from_soul=False)

        assert stored_identity_is_stale(MagicMock(), stored) is False

    def test_unreadable_soul_fails_open(self, monkeypatch):
        """checkable=False = SOUL.md exists but could not be read. Declaring
        staleness on unreadable input is not safe, so this must fail open."""
        stored = _stored_with_identity("CUSTOM SOUL IDENTITY")
        _patch_identity(
            monkeypatch, "DEFAULT HARDCODED IDENTITY",
            checkable=False, from_soul=False,
        )

        assert stored_identity_is_stale(MagicMock(), stored) is False

    def test_resolver_exception_fails_open(self, monkeypatch):
        stored = _stored_with_identity("CUSTOM SOUL IDENTITY")

        def _boom(agent):
            raise RuntimeError("resolver crashed")

        monkeypatch.setattr("agent.system_prompt.resolve_identity_block", _boom)

        assert stored_identity_is_stale(MagicMock(), stored) is False


class TestResolveIdentityBlockClassification:
    """Real-resolver pins for the absent/empty/unreadable provenance split
    (#68563 review): a readable-but-empty SOUL.md is the documented way to
    reset to the default personality, so it must stay checkable; only a
    failed read makes staleness unjudgeable."""

    @staticmethod
    def _resolver_agent():
        agent = MagicMock()
        agent.load_soul_identity = True
        agent.skip_context_files = False
        agent.context_compressor = None
        # No profile-scoped home: _agent_home must resolve to None so the
        # resolver falls back to the ambient home these tests configure
        # (a bare MagicMock would fabricate a nonexistent db-derived home
        # and flip `checkable` — see #50233 home scoping).
        agent._session_db = None
        return agent

    @pytest.fixture(autouse=True)
    def _no_seeding(self, monkeypatch):
        # ensure_hermes_home may seed a default SOUL.md on first run; these
        # tests pin the classification of a state the USER created, so the
        # first-run seeding is out of scope and disabled.
        monkeypatch.setattr(
            "hermes_cli.config.ensure_hermes_home", lambda: None
        )

    def test_readable_empty_soul_is_checkable_default(self):
        from hermes_constants import get_hermes_home

        from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

        soul = get_hermes_home() / "SOUL.md"
        soul.parent.mkdir(parents=True, exist_ok=True)
        soul.write_text("   \n\n  ", encoding="utf-8")

        ident = _REAL_RESOLVE_IDENTITY(self._resolver_agent())

        assert ident["checkable"] is True
        assert ident["from_soul"] is False
        assert ident["text"] == DEFAULT_AGENT_IDENTITY

    def test_undecodable_soul_is_not_checkable(self):
        from hermes_constants import get_hermes_home


        soul = get_hermes_home() / "SOUL.md"
        soul.parent.mkdir(parents=True, exist_ok=True)
        soul.write_bytes(b"\xff\xfe\x9c invalid utf-8 \x80")

        ident = _REAL_RESOLVE_IDENTITY(self._resolver_agent())

        assert ident["checkable"] is False
        assert ident["from_soul"] is False

    def test_absent_soul_is_checkable_default(self):
        from hermes_constants import get_hermes_home

        from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

        soul = get_hermes_home() / "SOUL.md"
        assert not soul.exists()

        ident = _REAL_RESOLVE_IDENTITY(self._resolver_agent())

        assert ident["checkable"] is True
        assert ident["from_soul"] is False
        assert ident["text"] == DEFAULT_AGENT_IDENTITY

    def test_absent_hermes_home_is_not_checkable(self, monkeypatch, tmp_path):
        """v6 class 12: HERMES_HOME itself missing/unmounted is a different
        failure mode than an existing HERMES_HOME with no SOUL.md inside it
        (the case above). Nothing here can confirm what identity applies, so
        this must NOT be checkable — fail open to reuse on restore, rather
        than confidently reporting "default identity"."""
        missing_home = tmp_path / "does_not_exist"
        assert not missing_home.exists()
        monkeypatch.setenv("HERMES_HOME", str(missing_home))

        ident = _REAL_RESOLVE_IDENTITY(self._resolver_agent())

        assert ident["checkable"] is False
        assert ident["from_soul"] is False


# ---------------------------------------------------------------------------
# PR #72253 redesign (v6) — pins for the compaction keep-prompt gate
# (agent/conversation_compression.py). §5 applies identity drift detection
# ONLY at this gate (the one place AGENTS.md:19-23 already carves out as the
# explicit exception), not on the restore hot path above. The gate now adds
# `and not stored_identity_is_stale(...)` to its keep condition (implemented
# in this diff); the tests below exercise the real, non-xfail behavior.
# See 72253-redesign-v6.md §6-D.
# ---------------------------------------------------------------------------


class TestCompactionIdentityDriftGate:
    """Real AIAgent + real SessionDB, mirroring
    tests/run_agent/test_in_place_compaction.py's harness, so compress_context
    runs its real locking/persistence path instead of a mocked stand-in."""

    def _make_agent(self, session_db, session_id, cached_prompt):
        import os
        from unittest.mock import patch as _patch

        with _patch.dict(os.environ, {"OPENROUTER_API_KEY": "tk"}):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="tk",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id=session_id,
                skip_context_files=True,
                skip_memory=True,
            )
        agent.compression_in_place = True
        agent._cached_system_prompt = cached_prompt

        def _fake_compress(messages, current_tokens=None, focus_topic=None, force=False):
            return [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary of prior turns"},
                {"role": "assistant", "content": "recent reply"},
            ]

        agent.context_compressor.compress = _fake_compress
        agent.context_compressor._last_compress_aborted = False
        agent.context_compressor._last_summary_error = None
        agent.context_compressor.compression_count = 1
        return agent

    def _seed(self, db, sid, n=8):
        db.create_session(sid, "cli", model="test/model")
        for i in range(n):
            db.append_message(
                session_id=sid,
                role="user" if i % 2 == 0 else "assistant",
                content=f"msg {i}",
            )

    def test_compaction_keeps_prompt_when_identity_unchanged(self, monkeypatch):
        """Existing keep-prompt optimization (KV-cache retention) must survive
        the redesign untouched — regression pin, expected to PASS today."""
        import tempfile
        from pathlib import Path

        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        identity = "CURRENT SOUL IDENTITY"
        stored = _stored_with_identity(identity)
        monkeypatch.setattr(
            "agent.system_prompt.resolve_identity_block",
            lambda agent: {"text": identity, "from_soul": True, "checkable": True},
        )
        rebuild = MagicMock(return_value="SHOULD_NOT_BE_USED")

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260804_120000_keep01"
            self._seed(db, sid)
            agent = self._make_agent(db, sid, stored)
            agent._build_system_prompt = rebuild

            _compressed, new_sp = compress_context(
                agent, [{"role": "user", "content": "x"}] * 8,
                approx_tokens=100_000, system_message="sys",
            )

            assert new_sp == stored
            rebuild.assert_not_called()
            assert db.get_session(sid)["system_prompt"] == stored
            db.close()

    def test_compaction_rebuilds_prompt_when_soul_changed(self, monkeypatch):
        """v6 §5: identity drift suppresses the keep optimization so the
        rebuilt (new-identity) prompt is what gets persisted at the
        compaction boundary — this is the one place AGENTS.md:19-23 already
        allows the system prompt to change mid-conversation.

        The keep gate in agent/conversation_compression.py now adds
        `and not stored_identity_is_stale(agent, cached_system_prompt)` to
        its existing (cached_system_prompt is not None, _memory_manager is
        None, _cached_prompt_reflects_builtin_memory) condition, so drift
        falls through to the rebuild branch instead of keeping stale bytes.
        """
        import tempfile
        from pathlib import Path

        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        stored = _stored_with_identity("OLD SOUL IDENTITY")
        new_prompt = _stored_with_identity("NEW SOUL IDENTITY")
        monkeypatch.setattr(
            "agent.system_prompt.resolve_identity_block",
            lambda agent: {
                "text": "NEW SOUL IDENTITY", "from_soul": True, "checkable": True
            },
        )
        rebuild = MagicMock(return_value=new_prompt)

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260804_120100_drift01"
            self._seed(db, sid)
            agent = self._make_agent(db, sid, stored)
            agent._build_system_prompt = rebuild

            _compressed, new_sp = compress_context(
                agent, [{"role": "user", "content": "x"}] * 8,
                approx_tokens=100_000, system_message="sys",
            )

            assert new_sp == new_prompt, (
                "expected the keep gate to be suppressed by identity drift "
                f"and rebuild to reflect it; got stale kept bytes: {new_sp!r}"
            )
            rebuild.assert_called_once()
            assert db.get_session(sid)["system_prompt"] == new_prompt
            db.close()

    def test_adopted_child_path_passes_through_on_drift(self, monkeypatch):
        """v6 §4.2 (F6): when another process has already rotated this
        session via its own compression, the adopting call returns the
        CHILD's own already-persisted prompt unchanged — never a rebuild,
        even if that prompt would be judged stale by the current resolver.
        ``_adopt_live_compression_child`` sets ``agent._cached_system_prompt``
        to the child's value before this call site reads it, so identity
        drift on top of that must not trigger a rebuild here. This documents
        the intentional pass-through rather than relying on it silently."""
        import tempfile
        from pathlib import Path

        from hermes_state import SessionDB
        import agent.conversation_compression as cc_module

        child_prompt = _stored_with_identity("CHILD PERSISTED IDENTITY")
        _patch_identity(monkeypatch, "SOME OTHER NEW IDENTITY")
        rebuild = MagicMock(return_value="SHOULD_NOT_BE_USED")

        def _fake_rotated(db, sid):
            return True

        def _fake_adopt(agent, db, sid):
            agent._cached_system_prompt = child_prompt
            return [{"role": "user", "content": "recovered from child"}]

        monkeypatch.setattr(
            cc_module, "_session_was_rotated_by_compression", _fake_rotated
        )
        monkeypatch.setattr(
            cc_module, "_adopt_live_compression_child", _fake_adopt
        )

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260804_120200_adopt01"
            self._seed(db, sid)
            agent = self._make_agent(db, sid, "PARENT STALE PROMPT")
            agent._build_system_prompt = rebuild

            _compressed, new_sp = cc_module.compress_context(
                agent, [{"role": "user", "content": "x"}] * 8,
                approx_tokens=100_000, system_message="sys",
            )

            assert new_sp == child_prompt, (
                "adoption path must pass through the child's persisted "
                f"prompt unchanged despite drift; got {new_sp!r}"
            )
            rebuild.assert_not_called()
            db.close()


# ---------------------------------------------------------------------------
# PR #72253 redesign (v6) — pass-through pins for the CAS (codex_app_server)
# compression path, which this PR intentionally leaves untouched (v6 §3.2).
# That path has no system-prompt write-back on `main` or here, so adding
# drift detection to it would change the in-flight prompt for one turn and
# then lose it on the next restore. These pins document that limitation
# rather than relying on it silently. See 72253-redesign-v6.md §6-D item 5.
# ---------------------------------------------------------------------------


class TestCasPassThroughOnDrift:
    """Mirrors tests/run_agent/test_codex_app_server_compaction.py's
    DummyAgent/FakeCodexSession pattern — the CAS dispatch in
    compress_context never acquires the compression lock or touches
    session_db, so a lightweight fake agent (rather than the real
    AIAgent+SessionDB harness above) is enough to reach every return form."""

    class _FakeCodexSession:
        def __init__(self, result):
            self.result = result
            self.calls = 0

        def compact_thread(self):
            self.calls += 1
            return self.result

        def close(self):
            pass

    @staticmethod
    def _cas_agent(cached_prompt, *, auto_compaction, codex_session):
        from types import SimpleNamespace

        agent = MagicMock()
        agent.api_mode = "codex_app_server"
        agent.codex_app_server_auto_compaction = auto_compaction
        agent.session_id = "cas-drift-session"
        agent.platform = "cli"
        agent._cached_system_prompt = cached_prompt
        agent._codex_session = codex_session
        agent._build_system_prompt = MagicMock(return_value="SHOULD_NOT_BE_BUILT")
        agent.context_compressor = SimpleNamespace(
            compression_count=0,
            last_compression_rough_tokens=0,
            last_prompt_tokens=0,
            last_completion_tokens=0,
            awaiting_real_usage_after_compression=False,
        )
        return agent

    @staticmethod
    def _run(agent, force):
        from agent.conversation_compression import compress_context

        return compress_context(
            agent, [{"role": "user", "content": "hi"}], "sys",
            approx_tokens=100_000, force=force,
        )

    def test_skip_native_mode_passes_through_on_drift(self, monkeypatch):
        stored = _stored_with_identity("OLD SOUL IDENTITY")
        _patch_identity(monkeypatch, "NEW SOUL IDENTITY")
        agent = self._cas_agent(
            stored, auto_compaction="native", codex_session=None
        )

        _messages, prompt = self._run(agent, force=False)

        assert prompt == stored
        agent._build_system_prompt.assert_not_called()

    def test_thread_absent_passes_through_on_drift(self, monkeypatch):
        stored = _stored_with_identity("OLD SOUL IDENTITY")
        _patch_identity(monkeypatch, "NEW SOUL IDENTITY")
        agent = self._cas_agent(
            stored, auto_compaction="hermes", codex_session=None
        )

        _messages, prompt = self._run(agent, force=True)

        assert prompt == stored
        agent._build_system_prompt.assert_not_called()

    def test_failure_passes_through_on_drift(self, monkeypatch):
        from agent.transports.codex_app_server_session import TurnResult

        stored = _stored_with_identity("OLD SOUL IDENTITY")
        _patch_identity(monkeypatch, "NEW SOUL IDENTITY")
        session = self._FakeCodexSession(TurnResult(interrupted=True))
        agent = self._cas_agent(
            stored, auto_compaction="hermes", codex_session=session
        )

        _messages, prompt = self._run(agent, force=True)

        assert prompt == stored
        assert session.calls == 1
        agent._build_system_prompt.assert_not_called()

    def test_success_passes_through_on_drift(self, monkeypatch):
        from agent.transports.codex_app_server_session import TurnResult

        stored = _stored_with_identity("OLD SOUL IDENTITY")
        _patch_identity(monkeypatch, "NEW SOUL IDENTITY")
        session = self._FakeCodexSession(TurnResult(compacted=True))
        agent = self._cas_agent(
            stored, auto_compaction="hermes", codex_session=session
        )

        _messages, prompt = self._run(agent, force=True)

        assert prompt == stored, (
            "CAS success has no system-prompt write-back (v6 F1 residual) "
            "— identity drift must not reach the returned prompt even on a "
            f"successful compaction; got {prompt!r}"
        )
        assert session.calls == 1
        agent._build_system_prompt.assert_not_called()


# ---------------------------------------------------------------------------
# TestSessionStartHookGuard was removed from this file (v6 §6-B). It pinned
# on_session_start firing/not-firing across the identity-stale, fresh-build,
# and preview-restart restore states. This PR only touches the compaction
# keep-prompt gate, not restore, so the identity-stale restore state it was
# guarding no longer exists here — it belongs to a follow-up PR that fixes
# the pre-existing on_session_start double-fire bug for the stale_runtime
# and null/empty stored-prompt classes in
# agent/conversation_loop.py's restore path (unrelated to SOUL.md identity).
# That follow-up PR must carry forward the preview-restart boundary case
# this class used to pin (a brand-new session that legitimately receives
# nonempty PARENT history but has no session DB): a guard keyed on history
# truthiness instead of the specific stale states would wrongly suppress
# that session's on_session_start hook.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
