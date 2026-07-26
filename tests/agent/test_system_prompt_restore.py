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
    """SOUL.md edits must reach continuing sessions (issue #68563).

    ``_stored_prompt_matches_runtime`` only rejects Model/Provider/cwd/
    Platform drift; identity content drift previously left the stale prompt
    reused verbatim forever. The check is anchored on the identity block
    plus ``HERMES_AGENT_HELP_GUIDANCE`` (the always-present next stable
    component), which supplies the boundary a bare substring lacks."""

    def test_edited_soul_rebuilds_and_persists(self, monkeypatch, caplog):
        stored = _stored_with_identity("OLD SOUL IDENTITY")
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)
        _patch_identity(monkeypatch, "NEW SOUL IDENTITY")

        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(
                agent, None, [{"role": "user", "content": "hi"}]
            )

        assert agent._cached_system_prompt == "BUILT_PROMPT"
        agent._build_system_prompt.assert_called_once()
        db.update_system_prompt.assert_called_once_with(
            "test-session-id", "BUILT_PROMPT"
        )
        assert any("stale identity" in r.getMessage() for r in caplog.records)

    def test_trailing_deletion_from_soul_is_detected(self, monkeypatch):
        """Deleting the TAIL of SOUL.md leaves the new block a prefix of the
        old one — a bare containment check would still "match". The anchor
        (help guidance immediately after the identity) catches it."""
        stored = _stored_with_identity("You are concise.\nNever disclose secrets.")
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)
        _patch_identity(monkeypatch, "You are concise.")

        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "hi"}]
        )

        assert agent._cached_system_prompt == "BUILT_PROMPT"
        agent._build_system_prompt.assert_called_once()

    def test_deleted_soul_falls_back_to_default_and_rebuilds(self, monkeypatch):
        """SOUL.md removed → the resolver returns the hardcoded default,
        which a SOUL-built stored prompt does not start with → rebuild."""
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

        assert agent._cached_system_prompt == "BUILT_PROMPT"
        agent._build_system_prompt.assert_called_once()

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


class TestSessionStartHookGuard:
    """on_session_start is documented "not on continuation": a stale-prompt
    fallthrough is a CONTINUING session getting its prompt rebuilt, so the
    hook must not re-fire there (it would duplicate session-scoped plugin
    work). Fresh builds keep firing it."""

    def test_stale_identity_fallthrough_does_not_fire_hook(self, monkeypatch):
        stored = _stored_with_identity("OLD SOUL IDENTITY")
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)
        _patch_identity(monkeypatch, "NEW SOUL IDENTITY")
        hook = MagicMock()
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)

        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "hi"}]
        )

        assert agent._cached_system_prompt == "BUILT_PROMPT"
        assert not any(
            c.args and c.args[0] == "on_session_start" for c in hook.call_args_list
        )

    def test_fresh_build_still_fires_hook(self, monkeypatch):
        agent = _make_agent(session_db=None)
        hook = MagicMock()
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)

        _restore_or_build_system_prompt(agent, None, [])

        assert any(
            c.args and c.args[0] == "on_session_start" for c in hook.call_args_list
        )

    def test_preview_restart_with_parent_history_still_fires_hook(self, monkeypatch):
        """The preview-restart shape: a brand-new session that deliberately
        receives nonempty PARENT history but has no session DB. Its start
        hook is legitimate — a guard keyed on history truthiness (instead of
        the stale states) would wrongly suppress it."""
        agent = _make_agent(session_db=None)
        hook = MagicMock()
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)

        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "from parent session"}]
        )

        assert agent._cached_system_prompt == "BUILT_PROMPT"
        assert any(
            c.args and c.args[0] == "on_session_start" for c in hook.call_args_list
        )


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
