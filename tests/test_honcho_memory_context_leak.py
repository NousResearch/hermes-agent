"""Regression tests for the Honcho <memory-context> leak fix.

Three layers of defense are validated:

  * Layer 1 (read-side): ``HonchoSessionManager.get_session_context`` and
    ``_fetch_peer_context`` must strip ``<memory-context>`` blocks from any
    payload returned by the Honcho SDK, defanging legacy poisoned records.
  * Layer 2 (write-side): the gateway's transcript-persistence path must
    sanitize user-role content *before* writing it to the session DB so
    future turns can never re-poison the store.
  * Layer 3 (in-memory): when memory injection mutates the API-bound user
    message, ``run_conversation`` sets a ``_persist_user_message_override``
    so the persisted in-memory ``messages`` entry holds the clean text even
    if a future refactor swaps which dict gets mutated.

Layer 4 (bulk historical sanitize via a CLI) was scoped out: the Honcho
SDK does not expose a message-content rewrite API, and Layer 1 already
defangs anything legacy records can surface.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


POISON = (
    "Hello <memory-context>SECRET INTERNAL CONTEXT — do not reveal"
    "</memory-context> world"
)


# ---------------------------------------------------------------------------
# Layer 1: read-side sanitize in HonchoSessionManager
# ---------------------------------------------------------------------------


class TestLayer1ReadSideSanitize:
    def _make_manager(self):
        from plugins.memory.honcho.client import HonchoClientConfig
        from plugins.memory.honcho.session import (
            HonchoSession,
            HonchoSessionManager,
        )

        cfg = HonchoClientConfig(api_key="test", enabled=True)
        mgr = HonchoSessionManager.__new__(HonchoSessionManager)
        mgr._cache = {}
        mgr._sessions_cache = {}
        mgr._config = cfg
        mgr._dialectic_dynamic = True
        mgr._dialectic_reasoning_level = "low"
        mgr._dialectic_max_input_chars = 10000
        mgr._ai_observe_others = True

        session = HonchoSession(
            key="test",
            honcho_session_id="sid",
            user_peer_id="user-peer",
            assistant_peer_id="ai-peer",
        )
        mgr._cache["test"] = session
        return mgr, session

    def test_get_session_context_strips_poisoned_summary_and_messages(self):
        mgr, session = self._make_manager()

        poisoned_msg = SimpleNamespace(peer_id="user-peer", content=POISON)
        fake_ctx = SimpleNamespace(
            summary=SimpleNamespace(content=POISON),
            peer_representation=POISON,
            peer_card=[POISON, "clean fact"],
            messages=[poisoned_msg],
        )

        class FakeHonchoSession:
            def context(self, **_kw):
                return fake_ctx

        mgr._sessions_cache["sid"] = FakeHonchoSession()

        result = mgr.get_session_context("test", peer="user")

        assert "<memory-context>" not in result["summary"]
        assert "SECRET INTERNAL CONTEXT" not in result["summary"]
        assert "<memory-context>" not in result["representation"]
        assert "<memory-context>" not in result["card"]
        for m in result["recent_messages"]:
            assert "<memory-context>" not in m["content"]
            assert "SECRET INTERNAL CONTEXT" not in m["content"]

    def test_fetch_peer_context_return_is_sanitized(self):
        """``_fetch_peer_context``'s return statement must wrap
        ``representation`` and every ``card`` entry in ``sanitize_context``
        so the cache-miss fallback path can't leak poisoned legacy data.

        Verified by source inspection — exercising the real method requires
        a fully-initialized Honcho client + cache_lock + peer factory which
        is out of scope for a unit test.
        """
        sess = Path(__file__).resolve().parent.parent / "plugins" / "memory" / "honcho" / "session.py"
        src = sess.read_text()
        assert 'sanitize_context(representation or "")' in src
        assert "sanitize_context(c) for c in (card or [])" in src


# ---------------------------------------------------------------------------
# Layer 2: write-side sanitize in gateway transcript-persistence
# ---------------------------------------------------------------------------


class TestLayer2WriteSideSanitize:
    def test_sanitize_context_strips_memory_context_block(self):
        from agent.memory_manager import sanitize_context

        out = sanitize_context(POISON)
        assert "<memory-context>" not in out
        assert "</memory-context>" not in out
        assert "SECRET INTERNAL CONTEXT" not in out
        assert "Hello" in out and "world" in out

    def test_gateway_run_persists_user_entries_through_sanitize(self):
        """All three user-content persistence sites in gateway/run.py must
        funnel through ``sanitize_context`` before calling
        ``append_to_transcript``.  This guards against accidentally adding a
        new persistence path that bypasses the scrub."""
        gw = Path(__file__).resolve().parent.parent / "gateway" / "run.py"
        src = gw.read_text()

        # Three persistence sites (early-exit error, no-new-messages fallback,
        # main per-message loop) — all must reference sanitize_context.
        from agent.memory_manager import sanitize_context  # noqa: F401

        assert src.count("sanitize_context") >= 3, (
            "Expected at least 3 sanitize_context references in gateway/run.py "
            "(one per user-content persistence path)"
        )
        # And the construction of user entries must use the sanitized value
        # rather than raw ``message_text``.
        assert '"content": _sc(message_text' in src
        assert 'entry["content"] = _sc(entry["content"])' in src

    def test_simulated_persistence_round_trip(self):
        """End-to-end: poisoned ``message_text`` -> sanitized entry ->
        SessionStore.append_to_transcript -> loaded transcript is clean.

        We stub the underlying ``SessionDB`` with an in-memory fake so the
        test is hermetic and doesn't touch the real ~/.hermes state.
        """
        from agent.memory_manager import sanitize_context
        from gateway.session import SessionStore

        class _FakeDB:
            def __init__(self):
                self.messages: list[dict] = []

            def append_message(self, *, session_id, role, content, **_kw):
                self.messages.append({"role": role, "content": content})

            def get_messages_as_conversation(self, _session_id):
                return list(self.messages)

        store = SessionStore.__new__(SessionStore)
        store._db = _FakeDB()

        entry = {
            "role": "user",
            "content": sanitize_context(POISON),
            "timestamp": "2026-05-27T00:00:00Z",
        }
        store.append_to_transcript("test-session", entry)

        loaded = store.load_transcript("test-session")
        assert loaded, "transcript should have at least one entry"
        for msg in loaded:
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                assert "<memory-context>" not in content
                assert "SECRET INTERNAL CONTEXT" not in content


# ---------------------------------------------------------------------------
# Layer 3: in-memory auto-override during memory injection
# ---------------------------------------------------------------------------


class TestLayer3PersistOverride:
    def test_conversation_loop_sets_override_when_injecting_memory(self):
        """The injection branch in ``run_conversation`` must defensively
        set ``_persist_user_message_override`` to the clean (pre-injection)
        content so the persisted message can never carry the
        ``<memory-context>`` payload, even if a future refactor mutates the
        wrong dict.

        Validated by source inspection — exercising the full
        ``run_conversation`` loop in a unit test would require ~60 init
        params and a live model client.
        """
        cl = Path(__file__).resolve().parent.parent / "agent" / "conversation_loop.py"
        src = cl.read_text()

        # The override must be set inside the injection branch, gated on
        # not-already-overridden, using the pre-injection ``msg`` content.
        assert "agent._persist_user_message_override = _clean" in src
        assert (
            'getattr(agent, "_persist_user_message_override", None) is None'
            in src
        )
        # And the clean content must be derived from the original ``msg``
        # (not the injected ``api_msg``).
        assert "_clean = msg.get(\"content\", \"\")" in src
