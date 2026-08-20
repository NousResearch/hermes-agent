"""Tests for the sharedSessions cross-session recall feature.

Added in PR introducing `shared_sessions` config field on HonchoClientConfig.
Exercises three paths:

  1. Config default — `shared_sessions` defaults to empty list and the merge
     path is a no-op.
  2. Config with a shared session — `get_session_context` calls into the
     shared session's `context()` method and merges the result under a
     `shared_context` key.
  3. Failure isolation — when a shared session's `context()` raises, the
     exception is swallowed at debug level and other shared sessions still
     contribute.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSessionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(peer_id: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(peer_id=peer_id, content=content)


def _make_context(summary_text: str | None = None, messages: list | None = None) -> SimpleNamespace:
    summary_obj = SimpleNamespace(content=summary_text) if summary_text else None
    return SimpleNamespace(
        summary=summary_obj,
        peer_representation=None,
        peer_card=None,
        messages=messages or [],
    )


def _build_manager(shared_sessions: list[str]) -> HonchoSessionManager:
    """Build a HonchoSessionManager with a mocked Honcho client + config."""
    config = HonchoClientConfig(
        host="hermes",
        workspace_id="hermes",
        peer_name="user",
        ai_peer="assistant",
        shared_sessions=shared_sessions,
    )
    honcho_mock = MagicMock()
    return HonchoSessionManager(honcho=honcho_mock, config=config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_shared_sessions_default_empty():
    """Default config has empty shared_sessions; no merge work happens."""
    config = HonchoClientConfig(host="hermes", workspace_id="hermes")
    assert config.shared_sessions == []
    assert isinstance(config.shared_sessions, list)

    manager = HonchoSessionManager(config=config)
    assert manager._shared_sessions == []


def test_shared_sessions_field_accepts_explicit_list():
    """Explicit list is preserved through dataclass init."""
    config = HonchoClientConfig(
        host="hermes",
        workspace_id="hermes",
        shared_sessions=["canonical-knowledge", "team-decisions"],
    )
    assert config.shared_sessions == ["canonical-knowledge", "team-decisions"]


def test_get_session_context_merges_shared_session(monkeypatch):
    """When shared_sessions is set and the current session resolves, the
    result includes a shared_context entry for each configured session."""
    manager = _build_manager(shared_sessions=["canonical"])

    # Mock the current-session lookup so get_session_context proceeds.
    current_session = SimpleNamespace(
        honcho_session_id="current",
        user_peer_id="user",
        assistant_peer_id="assistant",
    )
    manager._cache["current"] = current_session

    # Mock the honcho_session for the current session.
    honcho_session_mock = MagicMock()
    honcho_session_mock.context.return_value = _make_context(
        summary_text="current session summary",
        messages=[_make_message("user", "hello")],
    )
    manager._sessions_cache[current_session.honcho_session_id] = honcho_session_mock

    # Mock the shared session lookup.
    shared_session_mock = MagicMock()
    shared_session_mock.context.return_value = _make_context(
        summary_text="canonical content",
        messages=[
            _make_message("user", "Widget X uses protocol Y"),
            _make_message("assistant", "noted"),
        ],
    )
    manager.honcho.session = MagicMock(return_value=shared_session_mock)

    result = manager.get_session_context("current", peer="user")

    assert "shared_context" in result
    assert len(result["shared_context"]) == 1
    entry = result["shared_context"][0]
    assert entry["session_id"] == "canonical"
    assert entry["summary"] == "canonical content"
    assert len(entry["recent_messages"]) == 2
    assert entry["recent_messages"][0]["content"] == "Widget X uses protocol Y"


def test_shared_session_exception_is_isolated(monkeypatch, caplog):
    """A shared session raising during context() does not break the response,
    and other shared sessions still contribute."""
    manager = _build_manager(shared_sessions=["bad-id", "good-id"])

    current_session = SimpleNamespace(
        honcho_session_id="current",
        user_peer_id="user",
        assistant_peer_id="assistant",
    )
    manager._cache["current"] = current_session

    honcho_session_mock = MagicMock()
    honcho_session_mock.context.return_value = _make_context(summary_text="ok")
    manager._sessions_cache[current_session.honcho_session_id] = honcho_session_mock

    def session_factory(sid: str):
        s = MagicMock()
        if sid == "bad-id":
            s.context.side_effect = RuntimeError("simulated network failure")
        else:
            s.context.return_value = _make_context(
                summary_text="good content",
                messages=[_make_message("user", "still works")],
            )
        return s

    manager.honcho.session = MagicMock(side_effect=session_factory)

    result = manager.get_session_context("current", peer="user")

    # The good session contributed; the bad one was silently isolated.
    assert "shared_context" in result
    sids = [e["session_id"] for e in result["shared_context"]]
    assert "good-id" in sids
    assert "bad-id" not in sids


def test_shared_sessions_empty_list_skips_merge_loop():
    """Empty list means the merge code path is skipped entirely — result
    does not contain a shared_context key."""
    manager = _build_manager(shared_sessions=[])

    current_session = SimpleNamespace(
        honcho_session_id="current",
        user_peer_id="user",
        assistant_peer_id="assistant",
    )
    manager._cache["current"] = current_session

    honcho_session_mock = MagicMock()
    honcho_session_mock.context.return_value = _make_context(summary_text="ok")
    manager._sessions_cache[current_session.honcho_session_id] = honcho_session_mock

    result = manager.get_session_context("current", peer="user")
    assert "shared_context" not in result
