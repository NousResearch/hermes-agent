"""Regression test: stale-stream client rebuild must not fail when
_client_kwargs["api_key"] is stale/missing for chat_completions providers.

PRODUCTION BUG PATTERN (yunwu / api.wlai.vip):
  WARNING run_agent: Failed to rebuild shared OpenAI client (stale_stream_pool_cleanup)
      provider=yunwu base_url=https://api.wlai.vip model=claude-sonnet-4-6
      error=The api_key client option must be set either by passing api_key to
            the client or by setting the OPENAI_API_KEY environment variable

Sequence:
  1. Agent initialised for a chat_completions provider (e.g. yunwu / wlai.vip).
  2. A mid-session credential swap (pool rotation, _swap_credential, or a path
     that rebuilds _client_kwargs without carrying the api_key forward) leaves
     _client_kwargs["api_key"] as None or absent while self.api_key still holds
     the correct value.
  3. The outer poll loop in interruptible_streaming_api_call() detects a stale
     stream and calls agent._replace_primary_openai_client(reason=
     "stale_stream_pool_cleanup").
  4. _replace_primary_openai_client reads _client_kwargs["api_key"] → None/missing
     → OpenAI SDK raises "api_key must be set" → client rebuild fails → every
     subsequent request on that agent fails until a full restart.

Fix: _replace_primary_openai_client (and _ensure_primary_openai_client) now
refresh _client_kwargs["api_key"] from self.api_key before calling
_create_openai_client, so the dict can never be stale at rebuild time.

NOTE: the similar-sounding #28161 fix (test_28161_anthropic_stream_pool_cleanup)
covers the anthropic_messages path by NOT calling _replace_primary_openai_client
at all. This test covers the separate chat_completions path where
_replace_primary_openai_client IS the correct rebuild mechanism.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_completions_agent(api_key: str = "sk-yunwu-real-key", **kwargs):
    """Minimal AIAgent wired for chat_completions (OpenAI-compat) mode."""
    from run_agent import AIAgent

    defaults = dict(
        api_key=api_key,
        base_url="https://api.wlai.vip/v1",
        model="claude-sonnet-4-6",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    defaults.update(kwargs)
    agent = AIAgent(**defaults)
    # Confirm we're in the right mode — this test is only for chat_completions.
    assert agent.api_mode == "chat_completions", (
        f"Expected chat_completions api_mode, got {agent.api_mode!r}. "
        "The test is only meaningful for OpenAI-compat providers."
    )
    return agent


# ---------------------------------------------------------------------------
# Core regression: api_key missing from _client_kwargs at rebuild time
# ---------------------------------------------------------------------------


class TestStaleStreamPoolCleanupChatCompletions:
    """_replace_primary_openai_client must succeed even when
    _client_kwargs["api_key"] is None/missing at the moment the stale-stream
    cleanup fires — self.api_key is the authoritative source."""

    def test_replace_succeeds_when_client_kwargs_api_key_is_none(self, monkeypatch):
        """Simulate the production failure: _client_kwargs["api_key"] is None
        (e.g. after a code path that rebuilds the dict without forwarding the
        key), but self.api_key is correct.

        Before the fix: _replace_primary_openai_client calls
        _create_openai_client(self._client_kwargs) → OpenAI(api_key=None) →
        raises "The api_key client option must be set…".

        After the fix: _replace_primary_openai_client refreshes
        _client_kwargs["api_key"] from self.api_key before the call → succeeds.
        """
        agent = _make_chat_completions_agent(api_key="sk-yunwu-real-key")

        # Corrupt _client_kwargs as the bug does: api_key replaced with None.
        agent._client_kwargs["api_key"] = None

        # self.api_key still holds the correct value (as it does in production).
        assert agent.api_key == "sk-yunwu-real-key"

        constructed_kwargs: list[dict] = []

        def fake_openai(**kwargs):
            constructed_kwargs.append(dict(kwargs))
            mock_client = MagicMock()
            mock_client._closed = False
            return mock_client

        monkeypatch.setattr("run_agent.OpenAI", fake_openai)

        result = agent._replace_primary_openai_client(reason="stale_stream_pool_cleanup")

        assert result is True, (
            "_replace_primary_openai_client returned False — rebuild failed. "
            "This reproduces the stale_stream_pool_cleanup production bug where "
            "_client_kwargs['api_key'] is None and the OpenAI SDK raises."
        )
        assert constructed_kwargs, "OpenAI constructor was never called"
        used_key = constructed_kwargs[-1].get("api_key")
        assert used_key == "sk-yunwu-real-key", (
            f"OpenAI was called with api_key={used_key!r} instead of "
            f"'sk-yunwu-real-key'. The fix must forward self.api_key into "
            f"_client_kwargs before rebuilding."
        )
        # The dict must be patched in-place too.
        assert agent._client_kwargs.get("api_key") == "sk-yunwu-real-key", (
            "_client_kwargs['api_key'] was not updated in-place by the fix."
        )

    def test_replace_succeeds_when_client_kwargs_missing_api_key(self, monkeypatch):
        """Same as above but api_key key is entirely absent from _client_kwargs."""
        agent = _make_chat_completions_agent(api_key="sk-real-key-2")
        agent._client_kwargs.pop("api_key", None)
        assert "api_key" not in agent._client_kwargs

        constructed_kwargs: list[dict] = []

        def fake_openai(**kwargs):
            constructed_kwargs.append(dict(kwargs))
            mock_client = MagicMock()
            mock_client._closed = False
            return mock_client

        monkeypatch.setattr("run_agent.OpenAI", fake_openai)

        result = agent._replace_primary_openai_client(reason="stale_stream_pool_cleanup")

        assert result is True
        assert constructed_kwargs[-1].get("api_key") == "sk-real-key-2"

    def test_ensure_primary_succeeds_when_client_kwargs_api_key_is_none(
        self, monkeypatch
    ):
        """_ensure_primary_openai_client has the same rebuild logic and must
        also refresh api_key from self.api_key before rebuilding."""
        agent = _make_chat_completions_agent(api_key="sk-ensure-key")

        # Force the client to appear closed so _ensure triggers a rebuild.
        closed_mock = MagicMock()
        closed_mock._closed = True
        agent.client = closed_mock

        with patch.object(
            agent, "_is_openai_client_closed", return_value=True
        ):
            # Corrupt _client_kwargs.
            agent._client_kwargs["api_key"] = None

            constructed_kwargs: list[dict] = []

            def fake_openai(**kwargs):
                constructed_kwargs.append(dict(kwargs))
                mock_client = MagicMock()
                mock_client._closed = False
                return mock_client

            monkeypatch.setattr("run_agent.OpenAI", fake_openai)

            new_client = agent._ensure_primary_openai_client(
                reason="stale_stream_pool_cleanup"
            )

        assert new_client is not None
        assert constructed_kwargs[-1].get("api_key") == "sk-ensure-key", (
            "_ensure_primary_openai_client did not refresh api_key from self.api_key"
        )

    def test_client_kwargs_api_key_stays_in_sync_after_swap(self, monkeypatch):
        """After a credential swap that updates self.api_key,
        _replace_primary_openai_client must use the NEW key, not whatever
        was left in _client_kwargs by the previous write."""
        agent = _make_chat_completions_agent(api_key="sk-old-key")
        # Simulate a credential rotation that updated self.api_key but left
        # _client_kwargs with the old value (the race this fix closes).
        agent.api_key = "sk-rotated-key"
        agent._client_kwargs["api_key"] = "sk-old-key"  # stale

        constructed_kwargs: list[dict] = []

        def fake_openai(**kwargs):
            constructed_kwargs.append(dict(kwargs))
            mock_client = MagicMock()
            mock_client._closed = False
            return mock_client

        monkeypatch.setattr("run_agent.OpenAI", fake_openai)

        result = agent._replace_primary_openai_client(reason="stale_stream_pool_cleanup")

        assert result is True
        used_key = constructed_kwargs[-1].get("api_key")
        assert used_key == "sk-rotated-key", (
            f"Expected rotated key 'sk-rotated-key', got {used_key!r}. "
            "The rebuild should use self.api_key (current) not the stale dict value."
        )
