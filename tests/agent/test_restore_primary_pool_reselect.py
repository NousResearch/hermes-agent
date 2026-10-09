# Copyright 2025 Nous Research (Licensed under the Apache License, Version 2.0)
"""Test that _restore_primary_runtime re-selects from the credential pool
instead of using a stale snapshot key.

Bug: when a credential pool entry is revoked/marked-exhausted during a turn,
_restore_primary_runtime restores the original (now-stale) api_key from the
construction-time snapshot. The next turn immediately hits the same error,
exhausting remaining entries and falling through to cross-provider fallback.
"""

import time
from unittest.mock import MagicMock

import pytest

from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    PooledCredential,
)


def _make_entry(
    label: str,
    access_token: str,
    *,
    source: str = "device_code",
    priority: int = 0,
    last_status: str | None = None,
    last_status_at: float | None = None,
) -> dict:
    return {
        "id": label,
        "label": label,
        "provider": "openai-codex",
        "auth_type": AUTH_TYPE_OAUTH,
        "source": source,
        "priority": priority,
        "access_token": access_token,
        "refresh_token": f"rt-{label}",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "last_status": last_status,
        "last_status_at": last_status_at,
    }


def _build_mock_pool(entries: list[dict], *, strategy: str = "round_robin"):
    """Build a mock CredentialPool with the given entries."""
    from agent.credential_pool import CredentialPool

    pool = CredentialPool(
        provider="openai-codex",
        entries=[PooledCredential.from_dict("openai-codex", e) for e in entries],
    )
    pool._strategy = strategy
    return pool


class TestRestorePrimaryPoolReselect:
    """_restore_primary_runtime should re-select from the credential pool."""

    def _make_agent(self, pool):
        """Create a minimal AIAgent with the given credential pool."""
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent.model = "gpt-5.5"
        agent.provider = "openai-codex"
        agent.base_url = "https://chatgpt.com/backend-api/codex"
        agent.api_mode = "codex_responses"
        agent.api_key = "original-key-entry-1"
        agent._client_kwargs = {
            "api_key": "original-key-entry-1",
            "base_url": "https://chatgpt.com/backend-api/codex",
        }
        agent._credential_pool = pool
        agent._fallback_activated = True
        agent._fallback_index = 1
        agent._rate_limited_until = 0
        agent._use_prompt_caching = False
        agent._use_native_cache_layout = False
        agent.context_compressor = MagicMock()
        agent.context_compressor.update_model = MagicMock()

        # Snapshot the original state
        agent._primary_runtime = {
            "model": "gpt-5.5",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
            "api_key": "original-key-entry-1",
            "client_kwargs": {
                "api_key": "original-key-entry-1",
                "base_url": "https://chatgpt.com/backend-api/codex",
            },
            "use_prompt_caching": False,
            "use_native_cache_layout": False,
            "compressor_model": "gpt-5.5",
            "compressor_base_url": "https://chatgpt.com/backend-api/codex",
            "compressor_api_key": "original-key-entry-1",
            "compressor_provider": "openai-codex",
            "compressor_context_length": 128000,
            "compressor_threshold_tokens": 0.8,
        }

        # Mock client creation methods
        agent._create_openai_client = MagicMock(return_value=MagicMock())
        agent._apply_client_headers_for_base_url = MagicMock()
        agent._replace_primary_openai_client = MagicMock(return_value=True)

        return agent


    def test_restore_uses_freshest_available_entry(self):
        """When multiple entries are available, restore should select the pool's best pick."""
        entries = [
            _make_entry("entry-1", "key-1", priority=0,
                         last_status="exhausted", last_status_at=time.time() + 3600),
            _make_entry("entry-2", "key-2", priority=1),
            _make_entry("entry-3", "key-3", priority=2),
        ]
        pool = _build_mock_pool(entries)

        agent = self._make_agent(pool)
        result = agent._restore_primary_runtime()

        assert result is True
        # entry-1 is exhausted, so pool should select entry-2
        assert agent.api_key == "key-2"
        assert agent._client_kwargs["api_key"] == "key-2"

    def test_pool_selection_ticket_is_read_only_until_commit_and_compensates(self):
        """Prepared restore selection owns cursor/count changes and can abort them."""
        pool = _build_mock_pool(
            [
                _make_entry("entry-1", "key-1", priority=0),
                _make_entry("entry-2", "key-2", priority=1),
            ],
            strategy="fill_first",
        )
        before_entries = [entry.to_dict() for entry in pool.entries()]
        before_epoch = pool._mutation_epoch
        before_current = pool.current()

        ticket = pool.prepare_selection(model="gpt-5.5")

        assert ticket is not None
        assert ticket.candidate.id == "entry-1"
        assert [entry.to_dict() for entry in pool.entries()] == before_entries
        assert pool._mutation_epoch == before_epoch
        assert pool.current() is before_current

        selected = ticket.commit()
        assert selected.id == "entry-1"
        assert pool.current().id == "entry-1"
        assert pool.current().request_count == 1

        ticket.abort()
        assert pool.current() is None
        assert [entry.request_count for entry in pool.entries()] == [0, 0]


    def test_selection_ticket_abort_preserves_interleaved_round_robin_selection(
        self,
        monkeypatch,
    ):
        """Abort removes only its selection, not one interleaved after validation."""
        pool = _build_mock_pool(
            [
                _make_entry("entry-1", "key-1", priority=0),
                _make_entry("entry-2", "key-2", priority=1),
            ],
            strategy="round_robin",
        )
        monkeypatch.setattr(pool, "_persist", MagicMock())
        ticket = pool.prepare_selection(model="gpt-5.5")
        assert ticket is not None
        assert ticket.candidate.id == "entry-1"

        validate = pool._validate_selection_basis
        interleaved_state = {}

        def validate_then_interleave(selection_ticket):
            validate(selection_ticket)
            concurrent = pool.select(model="gpt-5.5")
            assert concurrent is not None
            interleaved_state["cursor"] = pool.current().id
            interleaved_state["rows"] = [
                (entry.id, entry.priority, entry.request_count)
                for entry in pool.entries()
            ]

        monkeypatch.setattr(
            pool,
            "_validate_selection_basis",
            validate_then_interleave,
        )

        selected = ticket.commit()
        assert selected.id == "entry-1"
        assert [
            (entry.id, entry.priority, entry.request_count)
            for entry in pool.entries()
        ] == [
            ("entry-2", 0, 0),
            ("entry-1", 1, 2),
        ]

        ticket.abort()

        assert interleaved_state == {
            "cursor": "entry-1",
            "rows": [
                ("entry-2", 0, 0),
                ("entry-1", 1, 1),
            ],
        }
        assert pool.current().id == interleaved_state["cursor"]
        assert [
            (entry.id, entry.priority, entry.request_count)
            for entry in pool.entries()
        ] == interleaved_state["rows"]


    def test_selection_abort_keeps_other_refresh_but_removes_owned_rotation(
        self,
        monkeypatch,
    ):
        """A downstream failure preserves refreshed tokens, not failed selection state."""
        from agent.agent_runtime_helpers import _CredentialRevertTransitionTicket

        pool = _build_mock_pool(
            [
                _make_entry("entry-1", "key-1", priority=0),
                _make_entry("entry-2", "stale-key-2", priority=1),
            ],
            strategy="round_robin",
        )
        monkeypatch.setattr(pool, "_persist", MagicMock())
        monkeypatch.setattr(
            pool,
            "_entry_needs_refresh",
            lambda entry: (
                entry.id == "entry-2"
                and entry.access_token != "refreshed-key-2"
            ),
        )
        refresh_calls = []

        def refresh_other_entry(entry, *, force):
            assert force is False
            refresh_calls.append(entry.id)
            return pool._adopt(
                entry,
                persist=False,
                access_token="refreshed-key-2",
                refresh_token="refreshed-rt-2",
            )

        monkeypatch.setattr(pool, "_refresh_entry", refresh_other_entry)
        selection_ticket = pool.prepare_selection(model="gpt-5.5")
        assert selection_ticket is not None
        assert selection_ticket.candidate.id == "entry-1"

        class FailingSwapTicket:
            def __init__(self):
                self.abort_calls = 0

            def commit(self, _entry):
                raise RuntimeError("downstream swap failed")

            def abort(self):
                self.abort_calls += 1

        swap_ticket = FailingSwapTicket()
        owner_ticket = _CredentialRevertTransitionTicket(
            selection_ticket,
            swap_ticket,
        )

        with pytest.raises(RuntimeError, match="downstream swap failed"):
            owner_ticket.commit()
        owner_ticket.abort()

        rows = pool.entries()
        assert refresh_calls == ["entry-2"]
        assert swap_ticket.abort_calls == 1
        assert pool.current() is None
        assert [entry.id for entry in rows] == ["entry-1", "entry-2"]
        assert [entry.request_count for entry in rows] == [0, 0]
        assert [entry.priority for entry in rows] == [0, 1]
        refreshed = next(entry for entry in rows if entry.id == "entry-2")
        assert refreshed.access_token == "refreshed-key-2"
        assert refreshed.refresh_token == "refreshed-rt-2"





    def test_restore_updates_base_url_from_pool_entry(self):
        """If pool entry has a different base_url, restore should update it."""
        entries = [
            {
                **_make_entry("entry-1", "key-1", priority=0),
                "base_url": "https://custom-endpoint.example.com/v1",
            },
        ]
        pool = _build_mock_pool(entries)

        agent = self._make_agent(pool)
        result = agent._restore_primary_runtime()

        assert result is True
        assert "custom-endpoint.example.com" in agent.base_url
        assert "custom-endpoint.example.com" in agent._client_kwargs["base_url"]
