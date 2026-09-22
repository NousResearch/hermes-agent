"""A fallback session must come back as soon as ANY pooled credential of the primary provider
can serve again.

Measured live (2026-09-22), five homes, ~50 minutes of paid Fireworks traffic: account A of the
``opencode-go`` pool hit its weekly cap, so the 429 carried a ``reset_at`` ~5.1 days out and
``_arm_rate_limit_cooldown`` armed ``_rate_limited_until`` from it. The second account's rows were
deployed minutes later and were clean on every window, yet the sessions kept billing the paid
fallback for the rest of their lives: ``restore_primary_runtime`` returned False on that
session-wide cooldown *before* asking the credential pool, so account B stayed idle. Only a NEW
process came back to Go.

The cooldown is credential-scoped data (the reset belongs to the credential that 429'd) stored on
a session-wide slot. The pool already knows the difference (``has_available``); the restore gate
must ask it before honouring the cooldown, exactly as ``_pool_may_recover_from_rate_limit`` does
when deciding whether a 429 is worth a fallback at all.

Fail-closed: a missing/mismatched/unreadable pool, or a pool whose every entry is still benched,
keeps the session on the fallback (a single-credential pool would just re-hit the same cap).
"""

import json
import time
from unittest.mock import MagicMock, patch

from agent.credential_pool import (
    STATUS_EXHAUSTED,
    STATUS_OK,
    CredentialPool,
    PooledCredential,
)
from run_agent import AIAgent

_WEEKLY_WINDOW_S = 441504  # observed: 441504 s (7358.4 min, backoff#1, provider reset)


# =============================================================================
# Helpers
# =============================================================================

def _entry(provider="custom", id="cred-a", status=STATUS_OK, reset_at=None):
    return PooledCredential(
        provider=provider,
        id=id,
        label=f"label-{id}",
        auth_type="api_key",
        priority=1,
        source="manual",
        access_token="«redacted:sk-…»",
        last_status=status,
        last_status_at=time.time(),
        last_error_code=429 if status == STATUS_EXHAUSTED else None,
        last_error_reset_at=reset_at,
    )


def _capped_account_a_pool(provider="custom", secondary_status=STATUS_OK):
    """Account A benched for the rest of its weekly window; account B healthy (the live shape)."""
    return CredentialPool(
        provider,
        [
            _entry(id="account-a", status=STATUS_EXHAUSTED, reset_at=time.time() + _WEEKLY_WINDOW_S),
            _entry(id="account-b", status=secondary_status),
        ],
    )


def _make_tool_defs(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _make_agent(fallback_model=None):
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-12345678",
            base_url="https://my-llm.example.com/v1",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _activate_fallback(agent, base_url="https://openrouter.ai/api/v1"):
    mock_client = MagicMock()
    mock_client.api_key = "fallback-key-1234"
    mock_client.base_url = base_url
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(mock_client, None),
    ):
        assert agent._try_activate_fallback() is True
    assert agent._fallback_activated is True


def _forced_fallback_session(agent):
    """Put the session in the live end-state: cross-provider fallback active, the primary's
    weekly-cap cooldown armed from account A's reset_at, and the primary pool slot empty — a
    cross-provider fallback clears it to avoid contamination, and it stays empty when the fallback
    provider has no pool rows of its own."""
    primary_model = agent.model
    _activate_fallback(agent)
    agent._rate_limited_until = time.monotonic() + _WEEKLY_WINDOW_S
    agent._credential_pool = None
    agent._credential_pool_entry_id = None
    return primary_model


# =============================================================================
# The unwind
# =============================================================================

class TestRestoreWhenPooledCredentialRecovers:
    FB = {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}

    def test_returns_to_primary_although_the_capped_credential_is_benched_for_days(self):
        agent = _make_agent(fallback_model=self.FB)
        primary_model = _forced_fallback_session(agent)
        pool = _capped_account_a_pool()

        with (
            patch("agent.credential_pool.load_pool", return_value=pool),
            patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
        ):
            assert agent._restore_primary_runtime() is True

        assert agent._fallback_activated is False
        assert agent.provider == "custom"
        assert agent.model == primary_model

    def test_stale_cooldown_is_cleared_so_the_next_rate_limit_can_arm_again(self):
        agent = _make_agent(fallback_model=self.FB)
        _forced_fallback_session(agent)

        with (
            patch("agent.credential_pool.load_pool", return_value=_capped_account_a_pool()),
            patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
        ):
            assert agent._restore_primary_runtime() is True

        # The 5-day stamp belonged to account A's window; measured against the primary now proved
        # able to serve, keeping it would block the very next restore for the rest of the week.
        assert agent._rate_limited_until == 0

    def test_single_credential_pool_stays_on_fallback(self):
        """One benched credential = nowhere to go: restoring would re-hit the same weekly cap."""
        agent = _make_agent(fallback_model=self.FB)
        _forced_fallback_session(agent)
        pool = CredentialPool(
            "custom",
            [_entry(id="account-a", status=STATUS_EXHAUSTED, reset_at=time.time() + _WEEKLY_WINDOW_S)],
        )

        with patch("agent.credential_pool.load_pool", return_value=pool):
            assert agent._restore_primary_runtime() is False
        assert agent._fallback_activated is True

    def test_whole_pool_benched_stays_on_fallback(self):
        agent = _make_agent(fallback_model=self.FB)
        _forced_fallback_session(agent)
        pool = _capped_account_a_pool(secondary_status=STATUS_EXHAUSTED)
        pool._entries[1].last_error_reset_at = time.time() + _WEEKLY_WINDOW_S
        pool._entries[1].last_error_code = 429

        with patch("agent.credential_pool.load_pool", return_value=pool):
            assert agent._restore_primary_runtime() is False
        assert agent._fallback_activated is True

    def test_unloadable_primary_pool_stays_on_fallback(self):
        """Fail closed: no pool to consult means the cooldown is honoured, as before."""
        agent = _make_agent(fallback_model=self.FB)
        _forced_fallback_session(agent)

        with patch("agent.credential_pool.load_pool", return_value=None):
            assert agent._restore_primary_runtime() is False
        assert agent._fallback_activated is True

    def test_pool_read_failure_stays_on_fallback(self):
        agent = _make_agent(fallback_model=self.FB)
        _forced_fallback_session(agent)

        with patch("agent.credential_pool.load_pool", side_effect=RuntimeError("boom")):
            assert agent._restore_primary_runtime() is False
        assert agent._fallback_activated is True

    def test_transient_cooldown_with_a_healthy_pool_still_recovers(self):
        """The 60s transient cooldown is provider-wide, but it too is armed from a credential's
        429: with a second credential available the session returns immediately instead of
        waiting out a cooldown it does not need."""
        agent = _make_agent(fallback_model=self.FB)
        _activate_fallback(agent)
        agent._rate_limited_until = time.monotonic() + 60
        agent._credential_pool = CredentialPool("openrouter", [_entry(provider="openrouter", id="fb")])

        with (
            patch("agent.credential_pool.load_pool", return_value=_capped_account_a_pool()),
            patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
        ):
            assert agent._restore_primary_runtime() is True
        assert agent._fallback_activated is False

    def test_restores_when_the_attached_pool_belongs_to_the_fallback_provider(self):
        """A fallback provider that DOES have pool rows leaves its own pool attached; the primary
        pool is then read through the reset-aware gate's prefetch."""
        agent = _make_agent(fallback_model=self.FB)
        _forced_fallback_session(agent)
        agent._credential_pool = CredentialPool("fireworks", [_entry(provider="fireworks", id="fb")])

        with (
            patch("agent.credential_pool.load_pool", return_value=_capped_account_a_pool()),
            patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
        ):
            assert agent._restore_primary_runtime() is True
        assert agent._fallback_activated is False
        assert agent._credential_pool is None or agent._credential_pool.provider == "custom"


# =============================================================================
# End to end: real auth.json, real pool, real credential swap
# =============================================================================

class TestPooledRecoveryEndToEnd:
    """No pool mocks: the session reads the real ``auth.json`` through ``load_pool`` and must come
    back on account B's key, the one a fresh process would have picked."""

    PROVIDER = "openrouter"
    BASE_URL = "https://openrouter.ai/api/v1"
    MODEL = "anthropic/claude-sonnet-4"
    FB = {"provider": "fireworks", "model": "accounts/fireworks/models/deepseek-v4p1-flash"}

    @staticmethod
    def _write_auth_store(tmp_path, entries):
        home = tmp_path / "hermes"
        home.mkdir(parents=True, exist_ok=True)
        (home / "auth.json").write_text(
            json.dumps({"version": 1, "credential_pool": {"openrouter": entries}}), encoding="utf-8"
        )
        return home

    @staticmethod
    def _pooled(cred_id, token, status, reset_at=None):
        return {
            "id": cred_id,
            "label": cred_id,
            "auth_type": "api_key",
            "priority": 0,
            "source": "manual",
            "access_token": token,
            "base_url": "https://openrouter.ai/api/v1",
            "last_status": status,
            "last_status_at": time.time(),
            "last_error_code": 429 if status == STATUS_EXHAUSTED else None,
            "last_error_reset_at": reset_at,
        }

    def _agent(self):
        with (
            patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
        ):
            agent = AIAgent(
                api_key="sk-account-a",
                base_url=self.BASE_URL,
                provider=self.PROVIDER,
                model=self.MODEL,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                fallback_model=self.FB,
            )
            agent.client = MagicMock()
            return agent

    def _end_to_end(self, tmp_path, monkeypatch, entries):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        self._write_auth_store(tmp_path, entries)
        agent = self._agent()

        # The live state: cross-provider fallback active (its client construction is not what this
        # test is about), the primary pool cleared by the fallback path, and the weekly-cap cooldown
        # armed from account A's reset_at.
        _activate_fallback(agent, base_url="https://api.fireworks.ai/inference/v1")
        agent._credential_pool = None
        agent._credential_pool_entry_id = None
        agent._rate_limited_until = time.monotonic() + _WEEKLY_WINDOW_S

        with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
            restored = agent._restore_primary_runtime()
        return agent, restored

    def test_returns_on_account_b_when_only_account_a_is_capped(self, tmp_path, monkeypatch):
        entries = [
            self._pooled("account-a", "sk-account-a", STATUS_EXHAUSTED, reset_at=time.time() + _WEEKLY_WINDOW_S),
            self._pooled("account-b", "sk-account-b", STATUS_OK),
        ]
        agent, restored = self._end_to_end(tmp_path, monkeypatch, entries)

        assert restored is True
        assert agent._fallback_activated is False
        assert (agent.provider, agent.model) == (self.PROVIDER, self.MODEL)
        assert agent._credential_pool_entry_id == "account-b"
        assert agent.api_key == "sk-account-b"

    def test_stays_on_fallback_when_account_a_is_the_only_credential(self, tmp_path, monkeypatch):
        entries = [
            self._pooled("account-a", "sk-account-a", STATUS_EXHAUSTED, reset_at=time.time() + _WEEKLY_WINDOW_S),
        ]
        agent, restored = self._end_to_end(tmp_path, monkeypatch, entries)

        assert restored is False
        assert agent._fallback_activated is True
        assert agent.provider == "fireworks"

