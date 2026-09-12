"""AIAgent setup coverage for expiry-aware persisted credential pins."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json

import pytest

from hermes_state import SessionDB

agent_init = importlib.import_module("agent.agent_init")
SessionCredentialBindingError = agent_init.SessionCredentialBindingError


class StopBeforeFirstModelRequest(Exception):
    """Stops setup exactly where the model client would be constructed."""


@dataclass
class Credential:
    id: str
    account_id: str
    api_key: str
    base_url: str
    runtime_api_key: str | None = None
    runtime_base_url: str | None = None


class ExpiryAwarePool:
    strategy = "expiry_aware"

    def __init__(self, selected: Credential, entries: list[Credential]):
        self.selected = selected
        self.entries = {entry.id: entry for entry in entries}
        self.select_calls = 0
        self.exact_calls: list[str] = []
        self.usage_lookup = None

    def set_expiry_aware_usage_lookup(self, lookup) -> None:
        self.usage_lookup = lookup

    def select(self) -> Credential:
        self.select_calls += 1
        return self.selected

    def select_exact(self, entry_id: str) -> Credential | None:
        self.exact_calls.append(entry_id)
        return self.entries.get(entry_id)


class OtherStrategyPool:
    strategy = "round_robin"

    def select(self) -> Credential:
        raise AssertionError("non-expiry-aware pools must not be selected here")


class FakeAgent:
    def _read_reasoning_echo_from_config(self) -> bool:
        return False


def _session_db(tmp_path, session_id: str) -> SessionDB:
    db = SessionDB(tmp_path / "sessions.db")
    db.create_session(session_id, "test")
    return db


def _bind(db: SessionDB, session_id: str, credential: Credential) -> None:
    db.get_or_bind_session_credential(
        session_id,
        provider="openai-codex",
        entry_id=credential.id,
        account_id=credential.account_id,
    )


def _run_setup(monkeypatch, db, pool, session_id: str):
    captured: dict[str, object] = {}

    monkeypatch.setattr(agent_init, "_resolve_api_mode", lambda *args: None)
    monkeypatch.setattr(agent_init, "_finalize_routing", lambda *args: None)
    monkeypatch.setattr(agent_init, "_set_defaults", lambda *args: None)
    monkeypatch.setattr(agent_init, "_init_prompt_cache_config", lambda *args: None)
    monkeypatch.setattr(agent_init, "_init_turn_state", lambda *args: None)
    monkeypatch.setattr(agent_init, "_setup_logging", lambda *args: None)

    def stop_at_client(agent, api_key, base_url, fallback_model):
        captured.update(
            api_key=api_key,
            base_url=base_url,
            fallback_model=fallback_model,
            credential=getattr(agent, "_expiry_aware_session_credential", None),
        )
        raise StopBeforeFirstModelRequest

    monkeypatch.setattr(agent_init, "_build_client", stop_at_client)
    with pytest.raises(StopBeforeFirstModelRequest):
        agent_init.init_agent(
            FakeAgent(),
            provider="openai-codex",
            api_key="caller-key",
            base_url="https://caller.invalid/v1",
            fallback_model="fallback",
            credential_pool=pool,
            session_db=db,
            session_id=session_id,
        )
    return captured


def test_initial_setup_selects_once_binds_and_builds_client_from_selected_credential(
    tmp_path, monkeypatch
):
    selected = Credential("entry-a", "account-a", "selected-key", "https://a.invalid/v1")
    pool = ExpiryAwarePool(selected, [selected])
    db = _session_db(tmp_path, "new-session")

    captured = _run_setup(monkeypatch, db, pool, "new-session")

    assert pool.select_calls == 1
    assert pool.exact_calls == []
    assert captured == {
        "api_key": "selected-key",
        "base_url": "https://a.invalid/v1",
        "fallback_model": None,
        "credential": selected,
    }
    config = json.loads(db.get_session("new-session")["model_config"])
    assert config["credential_binding"] == {
        "provider": "openai-codex",
        "entry_id": "entry-a",
        "account_id": "account-a",
    }


def test_resume_uses_pinned_credential_even_when_another_has_better_quota(
    tmp_path, monkeypatch
):
    pinned = Credential(
        "entry-pinned",
        "account-a",
        "stored-key",
        "https://stored.invalid/v1",
        runtime_api_key="fresh-key-a",
        runtime_base_url="https://refreshed.invalid/v1",
    )
    better = Credential("entry-better", "account-b", "better-key", "https://better.invalid/v1")
    pool = ExpiryAwarePool(better, [pinned, better])
    db = _session_db(tmp_path, "resume-session")
    _bind(db, "resume-session", pinned)

    captured = _run_setup(monkeypatch, db, pool, "resume-session")

    assert pool.select_calls == 0
    assert pool.exact_calls == ["entry-pinned"]
    assert captured["credential"] is pinned
    assert captured["api_key"] == "fresh-key-a"
    assert captured["base_url"] == "https://refreshed.invalid/v1"
    assert captured["fallback_model"] is None
    assert pool.usage_lookup is None


def test_resume_does_not_install_or_fetch_expiry_aware_usage(tmp_path, monkeypatch):
    pinned = Credential("entry-pinned", "account-a", "stored-key", "https://stored.invalid/v1")
    pool = ExpiryAwarePool(pinned, [pinned])
    db = _session_db(tmp_path, "pinned-no-usage")
    _bind(db, "pinned-no-usage", pinned)
    calls = []

    def fail_if_called(_entry):
        calls.append(_entry)
        raise AssertionError("pinned resume must not fetch quota telemetry")

    monkeypatch.setattr("agent.account_usage.fetch_codex_expiry_aware_usage", fail_if_called)
    _run_setup(monkeypatch, db, pool, "pinned-no-usage")

    assert pool.select_calls == 0
    assert pool.usage_lookup is None
    assert calls == []


def test_new_setup_honors_concurrent_binding_winner_and_rebuilds_client(
    tmp_path, monkeypatch
):
    selected = Credential("entry-selected", "account-selected", "selected-key", "https://selected.invalid/v1")
    winner = Credential("entry-winner", "account-winner", "winner-key", "https://winner.invalid/v1")
    pool = ExpiryAwarePool(selected, [selected, winner])
    backing_db = _session_db(tmp_path, "racing-session")

    class ConcurrentWinnerDB:
        def get_session(self, session_id):
            return backing_db.get_session(session_id)

        def get_or_bind_session_credential(self, session_id, *, provider, entry_id, account_id):
            _bind(backing_db, session_id, winner)
            return backing_db.get_or_bind_session_credential(
                session_id, provider=provider, entry_id=entry_id, account_id=account_id
            )

    captured = _run_setup(monkeypatch, ConcurrentWinnerDB(), pool, "racing-session")

    assert pool.select_calls == 1
    assert pool.exact_calls == ["entry-winner"]
    assert captured["credential"] is winner
    assert captured["api_key"] == "winner-key"
    assert captured["base_url"] == "https://winner.invalid/v1"


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([], "no longer available"),
        ([Credential("entry-a", "different-account", "key", "https://a.invalid/v1")], "account mismatch"),
    ],
)
def test_resume_fails_explicitly_for_removed_or_mismatched_pinned_credential(
    tmp_path, entries, message
):
    pinned = Credential("entry-a", "account-a", "pinned-key", "https://pinned.invalid/v1")
    pool = ExpiryAwarePool(pinned, entries)
    db = _session_db(tmp_path, "invalid-resume")
    _bind(db, "invalid-resume", pinned)

    with pytest.raises(SessionCredentialBindingError, match=message):
        agent_init.prepare_expiry_aware_session_credential(
            pool, db, "invalid-resume", "openai-codex"
        )

    assert pool.select_calls == 0
    assert pool.exact_calls == ["entry-a"]


def test_other_pool_strategies_keep_existing_client_setup(tmp_path, monkeypatch):
    db = _session_db(tmp_path, "other-strategy")

    captured = _run_setup(monkeypatch, db, OtherStrategyPool(), "other-strategy")

    assert captured == {
        "api_key": "caller-key",
        "base_url": "https://caller.invalid/v1",
        "fallback_model": "fallback",
        "credential": None,
    }


def test_resume_keeps_pin_after_strategy_changed(tmp_path):
    pinned = Credential("entry-a", "account-a", "key", "https://a.invalid/v1")
    pool = ExpiryAwarePool(pinned, [pinned])
    pool.strategy = "fill_first"
    db = _session_db(tmp_path, "changed-strategy")
    _bind(db, "changed-strategy", pinned)
    assert agent_init.prepare_expiry_aware_session_credential(
        pool, db, "changed-strategy", "openai-codex"
    ) is pinned
    assert pool.select_calls == 0


@pytest.mark.parametrize("provider", ["openai-codex", "anthropic"])
def test_resume_with_missing_pool_cannot_drop_pin(tmp_path, provider):
    pinned = Credential("entry-a", "account-a", "key", "https://a.invalid/v1")
    db = _session_db(tmp_path, "missing-pool")
    _bind(db, "missing-pool", pinned)
    with pytest.raises(SessionCredentialBindingError):
        agent_init.prepare_expiry_aware_session_credential(
            None, db, "missing-pool", provider
        )
