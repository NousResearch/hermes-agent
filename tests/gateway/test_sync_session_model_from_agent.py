"""Regression: gateway fallback sync must not shadow an explicit /model choice.

``_sync_session_model_from_agent`` runs after every gateway turn to persist
whatever provider a fallback chain actually used. ``update_session_model``
runs when the user explicitly switches models with ``/model``. Both used to
write to different shapes of ``model_config`` (a nested ``gateway_runtime``
dict vs. flat top-level keys), and ``session_gateway_runtime()`` — the single
reader both CLI resume and the TUI gateway use — preferred the nested shape
unconditionally. A fallback sync landing after an explicit /model switch
would permanently shadow the user's choice on the next resume, silently
routing back to the stale fallback provider.
"""

from types import SimpleNamespace

import pytest

from hermes_state import SessionDB


def _make_runner(db):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._session_db = SimpleNamespace(_db=db)
    return runner


def _agent(model, provider=None, base_url=None, api_mode=None):
    return SimpleNamespace(
        model=model, provider=provider, base_url=base_url, api_mode=api_mode,
    )


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    database = SessionDB(db_path=tmp_path / "state.db")
    database.create_session(session_id="s1", source="telegram", model="m0")
    return database


def test_fallback_sync_writes_flat_keys_not_nested(db):
    """The automatic sync must use the SAME shape as the explicit /model
    persist — no more separate nested gateway_runtime dict."""
    runner = _make_runner(db)
    runner._sync_session_model_from_agent(
        "s1", _agent("claude-x", provider="anthropic", base_url="https://api.anthropic.com"),
    )

    meta = db.get_session("s1")
    assert meta["model"] == "claude-x"
    runtime = SessionDB.session_gateway_runtime(meta)
    assert runtime["provider"] == "anthropic"
    assert runtime["base_url"] == "https://api.anthropic.com"

    import json
    config = json.loads(meta["model_config"])
    assert "gateway_runtime" not in config
    assert config["provider"] == "anthropic"


def test_explicit_model_switch_after_fallback_sync_wins_on_resume(db):
    """The exact bug scenario: a background fallback sync writes first, then
    the user explicitly switches models. Resume must see the user's choice,
    not the earlier fallback."""
    runner = _make_runner(db)

    # Provider fallback happens mid-turn; the gateway syncs it automatically.
    runner._sync_session_model_from_agent(
        "s1", _agent("gpt-5-mini", provider="openai", base_url="https://api.openai.com/v1"),
    )

    # The user then explicitly switches models via /model.
    db.update_session_model("s1", "claude-opus", provider="anthropic")

    meta = db.get_session("s1")
    runtime = SessionDB.session_gateway_runtime(meta)
    assert runtime["provider"] == "anthropic", (
        "explicit /model choice was shadowed by the earlier automatic "
        "fallback sync"
    )


def test_fallback_sync_after_explicit_switch_updates_runtime(db):
    """The reverse order: an explicit switch, then a later fallback event
    (e.g. that provider goes down) must also be visible on resume."""
    runner = _make_runner(db)

    db.update_session_model("s1", "claude-opus", provider="anthropic")
    runner._sync_session_model_from_agent(
        "s1", _agent("gpt-5-mini", provider="openai", base_url="https://api.openai.com/v1"),
    )

    meta = db.get_session("s1")
    runtime = SessionDB.session_gateway_runtime(meta)
    assert runtime["provider"] == "openai"
    assert meta["model"] == "gpt-5-mini"


def test_unchanged_state_is_a_noop(db, monkeypatch):
    """No write when model/provider/base_url/api_mode already match."""
    runner = _make_runner(db)
    runner._sync_session_model_from_agent(
        "s1", _agent("claude-x", provider="anthropic", base_url="https://a"),
    )

    calls = []
    monkeypatch.setattr(
        db, "update_session_gateway_runtime",
        lambda *a, **k: calls.append((a, k)),
    )
    runner._sync_session_model_from_agent(
        "s1", _agent("claude-x", provider="anthropic", base_url="https://a"),
    )
    assert calls == []


def test_missing_session_row_is_silent_noop(db):
    runner = _make_runner(db)
    # Must not raise for a session that doesn't exist.
    runner._sync_session_model_from_agent("nonexistent", _agent("claude-x", provider="anthropic"))
