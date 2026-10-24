"""base-red / fix-green receipt for NousResearch/hermes-agent#75549.

Bug under test: a gateway turn served by a FALLBACK provider persists its real runtime route
through ``GatewayRunner._sync_session_model_from_agent()`` into
``model_config.gateway_runtime`` (provider / base_url / api_mode / fallback_active). Neither
/status nor /usage reads that snapshot:

* ``_status_model_route()`` builds its last-resort route from ``session_row.billing_provider``
  and then fills the provider from the CONFIGURED ``model.provider`` -> it prints the model the
  session actually used next to a provider that never served it.
* ``_persisted_billing_route()`` needs ``billing_provider <> ''`` (no billed call yet), so /usage
  falls through to ``_configured_provider()`` and fetches ACCOUNT usage for the configured
  endpoint - the account whose credits just failed over.

The canonical reader for the snapshot already exists: ``SessionDB.session_gateway_runtime()``
(hermes_state_gateway.py). These tests assert the COMMANDS (not a helper) report the served route,
using the real writer and a real SessionDB - no hand-written model_config, no mocked SessionDB.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from hermes_state import AsyncSessionDB, SessionDB

FALLBACK_PROVIDER = "fallback-prov"
FALLBACK_BASE_URL = "https://fallback.example/v1"
CONFIGURED_PROVIDER = "cfg-primary"
CONFIGURED_BASE_URL = "https://cfg.example/v1"
CONFIGURED_MODEL = "cfg-primary-model"
MODEL = "fallback-model"
BILLING_PROVIDER = "billing-prov"


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_source(), message_id="m1")


def _configured_config() -> dict:
    """What ``_load_gateway_config()`` returns: the CONFIGURED route, never the served one."""
    return {
        "model": {
            "model": CONFIGURED_MODEL,
            "provider": CONFIGURED_PROVIDER,
            "base_url": CONFIGURED_BASE_URL,
        }
    }


@pytest.fixture
def fallback_served_session(tmp_path):
    """A session whose real runtime route is a fallback, persisted by the REAL writer."""
    from gateway.run import GatewayRunner

    db = SessionDB(db_path=tmp_path / "state.db")
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    entry = store.get_or_create_session(_source())
    db.create_session(entry.session_id, "telegram", model=MODEL)

    writer = object.__new__(GatewayRunner)
    writer._session_db = AsyncSessionDB(db)
    served_by = SimpleNamespace(
        model=MODEL,
        provider=FALLBACK_PROVIDER,
        base_url=FALLBACK_BASE_URL,
        api_mode="chat",
        _fallback_activated=True,
    )
    writer._sync_session_model_from_agent(entry.session_id, served_by)

    try:
        yield db, entry, store
    finally:
        db.close()


def _runner(entry, db, store):
    """GatewayRunner with the real stores wired and no resident/cached agent."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db = AsyncSessionDB(db)
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._agent_cache = {}
    runner._agent_cache_lock = MagicMock()
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_a, **_k: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *_a, **_k: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


def test_fixture_persists_the_runtime_snapshot_the_real_writer_writes(fallback_served_session):
    """Precondition guard: proves the state under test is what the gateway really writes.

    Passes on both base and fix - if this ever fails, the regression below is measuring
    something else (tune it before trusting the red/green result)."""
    db, entry, _store = fallback_served_session
    row = db.get_session(entry.session_id)
    runtime = json.loads(row["model_config"])["gateway_runtime"]
    assert runtime["provider"] == FALLBACK_PROVIDER
    assert runtime["base_url"] == FALLBACK_BASE_URL
    assert runtime["fallback_active"] is True
    # No API call has been billed against the fallback route yet -> no persisted billing route.
    assert not (row.get("billing_provider") or "")
    assert db.get_recent_session_model_route(entry.session_id) in (None, {}) or not (
        db.get_recent_session_model_route(entry.session_id) or {}
    ).get("billing_provider")


@pytest.mark.asyncio
async def test_status_reports_the_provider_that_actually_served_the_session(
    fallback_served_session, monkeypatch
):
    """RED on base: /status claims the configured provider served a fallback-served session."""
    db, entry, store = fallback_served_session
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda *_a, **_k: _configured_config())

    out = await _runner(entry, db, store)._handle_message(_event("/status"))

    assert f"**Model:** `{MODEL}` ({FALLBACK_PROVIDER})" in out, (
        "a session served by the fallback route must report that provider"
    )
    assert CONFIGURED_PROVIDER not in out, (
        "/status must not name the configured provider for a session it never served"
    )


def test_status_route_carries_the_snapshot_base_url(fallback_served_session, monkeypatch):
    """RED on base: the winning route must pair the snapshot provider WITH its base_url.

    The route dict feeds the context-window lookup (``_resolve_route_context``), so a provider
    without its endpoint silently queries the wrong runtime."""
    from gateway.slash_commands_status import _status_model_route

    db, entry, _store = fallback_served_session
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda *_a, **_k: _configured_config())
    session_row = db.get_session(entry.session_id)

    model, provider, _used, _total, route = _status_model_route(None, {}, {}, session_row, entry)

    assert (model, provider) == (MODEL, FALLBACK_PROVIDER)
    assert route.get("base_url") == FALLBACK_BASE_URL, (
        "provider and base_url must come from the same source (never a mixed pair)"
    )


@pytest.mark.asyncio
async def test_usage_fetches_account_usage_for_the_served_route(
    fallback_served_session, monkeypatch
):
    """RED on base: /usage bills the account of the configured provider, not the served one."""
    db, entry, store = fallback_served_session
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda *_a, **_k: _configured_config())
    monkeypatch.setattr("agent.account_usage.nous_credits_lines", lambda markdown=False: [])
    captured = {}

    def _fake_fetch(provider, base_url=None, api_key=None):
        captured.update(provider=provider, base_url=base_url)
        return None

    monkeypatch.setattr("gateway.slash_commands_status.fetch_account_usage", _fake_fetch)

    await _runner(entry, db, store)._handle_usage_command(_event("/usage"))

    assert captured.get("provider") == FALLBACK_PROVIDER, (
        "/usage must query the provider that actually served the session"
    )
    assert captured.get("base_url") == FALLBACK_BASE_URL, (
        "provider and base_url must be taken as a PAIR from one source"
    )

@pytest.fixture
def billing_provider_without_endpoint(tmp_path):
    """A session whose billing row names a provider but carries no endpoint URL.

    Realistic: ``billing_provider`` lands on the first accounted call, while ``billing_base_url``
    stays empty for providers served by their own default endpoint. The runtime snapshot then
    describes a DIFFERENT (fallback) provider, so its base_url must never be adopted here."""
    from gateway.run import GatewayRunner

    db = SessionDB(db_path=tmp_path / "state.db")
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    entry = store.get_or_create_session(_source())
    db.create_session(entry.session_id, "telegram", model=MODEL)

    writer = object.__new__(GatewayRunner)
    writer._session_db = AsyncSessionDB(db)
    writer._sync_session_model_from_agent(
        entry.session_id,
        SimpleNamespace(
            model=MODEL,
            provider=FALLBACK_PROVIDER,
            base_url=FALLBACK_BASE_URL,
            api_mode="chat",
            _fallback_activated=True,
        ),
    )
    db.update_session_billing_route(entry.session_id, provider=BILLING_PROVIDER, base_url="")

    try:
        yield db, entry, store
    finally:
        db.close()


@pytest.mark.asyncio
async def test_usage_never_pairs_one_sources_provider_with_another_sources_endpoint(
    billing_provider_without_endpoint, monkeypatch
):
    """Design guard (green on base, RED on the #75549 hunk): gap-filling provider and base_url
    independently adopts a fallback endpoint for a billing provider - an account that never
    served the session."""
    db, entry, store = billing_provider_without_endpoint
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda *_a, **_k: _configured_config())
    monkeypatch.setattr("agent.account_usage.nous_credits_lines", lambda markdown=False: [])
    captured = {}

    def _fake_fetch(provider, base_url=None, api_key=None):
        captured.update(provider=provider, base_url=base_url)
        return None

    monkeypatch.setattr("gateway.slash_commands_status.fetch_account_usage", _fake_fetch)

    await _runner(entry, db, store)._handle_usage_command(_event("/usage"))

    assert captured.get("provider") == BILLING_PROVIDER
    assert captured.get("base_url") != FALLBACK_BASE_URL, (
        "an endpoint belonging to another provider must never be paired with this one"
    )
