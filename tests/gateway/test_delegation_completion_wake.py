"""``gateway.delegation_completion_wake``: an async-delegation completion may wake an IDLE
api_server session (personal single-user gateways) while every other case keeps the #85957
durable persist-only row."""
import asyncio
from types import SimpleNamespace

import pytest

import gateway.platforms.api_server_runs as runs_mod
import gateway.wake as wake_mod
from gateway.run_config_loaders import GatewayConfigLoadersMixin
from gateway.run_notifications import GatewayNotificationsMixin
from gateway.wake import persist_delegation_delivery
from hermes_state import SessionDB


class _Gateway(GatewayNotificationsMixin, GatewayConfigLoadersMixin):
    def __init__(self, statuses, approvals):
        self._run_statuses = statuses
        self._run_approval_sessions = approvals


@pytest.fixture
def wake_on(monkeypatch):
    monkeypatch.setattr("gateway.run._load_gateway_config",
                        lambda: {"gateway": {"delegation_completion_wake": True}})


@pytest.fixture
def wake_default_off(monkeypatch):
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})


def _adapter(statuses=(), approvals=(), db=None):
    return SimpleNamespace(
        _run_statuses={s["run_id"]: s for s in statuses},
        _run_approval_sessions=set(approvals),
        _ensure_session_db=lambda: db)


def test_default_off_keeps_persist_only(wake_default_off):
    gw = _Gateway({}, set())
    assert gw._delegation_completion_wake_allowed(_adapter(), "s1") is False


def test_wake_requires_idle_session(wake_on):
    busy = _adapter(statuses=[{"run_id": "r1", "session_id": "s1", "status": "running"}])
    gw = _Gateway({}, set())
    assert gw._delegation_completion_wake_allowed(busy, "s1") is False
    gated = _adapter(statuses=[{"run_id": "r2", "session_id": "s1", "status": "completed"}],
                     approvals=["r2"])
    assert gw._delegation_completion_wake_allowed(gated, "s1") is False
    idle = _adapter(statuses=[{"run_id": "r3", "session_id": "s2", "status": "completed"}])
    assert gw._delegation_completion_wake_allowed(idle, "s1") is True


def test_session_has_live_run_shapes():
    adapter = _adapter(
        statuses=[{"run_id": "a", "session_id": "s1", "status": "running"},
                  {"run_id": "b", "session_id": "s1", "status": "queued"},
                  {"run_id": "c", "session_id": "s2", "status": "completed"},
                  {"run_id": "d", "session_id": "s3", "status": "completed"}],
        approvals=["d"])
    assert runs_mod.session_has_live_run(adapter, "s1") is True
    assert runs_mod.session_has_live_run(adapter, "s2") is False
    assert runs_mod.session_has_live_run(adapter, "s3") is True   # approval gate = busy
    assert runs_mod.session_has_live_run(adapter, "s9") is False


@pytest.mark.asyncio
async def test_routing_wakes_idle_session(wake_on, monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="api_server")
    adapter = _adapter(db=db)
    captured = {}

    async def fake_wake(adapter, *, text, session_id, **kw):
        captured["wake"] = session_id

    monkeypatch.setattr(wake_mod, "deliver_wake", fake_wake)
    ok = await GatewayNotificationsMixin._self_post_api_server(
        _Gateway({}, set()), adapter, "RESULTS", "s1",
        {"type": "async_delegation", "delegation_id": "d1"})
    assert ok is True
    assert captured.get("wake") == "s1"
    assert db.get_messages("s1") == []  # woken: no duplicate delivery row


@pytest.mark.asyncio
async def test_routing_persists_when_busy_even_with_wake_on(wake_on, monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="api_server")
    adapter = _adapter(statuses=[{"run_id": "r1", "session_id": "s1", "status": "running"}], db=db)

    async def refuse_wake(*a, **k):
        raise AssertionError("a busy session must never be woken")

    monkeypatch.setattr(wake_mod, "deliver_wake", refuse_wake)
    ok = await GatewayNotificationsMixin._self_post_api_server(
        _Gateway({}, set()), adapter, "RESULTS", "s1",
        {"type": "async_delegation", "delegation_id": "d1"})
    assert ok is True
    rows = db.get_messages("s1")
    assert len(rows) == 1  # the durable #85957 delivery row


@pytest.mark.asyncio
async def test_routing_persists_when_disabled(tmp_path, wake_default_off, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="api_server")
    adapter = _adapter(db=db)

    async def refuse_wake(*a, **k):
        raise AssertionError("default off must never wake")

    monkeypatch.setattr(wake_mod, "deliver_wake", refuse_wake)
    ok = await GatewayNotificationsMixin._self_post_api_server(
        _Gateway({}, set()), adapter, "RESULTS", "s1",
        {"type": "async_delegation", "delegation_id": "d1"})
    assert ok is True
    assert len(db.get_messages("s1")) == 1
    assert persist_delegation_delivery is not None
