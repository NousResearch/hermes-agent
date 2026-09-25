"""PG-selected runtime delivery-ledger wiring: selector, recovery flow, and SQLite fallback.

Exercises ``selected_delivery_ledger`` (the runtime seam) against the real disposable PG18
target: a healthy selection returns a working ledger whose runtime recovery methods
(``sweep_failed_for_runtime`` / ``pending_retries``) mirror the SQLite contract; an
unreachable selection fails typed without opening ``state.db``; a SQLite home keeps the
legacy module path (``None``) and the compatibility wrapper still round-trips.
"""
from __future__ import annotations

import pytest

from gateway.delivery_ledger import RECONNECTED_MARKER
from gateway.delivery_ledger_adapter import SqliteDeliveryLedger, selected_delivery_ledger
from gateway.delivery_ledger_postgresql import PostgreSQLDeliveryLedger
from state_store import StateStoreConfigurationError
from state_store_runtime_readiness import trap_state_db_opens

_DSN_ENV = "OWNED_DELIVERY_DSN"


def _pg_home(tmp_path, monkeypatch, *, dsn_value: str):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n"
        "    dsn_env: OWNED_DELIVERY_DSN\n    connect_timeout_seconds: 1\n    pool_max_size: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv(_DSN_ENV, dsn_value)
    return home


@pytest.mark.integration
def test_selected_pg_ledger_runtime_recovery_flow(postgresql_delivery_target, tmp_path, monkeypatch):
    _pg_home(tmp_path, monkeypatch, dsn_value=postgresql_delivery_target.dsn)
    import gateway.delivery_ledger_postgresql as delivery_postgresql

    # Exercise the runtime selector against the fixture's marker-owned tenant;
    # no selector test creates or destroys a shared/root ledger schema.
    monkeypatch.setattr(delivery_postgresql, "postgresql_delivery_ledger_schema", lambda: postgresql_delivery_target.schema)

    ledger = selected_delivery_ledger()
    assert ledger is not None
    assert isinstance(ledger, PostgreSQLDeliveryLedger)
    try:
        # A reconnect-only rejection is claimed by the runtime sweep with a visible marker.
        receipt = ledger.record_obligation(
            obligation_id="rt-reconnect", session_key="s", platform="slack", chat_id="c",
            thread_id=None, content="body",
        )
        claim = ledger.mark_attempting(receipt)
        assert claim is not None
        assert ledger.mark_failed(claim, "send_path_degraded")

        claimed = ledger.sweep_failed_for_runtime("slack")
        assert len(claimed) == 1
        row = claimed[0]
        assert row["obligation_id"] == "rt-reconnect"
        assert row["needs_marker"] is True
        assert row["runtime_recovery"] is True
        assert row["marker"] == RECONNECTED_MARKER

        # A plain transient rejection (not reconnect, not flood, not dead) arms a backoff timer:
        # pending_retries lists it with a not_before deadline.
        receipt2 = ledger.record_obligation(
            obligation_id="rt-backoff", session_key="s", platform="slack", chat_id="c",
            thread_id=None, content="body2",
        )
        claim2 = ledger.mark_attempting(receipt2)
        assert claim2 is not None
        assert ledger.mark_failed(claim2, "500 internal error")

        pending = ledger.pending_retries()
        slack = [p for p in pending if p["platform"] == "slack" and p["profile"] == "default"]
        assert slack
        assert all("not_before" in p for p in slack)
    finally:
        ledger.close()


@pytest.mark.integration
def test_selected_pg_unreachable_raises_without_state_db(tmp_path, monkeypatch):
    home = _pg_home(tmp_path, monkeypatch, dsn_value="postgresql://fixture/only")

    import psycopg as _psycopg

    with trap_state_db_opens(home) as events:
        with pytest.raises((StateStoreConfigurationError, _psycopg.Error)):
            selected_delivery_ledger()
    assert events == []
    assert not (home / "state.db").exists()


@pytest.mark.integration
def test_sqlite_home_selected_returns_none_and_sqlite_smoke(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("state_store:\n  backend: sqlite\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert selected_delivery_ledger() is None

    ledger = SqliteDeliveryLedger()
    receipt = ledger.record_obligation(
        obligation_id="sqlite-smoke", session_key="s", platform="slack", chat_id="c",
        thread_id=None, content="body",
    )
    claim = ledger.mark_attempting(receipt)
    assert claim is not None
    assert ledger.mark_delivered(claim)
