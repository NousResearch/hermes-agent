"""PostgreSQL selected-runtime readiness and no-SQLite-fallback contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from state_store_runtime_readiness import (
    PostgreSQLRuntimeActivationError,
    StateDbOpenAttempt,
    inspect_runtime_activation,
    static_raw_state_db_inventory,
    trap_state_db_opens,
    write_runtime_callsite_report,
)


_PG_CONFIG = {
    "state_store": {
        "backend": "postgresql",
        "postgresql": {
            "dsn_env": "HERMES_STATE_STORE_TEST_DSN",
            "connect_timeout_seconds": 3,
            "pool_max_size": 2,
        },
    },
}


def test_static_inventory_reports_every_approved_runtime_raw_opener():
    inventory = static_raw_state_db_inventory()
    unported = {(item.path, item.symbol, item.operation) for item in inventory if item.classification == "unported-legacy-runtime"}

    assert ("cli.py", "HermesCLI._init_session_store", "SessionDB") not in unported
    assert ("gateway/delivery_ledger.py", "_connect", "open_db") in unported
    assert ("tools/async_delegation.py", "_connect", "open_db") in unported
    assert ("tools/bot_live_delivery.py", "find_canonical_owner", "SessionDB") in unported
    assert ("tools/bot_live_delivery.py", "_matches", "SessionDB") in unported
    assert ("tui_gateway/server.py", "_get_db", "acquire") in unported
    assert ("tui_gateway/methods_session.py", "_profile_session_db", "acquire") in unported
    assert ("tui_gateway/compute_host.py", "ComputeHost._build_server_session", "acquire") in unported
    assert ("tui_gateway/session_workdir.py", "_workdir_owner_db", "acquire") in unported
    assert ("gateway/platforms/api_server.py", "APIServerAdapter._open_and_cache_session_db", "acquire") in unported
    assert any(
        item.path == "state_store.py" and item.symbol == "resolve_contextual_session_search_store"
        and item.classification == "backend-neutral-contextual-contract"
        for item in inventory
    )


def test_runtime_file_open_trap_reports_exact_legacy_sqlite_caller(tmp_path):
    from hermes_state import SessionDB

    with pytest.raises(StateDbOpenAttempt) as caught:
        with trap_state_db_opens(tmp_path):
            SessionDB(db_path=tmp_path / "state.db")

    event = caught.value.event
    assert event.path == str((tmp_path / "state.db").resolve())
    assert event.operation == "sqlite3.connect"
    assert event.caller_path.endswith("hermes_cli/sqlite_safe_read.py")
    assert event.caller_symbol == "connect_tracked"
    # SQLite test isolation materializes the fixture before the native connect;
    # the trap proves the connect itself did not proceed.
    assert (tmp_path / "state.db").exists()


def test_selected_postgresql_profile_fails_before_state_db_open_and_reports_tenant(tmp_path, monkeypatch):
    profile_home = tmp_path / ".hermes" / "profiles" / "pg-sandbox"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n    dsn_env: HERMES_STATE_STORE_TEST_DSN\n    connect_timeout_seconds: 3\n    pool_max_size: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", "postgresql://fixture/only")

    from hermes_state import SessionDB

    with trap_state_db_opens(profile_home) as events:
        with pytest.raises(PostgreSQLRuntimeActivationError) as caught:
            SessionDB()

    report = caught.value.report
    assert report.selected_backend == "postgresql"
    assert report.profile_home == str(profile_home.resolve())
    assert report.profile_name == "pg-sandbox"
    assert report.tenant_schema and report.tenant_schema.startswith("hermes_state_store_tenant_")
    assert "alembic-core-v25-catalog" in report.supported_capabilities
    assert "cli-fresh-resume-session-contract" in report.supported_capabilities
    assert "contextual-session-search-contract" in report.supported_capabilities
    assert "contextual-session-search-contract" not in report.missing_capabilities
    assert "gateway-session-routing-transcript" in report.supported_capabilities
    assert "acp-session-transcript-lifecycle" in report.supported_capabilities
    assert "gateway-delivery-ledger-routing" in report.supported_capabilities
    assert "async-delegation-ledger-routing" in report.supported_capabilities
    assert "tui-api-session-runtime" in report.missing_capabilities
    assert "cron-session-transcript-lifecycle" in report.missing_capabilities
    assert report.raw_state_db_openers
    assert events == []
    assert not (profile_home / "state.db").exists()


def test_selected_postgresql_async_dispatch_refuses_before_runner_or_state_db_side_effect(tmp_path, monkeypatch):
    """The delegate_task background path cannot execute or fall back to SQLite under PG.

    The runner is the external-subagent side effect.  Durable dispatch must reject
    before it reaches the executor when its only complete ledger is unavailable.
    """
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import async_delegation as ad

    home = tmp_path / ".hermes" / "profiles" / "pg-sandbox"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n    dsn_env: HERMES_STATE_STORE_TEST_DSN\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", "postgresql://fixture/only")
    runner_calls = []
    token = set_hermes_home_override(str(home))
    try:
        with trap_state_db_opens(home) as events:
            with pytest.raises(PostgreSQLRuntimeActivationError) as caught:
                ad.dispatch_async_delegation(
                    goal="must not run", context=None, toolsets=None, role="leaf", model=None,
                    session_key="", runner=lambda: runner_calls.append("called") or {"status": "completed"},
                )
        assert "async-delegation-ledger-routing" in caught.value.report.supported_capabilities
        assert runner_calls == []
        assert events == []
        assert not (home / "state.db").exists()
    finally:
        ad._reset_for_tests()
        reset_hermes_home_override(token)


@pytest.mark.parametrize("module_name", ["gateway.delivery_ledger", "tools.async_delegation"])
def test_selected_postgresql_blocks_raw_ledger_openers_before_state_db_side_effect(tmp_path, monkeypatch, module_name):
    import importlib
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n    dsn_env: HERMES_STATE_STORE_TEST_DSN\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", "postgresql://fixture/only")
    if module_name == "gateway.delivery_ledger":
        # The gateway delivery ledger is process-scoped so multiplexed profile overrides
        # cannot split boot recovery from live writes; bind its launch home explicitly.
        monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(str(home))
    try:
        opener = importlib.import_module(module_name)._connect
        with trap_state_db_opens(home) as events:
            with pytest.raises(PostgreSQLRuntimeActivationError):
                opener()
        assert events == []
        assert not (home / "state.db").exists()
    finally:
        reset_hermes_home_override(token)


def test_default_sqlite_runtime_is_unchanged(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_state
    from hermes_state import SessionDB

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    db = SessionDB()
    try:
        assert db.db_path == home / "state.db"
        assert (home / "state.db").exists()
    finally:
        db.close()


def test_machine_readable_report_exposes_remaining_callsites_without_secret(tmp_path):
    output = tmp_path / "remaining-callsites.json"
    report = write_runtime_callsite_report(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert report.selected_backend == "sqlite"
    assert payload["ready"] is True
    assert any(item["classification"] == "unported-legacy-runtime" for item in payload["raw_state_db_openers"])
    assert "postgresql://" not in output.read_text(encoding="utf-8")


def test_checked_in_callsite_report_is_the_static_opener_baseline():
    report_path = Path(__file__).parent.parent / "website" / "docs" / "developer-guide" / "state-store-postgresql-runtime-callsite-report.json"
    checked_in = json.loads(report_path.read_text(encoding="utf-8"))["raw_state_db_openers"]
    generated = [
        {"path": item.path, "symbol": item.symbol, "operation": item.operation, "classification": item.classification}
        for item in static_raw_state_db_inventory()
    ]

    assert generated == checked_in


def test_postgresql_report_is_actionable_without_opening_a_database(tmp_path):
    profile_home = tmp_path / ".hermes"
    profile_home.mkdir()

    report = inspect_runtime_activation(
        _PG_CONFIG,
        home=profile_home,
        secret_lookup=lambda name: "postgresql://fixture/only" if name == "HERMES_STATE_STORE_TEST_DSN" else None,
    )

    assert report.selected_backend == "postgresql"
    assert report.ready is False
    assert report.tenant_schema and report.tenant_schema.startswith("hermes_state_store_tenant_")
    assert not (profile_home / "state.db").exists()


def test_nonsecret_sandbox_fixture_selects_postgresql_without_embedding_a_dsn(tmp_path):
    import yaml

    fixture = Path(__file__).parent / "fixtures" / "postgresql-state-store-runtime-sandbox-config.yaml"
    config = yaml.safe_load(fixture.read_text(encoding="utf-8"))
    report = inspect_runtime_activation(
        config,
        home=tmp_path,
        secret_lookup=lambda name: "postgresql://fixture/only" if name == "HERMES_STATE_STORE_TEST_DSN" else None,
    )

    assert report.selected_backend == "postgresql"
    assert "postgresql://" not in fixture.read_text(encoding="utf-8")
