from sqlalchemy import create_engine, text

from app.migration_bootstrap import run_startup_schema_bootstrap


def test_bootstrap_adds_missing_policy_columns(tmp_path):
    db_path = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE tenant_policies (
                    id INTEGER PRIMARY KEY,
                    tenant_id INTEGER NOT NULL,
                    min_confidence FLOAT NOT NULL DEFAULT 0.22,
                    force_handoff_keywords_json TEXT NOT NULL DEFAULT '[]',
                    pii_redaction_enabled BOOLEAN NOT NULL DEFAULT 1,
                    max_citations INTEGER NOT NULL DEFAULT 5,
                    created_at DATETIME,
                    updated_at DATETIME
                )
                """
            )
        )

    summary = run_startup_schema_bootstrap(engine)
    assert summary["dialect"] == "sqlite"
    assert int(summary["applied"]) >= 1

    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info('tenant_policies')")).fetchall()
    cols = {r[1] for r in rows}

    assert "policy_rules_json" in cols
    assert "policy_pack" in cols
    assert "daily_query_budget" in cols
    assert "daily_run_budget" in cols
    assert "daily_cost_budget_usd" in cols
    assert "max_top_k" in cols
    assert "max_question_chars" in cols
    assert "daily_external_api_budget" in cols
    assert "external_api_timeout_cap_seconds" in cols
    assert "public_api_allowlist_json" in cols


def test_bootstrap_creates_public_api_providers_table(tmp_path):
    db_path = tmp_path / "legacy-no-provider-table.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)

    # no pre-created tables needed; bootstrap should create provider registry table
    summary = run_startup_schema_bootstrap(engine)
    assert summary["dialect"] == "sqlite"

    with engine.begin() as conn:
        tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='public_api_providers'"))
        row = tables.fetchone()
        assert row is not None

        cols = conn.execute(text("PRAGMA table_info('public_api_providers')")).fetchall()
        col_names = {r[1] for r in cols}

    assert "name" in col_names
    assert "base_url" in col_names
    assert "tenant_scope" in col_names
    assert "sample_query_json" in col_names
