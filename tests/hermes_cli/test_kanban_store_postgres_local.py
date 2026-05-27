import os
import uuid

import pytest

from hermes_cli.kanban_store_factory import create_kanban_store


pytestmark = pytest.mark.skipif(
    not os.environ.get("HERMES_TEST_POSTGRES_DSN"),
    reason="set HERMES_TEST_POSTGRES_DSN to run local PostgreSQL Kanban store tests",
)


def _dsn_with_schema(base: str, schema: str) -> str:
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}options=-csearch_path%3D{schema}"


def test_postgres_store_initializes_and_supports_basic_task_flow(monkeypatch):
    base_dsn = os.environ["HERMES_TEST_POSTGRES_DSN"]
    schema = "test_kanban_" + uuid.uuid4().hex[:12]
    monkeypatch.setenv("HERMES_KANBAN_POSTGRES_DSN", _dsn_with_schema(base_dsn, schema))
    store = create_kanban_store("postgres")

    with store.connect(board="contract") as conn:
        task_id = store.create_task(
            conn,
            title="postgres contract task",
            body="created in local postgres contract test",
            assignee="qa",
            created_by="pytest",
            initial_status="running",
            board="contract",
        )
        listed = store.list_tasks(conn, status="ready")
        claimed = store.claim_task(conn, task_id, ttl_seconds=30, claimer="pytest")
        assert claimed is not None
        assert claimed.id == task_id
        assert any(task.id == task_id for task in listed)
        assert store.complete_task(conn, task_id, summary="done from postgres contract") is True
        assert store.get_task(conn, task_id).status == "done"
        stats = store.board_stats(conn)
        assert stats["done"] >= 1

    # cleanup schema explicitly
    import psycopg
    with psycopg.connect(base_dsn, autocommit=True) as cleanup:
        cleanup.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
