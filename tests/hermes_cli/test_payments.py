"""Tests for the dashboard Payments adapter, API surface, and Slack helpers."""

from __future__ import annotations

import importlib
import json
import sqlite3
import io
from pathlib import Path
import runpy
from contextlib import redirect_stdout

import pytest


SCHEMA_SQL = """
CREATE TABLE payment_requests (
    payment_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence TEXT NOT NULL,
    captured_at TEXT,
    updated_at TEXT,
    vendor TEXT,
    title TEXT,
    amount_value REAL,
    amount_currency TEXT,
    amount_display TEXT,
    due_date TEXT,
    payee_name TEXT,
    account_holder TEXT,
    account_number TEXT,
    sort_code TEXT,
    iban TEXT,
    swift TEXT,
    routing_number TEXT,
    payment_reference TEXT,
    invoice_number TEXT,
    billing_address TEXT,
    tax_details TEXT,
    preview_text TEXT,
    warnings_json TEXT,
    attachments_json TEXT,
    original_json TEXT,
    raw_text_path TEXT,
    materialized_path TEXT,
    review_note TEXT
)
"""

INBOX_ITEMS_SCHEMA_SQL = """
CREATE TABLE inbox_items (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_account TEXT,
    source_thread_id TEXT,
    source_message_id TEXT,
    source_url TEXT,
    captured_at TEXT,
    updated_at TEXT,
    last_classified_at TEXT,
    workflow_type TEXT NOT NULL,
    queue TEXT NOT NULL,
    status TEXT NOT NULL,
    title TEXT,
    summary TEXT,
    counterparty TEXT,
    action_required INTEGER,
    confidence REAL,
    amount_value REAL,
    amount_currency TEXT,
    amount_display TEXT,
    due_date TEXT,
    meeting_dates_json TEXT,
    meeting_timezone TEXT,
    reference_id TEXT,
    receipt_like INTEGER,
    invoice_like INTEGER,
    calendar_like INTEGER,
    warning_flags_json TEXT,
    operator_notes TEXT,
    raw_payload_path TEXT,
    materialized_path TEXT,
    manual_status INTEGER,
    manual_notes INTEGER,
    reviewed_at TEXT,
    reviewed_by TEXT,
    entities_json TEXT,
    artifacts_json TEXT
)
"""


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: home)
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: home / "config.yaml")
    monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: home / ".env")
    return home


@pytest.fixture
def canonical_db(tmp_path, monkeypatch):
    db_path = tmp_path / "payments-review.db"
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA_SQL)
    conn.commit()
    conn.close()
    monkeypatch.setenv("PAYMENTS_REVIEW_DB_PATH", str(db_path))
    monkeypatch.delenv("PAYMENTS_REVIEW_PLUGIN_ROOT", raising=False)
    return db_path


def _insert_payment(db_path: Path, **overrides) -> None:
    row = {
        "payment_id": "pay-1",
        "source": "gmail",
        "status": "new",
        "confidence": "high",
        "captured_at": "2026-07-01T10:00:00+00:00",
        "updated_at": "2026-07-01T10:00:00+00:00",
        "vendor": "Example Vendor",
        "title": "Invoice INV-1001",
        "amount_value": 42.0,
        "amount_currency": "GBP",
        "amount_display": "GBP 42.00",
        "due_date": "2026-07-05",
        "payee_name": "Example Vendor Ltd",
        "account_holder": "Example Vendor Ltd",
        "account_number": "12345678",
        "sort_code": "12-34-56",
        "iban": "GB11TEST00000012345678",
        "swift": "TESTGB2L",
        "routing_number": "",
        "payment_reference": "ACME-1001",
        "invoice_number": "INV-1001",
        "billing_address": "1 Example Street",
        "tax_details": "VAT GB123",
        "preview_text": "Amount due GBP 42.00 by 2026-07-05",
        "warnings_json": json.dumps([]),
        "attachments_json": json.dumps(["invoice.pdf"]),
        "original_json": json.dumps(
            {
                "label": "Acme Billing <billing@example.test>",
                "url": "https://mail.google.com/mail/u/0/#inbox/pay-1",
                "message_id": "msg-1",
                "thread_id": "thread-1",
            }
        ),
        "raw_text_path": "/tmp/pay-1.txt",
        "materialized_path": "/tmp/pay-1.md",
        "review_note": "",
    }
    row.update(overrides)
    columns = ", ".join(row)
    placeholders = ", ".join("?" for _ in row)
    conn = sqlite3.connect(db_path)
    conn.execute(
        f"INSERT OR REPLACE INTO payment_requests ({columns}) VALUES ({placeholders})",
        tuple(row.values()),
    )
    conn.commit()
    conn.close()


def _create_inbox_items_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(INBOX_ITEMS_SCHEMA_SQL)
    conn.commit()
    conn.close()


def _insert_inbox_item(db_path: Path, **overrides) -> None:
    row = {
        "id": "pay-1",
        "source": "gmail",
        "source_account": "default",
        "source_thread_id": "thread-1",
        "source_message_id": "msg-1",
        "source_url": "https://mail.google.com/mail/u/0/#inbox/thread-1",
        "captured_at": "2026-07-01T10:00:00+00:00",
        "updated_at": "2026-07-01T10:00:00+00:00",
        "last_classified_at": "2026-07-01T10:00:00+00:00",
        "workflow_type": "payment_request",
        "queue": "payments",
        "status": "new",
        "title": "Invoice INV-1001",
        "summary": "Amount due GBP 42.00 by 2026-07-05",
        "counterparty": "Example Vendor",
        "action_required": 1,
        "confidence": 0.95,
        "amount_value": 42.0,
        "amount_currency": "GBP",
        "amount_display": "GBP 42.00",
        "due_date": "2026-07-05",
        "meeting_dates_json": json.dumps([]),
        "meeting_timezone": "",
        "reference_id": "ACME-1001",
        "receipt_like": 0,
        "invoice_like": 1,
        "calendar_like": 0,
        "warning_flags_json": json.dumps([]),
        "operator_notes": "Keep manual note",
        "raw_payload_path": "/tmp/pay-1.txt",
        "materialized_path": "/tmp/pay-1.md",
        "manual_status": 0,
        "manual_notes": 1,
        "reviewed_at": None,
        "reviewed_by": None,
        "entities_json": json.dumps(
            {
                "sender": "Acme Billing <billing@example.test>",
                "organization": "Example Vendor",
                "payee_name": "Example Vendor Ltd",
                "account_holder": "Example Vendor Ltd",
                "account_number": "12345678",
                "sort_code": "12-34-56",
                "iban": "GB11TEST00000012345678",
                "swift": "TESTGB2L",
                "payment_reference": "ACME-1001",
                "invoice_number": "INV-1001",
                "billing_address": "1 Example Street",
                "tax_details": "VAT GB123",
            }
        ),
        "artifacts_json": json.dumps(
            {
                "label": "Acme Billing <billing@example.test>",
                "attachments": ["invoice.pdf"],
                "original_thread_url": "https://mail.google.com/mail/u/0/#inbox/pay-1",
            }
        ),
    }
    row.update(overrides)
    columns = ", ".join(row)
    placeholders = ", ".join("?" for _ in row)
    conn = sqlite3.connect(db_path)
    conn.execute(
        f"INSERT OR REPLACE INTO inbox_items ({columns}) VALUES ({placeholders})",
        tuple(row.values()),
    )
    conn.commit()
    conn.close()


class _RuntimeStub:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def update_payment_request_status(self, payload: dict[str, str]) -> str:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE payment_requests SET status = ?, updated_at = ? WHERE payment_id = ?",
            (payload["status"], "2026-07-04T12:00:00+00:00", payload["payment_id"]),
        )
        changed = conn.total_changes
        conn.commit()
        conn.close()
        if not changed:
            return json.dumps({"ok": False, "error": f"Unknown payment_id: {payload['payment_id']}"})
        return json.dumps({"ok": True, "payment": {"payment_id": payload["payment_id"]}})

    def ingest_payments_from_gmail(self, payload: dict[str, object]) -> str:
        _insert_payment(
            self.db_path,
            payment_id="pay-2",
            status="needs_review",
            vendor="Trainline",
            title="Trainline booking confirmation",
            amount_value=55.0,
            amount_currency="GBP",
            amount_display="GBP 55.00",
            due_date="2026-07-07",
            warnings_json=json.dumps(["This looks like a receipt"]),
            preview_text="Your booking receipt is attached",
        )
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE payment_requests SET status = ?, updated_at = ? WHERE payment_id = ?",
            ("paid", "2026-07-04T12:05:00+00:00", "pay-1"),
        )
        conn.commit()
        conn.close()
        return json.dumps(
            {
                "ok": True,
                "query": payload["query_override"],
                "thread_count": 2,
                "payments": [
                    {
                        "payment_id": "pay-2",
                        "source": "gmail",
                        "status": "needs_review",
                        "amount": {"display": "GBP 55.00"},
                    },
                    {
                        "payment_id": "pay-1",
                        "source": "gmail",
                        "status": "paid",
                        "duplicate": True,
                        "amount": {"display": "GBP 42.00"},
                    },
                ],
            }
        )


class TestPaymentsStore:
    def test_list_requests_reads_canonical_store(self, isolated_home, canonical_db, monkeypatch):
        (isolated_home / "google_token.json").write_text('{"token": "x"}', encoding="utf-8")
        monkeypatch.setattr("hermes_cli.config.load_env", lambda: {"EMAIL_ADDRESS": "bills@example.test"})
        _insert_payment(canonical_db, review_note="Keep manual note")

        import hermes_cli.payments as payments

        importlib.reload(payments)
        result = payments.list_payment_requests()

        gmail = next(source for source in result["sources"] if source["id"] == "gmail")
        email = next(source for source in result["sources"] if source["id"] == "email")
        request = result["requests"][0]

        assert gmail["connected"] is True
        assert email["connected"] is True
        assert request["id"] == "pay-1"
        assert request["operator_status"] == "needs_review"
        assert request["source_label"] == "Gmail"
        assert request["amount"]["display"] == "GBP 42.00"
        assert request["review_note"] == "Keep manual note"
        assert request["raw_text_path"] == "/tmp/pay-1.txt"

    def test_list_requests_falls_back_to_immutable_read_only_store(
        self, isolated_home, canonical_db, monkeypatch
    ):
        (isolated_home / "google_token.json").write_text('{"token": "x"}', encoding="utf-8")
        monkeypatch.setattr("hermes_cli.config.load_env", lambda: {"EMAIL_ADDRESS": "bills@example.test"})
        _insert_payment(canonical_db)

        import hermes_cli.payments as payments

        importlib.reload(payments)

        real_connect = payments.sqlite3.connect
        attempts: list[tuple[str, bool]] = []

        class _SchemaProbeFailure:
            row_factory = None

            def execute(self, query, *args, **kwargs):
                if "sqlite_master" in query:
                    raise sqlite3.OperationalError("unable to open database file")
                raise AssertionError("unexpected query before immutable fallback")

        def fake_connect(database, *args, **kwargs):
            attempts.append((str(database), bool(kwargs.get("uri"))))
            if str(database) == str(canonical_db):
                return _SchemaProbeFailure()
            return real_connect(database, *args, **kwargs)

        monkeypatch.setattr(payments.sqlite3, "connect", fake_connect)
        result = payments.list_payment_requests()

        assert result["requests"][0]["id"] == "pay-1"
        assert attempts[0] == (str(canonical_db), False)
        assert attempts[1] == (f"file:{canonical_db}?mode=ro&immutable=1", True)

    def test_list_requests_reads_inbox_items_compatibility_store(self, isolated_home, tmp_path, monkeypatch):
        db_path = tmp_path / "inbox-items.db"
        _create_inbox_items_db(db_path)
        _insert_inbox_item(db_path)
        monkeypatch.setenv("PAYMENTS_REVIEW_DB_PATH", str(db_path))
        (isolated_home / "google_token.json").write_text('{"token": "x"}', encoding="utf-8")
        monkeypatch.setattr("hermes_cli.config.load_env", lambda: {"EMAIL_ADDRESS": "bills@example.test"})

        import hermes_cli.payments as payments

        importlib.reload(payments)
        result = payments.list_payment_requests()

        request = result["requests"][0]
        assert request["id"] == "pay-1"
        assert request["payment_id"] == "pay-1"
        assert request["operator_status"] == "needs_review"
        assert request["vendor"] == "Example Vendor"
        assert request["amount"]["display"] == "GBP 42.00"
        assert request["review_note"] == "Keep manual note"
        assert request["attachments"] == ["invoice.pdf"]
        assert request["original"]["thread_id"] == "thread-1"

    def test_update_status_round_trips_through_canonical_store(self, canonical_db, monkeypatch):
        _insert_payment(canonical_db)

        import hermes_cli.payments as payments

        importlib.reload(payments)

        updated = payments.update_payment_status("pay-1", "paid")

        assert updated["status"] == "paid"
        assert updated["operator_status"] == "paid"

        listed = payments.list_payment_requests()["requests"][0]
        assert listed["status"] == "paid"
        assert listed["updated_at"] != "2026-07-01T10:00:00+00:00"

    def test_update_status_round_trips_through_inbox_items_store(self, tmp_path, monkeypatch):
        db_path = tmp_path / "inbox-items.db"
        _create_inbox_items_db(db_path)
        _insert_inbox_item(db_path)
        monkeypatch.setenv("PAYMENTS_REVIEW_DB_PATH", str(db_path))

        import hermes_cli.payments as payments

        importlib.reload(payments)

        updated = payments.update_payment_status("pay-1", "paid")

        assert updated["status"] == "paid"
        assert updated["operator_status"] == "paid"

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT status, manual_status, reviewed_at FROM inbox_items WHERE id = ?",
            ("pay-1",),
        ).fetchone()
        conn.close()
        assert row[0] == "paid"
        assert row[1] == 1
        assert row[2]

    def test_list_inbox_items_filters_queue_and_workflow(self, tmp_path, monkeypatch):
        db_path = tmp_path / "inbox-items.db"
        _create_inbox_items_db(db_path)
        _insert_inbox_item(db_path, id="pay-1", workflow_type="payment_request", queue="payments")
        _insert_inbox_item(
            db_path,
            id="meet-1",
            workflow_type="meeting_request",
            queue="calendar",
            title="Can we meet next week?",
            summary="Meeting request",
            counterparty="Alice",
            amount_value=None,
            amount_currency="",
            amount_display="",
            due_date=None,
            receipt_like=0,
            invoice_like=0,
            calendar_like=1,
            warning_flags_json=json.dumps([]),
            operator_notes="",
            entities_json=json.dumps({"sender": "Alice <alice@example.test>"}),
            artifacts_json=json.dumps({"original_thread_url": "https://mail.google.com/mail/u/0/#inbox/meet-1"}),
        )
        monkeypatch.setenv("PAYMENTS_REVIEW_DB_PATH", str(db_path))

        import hermes_cli.payments as payments

        importlib.reload(payments)

        only_payments = payments.list_inbox_items(queue="payments")
        only_meetings = payments.list_inbox_items(workflow_type="meeting_request")

        assert [item["id"] for item in only_payments["items"]] == ["pay-1"]
        assert [item["id"] for item in only_meetings["items"]] == ["meet-1"]

    def test_payments_prefer_legacy_table_when_shadow_table_also_exists(self, tmp_path, monkeypatch):
        db_path = tmp_path / "both.db"
        conn = sqlite3.connect(db_path)
        conn.execute(SCHEMA_SQL)
        conn.execute(INBOX_ITEMS_SCHEMA_SQL)
        conn.commit()
        conn.close()
        _insert_payment(db_path, payment_id="pay-legacy", vendor="Legacy Vendor")
        _insert_inbox_item(db_path, id="pay-shadow", counterparty="Shadow Vendor")
        monkeypatch.setenv("PAYMENTS_REVIEW_DB_PATH", str(db_path))

        import hermes_cli.payments as payments

        importlib.reload(payments)

        listed = payments.list_payment_requests()["requests"]
        assert [item["id"] for item in listed] == ["pay-legacy"]

        inbox = payments.list_inbox_items(queue="payments")["items"]
        assert [item["id"] for item in inbox] == ["pay-shadow"]

    def test_update_inbox_item_status_updates_canonical_row(self, tmp_path, monkeypatch):
        db_path = tmp_path / "inbox-items.db"
        _create_inbox_items_db(db_path)
        _insert_inbox_item(db_path, id="meet-1", workflow_type="meeting_request", queue="calendar")
        monkeypatch.setenv("PAYMENTS_REVIEW_DB_PATH", str(db_path))

        import hermes_cli.payments as payments

        importlib.reload(payments)

        updated = payments.update_inbox_item_status("meet-1", "ignored")

        assert updated["id"] == "meet-1"
        assert updated["status"] == "ignored"
        assert updated["manual_status"] is True

    def test_shadow_sync_mirrors_legacy_payments_into_inbox_items(self, canonical_db, monkeypatch):
        _insert_payment(canonical_db)

        import hermes_cli.payments as payments

        importlib.reload(payments)
        monkeypatch.setattr(payments, "_payments_review_runtime", lambda: _RuntimeStub(canonical_db))

        result = payments.sync_gmail_payment_requests_shadow(query="invoice", max_results=5)

        shadow = result["shadow"]
        assert shadow["mirrored"] == 2
        assert shadow["parity"]["payments_count"] == 2
        assert shadow["parity"]["shadow_count"] == 2
        assert shadow["parity"]["parity_ok"] is True

        conn = sqlite3.connect(canonical_db)
        rows = conn.execute(
            "SELECT id, workflow_type, queue, status FROM inbox_items ORDER BY id"
        ).fetchall()
        conn.close()
        assert rows == [
            ("pay-1", "payment_receipt", "payments", "paid"),
            ("pay-2", "payment_receipt", "payments", "needs_review"),
        ]
        snapshot = payments.load_shadow_report_snapshot()
        assert snapshot["updated_at"]
        assert snapshot["result"]["shadow"]["parity"]["parity_ok"] is True

    def test_sync_gmail_payment_requests_uses_canonical_runtime(self, canonical_db, monkeypatch):
        _insert_payment(canonical_db)

        import hermes_cli.payments as payments

        importlib.reload(payments)
        monkeypatch.setattr(payments, "_payments_review_runtime", lambda: _RuntimeStub(canonical_db))

        result = payments.sync_gmail_payment_requests(query="invoice", max_results=5)

        assert result["query"] == "invoice"
        assert result["fetched"] == 2
        assert result["imported"] == 1
        assert result["updated"] == 1
        listed = payments.list_payment_requests()["requests"]
        assert {item["id"] for item in listed} == {"pay-1", "pay-2"}
        assert next(item for item in listed if item["id"] == "pay-1")["status"] == "paid"

    def test_install_gmail_sync_cron_script_and_job(self, isolated_home, canonical_db, monkeypatch):
        import cron.jobs as cron_jobs
        import hermes_cli.payments as payments

        cron_dir = isolated_home / "cron"
        scripts_dir = isolated_home / "scripts"
        cron_dir.mkdir(exist_ok=True)
        (cron_dir / "output").mkdir(exist_ok=True)
        scripts_dir.mkdir(exist_ok=True)
        monkeypatch.setattr(cron_jobs, "HERMES_DIR", isolated_home)
        monkeypatch.setattr(cron_jobs, "CRON_DIR", cron_dir)
        monkeypatch.setattr(cron_jobs, "JOBS_FILE", cron_dir / "jobs.json")
        monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", cron_dir / "output")

        importlib.reload(payments)

        script_path = payments.install_gmail_sync_cron_script(query="invoice newer_than:7d", max_results=7)
        assert script_path == scripts_dir / payments.PAYMENTS_GMAIL_SYNC_CRON_SCRIPT
        script_text = script_path.read_text(encoding="utf-8")
        assert "invoice newer_than:7d" in script_text
        assert "max_results=7" in script_text

        created = payments.ensure_gmail_sync_cron_job(
            schedule="every 2h",
            query="invoice newer_than:7d",
            max_results=7,
        )
        assert created["name"] == payments.PAYMENTS_GMAIL_SYNC_CRON_JOB_NAME
        assert created["script"] == payments.PAYMENTS_GMAIL_SYNC_CRON_SCRIPT
        assert created["no_agent"] is True
        assert created["deliver"] == "local"

        updated = payments.ensure_gmail_sync_cron_job(
            schedule="every 3h",
            query="receipt newer_than:30d",
            max_results=9,
        )
        assert updated["id"] == created["id"]
        jobs = cron_jobs.list_jobs(include_disabled=True)
        assert len(jobs) == 1
        assert jobs[0]["script"] == payments.PAYMENTS_GMAIL_SYNC_CRON_SCRIPT
        assert jobs[0]["schedule_display"] == "every 180m"
        assert "receipt newer_than:30d" in script_path.read_text(encoding="utf-8")
        assert "max_results=9" in script_path.read_text(encoding="utf-8")

    def test_install_gmail_shadow_sync_cron_script_and_job(self, isolated_home, canonical_db, monkeypatch):
        import cron.jobs as cron_jobs
        import hermes_cli.payments as payments

        cron_dir = isolated_home / "cron"
        scripts_dir = isolated_home / "scripts"
        cron_dir.mkdir(exist_ok=True)
        (cron_dir / "output").mkdir(exist_ok=True)
        scripts_dir.mkdir(exist_ok=True)
        monkeypatch.setattr(cron_jobs, "HERMES_DIR", isolated_home)
        monkeypatch.setattr(cron_jobs, "CRON_DIR", cron_dir)
        monkeypatch.setattr(cron_jobs, "JOBS_FILE", cron_dir / "jobs.json")
        monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", cron_dir / "output")

        importlib.reload(payments)

        script_path = payments.install_gmail_shadow_sync_cron_script(
            query="invoice newer_than:7d", max_results=7
        )
        assert script_path == scripts_dir / payments.PAYMENTS_GMAIL_SHADOW_SYNC_CRON_SCRIPT
        script_text = script_path.read_text(encoding="utf-8")
        assert "sync_gmail_payment_requests_shadow" in script_text
        assert "invoice newer_than:7d" in script_text
        assert "max_results=7" in script_text

        created = payments.ensure_gmail_shadow_sync_cron_job(
            schedule="every 6h",
            query="invoice newer_than:7d",
            max_results=7,
        )
        assert created["name"] == payments.PAYMENTS_GMAIL_SHADOW_SYNC_CRON_JOB_NAME
        assert created["script"] == payments.PAYMENTS_GMAIL_SHADOW_SYNC_CRON_SCRIPT
        assert created["no_agent"] is True

    def test_runtime_loader_supports_relative_imports(self, tmp_path, monkeypatch):
        plugins_root = tmp_path / "plugins"
        plugin_root = plugins_root / "payments_review"
        plugin_root.mkdir(parents=True)
        (plugin_root / "validate.py").write_text(
            "def marker():\n    return 'ok'\n",
            encoding="utf-8",
        )
        sibling_root = plugins_root / "google_workspace_cli"
        sibling_root.mkdir()
        (sibling_root / "__init__.py").write_text("", encoding="utf-8")
        (sibling_root / "runtime.py").write_text(
            "def sibling_marker():\n    return 'sibling-ok'\n",
            encoding="utf-8",
        )
        (plugin_root / "runtime.py").write_text(
            "from .validate import marker\n"
            "from google_workspace_cli.runtime import sibling_marker\n\n"
            "VALUE = marker()\n"
            "SIBLING = sibling_marker()\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("PAYMENTS_REVIEW_PLUGIN_ROOT", str(plugin_root))
        monkeypatch.delenv("PAYMENTS_REVIEW_DB_PATH", raising=False)

        import hermes_cli.payments as payments

        importlib.reload(payments)
        runtime = payments._payments_review_runtime()

        assert runtime.VALUE == "ok"
        assert runtime.SIBLING == "sibling-ok"

    def test_slack_helpers_render_summary_and_modal(self, monkeypatch):
        import hermes_cli.payments as payments

        importlib.reload(payments)
        monkeypatch.setattr(
            payments,
            "list_payment_requests",
            lambda: {
                "sources": [],
                "storage_path": "/tmp/payments-review.db",
                "requests": [
                    {
                        "id": "pay-1",
                        "payment_id": "pay-1",
                        "source": "gmail",
                        "source_label": "Gmail",
                        "status": "needs_review",
                        "operator_status": "needs_review",
                        "confidence": "high",
                        "received_at": "2026-07-01T10:00:00+00:00",
                        "updated_at": "2026-07-01T10:00:00+00:00",
                        "vendor": "Acme",
                        "title": "Invoice INV-1001",
                        "amount": {"value": 42.0, "currency": "GBP", "display": "GBP 42.00"},
                        "due_date": "2026-07-05",
                        "payee_name": "",
                        "account_holder": "",
                        "account_number": "",
                        "sort_code": "",
                        "iban": "",
                        "swift": "",
                        "routing_number": "",
                        "payment_reference": "",
                        "invoice_number": "INV-1001",
                        "billing_address": "",
                        "tax_details": "",
                        "preview_text": "Invoice attached",
                        "warnings": [],
                        "attachments": [],
                        "original": {"label": "Acme", "url": "", "message_id": "", "thread_id": ""},
                        "looks_paid": False,
                    },
                    {
                        "id": "pay-2",
                        "payment_id": "pay-2",
                        "source": "gmail",
                        "source_label": "Gmail",
                        "status": "paid",
                        "operator_status": "paid",
                        "confidence": "medium",
                        "received_at": "2026-07-01T10:00:00+00:00",
                        "updated_at": "2026-07-01T10:00:00+00:00",
                        "vendor": "Uber",
                        "title": "Trip receipt",
                        "amount": {"value": 18.0, "currency": "GBP", "display": "GBP 18.00"},
                        "due_date": None,
                        "payee_name": "",
                        "account_holder": "",
                        "account_number": "",
                        "sort_code": "",
                        "iban": "",
                        "swift": "",
                        "routing_number": "",
                        "payment_reference": "",
                        "invoice_number": "",
                        "billing_address": "",
                        "tax_details": "",
                        "preview_text": "Thanks for riding",
                        "warnings": ["Payment already been made"],
                        "attachments": [],
                        "original": {"label": "Uber", "url": "", "message_id": "", "thread_id": ""},
                        "looks_paid": True,
                    },
                ],
            },
        )

        markdown = payments.render_slack_canvas_markdown(dashboard_url="https://example.test/payments")
        blocks = payments.build_slack_mobile_blocks()
        modal = payments.build_slack_status_modal_view(
            {
                "id": "pay-1",
                "vendor": "Acme",
                "title": "Invoice INV-1001",
                "amount": {"display": "GBP 42.00"},
                "due_date": "2026-07-05",
                "status": "needs_review",
                "operator_status": "needs_review",
                "looks_paid": False,
            },
            private_metadata='{"channel_id":"C1","message_ts":"123.45"}',
        )

        assert "Needs review" in markdown
        assert "Recently paid" in markdown
        assert "receipt-like" in markdown
        assert any(
            block.get("accessory", {}).get("action_id") == "payments_open_status_modal"
            for block in blocks
            if block.get("type") == "section"
        )
        assert modal["private_metadata"] == '{"channel_id":"C1","message_ts":"123.45"}'
        assert modal["blocks"][2]["elements"][2]["action_id"] == "payments_mark_paid"


class TestPaymentsApi:
    @pytest.fixture(autouse=True)
    def _setup_client(self, isolated_home, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", isolated_home / "state.db")

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def test_list_and_update_payments(self, canonical_db, monkeypatch):
        _insert_payment(canonical_db)

        import hermes_cli.payments as payments

        importlib.reload(payments)
        monkeypatch.setattr(payments, "_payments_review_runtime", lambda: _RuntimeStub(canonical_db))

        listed = self.client.get("/api/payments")
        assert listed.status_code == 200
        payload = listed.json()
        assert payload["requests"][0]["id"] == "pay-1"
        assert payload["requests"][0]["operator_status"] == "needs_review"

        updated = self.client.post("/api/payments/pay-1/status", json={"status": "ready_to_pay"})
        assert updated.status_code == 200
        assert updated.json()["status"] == "ready_to_pay"

    def test_update_rejects_unknown_status(self):
        updated = self.client.post("/api/payments/missing/status", json={"status": "bogus"})
        assert updated.status_code == 400

    def test_sync_payments_endpoint(self, canonical_db, monkeypatch):
        import hermes_cli.payments as payments

        importlib.reload(payments)

        def fake_sync(query, max_results):
            assert query == payments.DEFAULT_GMAIL_QUERY
            assert max_results == payments.DEFAULT_GMAIL_MAX_RESULTS
            return {
                "source": "gmail",
                "query": query,
                "fetched": 2,
                "imported": 1,
                "updated": 1,
                "requests": [],
            }

        monkeypatch.setattr(payments, "sync_gmail_payment_requests", fake_sync)
        resp = self.client.post("/api/payments/sync", json={"source": "gmail"})
        assert resp.status_code == 200
        assert resp.json()["updated"] == 1

    def test_shadow_sync_and_report_endpoints(self, canonical_db, monkeypatch):
        _insert_payment(canonical_db)

        import hermes_cli.payments as payments

        importlib.reload(payments)
        monkeypatch.setattr(payments, "_payments_review_runtime", lambda: _RuntimeStub(canonical_db))

        sync_response = self.client.post("/api/payments/shadow-sync", json={"source": "gmail"})
        assert sync_response.status_code == 200
        assert sync_response.json()["shadow"]["parity"]["parity_ok"] is True

        report_response = self.client.get("/api/payments/shadow-report")
        assert report_response.status_code == 200
        payload = report_response.json()
        assert payload["parity"]["parity_ok"] is True
        assert payload["snapshot"]["result"]["shadow"]["mirrored"] == 2

    def test_shadow_schedule_endpoint(self, isolated_home, canonical_db, monkeypatch):
        import cron.jobs as cron_jobs
        import hermes_cli.payments as payments

        cron_dir = isolated_home / "cron"
        scripts_dir = isolated_home / "scripts"
        cron_dir.mkdir(exist_ok=True)
        (cron_dir / "output").mkdir(exist_ok=True)
        scripts_dir.mkdir(exist_ok=True)
        monkeypatch.setattr(cron_jobs, "HERMES_DIR", isolated_home)
        monkeypatch.setattr(cron_jobs, "CRON_DIR", cron_dir)
        monkeypatch.setattr(cron_jobs, "JOBS_FILE", cron_dir / "jobs.json")
        monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", cron_dir / "output")

        importlib.reload(payments)

        response = self.client.post(
            "/api/payments/shadow-schedule",
            json={"schedule": "every 6h", "run_now": False},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["job"]["script"] == payments.PAYMENTS_GMAIL_SHADOW_SYNC_CRON_SCRIPT

    def test_list_inbox_items_endpoint(self, tmp_path, monkeypatch):
        db_path = tmp_path / "inbox-items.db"
        _create_inbox_items_db(db_path)
        _insert_inbox_item(db_path, id="pay-1", workflow_type="payment_request", queue="payments")
        _insert_inbox_item(db_path, id="meet-1", workflow_type="meeting_request", queue="calendar")
        monkeypatch.setenv("PAYMENTS_REVIEW_DB_PATH", str(db_path))

        response = self.client.get("/api/inbox-items?queue=payments")
        assert response.status_code == 200
        payload = response.json()
        assert [item["id"] for item in payload["items"]] == ["pay-1"]

    def test_update_inbox_item_status_endpoint(self, tmp_path, monkeypatch):
        db_path = tmp_path / "inbox-items.db"
        _create_inbox_items_db(db_path)
        _insert_inbox_item(db_path, id="meet-1", workflow_type="meeting_request", queue="calendar")
        monkeypatch.setenv("PAYMENTS_REVIEW_DB_PATH", str(db_path))

        response = self.client.post("/api/inbox-items/meet-1/status", json={"status": "ignored"})
        assert response.status_code == 200
        assert response.json()["status"] == "ignored"

    def test_ops_services_endpoint_includes_defaults_and_config_overrides(self, isolated_home):
        (isolated_home / "config.yaml").write_text(
            """
dashboard:
  ops_links:
    - id: grafana
      title: Grafana Cloud
      url: https://grafana.example.test
      description: Shared Grafana endpoint
      group: Observability
      tags: [grafana, cloud]
    - id: n8n
      title: n8n
      url: http://automation.example.test
      description: Workflow automation
      group: Automation
      tags: [automation]
""".strip()
            + "\n",
            encoding="utf-8",
        )

        response = self.client.get("/api/ops/services")
        assert response.status_code == 200
        payload = response.json()
        services = {item["id"]: item for item in payload["services"]}

        assert services["payments"]["url"].endswith(":9121/payments")
        assert services["grafana"]["title"] == "Grafana Cloud"
        assert services["grafana"]["url"] == "https://grafana.example.test"
        assert services["n8n"]["group"] == "Automation"

    def test_webhub_ops_summary_endpoint(self, isolated_home, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

        import plugins.observability.webhub as webhub

        importlib.reload(webhub)
        webhub.on_post_api_request(
            session_id="s1",
            turn_id="t1",
            api_request_id="r1",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-sonnet-4.5",
            api_mode="chat_completions",
            api_duration=0.75,
            usage={
                "input_tokens": 40,
                "output_tokens": 10,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
                "request_count": 1,
                "total_tokens": 50,
            },
        )

        response = self.client.get("/api/ops/webhub")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"]["openrouter_configured"] is True
        assert payload["status"]["slack_configured"] is True
        assert payload["summary"]["requests"]["total"] == 1
        assert payload["links"]["channels_url"].endswith(":9121/channels")

    def test_webhub_prometheus_endpoint(self, isolated_home):
        response = self.client.get("/api/ops/webhub/prometheus")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "hermes_webhub_requests_total" in response.text


class TestPaymentsSlackSurfaceScript:
    def test_find_existing_mobile_inbox_ts_prefers_latest_matching_text(self):
        module = runpy.run_path(
            str(Path(__file__).resolve().parents[2] / "scripts" / "payments-slack-surface.py")
        )

        find_ts = module["_find_existing_mobile_inbox_ts"]
        mobile_text = module["MOBILE_INBOX_TEXT"]

        messages = [
            {"ts": "300.0", "text": "something else"},
            {"ts": "200.0", "text": mobile_text},
            {"ts": "100.0", "text": mobile_text},
        ]

        assert find_ts(messages) == "200.0"

    def test_find_existing_mobile_inbox_ts_returns_empty_when_missing(self):
        module = runpy.run_path(
            str(Path(__file__).resolve().parents[2] / "scripts" / "payments-slack-surface.py")
        )

        find_ts = module["_find_existing_mobile_inbox_ts"]

        assert find_ts([{"ts": "100.0", "text": "no match"}]) == ""

    def test_print_sync_summary_formats_counts(self):
        module = runpy.run_path(
            str(Path(__file__).resolve().parents[2] / "scripts" / "payments-slack-surface.py")
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            module["_print_sync_summary"]({"fetched": 12, "imported": 3, "updated": 9})

        assert (
            buffer.getvalue().strip()
            == "synced gmail payments fetched=12 imported=3 updated=9"
        )
