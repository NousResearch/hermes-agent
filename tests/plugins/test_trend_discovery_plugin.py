"""Tests for the Trend Discovery Center plugin."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from plugins.trend_discovery.health import health_check
from plugins.trend_discovery.knowledge import build_digest, write_review_queue
from plugins.trend_discovery.notifications import notify, watchdog
from plugins.trend_discovery.scanner import TrendScanner
from plugins.trend_discovery.store import TrendDiscoveryStore
from plugins.trend_discovery.cli import _handle_sources


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture()
def store(hermes_home):
    path = hermes_home / "trend-discovery" / "test.db"
    store = TrendDiscoveryStore(path)
    store.init()
    return store


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - stdlib hook
        if self.path == "/rss":
            body = b"""<?xml version="1.0"?>
            <rss><channel>
              <item>
                <title>Acme Agents launches workflow automation</title>
                <link>https://example.test/acme-agents</link>
                <description>New startup building agentic workflow tools.</description>
              </item>
            </channel></rss>"""
            self.send_response(200)
            self.send_header("Content-Type", "application/rss+xml")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/json":
            body = json.dumps(
                {
                    "items": [
                        {
                            "title": "RobotGrid raises seed funding",
                            "url": "https://example.test/robotgrid",
                            "summary": "Robotics automation startup funding.",
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/page":
            body = b"<html><head><title>Climate AI Tool</title><meta name='description' content='Climate technology software startup'></head></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(500)
        self.end_headers()

    def log_message(self, *_args):
        return


@pytest.fixture()
def http_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()


def test_store_seeds_full_plan(store):
    snapshot = store.status_snapshot()
    assert len(snapshot["phases"]) == 5
    assert len(snapshot["issues"]) == 53
    rows = store.compliance_rows()
    assert rows[-1]["issue_id"] == "PROJECT_TOTAL"
    assert rows[-1]["percent_complete"] == 0


def test_mark_all_issue_completion_updates_phase_total(store):
    for row in store.compliance_rows():
        if row["issue_id"].startswith("TD-000"):
            store.set_issue_complete(row["issue_id"], evidence={"test": True})
    rows = store.compliance_rows()
    p0 = [row for row in rows if row["issue_id"] == "P0_TOTAL"][0]
    assert p0["percent_complete"] == 100
    assert p0["remaining_percent"] == 0


def test_health_check_local_store(store):
    result = health_check(store)
    assert result["ok"] is True
    assert result["checks"]["database"]["ok"] is True
    assert result["checks"]["issues_seeded"]["count"] == 53


def test_scanner_inserts_from_multiple_sources_and_survives_failure(store, http_server):
    with store.connect() as conn:
        conn.execute("DELETE FROM sources")
        conn.execute(
            "INSERT INTO sources (name, adapter, url, priority, timeout_seconds) VALUES (?, ?, ?, ?, ?)",
            ("local-rss", "rss", f"{http_server}/rss", 1, 5),
        )
        conn.execute(
            "INSERT INTO sources (name, adapter, url, priority, timeout_seconds) VALUES (?, ?, ?, ?, ?)",
            ("local-json", "open_crawl", f"{http_server}/json", 2, 5),
        )
        conn.execute(
            "INSERT INTO sources (name, adapter, url, priority, timeout_seconds) VALUES (?, ?, ?, ?, ?)",
            ("broken", "rss", f"{http_server}/missing", 3, 5),
        )
    result = TrendScanner(store).scan(query="robotics automation startup")
    assert result["status"] == "success"
    assert result["inserted"] == 2
    assert any(item["source"] == "broken" and item["status"] == "failed" for item in result["sources"])
    snapshot = store.status_snapshot()
    assert snapshot["findings_count"] == 2


def test_circuit_breaker_opens_after_repeated_failures(store, http_server):
    with store.connect() as conn:
        conn.execute("DELETE FROM sources")
        conn.execute(
            "INSERT INTO sources (name, adapter, url, priority, timeout_seconds) VALUES (?, ?, ?, ?, ?)",
            ("broken", "rss", f"{http_server}/missing", 1, 5),
        )
    scanner = TrendScanner(store)
    for _ in range(3):
        scanner.scan(query="startup")
    with store.connect() as conn:
        row = conn.execute("SELECT failure_count, circuit_open_until FROM sources WHERE name='broken'").fetchone()
    assert row["failure_count"] >= 3
    assert row["circuit_open_until"]


def test_notifications_fallback_to_local_receipt(store):
    result = notify(store, "hello", target="env-webhook")
    assert result["status"] == "sent"
    assert result["evidence"]["fallback"] == "local"
    with store.connect() as conn:
        row = conn.execute("SELECT status, error FROM notifications ORDER BY created_at DESC LIMIT 1").fetchone()
    assert row["status"] == "sent"
    assert "HERMES_TD_WEBHOOK_URL" in row["error"]


def test_configure_values_are_persisted(store):
    store.set_config("notification.primary", "macos")
    store.set_config("notification.fallback", "local")
    assert store.get_config("notification.primary") == "macos"
    assert store.get_config("notification.fallback") == "local"


def test_source_admin_add_disable_enable_delete(store):
    class Args:
        source_action = "add"
        name = "local-test-rss"
        adapter = "rss"
        url = "https://example.test/feed.xml"
        priority = 12
        timeout = 7
        metadata = '{"scope": "test"}'
        json = False

    assert _handle_sources(store, Args()) == 0
    rows = store.status_snapshot()["sources"]
    added = [row for row in rows if row["name"] == "local-test-rss"][0]
    assert added["adapter"] == "rss"
    assert added["enabled"] == 1

    Args.source_action = "disable"
    assert _handle_sources(store, Args()) == 0
    disabled = [row for row in store.status_snapshot()["sources"] if row["name"] == "local-test-rss"][0]
    assert disabled["enabled"] == 0

    Args.source_action = "enable"
    assert _handle_sources(store, Args()) == 0
    enabled = [row for row in store.status_snapshot()["sources"] if row["name"] == "local-test-rss"][0]
    assert enabled["enabled"] == 1

    Args.source_action = "delete"
    assert _handle_sources(store, Args()) == 0
    assert not [row for row in store.status_snapshot()["sources"] if row["name"] == "local-test-rss"]


def test_watchdog_reports_no_successful_run(store):
    result = watchdog(store, notify_user=False)
    assert result["ok"] is False
    assert "no successful run recorded yet" in result["alerts"]


def test_digest_and_review_queue_writeback(store, http_server):
    with store.connect() as conn:
        conn.execute("DELETE FROM sources")
        conn.execute(
            "INSERT INTO sources (name, adapter, url, priority, timeout_seconds) VALUES (?, ?, ?, ?, ?)",
            ("local-rss", "rss", f"{http_server}/rss", 1, 5),
        )
    TrendScanner(store).scan(query="agentic workflow startup")
    digest = build_digest(store)
    assert "Acme Agents" in digest
    path = write_review_queue(store)
    assert path.exists()
    assert "Trend Discovery Digest" in path.read_text()


def test_plugin_discovery_when_enabled(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - trend-discovery\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import hermes_cli.plugins as plugins_mod

    monkeypatch.setattr(plugins_mod, "_plugin_manager", None)
    manager = plugins_mod.get_plugin_manager()
    manager.discover_and_load()
    loaded = manager._plugins.get("trend-discovery")
    assert loaded is not None
    assert loaded.enabled
    assert "trend-discovery" in manager._cli_commands
    assert plugins_mod is not None
    from hermes_cli.plugins_cmd import _plugin_exists

    assert _plugin_exists("trend-discovery") is True
