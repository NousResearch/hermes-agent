"""Phase2 P1: crashed/timed_out runs carry provider_error/infra_error telemetry.

Additive ``task_runs.metadata`` keys only — no migration, no filtering.
Covers the pure classifier, the crash-close hook (``_classify_dead_worker``)
and the timeout-close hook (``enforce_max_runtime``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def test_rate_limit_text_classifies_with_code_and_provider():
    hit = kbd._classify_provider_error_text(
        "opencode-go request failed: rate limit exceeded, retry later (429)"
    )
    assert hit is not None
    assert hit["reason"] == "rate_limit"
    assert hit["code"] == 429
    assert hit["provider"] == "opencode-go"


def test_billing_text_classifies():
    hit = kbd._classify_provider_error_text(
        "API error 402: insufficient credits, top up"
    )
    assert hit is not None
    assert hit["reason"] == "billing"
    assert hit["code"] == 402


def test_overloaded_and_timeout_texts():
    over = kbd._classify_provider_error_text("provider overloaded (503), backing off")
    assert over is not None and over["reason"] == "overloaded" and over["code"] == 503
    tmo = kbd._classify_provider_error_text(
        "connection timed out after 120s (openai-codex)"
    )
    assert tmo is not None and tmo["reason"] == "timeout"
    assert tmo["provider"] == "openai-codex"


def test_auth_model_not_found_and_context_texts():
    auth = kbd._classify_provider_error_text("401 unauthorized: invalid api key")
    assert auth is not None and auth["reason"] == "auth"
    mnf = kbd._classify_provider_error_text("model 'mimo-v2.5' not found for provider")
    assert mnf is not None and mnf["reason"] == "model_not_found" and mnf["code"] == 404
    ctx = kbd._classify_provider_error_text("request exceeds maximum context length")
    assert ctx is not None and ctx["reason"] == "context_overflow"


def test_benign_and_empty_texts_return_none():
    assert kbd._classify_provider_error_text("") is None
    assert kbd._classify_provider_error_text("   ") is None
    assert (
        kbd._classify_provider_error_text(
            "Finished the refactor; all 12 tests pass. See summary above."
        )
        is None
    )


def test_detail_is_capped_and_provider_optional():
    long_text = "x" * 5000 + " rate limit 429"
    hit = kbd._classify_provider_error_text(long_text)
    assert hit is not None
    assert len(hit["detail"]) <= kbd._PROVIDER_TELEMETRY_DETAIL_CHARS
    assert hit["provider"] is None
    assert set(hit) == {"reason", "code", "provider", "detail"}
    json.dumps(hit)  # JSON-safe


# ---------------------------------------------------------------------------
# Crash-close hook
# ---------------------------------------------------------------------------


def test_classify_dead_worker_attaches_provider_error(monkeypatch):
    monkeypatch.setattr(kbd, "_classify_worker_exit", lambda pid: ("nonzero_exit", 1))
    monkeypatch.setattr(
        kbd,
        "_worker_final_output",
        lambda task_id, board=None: "request failed: 429 rate limit exceeded",
    )
    dead = kbd._classify_dead_worker(424242, "lock", task_id="t_x", board=None)
    assert dead.event_kind == "crashed"
    assert dead.event_payload["provider_error"]["reason"] == "rate_limit"
    assert dead.event_payload["provider_error"]["code"] == 429
    assert "infra_error" not in dead.event_payload


def test_classify_dead_worker_attaches_infra_error_without_signature(monkeypatch):
    monkeypatch.setattr(kbd, "_classify_worker_exit", lambda pid: ("signaled", 9))
    monkeypatch.setattr(kbd, "_worker_final_output", lambda task_id, board=None: "")
    dead = kbd._classify_dead_worker(424243, "lock", task_id="t_y", board=None)
    assert dead.event_kind == "crashed"
    assert "provider_error" not in dead.event_payload
    infra = dead.event_payload["infra_error"]
    assert infra["reason"] == "worker_signaled"
    assert infra["code"] == 9


# ---------------------------------------------------------------------------
# Timeout-close hook (end-to-end into task_runs.metadata)
# ---------------------------------------------------------------------------


def _backdate_run(conn, tid, seconds_ago=30):
    old_started = int(time.time()) - seconds_ago
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (old_started, tid))
        conn.execute(
            "UPDATE task_runs SET started_at = ? "
            "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (old_started, tid),
        )


def _latest_run_metadata(conn, tid):
    row = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    assert row is not None and row["metadata"], "closing run must carry metadata"
    return json.loads(row["metadata"])


def test_timeout_close_carries_infra_error(kanban_home, monkeypatch):
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)
    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn, title="long job", assignee="worker", max_runtime_seconds=1
        )
        kb.claim_task(conn, tid)
        kbd._set_worker_pid(conn, tid, os.getpid())
        _backdate_run(conn, tid)
        assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: None)
        meta = _latest_run_metadata(conn, tid)
        assert meta["infra_error"]["reason"] == "max_runtime_exceeded"
        assert meta["limit_seconds"] == 1
    finally:
        conn.close()


def test_timeout_close_prefers_provider_error(kanban_home, monkeypatch):
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(
        kbd,
        "_worker_final_output",
        lambda task_id, board=None: "provider 503 overloaded, retrying",
    )
    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn, title="long job", assignee="worker", max_runtime_seconds=1
        )
        kb.claim_task(conn, tid)
        kbd._set_worker_pid(conn, tid, os.getpid())
        _backdate_run(conn, tid)
        assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: None)
        meta = _latest_run_metadata(conn, tid)
        assert meta["provider_error"]["reason"] == "overloaded"
        assert meta["provider_error"]["code"] == 503
        assert "infra_error" not in meta
    finally:
        conn.close()
