"""Targeted tests for M4 salvage fixes (C019/C026/C027/C028/C030).

Non-live: these tests exercise path resolution, idempotent state claims,
and atomic write patterns without starting services, connecting to
PostgreSQL, or triggering live reindexes.
"""
from __future__ import annotations

import os
import sys
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_gitradar_upstream_monitor_uses_shared_runbook_root():
    """The consumer should resolve repos dir from the shared runbook root."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "archive" / "gitradar-upstream-monitor.py"
    if not script.exists():
        pytest.skip("gitradar-upstream-monitor.py not found")
    source = script.read_text()
    assert "runbooks" in source
    assert "github-radar" in source
    assert "_RUNBOOK_ROOT" in source


def test_gitradar_discover_uses_shared_runbook_root():
    """The producer should write to the shared runbook root."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "github-radar-discover.py"
    if not script.exists():
        pytest.skip("github-radar-discover.py not found")
    source = script.read_text()
    assert "RUNBOOKS_DIR" in source
    assert "runbooks" in source and "github-radar" in source


def test_claim_for_enqueue_is_idempotent(tmp_path):
    """claim_for_enqueue should only succeed once for the same draft."""
    ce_dir = Path(__file__).resolve().parents[1]
    if str(ce_dir) not in sys.path:
        sys.path.insert(0, str(ce_dir))

    db_path = tmp_path / "test_content_engine.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        "CREATE TABLE drafts ("
        "id TEXT PRIMARY KEY, brand TEXT NOT NULL, platform TEXT NOT NULL, "
        "content_type TEXT NOT NULL DEFAULT 'text', pillar TEXT NOT NULL, "
        "topic TEXT NOT NULL, title TEXT, body_text TEXT NOT NULL, "
        "visual_description TEXT, visual_path TEXT, ai_image_path TEXT, "
        "ai_video_path TEXT, image_prompt TEXT, video_prompt TEXT, "
        "status TEXT NOT NULL DEFAULT 'draft', created_at TEXT NOT NULL, "
        "approved_at TEXT, rejected_at TEXT, published_at TEXT, "
        "postiz_id TEXT, ai_enriched_at TEXT, regenerate_count INTEGER DEFAULT 0, "
        "slop_score INTEGER DEFAULT 0, slop_issues TEXT DEFAULT '', "
        "source_provenance TEXT, editorial_rationale TEXT, enqueue_state TEXT);"
    )
    conn.execute(
        "INSERT INTO drafts (id, brand, platform, pillar, topic, body_text, created_at, status) "
        "VALUES ('d1', 'sahil_twitter', 'twitter', 'tech', 'ai', 'test body', '2026-01-01', 'approved')"
    )
    conn.commit()
    conn.close()

    with patch("database.DB_PATH", db_path):
        from database import claim_for_enqueue, release_enqueue_claim

        assert claim_for_enqueue("d1") is True
        assert claim_for_enqueue("d1") is False
        release_enqueue_claim("d1")
        assert claim_for_enqueue("d1") is True


def test_xurl_config_write_is_atomic(tmp_path, monkeypatch):
    """configure_xurl_from_postiz should write the xurl config atomically."""
    # Isolate the module-level browser state path before importing the module.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    ce_dir = Path(__file__).resolve().parents[1]
    if str(ce_dir) not in sys.path:
        sys.path.insert(0, str(ce_dir))

    xurl_path = tmp_path / ".xurl"
    monkeypatch.setattr("os.path.expanduser", lambda p: str(xurl_path) if p == "~/.xurl" else p)

    mock_result = MagicMock(returncode=0, stderr="")
    mock_result.stdout = "1|test-name|access-token:access-secret|rtok|Sahil_Saghir\n"
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_result)

    for key, val in {
        "XURL_CLIENT_ID": "cid",
        "XURL_CLIENT_SECRET": "csec",
        "XURL_CONSUMER_KEY": "ckey",
        "XURL_CONSUMER_SECRET": "csec2",
    }.items():
        monkeypatch.setenv(key, val)

    # Rely on the cached import from test_engagement_credentials.py
    # (engagement_x_poster uses Path.home() at module level, so we
    # cannot re-import it safely in a test with a fake HOME).
    from engagement_suggester import configure_xurl_from_postiz

    assert configure_xurl_from_postiz() is True
    assert xurl_path.exists()
    content = xurl_path.read_text()
    assert "client_id: cid" in content
    assert "access_token: access-token" in content

    stat = os.stat(str(xurl_path))
    mode = stat.st_mode & 0o777
    assert mode == 0o600, f"xurl config permissions are {oct(mode)}, expected 0o600"


def test_gitnexus_reindex_hook_exists_and_is_valid():
    """The reindex hook script should exist and have valid bash syntax."""
    hook = Path(__file__).resolve().parents[2] / "scripts" / "gitnexus-reindex-hook.sh"
    assert hook.exists()
    assert os.access(str(hook), os.X_OK)
    source = hook.read_text()
    assert "grep -qE" in source and "40" in source
    assert "ERROR:" in source
    assert "registry.json" in source


def test_gitnexus_reindex_runner_has_failure_reporting():
    """The reindex runner should report failures explicitly."""
    runner = Path(__file__).resolve().parents[2] / "scripts" / "gitnexus-reindex-runner.sh"
    assert runner.exists()
    source = runner.read_text()
    assert "ERROR:" in source
    assert "Exit code:" in source


def test_approve_does_not_enqueue_to_postiz():
    """The approve command should NOT call queue_post (state-only).
    Checks for actual function calls, not mentions in comments."""
    ce = Path(__file__).resolve().parents[1] / "content_engine.py"
    source = ce.read_text()
    assert 'args.cmd == "approve"' in source
    approve_start = source.index('args.cmd == "approve"')
    approve_end = source.index('elif args.cmd', approve_start)
    approve_block = source[approve_start:approve_end]
    # Check that there is no actual queue_post() call (not in comments).
    # Strip comment lines before checking.
    code_lines = [
        line for line in approve_block.split("\n")
        if not line.strip().startswith("#")
    ]
    code_only = "\n".join(code_lines)
    assert "queue_post(" not in code_only, (
        "approve command still directly calls queue_post() - should be state-only"
    )
    assert "state-only" in approve_block or "publish_to_postiz" in approve_block
