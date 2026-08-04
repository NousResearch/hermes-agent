import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "content_engine" / "publish_to_postiz.py"


def _load_module(monkeypatch, db_path):
    monkeypatch.setenv("CONTENT_ENGINE_DB_PATH", str(db_path))
    sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location("publish_to_postiz_runtime_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _draft_db(path, approved=False):
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE drafts (
        id TEXT, brand TEXT, platform TEXT, body_text TEXT, title TEXT,
        content_type TEXT, ai_image_path TEXT, visual_path TEXT,
        approved_at TEXT, pillar TEXT, status TEXT, postiz_id TEXT,
        enqueue_state TEXT)""")
    if approved:
        conn.execute("INSERT INTO drafts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            "draft-1", "sahil_twitter", "twitter", "hello", "title", "text",
            None, None, "now", "pillar", "approved", None, "pending"))
    conn.commit()
    conn.close()


def test_missing_database_fails_without_creating_it(monkeypatch, tmp_path):
    db = tmp_path / "missing.db"
    module = _load_module(monkeypatch, db)
    with pytest.raises(RuntimeError, match="database missing"):
        module.preflight()
    assert not db.exists()


def test_zero_drafts_still_checks_downstream_then_reports_no_work(monkeypatch, tmp_path, capsys):
    db = tmp_path / "content.db"
    _draft_db(db)
    module = _load_module(monkeypatch, db)
    calls = []
    monkeypatch.setattr(module, "check_publisher_readiness", lambda: calls.append(True))
    assert module.main([]) == 0
    assert calls == [True]
    assert "no-work" in capsys.readouterr().out


def test_preflight_reports_ready_idle_without_mutation(monkeypatch, tmp_path, capsys):
    db = tmp_path / "content.db"
    _draft_db(db)
    before = db.read_bytes()
    module = _load_module(monkeypatch, db)
    monkeypatch.setattr(module, "check_publisher_readiness", lambda: None)
    assert module.main(["--preflight"]) == 0
    assert db.read_bytes() == before
    assert "ready-idle" in capsys.readouterr().out


def test_dry_run_never_claims_or_queues(monkeypatch, tmp_path, capsys):
    db = tmp_path / "content.db"
    _draft_db(db, approved=True)
    before = db.read_bytes()
    module = _load_module(monkeypatch, db)
    monkeypatch.setattr(module, "check_publisher_readiness", lambda: None)
    monkeypatch.setattr(module, "claim_for_enqueue", lambda *_: pytest.fail("claim mutated state"))
    monkeypatch.setattr(module, "queue_post", lambda **_: pytest.fail("published"))
    assert module.main(["--dry-run"]) == 0
    assert db.read_bytes() == before
    assert "dry-run" in capsys.readouterr().out


def test_publisher_unavailable_is_distinct(monkeypatch, tmp_path, capsys):
    db = tmp_path / "content.db"
    _draft_db(db)
    module = _load_module(monkeypatch, db)
    monkeypatch.setattr(module, "check_publisher_readiness", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    assert module.main([]) == 2
    assert "publisher-unavailable" in capsys.readouterr().err