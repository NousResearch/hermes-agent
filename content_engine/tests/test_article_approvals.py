"""Durable long-form article approval state and migration tests."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import database as db


def _use_db(monkeypatch, tmp_path: Path) -> Path:
    path = tmp_path / "content.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    return path


def test_register_and_list_article_approval_has_durable_contract(monkeypatch, tmp_path):
    _use_db(monkeypatch, tmp_path)
    bundle = tmp_path / "x-bundle"
    bundle.mkdir()

    db.register_article_approval(
        article_id="article-x-1", bundle_path=bundle, brand="sahil_twitter",
        platform="twitter", created_at="2026-07-31T08:00:00+00:00",
    )

    assert db.list_article_approvals(status="pending") == [{
        "article_id": "article-x-1", "bundle_path": str(bundle),
        "brand": "sahil_twitter", "platform": "twitter", "status": "pending",
        "created_at": "2026-07-31T08:00:00+00:00", "approved_at": None,
    }]


def test_register_article_approval_is_idempotent_and_preserves_terminal_status(monkeypatch, tmp_path):
    path = _use_db(monkeypatch, tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    db.register_article_approval(article_id="a1", bundle_path=bundle, brand="sahil_linkedin", platform="linkedin")
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE article_approvals SET status='approved', approved_at='2026-07-31T09:00:00+00:00' WHERE article_id='a1'")
        conn.commit()

    db.register_article_approval(article_id="a1", bundle_path=bundle, brand="sahil_linkedin", platform="linkedin")

    assert db.list_article_approvals(status="pending") == []
    row = db.list_article_approvals(status="approved")[0]
    assert row["approved_at"] == "2026-07-31T09:00:00+00:00"


def test_existing_draft_approval_handlers_update_longform_state(monkeypatch, tmp_path):
    _use_db(monkeypatch, tmp_path)
    for article_id in ("approve-me", "reject-me"):
        db.insert_draft(article_id, "sahil_twitter", "twitter", "p", "t", "T", "Body", content_type="article")
        db.register_article_approval(
            article_id=article_id, bundle_path=tmp_path / article_id,
            brand="sahil_twitter", platform="twitter",
        )

    db.approve_draft("approve-me")
    db.reject_draft("reject-me")

    approved = db.list_article_approvals(status="approved")
    rejected = db.list_article_approvals(status="rejected")
    assert [row["article_id"] for row in approved] == ["approve-me"]
    assert approved[0]["approved_at"] is not None
    assert [row["article_id"] for row in rejected] == ["reject-me"]
    assert rejected[0]["approved_at"] is None


def test_migrate_existing_article_drafts_matches_x_and_linkedin_idempotently(monkeypatch, tmp_path):
    _use_db(monkeypatch, tmp_path)
    bundles = tmp_path / "articles"
    x_bundle = bundles / "2026-07-30-x-title"
    li_bundle = bundles / "2026-07-30-linkedin-title"
    for bundle, text in ((x_bundle, "# X title\n\nX body"), (li_bundle, "# LinkedIn title\n\nLinkedIn body")):
        bundle.mkdir(parents=True)
        (bundle / "article.md").write_text(text)
    db.insert_draft("x1", "sahil_twitter", "twitter", "p", "t", "X title", "# X title\n\nX body", content_type="article")
    db.insert_draft("li1", "sahil_linkedin", "linkedin", "p", "t", "LinkedIn title", "# LinkedIn title\n\nLinkedIn body", content_type="article")

    first = db.migrate_article_approvals(bundles)
    second = db.migrate_article_approvals(bundles)

    assert first == {"created": 2, "missing_bundles": [], "orphan_bundles": []}
    assert second == {"created": 0, "missing_bundles": [], "orphan_bundles": []}
    rows = db.list_article_approvals(status="pending")
    assert {(r["article_id"], r["platform"], r["bundle_path"]) for r in rows} == {
        ("x1", "twitter", str(x_bundle)), ("li1", "linkedin", str(li_bundle)),
    }


def test_migration_reports_db_filesystem_disagreement(monkeypatch, tmp_path):
    _use_db(monkeypatch, tmp_path)
    bundles = tmp_path / "articles"
    orphan = bundles / "2026-07-30-orphan"
    orphan.mkdir(parents=True)
    (orphan / "article.md").write_text("# Orphan\n\nNo DB row")
    db.insert_draft("missing", "sahil_twitter", "twitter", "p", "t", "Missing", "# Missing\n\nNo bundle", content_type="article")

    result = db.migrate_article_approvals(bundles)

    assert result["created"] == 0
    assert result["missing_bundles"] == ["missing"]
    assert result["orphan_bundles"] == [str(orphan)]
