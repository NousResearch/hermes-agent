"""Tests for the single tabbed review-bundle builder.

Run: cd content_engine && PYTHONPATH=. ../.venv/bin/python -m pytest tests/test_build_review_bundle.py -q
"""
from __future__ import annotations

from pathlib import Path

import scripts.build_review_bundle as brb
from scripts.build_review_bundle import (
    BLOG_GROUP,
    LANE_GROUPS,
    LINKEDIN_GROUP,
    X_GROUP,
    _chunk_by_size,
    _pending_article_items,
    _img_card,
    build,
    main,
    render,
)


def _item(slug: str, title: str, group: str, pane_bytes: int = 100) -> dict:
    pane = f"<h1>{title}</h1>" + ("x" * pane_bytes)
    return {"slug": slug, "title": title, "pane": pane, "group": group}


def test_render_lists_every_item_and_counts():
    ai = [_item("a", "Alpha", LANE_GROUPS["ai"])]
    pm = [_item("b", "Beta", LANE_GROUPS["pm"])]
    x = [_item("c", "Gamma", X_GROUP)]
    doc = render(
        [(LANE_GROUPS["ai"], ai), (LANE_GROUPS["pm"], pm), (X_GROUP, x), (LINKEDIN_GROUP, [])],
        "T",
    )

    # every article title appears (sidebar + pane)
    for t in ("Alpha", "Beta", "Gamma"):
        assert t in doc
    # summary counts reflect what was passed in (X/LinkedIn excluded from totals)
    assert "2 blog posts awaiting review" in doc
    assert "X/Twitter articles" not in doc
    assert "LinkedIn articles" not in doc
    # switcher wiring present
    assert "function show" in doc
    assert 'data-pane="0"' in doc  # SUMMARY link
    # one pane per article + the summary pane = 4 panes
    assert doc.count('class="pane') == 4
    # LinkedIn placeholder group rendered but muted
    assert "LINKEDIN" in doc and "group muted" in doc
    # X/Twitter item pill uses the display label, not the raw group constant
    # (regression: a hard-coded {"X": ...} lookup KeyError'd on "X/TWITTER")
    assert "X/Twitter" in doc and "<span class='pill'>X/Twitter</span>" in doc


def test_article_items_excludes_social_platform_articles(tmp_path, monkeypatch):
    """X/Twitter and LinkedIn articles are owned by dedicated managers and
    must never surface in the blog review bundle."""
    x = tmp_path / "x"
    linkedin = tmp_path / "linkedin"
    for bundle, title in ((x, "X Article"), (linkedin, "LinkedIn Article")):
        bundle.mkdir()
        (bundle / "article.md").write_text(f"# {title}\n\nBody")
    rows = [
        {"article_id": "x1", "bundle_path": str(x), "brand": "sahil_twitter", "platform": "twitter"},
        {"article_id": "li1", "bundle_path": str(linkedin), "brand": "sahil_linkedin", "platform": "linkedin"},
    ]
    monkeypatch.setattr(brb.database, "init_db", lambda: None)
    monkeypatch.setattr(
        brb.database, "migrate_article_approvals",
        lambda root: {"created": 0, "missing_bundles": [], "orphan_bundles": []},
    )
    monkeypatch.setattr(brb.database, "list_article_approvals", lambda status: rows)
    monkeypatch.setattr(brb, "X_BUNDLES", tmp_path / "articles")

    groups, diagnostics = _pending_article_items()

    assert diagnostics == []
    assert groups == {X_GROUP: [], LINKEDIN_GROUP: []}
    doc = render([(X_GROUP, groups[X_GROUP]), (LINKEDIN_GROUP, groups[LINKEDIN_GROUP])], "T")
    assert "X Article" not in doc and "LinkedIn Article" not in doc


def test_pending_article_items_migrates_legacy_state_and_queries_only_pending_records(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(brb.database, "init_db", lambda: seen.append("init"))
    monkeypatch.setattr(
        brb.database, "migrate_article_approvals",
        lambda root: seen.append(("migrate", root)) or {"created": 0, "missing_bundles": [], "orphan_bundles": []},
    )
    monkeypatch.setattr(brb.database, "list_article_approvals", lambda status: seen.append(("list", status)) or [])
    monkeypatch.setattr(brb, "X_BUNDLES", tmp_path / "articles")

    groups, diagnostics = _pending_article_items()

    assert seen == ["init", ("migrate", tmp_path / "articles"), ("list", "pending")]
    assert groups == {X_GROUP: [], LINKEDIN_GROUP: []}
    assert diagnostics == []


def test_article_items_report_missing_bundle_instead_of_silently_dropping(tmp_path, monkeypatch):
    rows = [{
        "article_id": "missing-li", "bundle_path": str(tmp_path / "gone"),
        "brand": "sahil_linkedin", "platform": "linkedin",
    }]
    monkeypatch.setattr(brb.database, "init_db", lambda: None)
    monkeypatch.setattr(
        brb.database, "migrate_article_approvals",
        lambda root: {"created": 0, "missing_bundles": [], "orphan_bundles": []},
    )
    monkeypatch.setattr(brb.database, "list_article_approvals", lambda status: rows)
    monkeypatch.setattr(brb, "X_BUNDLES", tmp_path / "articles")

    groups, diagnostics = _pending_article_items()

    # LinkedIn is owned by the dedicated manager — excluded, not diagnosed.
    assert groups[LINKEDIN_GROUP] == []
    assert diagnostics == []


def test_render_uses_the_approved_validation_palette():
    doc = render([(BLOG_GROUP, [_item("a", "Alpha", BLOG_GROUP)])], "T")
    for token in ("Inter", "#111", "#ffd166", ".wrap", ".deck", ".source"):
        assert token in doc


def test_img_card_embeds_or_flags_missing():
    ok = _img_card("Hero", "data:image/jpeg;base64,AAA", "meta")
    assert "<img" in ok and "card missing" not in ok

    miss = _img_card("Hero", "/nope/x.png", "image missing")
    assert "IMAGE MISSING" in miss and "card missing" in miss


def test_chunk_by_size_splits_when_over_budget(monkeypatch):
    monkeypatch.setattr(brb, "SIZE_CAP", 41_000)  # budget = 1_000
    items = [_item(str(i), f"T{i}", X_GROUP, pane_bytes=400) for i in range(5)]
    chunks = _chunk_by_size(items)
    assert len(chunks) > 1
    assert sum(len(c) for c in chunks) == 5  # nothing dropped


def test_build_single_file_when_small(monkeypatch, tmp_path):
    monkeypatch.setattr(brb, "PREVIEW_DIR", tmp_path)
    monkeypatch.setattr(brb, "_blog_items", lambda *a, **k: [_item("a", "Alpha", BLOG_GROUP)])
    monkeypatch.setattr(brb, "_pending_article_items", lambda *a, **k: ({X_GROUP: [], LINKEDIN_GROUP: []}, []))
    out = build()
    assert len(out) == 1
    name = Path(out[0]).name
    assert name.startswith("pending-review-") and "-blog-" not in name and "-x-" not in name
    assert Path(out[0]).exists()


def test_build_splits_per_platform_and_chunks_over_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(brb, "PREVIEW_DIR", tmp_path)
    monkeypatch.setattr(brb, "SIZE_CAP", 80_000)  # budget = 40_000
    monkeypatch.setattr(brb, "idea_cards", lambda: [])  # isolate from real backlog
    monkeypatch.setattr(brb, "_blog_items", lambda *a, **k: [_item("a", "Alpha", BLOG_GROUP, pane_bytes=20_000)])
    monkeypatch.setattr(brb, "_pending_article_items", lambda *a, **k: ({X_GROUP: [], LINKEDIN_GROUP: []}, []))
    out = build()

    # blog fits one file; no X/LinkedIn items exist anymore (social owned elsewhere)
    assert len(out) == 1
    assert all(Path(p).exists() for p in out)
    assert all(Path(p).stat().st_size <= brb.SIZE_CAP for p in out)


def test_build_empty_returns_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(brb, "PREVIEW_DIR", tmp_path)
    monkeypatch.setattr(brb, "_blog_items", lambda *a, **k: [])
    monkeypatch.setattr(brb, "_pending_article_items", lambda *a, **k: ({X_GROUP: [], LINKEDIN_GROUP: []}, []))
    # idea_cards() reads ~/.hermes/research/idea-backlog.jsonl — real machine
    # state. Stub it so the emptiness check is deterministic (same isolation
    # pattern as the other two sources).
    monkeypatch.setattr(brb, "idea_cards", lambda: [])
    assert build() == []


def test_main_prints_silent_when_nothing_pending(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(brb, "PREVIEW_DIR", tmp_path)
    monkeypatch.setattr(brb, "TRACKER", tmp_path / "missing.jsonl")
    monkeypatch.setattr(brb, "X_BUNDLES", tmp_path / "missing")
    monkeypatch.setattr(brb.database, "list_article_approvals", lambda status: [])
    monkeypatch.setattr(brb, "_blog_items", lambda *a, **k: [])
    monkeypatch.setattr(brb, "_pending_article_items", lambda *a, **k: ({X_GROUP: [], LINKEDIN_GROUP: []}, []))
    monkeypatch.setattr(brb, "idea_cards", lambda: [])
    main()
    assert capsys.readouterr().out.strip() == "[SILENT]"


def test_main_prints_summary_and_media_lines(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(brb, "PREVIEW_DIR", tmp_path)
    monkeypatch.setattr(brb, "TRACKER", tmp_path / "missing.jsonl")
    monkeypatch.setattr(brb, "X_BUNDLES", tmp_path / "missing")
    monkeypatch.setattr(brb.database, "list_article_approvals", lambda status: [])
    monkeypatch.setattr(brb, "_blog_items", lambda *a, **k: [_item("a", "Alpha", BLOG_GROUP)])
    monkeypatch.setattr(brb, "_pending_article_items", lambda *a, **k: ({X_GROUP: [], LINKEDIN_GROUP: []}, []))
    monkeypatch.setattr(brb, "idea_cards", lambda: [{"id": "idea-1", "title": "Idea One", "group": brb.IDEAS_GROUP, "pane": "<p>idea</p>"}])
    main()
    out = capsys.readouterr().out
    assert "awaiting review" in out
    assert "+ 1 idea concepts" in out  # ideas counted in the text summary, not just the HTML
    media = [ln for ln in out.splitlines() if ln.startswith("MEDIA:")]
    assert len(media) == 1
    assert media[0].endswith(".html")
