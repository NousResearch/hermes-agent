"""Tests for the single tabbed review-bundle builder.

Run: cd content_engine && PYTHONPATH=. ../.venv/bin/python -m pytest tests/test_build_review_bundle.py -q
"""
from __future__ import annotations

from pathlib import Path

import scripts.build_review_bundle as brb
from scripts.build_review_bundle import (
    BLOG_GROUP,
    LINKEDIN_GROUP,
    X_GROUP,
    _chunk_by_size,
    _article_items,
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
    blog = [_item("a", "Alpha", BLOG_GROUP), _item("b", "Beta", BLOG_GROUP)]
    x = [_item("c", "Gamma", X_GROUP)]
    doc = render([(BLOG_GROUP, blog), (X_GROUP, x), (LINKEDIN_GROUP, [])], "T")

    # every article title appears (sidebar + pane)
    for t in ("Alpha", "Beta", "Gamma"):
        assert t in doc
    # summary counts reflect what was passed in
    assert "2 blog posts + 1 X/Twitter articles + 0 LinkedIn articles" in doc
    # switcher wiring present
    assert "function show" in doc
    assert 'data-pane="0"' in doc  # SUMMARY link
    # one pane per article + the summary pane = 4 panes
    assert doc.count('class="pane') == 4
    # LinkedIn placeholder group rendered but muted
    assert "LINKEDIN" in doc and "group muted" in doc


def test_article_items_render_x_and_linkedin_under_real_platform_groups(tmp_path):
    x = tmp_path / "x"
    linkedin = tmp_path / "linkedin"
    for bundle, title in ((x, "X Article"), (linkedin, "LinkedIn Article")):
        bundle.mkdir()
        (bundle / "article.md").write_text(f"# {title}\n\nBody")
    rows = [
        {"article_id": "x1", "bundle_path": str(x), "brand": "sahil_twitter", "platform": "twitter"},
        {"article_id": "li1", "bundle_path": str(linkedin), "brand": "sahil_linkedin", "platform": "linkedin"},
    ]

    groups, diagnostics = _article_items(rows)

    assert diagnostics == []
    assert [item["slug"] for item in groups[X_GROUP]] == ["x1"]
    assert [item["slug"] for item in groups[LINKEDIN_GROUP]] == ["li1"]
    doc = render([(X_GROUP, groups[X_GROUP]), (LINKEDIN_GROUP, groups[LINKEDIN_GROUP])], "T")
    assert "1 X/Twitter articles + 1 LinkedIn articles" in doc
    assert "X Article" in doc and "LinkedIn Article" in doc


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


def test_article_items_report_missing_bundle_instead_of_silently_dropping(tmp_path):
    rows = [{
        "article_id": "missing-li", "bundle_path": str(tmp_path / "gone"),
        "brand": "sahil_linkedin", "platform": "linkedin",
    }]

    groups, diagnostics = _article_items(rows)

    assert groups[LINKEDIN_GROUP] == []
    assert diagnostics == [f"missing bundle for article missing-li: {tmp_path / 'gone'}"]


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
    monkeypatch.setattr(brb, "_blog_items", lambda *a, **k: [_item("a", "Alpha", BLOG_GROUP, pane_bytes=20_000)])
    monkeypatch.setattr(brb, "_pending_article_items", lambda *a, **k: ({
        X_GROUP: [_item(f"x{i}", f"X{i}", X_GROUP, pane_bytes=20_000) for i in range(6)],
        LINKEDIN_GROUP: [],
    }, []))
    out = build()

    # blog fits one file; X (120KB) must chunk into several
    assert len(out) >= 3
    assert all(Path(p).exists() for p in out)
    x_files = [p for p in out if "-x-" in Path(p).name]
    assert len(x_files) >= 2
    # every emitted file is under the cap (nothing that Discord would drop)
    assert all(Path(p).stat().st_size <= brb.SIZE_CAP for p in out)


def test_build_empty_returns_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(brb, "PREVIEW_DIR", tmp_path)
    monkeypatch.setattr(brb, "_blog_items", lambda *a, **k: [])
    monkeypatch.setattr(brb, "_pending_article_items", lambda *a, **k: ({X_GROUP: [], LINKEDIN_GROUP: []}, []))
    assert build() == []


def test_main_prints_silent_when_nothing_pending(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(brb, "PREVIEW_DIR", tmp_path)
    monkeypatch.setattr(brb, "TRACKER", tmp_path / "missing.jsonl")
    monkeypatch.setattr(brb, "X_BUNDLES", tmp_path / "missing")
    monkeypatch.setattr(brb.database, "list_article_approvals", lambda status: [])
    monkeypatch.setattr(brb, "_blog_items", lambda *a, **k: [])
    monkeypatch.setattr(brb, "_pending_article_items", lambda *a, **k: ({X_GROUP: [], LINKEDIN_GROUP: []}, []))
    main()
    assert capsys.readouterr().out.strip() == "[SILENT]"


def test_main_prints_summary_and_media_lines(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(brb, "PREVIEW_DIR", tmp_path)
    monkeypatch.setattr(brb, "TRACKER", tmp_path / "missing.jsonl")
    monkeypatch.setattr(brb, "X_BUNDLES", tmp_path / "missing")
    monkeypatch.setattr(brb.database, "list_article_approvals", lambda status: [])
    monkeypatch.setattr(brb, "_blog_items", lambda *a, **k: [_item("a", "Alpha", BLOG_GROUP)])
    monkeypatch.setattr(brb, "_pending_article_items", lambda *a, **k: ({X_GROUP: [], LINKEDIN_GROUP: []}, []))
    main()
    out = capsys.readouterr().out
    assert "awaiting review" in out
    media = [ln for ln in out.splitlines() if ln.startswith("MEDIA:")]
    assert len(media) == 1
    assert media[0].endswith(".html")
