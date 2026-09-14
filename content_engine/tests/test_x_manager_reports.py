"""Contract tests for X manager cron report scripts."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "content_engine"
CONTENT_ENGINE = Path(__file__).resolve().parents[1]


def _load(name: str):
    for path in (str(SCRIPTS), str(CONTENT_ENGINE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def artifact():
    xm = _load("x_manager_report").__dict__.get("xm")
    if xm is None:
        import x_manager as xm
    import time
    from datetime import datetime, timezone
    stamp_ms = int((time.time()-3600)*1000)
    sid = str((stamp_ms-1288834974657)<<22)
    source_url = f"https://x.com/example/status/{sid}"
    return xm.XArtifact(
        id="xm_quot_test",
        lane=xm.LANE_QUOTE_SCAN,
        brand="sahil_twitter",
        body="A concise recommendation.",
        pack=xm.ArgumentPack(
            claim="A claim",
            evidence="A source",
            mechanism="A mechanism",
            position="A position",
            context={
                "author": "example",
                "source_text": "The original post.",
                "source_url": source_url,
                "sources": [{"id": sid, "url": source_url, "author": "example",
                             "origin": "for_you",
                             "created_at": datetime.fromtimestamp(stamp_ms/1000,timezone.utc).isoformat()}],
            },
        ),
    )


def test_report_contains_original_and_recommendation(monkeypatch, tmp_path, artifact):
    report = _load("x_manager_report")
    monkeypatch.setattr(report, "REPORT_DIR", tmp_path)

    path = report.render_report(
        [artifact], lane="quote-scout", title="Quote-post recommendations"
    )
    text = path.read_text()

    assert "The original post." in text
    assert "A concise recommendation." in text
    assert "Open original" in text
    assert "Why this angle" in text
    assert text.count('<article class="card"') == 1


def test_report_mobile_wrap(monkeypatch,tmp_path,artifact):
    sync_playwright = pytest.importorskip('playwright.sync_api').sync_playwright
    report=_load('x_manager_report')
    monkeypatch.setattr(report,'REPORT_DIR',tmp_path)
    artifact.pack.context['context_status']={'state':'unknown','complete':False,'reason':'x'*300}
    path=report.render_report([artifact],lane='quote-scout',title='Review')
    with sync_playwright() as runner:
        browser=runner.chromium.launch(headless=True)
        page=browser.new_page()
        page.goto(path.as_uri())
        page.locator('details').evaluate_all('items => items.forEach(item => item.open = true)')
        for width in (1440,390):
            page.set_viewport_size({'width':width,'height':1000})
            assert page.locator('article').count()==1
            assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        browser.close()


def test_report_escapes_source_content(monkeypatch, tmp_path, artifact):
    report = _load("x_manager_report")
    monkeypatch.setattr(report, "REPORT_DIR", tmp_path)
    artifact.pack.context["source_text"] = '<script>alert("x")</script>'

    text = report.render_report(
        [artifact], lane="quote-scout", title="Review"
    ).read_text()

    assert "<script>" not in text
    assert "&lt;script&gt;" in text


def test_quote_scout_stdout_is_summary_plus_media(monkeypatch, capsys, artifact, tmp_path):
    scout = _load("x_quote_scout")
    report = tmp_path / "report.html"
    report.write_text("<html></html>")

    monkeypatch.setattr(scout, "_load_env", lambda: None)
    monkeypatch.setattr(scout, "_collect", lambda: [{"text": "x" * 80}])
    monkeypatch.setattr(scout, "_candidate_artifacts", lambda rows: ([artifact] * 5, [], []))
    monkeypatch.setattr(scout, "_merge_standalone_seeds", lambda seeds: None)
    monkeypatch.setattr(scout, "_already_reported", lambda items: False)
    monkeypatch.setattr(scout, "_record_reported", lambda items: None)
    monkeypatch.setattr(scout.xm, "stage_for_approval", lambda item: item.id)
    monkeypatch.setattr(scout, "render_report", lambda *args, **kwargs: report)

    scout.main()
    lines = capsys.readouterr().out.strip().splitlines()

    assert lines == [
        "X Manager · 5 post recommendations",
        "Original posts and recommended drafts are in the attached review.",
        f"MEDIA:{report}",
    ]


def test_morning_article_stdout_is_summary_plus_media(monkeypatch, capsys, artifact, tmp_path):
    morning = _load("x_morning_article")
    artifact.lane = morning.xm.LANE_ARTICLE
    report = tmp_path / "article.html"
    report.write_text("<html></html>")

    monkeypatch.setattr(morning, "_load_env", lambda: None)
    monkeypatch.setattr(morning, "_collect_signals", lambda: [{"summary": "x" * 80, "sources": artifact.pack.context["sources"]}])
    monkeypatch.setattr(morning, "_build_artifact", lambda signal: artifact)
    monkeypatch.setattr(morning, "_already_reported", lambda items: False)
    monkeypatch.setattr(morning, "_record_reported", lambda items: None)
    monkeypatch.setattr(morning.xm, "stage_for_approval", lambda item: item.id)
    monkeypatch.setattr(morning, "render_report", lambda *args, **kwargs: report)

    morning.main()
    lines = capsys.readouterr().out.strip().splitlines()

    assert lines == [
        "X Manager · 1 morning Article recommendation",
        "Source evidence and full drafts are in the attached review.",
        f"MEDIA:{report}",
    ]
