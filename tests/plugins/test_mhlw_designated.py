"""Tests for MHLW designated-substances monitor (no network)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "scrapling-feeds"


def _load_mhlw():
    pkg = "scrapling_feeds_mhlw_test"
    if pkg not in sys.modules:
        package = types.ModuleType(pkg)
        package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
        sys.modules[pkg] = package
        for stem in ("fetcher", "mhlw_designated"):
            mod_name = f"{pkg}.{stem}"
            spec = importlib.util.spec_from_file_location(
                mod_name, PLUGIN_DIR / f"{stem}.py"
            )
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = pkg
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
    return sys.modules[f"{pkg}.mhlw_designated"]


SAMPLE_COMMITTEE_HTML = """
<table>
<tr><th>回数</th><th>開催日</th><th>開催案内</th></tr>
<tr><td>令和７年度 第６回</td><td>2026年3月3日</td><td><a href="/stf/newpage_99999.html">開催</a></td></tr>
</table>
"""

SAMPLE_NOTICE_HTML = """
<title>薬事審議会　指定薬物部会を開催します</title>
<h1>薬事審議会　指定薬物部会を開催します</h1>
<h2>１　開催日時</h2>
<p>令和８年６月16日（火） 10：00 ～ 12：00</p>
"""


def test_parse_committee_page_finds_notice_link():
    mhlw = _load_mhlw()
    rows = mhlw.parse_committee_page(SAMPLE_COMMITTEE_HTML)
    urls = [r.get("url") for r in rows]
    assert any("newpage_99999" in (u or "") for u in urls)
    assert any(r.get("kind") == "meeting_schedule" for r in rows)


def test_parse_meeting_notice_extracts_datetime():
    mhlw = _load_mhlw()
    detail = mhlw.parse_meeting_notice_page(
        SAMPLE_NOTICE_HTML,
        url="https://www.mhlw.go.jp/stf/newpage_73769.html",
    )
    assert "指定薬物部会" in detail.get("title", "")
    assert "16日" in detail.get("meeting_when", "")


def test_state_baseline_suppresses_new_items(tmp_path, monkeypatch):
    mhlw = _load_mhlw()
    state_file = tmp_path / "mhlw_designated_state.json"
    monkeypatch.setattr(mhlw, "_state_path", lambda: state_file)
    monkeypatch.setattr(
        mhlw,
        "fetcher",
        type(
            "F",
            (),
            {
                "fetch_url": staticmethod(
                    lambda url: {
                        "success": True,
                        "body": SAMPLE_COMMITTEE_HTML,
                    }
                )
            },
        )(),
    )
    monkeypatch.setattr(mhlw, "scan_enforcement_announcements", lambda **_: [])

    first = mhlw.check_mhlw_designated(record_baseline=True, enrich_notices=False)
    assert first.get("new_count", 0) >= 1

    second = mhlw.check_mhlw_designated(enrich_notices=False)
    assert second.get("new_count") == 0
