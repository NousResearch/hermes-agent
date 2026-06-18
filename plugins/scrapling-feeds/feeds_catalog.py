"""Official government feed catalog — PRIMARY sources only.

URLs verified against publisher pages (2026-06). MOFA RSS may return 403 without
Scrapling/curl_cffi; disabled by default.
"""

from __future__ import annotations

from typing import Any

# Official catalog page references for traceability
CATALOG_DOCS = {
    "mod_rss": "https://www.mod.go.jp/j/rss/index.html",
    "cisa_subscribe": "https://www.cisa.gov/about/contact-us/subscribe-updates-cisa",
    "nisc_rss": "https://www.nisc.go.jp/rss/index.html",
}

GOV_FEEDS: dict[str, dict[str, Any]] = {
    "mod_updates": {
        "name": "防衛省 — 更新情報",
        "url": "https://www.mod.go.jp/j/rss/update.xml",
        "country": "JP",
        "agency": "防衛省",
        "category": "defense",
        "source_tier": "PRIMARY",
        "enabled_by_default": True,
        "catalog_doc": CATALOG_DOCS["mod_rss"],
    },
    "mod_press": {
        "name": "防衛省 — お知らせ（報道資料）",
        "url": "https://www.mod.go.jp/j/rss/news.xml",
        "country": "JP",
        "agency": "防衛省",
        "category": "defense",
        "source_tier": "PRIMARY",
        "enabled_by_default": True,
        "catalog_doc": CATALOG_DOCS["mod_rss"],
    },
    "nisc_index": {
        "name": "NISC — お知らせ（レガシー）",
        "url": "https://www.nisc.go.jp/rss/index.xml",
        "country": "JP",
        "agency": "内閣サイバーセキュリティセンター",
        "category": "cyber",
        "source_tier": "PRIMARY",
        "enabled_by_default": False,
        "catalog_doc": CATALOG_DOCS["nisc_rss"],
        "note": (
            "nisc.go.jp RSS は cyber.go.jp へリダイレクトされ XML 非提供（2026-06 確認）。"
            " 当面無効。"
        ),
    },
    "nisc_security": {
        "name": "NISC — セキュリティ情報（レガシー）",
        "url": "https://www.nisc.go.jp/rss/security.xml",
        "country": "JP",
        "agency": "内閣サイバーセキュリティセンター",
        "category": "cyber",
        "source_tier": "PRIMARY",
        "enabled_by_default": False,
        "catalog_doc": CATALOG_DOCS["nisc_rss"],
        "note": "同上 — cyber.go.jp 移行で RSS 終了の可能性。",
    },
    "digital_agency_news": {
        "name": "デジタル庁 — ニュース",
        "url": "https://digital.go.jp/rss/news.xml",
        "country": "JP",
        "agency": "デジタル庁",
        "category": "policy",
        "source_tier": "PRIMARY",
        "enabled_by_default": True,
        "catalog_doc": "https://www.digital.go.jp/",
    },
    "cisa_advisories_all": {
        "name": "CISA — Cybersecurity Advisories (all)",
        "url": "https://www.cisa.gov/cybersecurity-advisories/all.xml",
        "country": "US",
        "agency": "CISA",
        "category": "cyber",
        "source_tier": "PRIMARY",
        "enabled_by_default": True,
        "catalog_doc": CATALOG_DOCS["cisa_subscribe"],
        "note": (
            "CISA changed alert distribution in May 2025; feed may not list every "
            "release — cross-check cisa.gov for urgent items."
        ),
    },
    "cisa_ics_advisories": {
        "name": "CISA — ICS Advisories",
        "url": "https://www.cisa.gov/uscert/ics/advisories/advisories.xml",
        "country": "US",
        "agency": "CISA",
        "category": "ics",
        "source_tier": "PRIMARY",
        "enabled_by_default": True,
        "catalog_doc": CATALOG_DOCS["cisa_subscribe"],
    },
    "cisa_news": {
        "name": "CISA — News",
        "url": "https://www.cisa.gov/news.xml",
        "country": "US",
        "agency": "CISA",
        "category": "news",
        "source_tier": "PRIMARY",
        "enabled_by_default": True,
        "catalog_doc": CATALOG_DOCS["cisa_subscribe"],
    },
    "mofa_news": {
        "name": "外務省 — ニュース（RSS）",
        "url": "https://www.mofa.go.jp/rss/news.xml",
        "country": "JP",
        "agency": "外務省",
        "category": "foreign_policy",
        "source_tier": "PRIMARY",
        "enabled_by_default": False,
        "catalog_doc": "https://www.mofa.go.jp/",
        "note": "May require Scrapling Fetcher (site blocks plain urllib).",
    },
}


def list_feed_ids(*, enabled_only: bool = True) -> list[str]:
    ids: list[str] = []
    for fid, meta in GOV_FEEDS.items():
        if enabled_only and not meta.get("enabled_by_default", True):
            continue
        ids.append(fid)
    return ids


def get_feed(feed_id: str) -> dict[str, Any] | None:
    row = GOV_FEEDS.get(feed_id)
    return dict(row) if row else None
