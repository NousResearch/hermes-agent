"""e-Gov Law API — automatic primary law citations for Japan security PDB."""

from __future__ import annotations

import asyncio
from typing import Any

# 安全保障 PDB で常に照会する日本法一次資料（e-Gov Law API v2）
JAPAN_SECURITY_LAW_CATALOG: list[dict[str, Any]] = [
    {"search": "日本国憲法", "article": "9", "label": "憲法9条（平和主義）"},
    {"search": "自衛隊法", "article": "3", "label": "自衛隊の任務"},
    {"search": "武力攻撃事態等及び存立危機事態における我が国の平和と独立並びに国及び国民の安全の確保に関する法律", "article": "2", "label": "重要影響事態法等の定義"},
    {"search": "国家安全保障戦略", "article": None, "label": "国家安全保障戦略（法令検索）"},
    {"search": "サイバーセキュリティ基本法", "article": "1", "label": "サイバーセキュリティ基本法（目的）"},
]

EGOV_LAW_PORTAL = "https://laws.e-gov.go.jp"
SNIPPET_MAX_CHARS = 480


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Nested loop (e.g. gateway) — new thread would be heavy; use run_until_complete on new loop
    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


def _law_portal_url(law_id: str, article: str | None = None) -> str:
    base = f"{EGOV_LAW_PORTAL}/law/{law_id}"
    if article:
        return f"{base}#Mp-At_{article}"
    return base


async def _fetch_one_entry(
    entry: dict[str, Any],
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    from egov_law_mcp.api import EGovAPIClient, EGovAPIError
    from egov_law_mcp.tools.article import get_law_article
    from egov_law_mcp.tools.search import search_laws

    if client is None:
        client = EGovAPIClient()

    label = entry.get("label") or entry.get("search") or "法令"
    keyword = str(entry.get("search") or "")
    article = entry.get("article")

    try:
        search_result = await search_laws(keyword, limit=5, client=client)
    except Exception as exc:
        return {
            "success": False,
            "label": label,
            "search": keyword,
            "error": str(exc)[:300],
            "source_tier": "PRIMARY",
            "source_api": "e-Gov Law API v2",
        }

    laws = []
    if hasattr(search_result, "laws"):
        laws = search_result.laws
    elif isinstance(search_result, dict):
        laws = search_result.get("laws") or []

    if not laws:
        return {
            "success": False,
            "label": label,
            "search": keyword,
            "error": "法令未検出",
            "source_tier": "PRIMARY",
            "source_api": "e-Gov Law API v2",
        }

    law = laws[0]
    law_id = getattr(law, "law_id", None) or (law.get("law_id") if isinstance(law, dict) else "")
    law_name = getattr(law, "law_name", None) or (law.get("law_name") if isinstance(law, dict) else keyword)

    if not article:
        return {
            "success": True,
            "label": label,
            "law_id": law_id,
            "law_name": law_name,
            "article_number": None,
            "snippet": f"法令を特定: {law_name}（条文未指定）",
            "citation": f"[出典: e-Gov Law API v2 — {law_name}] {_law_portal_url(law_id)}",
            "source_url": _law_portal_url(law_id),
            "source_tier": "PRIMARY",
            "source_api": "e-Gov Law API v2",
        }

    try:
        art = await get_law_article(str(law_id), str(article), client=client)
    except Exception as exc:
        return {
            "success": False,
            "label": label,
            "law_id": law_id,
            "law_name": law_name,
            "article_number": str(article),
            "error": str(exc)[:300],
            "source_url": _law_portal_url(law_id, str(article)),
            "source_tier": "PRIMARY",
            "source_api": "e-Gov Law API v2",
        }

    content = (getattr(art, "content", None) or "")[:SNIPPET_MAX_CHARS]
    art_num = getattr(art, "article_number", None) or article
    return {
        "success": True,
        "label": label,
        "law_id": law_id,
        "law_name": getattr(art, "law_name", None) or law_name,
        "article_number": str(art_num),
        "snippet": content,
        "citation": (
            f"[出典: e-Gov Law API v2 — {law_name} 第{art_num}条] "
            f"{_law_portal_url(law_id, str(art_num))}"
        ),
        "source_url": _law_portal_url(law_id, str(art_num)),
        "source_tier": "PRIMARY",
        "source_api": "e-Gov Law API v2",
    }


async def fetch_security_law_citations_async(
    *,
    catalog: list[dict[str, Any]] | None = None,
    max_entries: int = 5,
) -> dict[str, Any]:
    entries = (catalog or JAPAN_SECURITY_LAW_CATALOG)[: max(1, min(max_entries, 8))]
    results: list[dict[str, Any]] = []
    try:
        from egov_law_mcp.api import EGovAPIClient

        client = EGovAPIClient()
    except ImportError as exc:
        return {
            "success": False,
            "skipped": True,
            "reason": f"egov-law-mcp not installed: {exc}",
            "citations": [],
        }

    for entry in entries:
        results.append(await _fetch_one_entry(entry, client=client))

    ok = sum(1 for r in results if r.get("success"))
    return {
        "success": ok > 0,
        "fetched": ok,
        "total": len(results),
        "citations": results,
        "source_api": "https://laws.e-gov.go.jp/api/2",
    }


def fetch_security_law_citations(
    *,
    catalog: list[dict[str, Any]] | None = None,
    max_entries: int = 5,
) -> dict[str, Any]:
    """Sync entry — e-Gov Law API から安全保障関連条文を取得。"""
    return _run_async(
        fetch_security_law_citations_async(catalog=catalog, max_entries=max_entries)
    )
