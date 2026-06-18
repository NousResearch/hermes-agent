"""MHLW 指定薬物部会・指定薬物施行/公表の定期監視（一次資料のみ）。"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from hermes_constants import get_hermes_home

from . import fetcher

# 一次資料（厚労省公式）
COMMITTEE_PAGE = "https://www.mhlw.go.jp/stf/shingi-yakuji_39213.html"
COMMITTEE_PAGE_DOC = "https://www.mhlw.go.jp/stf/shingi/shingi-yakuji_127874.html"
MHLW_BASE = "https://www.mhlw.go.jp"

ENFORCEMENT_SEARCH_QUERIES = (
    "site:mhlw.go.jp 指定薬物 新たに指定",
    "site:mhlw.go.jp 指定薬物 施行",
    "site:mhlw.go.jp 危険ドラッグ 指定薬物",
    "site:kanpou.go.jp 麻薬及び向精神薬取締法 指定",
)

_STATE_VERSION = 1


def _state_path() -> Path:
    path = get_hermes_home() / "scrapling-feeds" / "mhlw_designated_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_state() -> dict[str, Any]:
    path = _state_path()
    if not path.is_file():
        return {"version": _STATE_VERSION, "seen": {}, "last_check_at": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"version": _STATE_VERSION, "seen": {}, "last_check_at": None}
    if not isinstance(data.get("seen"), dict):
        data["seen"] = {}
    return data


def save_state(state: dict[str, Any]) -> None:
    state["version"] = _STATE_VERSION
    state["last_check_at"] = datetime.now(timezone.utc).isoformat()
    _state_path().write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _item_id(kind: str, url: str, title: str = "") -> str:
    raw = f"{kind}|{url}|{title}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _abs_url(href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    if href.startswith("http"):
        return href
    return urljoin(MHLW_BASE, href)


def parse_committee_page(html: str) -> list[dict[str, Any]]:
    """Parse 薬事審議会（指定薬物部会） table rows + 開催案内 links."""
    meetings: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for href in re.findall(r'href="(/stf/newpage_[^"]+\.html)"[^>]*>\s*開催', html):
        url = _abs_url(href)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        meetings.append(
            {
                "kind": "meeting_notice",
                "title": "指定薬物部会 開催案内",
                "url": url,
                "source_tier": "PRIMARY",
                "source_page": COMMITTEE_PAGE,
            }
        )

    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.DOTALL | re.IGNORECASE):
        if "開催日" in row_html and "<th" in row_html:
            continue
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.DOTALL | re.IGNORECASE)
        if len(cells) < 2:
            continue
        session_label = _strip_html(cells[0])
        meeting_date = _strip_html(cells[1])
        if not session_label or session_label in {"回数", "－", "-"}:
            continue
        if not meeting_date or meeting_date in {"開催日", "－", "-"}:
            continue
        notice_url = ""
        for href in re.findall(r'href="([^"]+)"', row_html):
            if "newpage_" in href:
                notice_url = _abs_url(href)
                break
        meetings.append(
            {
                "kind": "meeting_schedule",
                "session": session_label,
                "meeting_date": meeting_date,
                "title": f"指定薬物部会 {session_label}（{meeting_date}）",
                "url": notice_url or COMMITTEE_PAGE,
                "source_tier": "PRIMARY",
                "source_page": COMMITTEE_PAGE,
            }
        )

    return meetings


def parse_meeting_notice_page(html: str, *, url: str) -> dict[str, Any]:
    """Extract 開催日時 from a newpage announcement."""
    title_m = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
    title = _strip_html(title_m.group(1) if title_m else "")
    h1_m = re.search(r"<h1[^>]*>([^<]+)</h1>", html, re.IGNORECASE)
    if h1_m:
        title = _strip_html(h1_m.group(1)) or title

    meeting_when = ""
    for pattern in (
        r"１\s*開催日時\s*</h2>\s*<p[^>]*>([^<]+)",
        r"開催日時\s*</h2>\s*<p[^>]*>([^<]+)",
        r"１\s*開催日時[^<]*</[^>]+>\s*<[^>]+>([^<]{8,80})",
    ):
        m = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
        if m:
            meeting_when = _strip_html(m.group(1))
            break

    return {
        "kind": "meeting_notice",
        "title": title or "指定薬物部会 開催案内",
        "meeting_when": meeting_when,
        "url": url,
        "source_tier": "PRIMARY",
        "citation": f"[出典: 厚労省 指定薬物部会開催案内] {url}",
    }


def _ddgs_search(query: str, *, limit: int = 5) -> list[dict[str, str]]:
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        return []
    hits: list[dict[str, str]] = []
    try:
        with DDGS() as client:
            for i, row in enumerate(client.text(query, max_results=limit)):
                if i >= limit:
                    break
                url = str(row.get("href") or row.get("url") or "")
                if not url:
                    continue
                hits.append(
                    {
                        "title": str(row.get("title") or ""),
                        "url": url,
                        "description": str(row.get("body") or "")[:300],
                    }
                )
    except Exception:
        return []
    return hits


def scan_enforcement_announcements(*, limit_per_query: int = 4) -> list[dict[str, Any]]:
    """Site-constrained search for 指定薬物 designation / 施行 announcements."""
    items: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for query in ENFORCEMENT_SEARCH_QUERIES:
        for hit in _ddgs_search(query, limit=limit_per_query):
            url = hit.get("url") or ""
            if not url or url in seen_urls:
                continue
            if "mhlw.go.jp" not in url and "kanpou" not in url:
                continue
            seen_urls.add(url)
            title = hit.get("title") or ""
            items.append(
                {
                    "kind": "enforcement",
                    "title": title,
                    "url": url,
                    "summary": hit.get("description") or "",
                    "search_query": query,
                    "source_tier": "PRIMARY",
                    "citation": f"[出典: {title}] {url}",
                }
            )
    return items


def _enrich_meeting_notices(meetings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in meetings:
        url = str(row.get("url") or "")
        if row.get("kind") == "meeting_notice" and "newpage_" in url:
            fetched = fetcher.fetch_url(url)
            if not fetched.get("success"):
                out.append({**row, "fetch_error": fetched.get("error")})
                continue
            detail = parse_meeting_notice_page(fetched.get("body") or "", url=url)
            out.append({**row, **detail})
        else:
            out.append(row)
    return out


def check_mhlw_designated(
    *,
    record_baseline: bool = False,
    enrich_notices: bool = True,
    scan_enforcement: bool = True,
) -> dict[str, Any]:
    """Run full check; return new items vs persisted state."""
    state = load_state()
    seen: dict[str, Any] = dict(state.get("seen") or {})

    committee_fetch = fetcher.fetch_url(COMMITTEE_PAGE)
    committee_error = ""
    if committee_fetch.get("success"):
        meetings = parse_committee_page(committee_fetch.get("body") or "")
        if enrich_notices:
            meetings = _enrich_meeting_notices(meetings)
    else:
        committee_error = committee_fetch.get("error") or "fetch failed"

    enforcement: list[dict[str, Any]] = []
    if scan_enforcement:
        enforcement = scan_enforcement_announcements()

    all_items = meetings + enforcement
    new_items: list[dict[str, Any]] = []
    known_items: list[dict[str, Any]] = []

    for item in all_items:
        iid = _item_id(item.get("kind", ""), item.get("url", ""), item.get("title", ""))
        item["id"] = iid
        if iid in seen and not record_baseline:
            known_items.append(item)
        else:
            new_items.append(item)
            seen[iid] = {
                "first_seen_at": seen.get(iid, {}).get("first_seen_at")
                or datetime.now(timezone.utc).isoformat(),
                "kind": item.get("kind"),
                "title": item.get("title"),
                "url": item.get("url"),
            }

    if record_baseline or new_items or not state.get("last_check_at"):
        save_state({**state, "seen": seen})

    result: dict[str, Any] = {
        "success": bool(committee_fetch.get("success") or enforcement),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "committee_page": COMMITTEE_PAGE,
        "committee_doc": COMMITTEE_PAGE_DOC,
        "new_count": len(new_items),
        "known_count": len(known_items),
        "new_items": new_items,
        "known_items": known_items if record_baseline else known_items[:10],
        "meetings_found": len(meetings),
        "enforcement_found": len(enforcement),
        "methodology": (
            "厚労省 指定薬物部会ページ直読 + site: 制約検索（指定・施行公表）。"
            "ステルスクロールなし。"
        ),
    }
    if not committee_fetch.get("success"):
        result["committee_error"] = committee_error
    return result


def build_report_markdown(result: dict[str, Any], *, include_known: bool = False) -> str:
    lines = [
        "# 厚労省 指定薬物モニター",
        "",
        f"- **確認時刻**: {result.get('checked_at')}",
        f"- **部会ページ**: {COMMITTEE_PAGE}",
        f"- **新規**: {result.get('new_count', 0)} 件",
        "",
    ]
    new_items = result.get("new_items") or []
    if new_items:
        lines.extend(["## 🆕 新規検知", ""])
        meetings = [i for i in new_items if str(i.get("kind", "")).startswith("meeting")]
        enforce = [i for i in new_items if i.get("kind") == "enforcement"]
        if meetings:
            lines.append("### 指定薬物部会（開催・案内）")
            lines.append("")
            for item in meetings:
                lines.append(f"- **{item.get('title')}**")
                if item.get("meeting_when"):
                    lines.append(f"  - 開催: {item.get('meeting_when')}")
                elif item.get("meeting_date"):
                    lines.append(f"  - 日程: {item.get('meeting_date')} ({item.get('session', '')})")
                cite = item.get("citation") or item.get("url")
                lines.append(f"  - {cite}")
            lines.append("")
        if enforce:
            lines.append("### 指定薬物の指定・施行関連")
            lines.append("")
            for item in enforce:
                lines.append(f"- **{item.get('title')}**")
                if item.get("summary"):
                    lines.append(f"  - {item.get('summary')[:200]}")
                lines.append(f"  - {item.get('citation') or item.get('url')}")
            lines.append("")
    else:
        lines.extend(["## 状態", "", "_前回以降の新規なし（部会案内・施行公表）_", ""])

    if include_known and result.get("known_items"):
        lines.extend(["## 既知（直近）", ""])
        for item in (result.get("known_items") or [])[:8]:
            lines.append(f"- {item.get('title')} — {item.get('url')}")

    lines.extend(
        [
            "",
            "---",
            "_一次資料: 厚労省公式 / 官報検索 — Hermes scrapling-feeds_",
        ]
    )
    return "\n".join(lines)


def run_for_cron_stdout(**kwargs: Any) -> int:
    """Cron script entry — print markdown when new items exist."""
    only_new = kwargs.pop("only_new", True)
    result = check_mhlw_designated(**kwargs)
    if only_new and not result.get("new_count"):
        print("厚労省 指定薬物: 新規なし")
        return 0
    print(build_report_markdown(result))
    return 0 if result.get("success") else 1
