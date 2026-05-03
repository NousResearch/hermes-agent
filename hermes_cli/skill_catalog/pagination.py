"""Filter, sort, and slice catalog items (post-hub aggregation)."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple

from hermes_cli.skill_catalog.categories import normalize_category_key
from hermes_cli.skill_catalog.normalize import trust_rank

SortKey = Literal["relevance", "name", "trust"]


def filter_by_category(items: List[Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
    cat = normalize_category_key(category)
    if cat in ("", "all"):
        return items

    out: List[Dict[str, Any]] = []
    for item in items:
        cats = item.get("categories") or []
        if not isinstance(cats, list):
            continue
        if any(
            isinstance(c, dict)
            and cat
            in (
                normalize_category_key(str(c.get("code", ""))),
                normalize_category_key(str(c.get("name", ""))),
            )
            for c in cats
        ):
            out.append(item)
    return out


def sort_items(items: List[Dict[str, Any]], sort: str) -> List[Dict[str, Any]]:
    key = (sort or "relevance").strip().lower()
    if key not in ("relevance", "name", "trust"):
        key = "relevance"

    scoped = [dict(x) for x in items]

    if key == "name":
        scoped.sort(key=lambda x: str(x.get("name") or "").lower())
    elif key == "trust":
        scoped.sort(
            key=lambda x: (-trust_rank(str(x.get("trustLevel", ""))), str(x.get("name") or "").lower()),
        )
    else:
        scoped.sort(key=lambda x: int(x.get("_order", 0)))

    for row in scoped:
        row.pop("_order", None)
    return scoped


def paginate(
    items: List[Dict[str, Any]],
    *,
    page: int,
    page_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    total = len(items)
    page_size = max(1, min(page_size, 60))
    page = max(1, page)
    total_pages = (total + page_size - 1) // page_size if total else 0
    start = (page - 1) * page_size
    chunk = items[start : start + page_size]
    return chunk, {
        "page": page,
        "pageSize": page_size,
        "total": total,
        "totalPages": total_pages,
        "hasNext": start + page_size < total,
        "hasPrev": page > 1,
    }
