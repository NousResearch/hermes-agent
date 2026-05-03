"""Normalize ``SkillMeta`` → JSON-friendly catalog items + derived categories."""

from __future__ import annotations

from typing import Any, Dict, List, Set

from tools.skills_hub import SkillMeta

from hermes_cli.skill_catalog.categories import category_rules, list_categories, normalize_category_key


def _haystack(meta: SkillMeta) -> str:
    parts = [
        meta.name or "",
        meta.description or "",
        meta.identifier or "",
        " ".join(meta.tags or []),
        (meta.extra or {}).get("category", "") if isinstance(meta.extra, dict) else "",
    ]
    return "\n".join(parts).lower()


def derive_categories(meta: SkillMeta) -> List[Dict[str, str]]:
    """Map hub metadata to shop category chips via substring rules."""
    hay = _haystack(meta)
    codes: List[str] = []
    names_by_code: Dict[str, str] = {
        row["code"]: row["name"]
        for row in list_categories()
        if row.get("code") and row.get("code") != "all"
    }
    code_by_key: Dict[str, str] = {}
    for code, name in names_by_code.items():
        code_by_key[normalize_category_key(code)] = code
        code_by_key[normalize_category_key(name)] = code
    if isinstance(meta.extra, dict):
        source_category = str(meta.extra.get("category") or "").strip()
        matched_code = code_by_key.get(normalize_category_key(source_category))
        if matched_code:
            codes.append(matched_code)
    for row in category_rules():
        code = str(row.get("code", "")).strip()
        if not code:
            continue
        patterns = row.get("patterns") or []
        if not isinstance(patterns, list):
            continue
        for p in patterns:
            if not isinstance(p, str):
                continue
            needle = p.lower().strip()
            if needle and needle in hay:
                codes.append(code)
                break
    # De-dupe preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    if not uniq:
        uniq = ["productivity"]
    return [{"code": c, "name": names_by_code.get(c, c)} for c in uniq]


_TRUST_RANK = {"builtin": 3, "trusted": 2, "community": 1}


def trust_rank(level: str) -> int:
    return _TRUST_RANK.get((level or "").lower(), 0)


def meta_to_item(meta: SkillMeta, order: int) -> Dict[str, Any]:
    ident = (meta.identifier or "").strip() or meta.name
    cats = derive_categories(meta)
    return {
        "id": ident,
        "identifier": ident,
        "name": meta.name,
        "title": meta.name,
        "description": meta.description or "",
        "tags": list(meta.tags or []),
        "source": meta.source or "",
        "sourceLabel": meta.source or "",
        "trustLevel": meta.trust_level or "community",
        "repo": meta.repo,
        "path": meta.path,
        "categories": cats,
        "installCommand": f"hermes skills install {ident}",
        "_order": order,
    }


def dedupe_metas(items: List[SkillMeta]) -> List[SkillMeta]:
    """Prefer higher trust when identifiers collide."""
    by_key: Dict[str, SkillMeta] = {}
    for r in items:
        key = ((r.identifier or "").strip() or (r.name or "").strip() or "").lower()
        if not key:
            continue
        if key not in by_key:
            by_key[key] = r
        elif trust_rank(r.trust_level) > trust_rank(by_key[key].trust_level):
            by_key[key] = r
    return list(by_key.values())
