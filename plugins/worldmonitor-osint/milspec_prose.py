"""MILSPEC-aligned prose helpers — primary-source discipline for PDB reports."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

# Traceability: MIL-STD-498 style source labeling (open-source PDB, not classified product).
SOURCE_TIER_PRIMARY = "PRIMARY"
SOURCE_TIER_SECONDARY = "SECONDARY"
SOURCE_TIER_AGGREGATOR = "AGGREGATOR"
SOURCE_TIER_UNVERIFIED = "UNVERIFIED"

_PRIMARY_HOST_SUFFIXES = (
    ".go.jp",
    ".gov",
    ".mil",
    ".int",
    "e-gov.go.jp",
    "un.org",
    "nato.int",
    "mod.go.jp",
    "mofa.go.jp",
    "defense.gov",
    "state.gov",
    "whitehouse.gov",
    "congress.gov",
    "europa.eu",
)

_MILSPEC_LLM_SYSTEM = (
    "You are a Japanese national-security briefer writing MILSPEC-aligned PDB-style prose. "
    "Rules (mandatory):\n"
    "1. Every factual claim MUST cite a traceable source: [出典: URL] or [出典: 法令・公文書 ID] "
    "or [出典: Shinka scenario_id + evidence_block].\n"
    "2. Do NOT invent events, numbers, or policy positions. If a claim lacks a primary or "
    "secondary source in the input, label it UNVERIFIED and do not present it as fact.\n"
    "3. Prefer primary sources (government, treaty body, official gazette) over media.\n"
    "4. Separate OBSERVED (sourced) from ASSESSMENT (analytic judgment, clearly labeled).\n"
    "5. No marketing language. Concise bullet prose. Japanese output."
)


def _host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def classify_source_tier(url: str) -> str:
    """Classify a headline URL for PDB provenance labeling."""
    u = (url or "").strip()
    if not u:
        return SOURCE_TIER_UNVERIFIED
    host = _host(u)
    if not host:
        return SOURCE_TIER_UNVERIFIED
    for suffix in _PRIMARY_HOST_SUFFIXES:
        if host == suffix.lstrip(".") or host.endswith(suffix):
            return SOURCE_TIER_PRIMARY
    if "worldmonitor" in host:
        return SOURCE_TIER_AGGREGATOR
    return SOURCE_TIER_SECONDARY


def format_cited_headline(item: dict[str, Any]) -> str:
    title = (item.get("title") or "").strip()
    url = (item.get("url") or "").strip()
    tier = classify_source_tier(url)
    cat = item.get("threat_category") or item.get("category") or "general"
    backfill = item.get("backfill_method")
    suffix = ""
    if backfill and backfill not in {"already_primary", "no_terms"}:
        suffix = f" _(裏取り: {backfill})_"
    if url:
        return f"- [{tier}/{cat}] {title} — [出典: {url}]{suffix}"
    return f"- [{SOURCE_TIER_UNVERIFIED}/{cat}] {title} — [出典: 要一次資料裏取り]{suffix}"


def build_egov_citations_block(enrichment: dict[str, Any]) -> list[str]:
    egov = enrichment.get("egov") or {}
    citations = egov.get("citations") or []
    lines = ["## JAPAN LAW PRIMARY SOURCES（e-Gov Law API v2）", ""]
    if egov.get("skipped"):
        lines.append(f"_スキップ: {egov.get('reason', 'egov-law-mcp 未利用')}_")
        lines.append("")
        return lines
    if not citations:
        lines.append("_条文取得なし_")
        lines.append("")
        return lines
    for row in citations:
        if row.get("success"):
            label = row.get("label") or row.get("law_name")
            snippet = (row.get("snippet") or "").strip()
            cite = row.get("citation") or row.get("source_url") or ""
            lines.append(f"### {label}")
            if snippet:
                lines.append(snippet)
            lines.append(f"- {cite}")
            lines.append("")
        else:
            lines.append(
                f"- ⚠ {row.get('label') or row.get('search')}: "
                f"{row.get('error', '取得失敗')}"
            )
    lines.append(f"_API: {egov.get('source_api', 'e-Gov Law API v2')}_")
    lines.append("")
    return lines


def build_github_provenance_block(enrichment: dict[str, Any]) -> list[str]:
    gh = enrichment.get("github") or {}
    if gh.get("skipped"):
        return []
    lines = ["## TOOLCHAIN PROVENANCE（GitHub 一次メタデータ）", ""]
    for repo in gh.get("toolchain_repos") or []:
        cite = repo.get("citation") or repo.get("html_url") or ""
        role = repo.get("role") or ""
        lines.append(f"- **{repo.get('repo')}** — {role}")
        lines.append(f"  {cite}")
    topic_hits = gh.get("topic_search_hits") or []
    if topic_hits:
        lines.append("")
        lines.append("### GitHub topic search（参考・二次）")
        for hit in topic_hits:
            lines.append(f"- {hit.get('citation') or hit.get('html_url')}")
    lines.append("")
    return lines


def build_gov_feeds_block(enrichment: dict[str, Any]) -> list[str]:
    gov = enrichment.get("gov_feeds") or {}
    if gov.get("skipped"):
        return []
    lines = ["## GOVERNMENT FEEDS（PRIMARY — 公式 RSS/Atom）", ""]
    if not gov.get("success") and not gov.get("feeds"):
        lines.append(f"_取得失敗: {gov.get('reason') or gov.get('error', '不明')}_")
        lines.append("")
        return lines
    stats = (
        f"feeds {gov.get('feeds_ok', 0)}/{gov.get('feed_count', 0)}, "
        f"entries {gov.get('total_entries', 0)}, "
        f"window {gov.get('window_hours', 24)}h"
    )
    backend = "Scrapling" if gov.get("scrapling_available") else "urllib"
    lines.append(f"_直読バックエンド: {backend}; {stats}_")
    lines.append("")
    for feed in gov.get("feeds") or []:
        if not feed.get("success"):
            lines.append(f"- ⚠ **{feed.get('name') or feed.get('feed_id')}**: {feed.get('error')}")
            continue
        entries = feed.get("entries") or []
        if not entries:
            lines.append(f"- _{feed.get('name')}: 対象期間内の新規項目なし_")
            continue
        lines.append(f"### {feed.get('name')} ({feed.get('agency')})")
        for entry in entries[:8]:
            title = (entry.get("title") or "").strip()
            cite = entry.get("citation") or entry.get("url") or ""
            pub = entry.get("published_at") or ""
            pub_bit = f" ({pub})" if pub else ""
            lines.append(f"- [{feed.get('source_tier', 'PRIMARY')}] {title}{pub_bit}")
            lines.append(f"  {cite}")
        lines.append("")
    catalog = gov.get("catalog_docs") or {}
    if catalog:
        lines.append(f"_出典カタログ: {', '.join(catalog.values())}_")
    lines.append("")
    return lines


def build_backfill_notes_block(enrichment: dict[str, Any]) -> list[str]:
    rows = enrichment.get("headline_backfill") or []
    if not rows:
        return []
    lines = ["## PRIMARY BACKFILL（公式ドメイン site: 検索）", ""]
    stats = (enrichment.get("stats") or {})
    lines.append(
        f"- 裏取り試行: {stats.get('headlines_backfilled', len(rows))} 件 / "
        f"PRIMARY 解決: {stats.get('headlines_primary_resolved', 0)} 件"
    )
    lines.append("")
    for row in rows[:8]:
        method = row.get("backfill_method") or "?"
        tier = row.get("source_tier") or SOURCE_TIER_UNVERIFIED
        if row.get("primary_url"):
            lines.append(
                f"- [{tier}] {row.get('title', '')[:90]} → "
                f"[出典: {row.get('primary_url')}] ({method})"
            )
        else:
            lines.append(
                f"- [{tier}] {row.get('title', '')[:90]} — 一次資料未解決 ({method})"
            )
    lines.append("")
    return lines


def build_key_developments_lines(threats: dict[str, Any]) -> list[str]:
    headlines = threats.get("high_threat_headlines") or []
    if headlines:
        lines: list[str] = []
        by_cat: dict[str, list[dict[str, Any]]] = {}
        for item in headlines:
            if not isinstance(item, dict):
                continue
            key = str(item.get("threat_category") or "general")
            by_cat.setdefault(key, []).append(item)
        for cat in sorted(by_cat):
            lines.append(f"### {cat}")
            for item in by_cat[cat][:8]:
                lines.append(format_cited_headline(item))
            lines.append("")
        return lines

    by_cat = threats.get("high_threat_by_category") or {}
    if not by_cat:
        return ["_HIGH 分類の新規見出しなし。一次資料による新規事実の追加なし。_"]
    lines = []
    for cat, titles in sorted(by_cat.items()):
        lines.append(f"### {cat}")
        for title in titles[:8]:
            lines.append(
                f"- [{SOURCE_TIER_UNVERIFIED}/{cat}] {title} — [出典: 要一次資料裏取り]"
            )
        lines.append("")
    return lines


def build_shinka_evidence_lines(fusion: dict[str, Any]) -> list[str]:
    shinka = fusion.get("shinka_milspec") or {}
    if not shinka.get("success"):
        err = (shinka.get("error") or "評価未完了")[:200]
        return [f"- Shinka MILSPEC: 未完了 ({err})"]

    lines: list[str] = []
    for item in shinka.get("runs") or []:
        sid = item.get("scenario_id") or "?"
        score = item.get("total_score")
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        if score is None and isinstance(result.get("score"), dict):
            score = result["score"].get("total")
        verified = result.get("verified")
        evidence = result.get("evidence_blocks")
        lines.append(
            f"- `{sid}`: score={score}, verified={verified}, "
            f"evidence_blocks={evidence!r}"
        )
        kjs = result.get("key_judgments")
        if isinstance(kjs, list) and kjs:
            for kj in kjs[:3]:
                if isinstance(kj, str) and kj.strip():
                    lines.append(f"  - KJ [出典: Shinka/{sid}]: {kj.strip()[:240]}")
    return lines or ["- （シナリオ結果なし）"]


def build_provenance_section(
    fusion: dict[str, Any],
    threats: dict[str, Any],
    enrichment: dict[str, Any] | None = None,
) -> list[str]:
    primary = fusion.get("primary_sources") or {}
    egov = primary.get("egov_law_mcp") or {}
    wm = fusion.get("worldmonitor") or {}
    wm_tier = (wm.get("tier") or wm.get("tier_mode") or fusion.get("wm_tier") or "unknown")

    headlines = threats.get("high_threat_headlines") or []
    tier_counts = {SOURCE_TIER_PRIMARY: 0, SOURCE_TIER_SECONDARY: 0, SOURCE_TIER_AGGREGATOR: 0, SOURCE_TIER_UNVERIFIED: 0}
    for item in headlines:
        if isinstance(item, dict):
            tier_counts[classify_source_tier(str(item.get("url") or ""))] += 1

    lines = [
        "## SOURCE INTEGRITY（一次資料規律）",
        "",
        "- **規律**: 事実記述は信頼できる一次資料（政府公文書・法令・条約機関公表）を優先。"
        " メディア経由の記述は二次資料として明示し、単独では政策判断根拠にしない。",
        f"- **World Monitor 層**: tier={wm_tier}（集約 OSINT；見出しは裏取り対象）",
        f"- **見出しソース内訳**: PRIMARY={tier_counts[SOURCE_TIER_PRIMARY]}, "
        f"SECONDARY={tier_counts[SOURCE_TIER_SECONDARY]}, "
        f"AGGREGATOR={tier_counts[SOURCE_TIER_AGGREGATOR]}, "
        f"UNVERIFIED={tier_counts[SOURCE_TIER_UNVERIFIED]}",
        f"- **Shinka source_mode**: {primary.get('shinka_source_mode') or fusion.get('source_mode') or '?'}",
    ]
    enrich = enrichment or {}
    stats = enrich.get("stats") or {}
    if stats:
        lines.append(
            f"- **自動裏取り**: e-Gov条文={stats.get('egov_citations_ok', 0)}件, "
            f"見出しPRIMARY解決={stats.get('headlines_primary_resolved', 0)}件 "
            f"({stats.get('methodology', '')})"
        )
    lines.extend(
        [
        "- **日本法一次資料**: e-Gov Law API v2（`egov-law-mcp` / laws.e-gov.go.jp）",
        "",
        "### 追加照会先",
        "",
        "- 防衛省・外務省・内閣官房の公表（mod.go.jp / mofa.go.jp / cas.go.jp）",
        "- 国連安保理決議・NATO公式声明（該当時）",
        "",
        ]
    )
    return lines


def derive_watchlist(threats: dict[str, Any], fusion: dict[str, Any]) -> list[str]:
    """Evidence-traceable watch items only — no unsourced geopolitical boilerplate."""
    items: list[str] = []
    for idx, item in enumerate((threats.get("high_threat_headlines") or [])[:6], start=1):
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "")[:100]
        url = (item.get("url") or "").strip()
        cite = f"[出典: {url}]" if url else "[出典: 要一次資料裏取り]"
        items.append(f"{idx}. 追跡: {title} — {cite}")

    shinka = fusion.get("shinka_milspec") or {}
    base = len(items)
    for run in (shinka.get("runs") or [])[:4]:
        result = run.get("result") if isinstance(run.get("result"), dict) else {}
        if result.get("verified") is False or result.get("error"):
            sid = run.get("scenario_id") or "?"
            items.append(
                f"{base + 1}. Shinka整合性: `{sid}` — [出典: Shinka evaluate / allowlist]"
            )
            base += 1

    if not items:
        items.append(
            "1. 新規 HIGH シグナルなし — 一次資料チャネル（政府公表・e-Gov）の定例監視を継続"
        )
    return items[:8]


def extract_executive_summary_text(exec_sum: Any) -> str:
    if not isinstance(exec_sum, dict):
        return ""
    for key in ("summary_ja", "text", "content"):
        val = exec_sum.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def build_llm_user_context(
    *,
    topic: str,
    slot: str,
    threats: dict[str, Any],
    fusion: dict[str, Any],
    enrichment: dict[str, Any] | None = None,
) -> str:
    lines = [
        f"Slot: {slot}",
        f"Topic: {topic}",
        f"HIGH threats: {threats.get('unique_high_threat_count')}",
        "",
        "World Monitor HIGH headlines (cite URLs when used):",
    ]
    for item in (threats.get("high_threat_headlines") or [])[:12]:
        if isinstance(item, dict):
            lines.append(format_cited_headline(item))
    lines.append("")
    lines.append("Shinka MILSPEC runs:")
    lines.extend(build_shinka_evidence_lines(fusion))
    lines.append("")
    lines.append("Elevated CII:")
    for row in (threats.get("elevated_cii_regions") or [])[:8]:
        lines.append(
            f"- {row.get('region')}: score={row.get('combinedScore')} ({row.get('trend')})"
        )
    enrich = enrichment or {}
    egov = enrich.get("egov") or {}
    if egov.get("citations"):
        lines.append("")
        lines.append("e-Gov primary law citations (use these as PRIMARY sources):")
        for row in egov.get("citations") or []:
            if row.get("success"):
                lines.append(row.get("citation") or str(row.get("snippet", ""))[:200])
    return "\n".join(lines)


def synthesize_pdb_executive_summary(
    *,
    topic: str,
    slot: str,
    threats: dict[str, Any],
    fusion: dict[str, Any],
    enrichment: dict[str, Any] | None = None,
    max_tokens: int = 1400,
) -> dict[str, Any]:
    """MILSPEC PDB executive summary via Hermes auxiliary LLM."""
    import importlib.util
    from pathlib import Path

    prov_path = Path(__file__).resolve().parent.parent / "shinka-osint" / "providers.py"
    spec = importlib.util.spec_from_file_location("wm_shinka_providers", prov_path)
    if spec is None or spec.loader is None:
        return {"success": False, "skipped": True, "reason": "shinka providers unavailable"}
    shinka_providers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shinka_providers)

    resolved = shinka_providers.resolve_llm(require_auth=True)
    if resolved is None:
        return {
            "success": False,
            "skipped": True,
            "reason": "No Hermes LLM auth — MILSPEC scores/report body still generated without LLM.",
        }

    try:
        from agent.auxiliary_client import resolve_provider_client
    except ImportError as exc:
        return {"success": False, "skipped": True, "reason": f"auxiliary_client unavailable: {exc}"}

    client, model = resolve_provider_client(
        resolved.provider_id,
        model=resolved.model,
        explicit_api_key=resolved.api_key,
        explicit_base_url=resolved.base_url,
        task="worldmonitor_pdb_summary",
    )
    if client is None or not model:
        return {
            "success": False,
            "skipped": True,
            "reason": f"Could not build client for provider {resolved.provider_id}",
        }

    user_msg = build_llm_user_context(
        topic=topic,
        slot=slot,
        threats=threats,
        fusion=fusion,
        enrichment=enrichment,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _MILSPEC_LLM_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=max(256, min(max_tokens, 4096)),
            temperature=0.1,
        )
        content = ""
        if response.choices:
            content = (response.choices[0].message.content or "").strip()
        return {
            "success": bool(content),
            "skipped": False,
            "provider_id": resolved.provider_id,
            "model": model,
            "summary_ja": content,
            "milspec_primary_source_rule": True,
        }
    except Exception as exc:
        return {
            "success": False,
            "skipped": True,
            "provider_id": resolved.provider_id,
            "model": model,
            "reason": str(exc),
        }
