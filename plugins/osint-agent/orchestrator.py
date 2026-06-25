"""Unified OSINT brief — SitDeck + WM PDB + government RSS + MHLW."""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from . import plugin_loader

SLOT_LABELS = {
    "morning": ("朝次", "08:00"),
    "evening": ("夕次", "18:00"),
}


def _reports_dir() -> Path:
    path = get_hermes_home() / "osint-agent" / "briefs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _sitdeck_section(*, headless: bool = True) -> dict[str, Any]:
    try:
        plugin_loader.load_plugin_modules(
            "sitdeck-osint",
            ("credentials", "browser_crawl"),
        )
        creds_mod = plugin_loader.get_module(
            "sitdeck-osint",
            "credentials",
            stems_chain=("credentials", "browser_crawl"),
        )
        crawl_mod = plugin_loader.get_module(
            "sitdeck-osint",
            "browser_crawl",
            stems_chain=("credentials", "browser_crawl"),
        )
        status = creds_mod.credential_status()
        if not status.get("email_configured") or not status.get("password_configured"):
            return {"skipped": True, "reason": "SITDECK credentials not in .env"}
        crawl = crawl_mod.crawl_dashboard(headless=headless)
        if not crawl.get("success"):
            return {"success": False, "crawl": crawl}
        digest = crawl_mod.build_digest(crawl)
        return {"success": True, "digest": digest, "api_capture_count": crawl.get("api_capture_count")}
    except Exception as exc:
        return {"skipped": True, "reason": str(exc)[:200]}


def _mhlw_section(*, cron_mode: bool = False) -> dict[str, Any]:
    try:
        plugin_loader.load_plugin_modules(
            "scrapling-feeds",
            ("feeds_catalog", "fetcher", "rss_parse", "gov_digest", "mhlw_designated"),
        )
        mhlw = plugin_loader.get_module(
            "scrapling-feeds",
            "mhlw_designated",
            stems_chain=(
                "feeds_catalog",
                "fetcher",
                "rss_parse",
                "gov_digest",
                "mhlw_designated",
            ),
        )
        result = mhlw.check_mhlw_designated(record_baseline=False, scan_enforcement=True)
        new_items = result.get("new_items") or []
        lines = ["## 厚労省 指定薬物（新規検知）", ""]
        if not new_items:
            lines.append("_新規項目なし（前回チェック以降）_")
        else:
            for item in new_items[:20]:
                title = item.get("title") or item.get("url") or "?"
                url = item.get("url") or ""
                lines.append(f"- {title}" + (f" — {url}" if url else ""))
        return {
            "success": bool(result.get("success")),
            "new_count": len(new_items),
            "markdown": "\n".join(lines),
            "raw": result,
        }
    except Exception as exc:
        return {"skipped": True, "reason": str(exc)[:200]}


def _gov_feeds_markdown(hours: int = 24) -> tuple[str, dict[str, Any]]:
    try:
        pkg = plugin_loader.load_plugin_modules(
            "worldmonitor-osint",
            (
                "api",
                "auth_setup",
                "free_web",
                "milspec_prose",
                "primary_backfill",
            ),
        )
        backfill_mod = sys.modules[f"{pkg}.primary_backfill"]
        digest = backfill_mod._fetch_gov_feeds_digest(hours=hours, max_per_feed=6)
    except Exception as exc:
        return "", {"skipped": True, "reason": str(exc)[:200]}

    if digest.get("skipped"):
        return "", digest

    lines = ["## 官公庁 RSS（一次資料）", ""]
    feeds = digest.get("feeds") or []
    shown = 0
    for feed in feeds:
        if not feed.get("success"):
            continue
        for entry in (feed.get("entries") or [])[:3]:
            cite = entry.get("citation") or entry.get("title") or ""
            lines.append(f"- {cite}")
            shown += 1
    if shown == 0:
        lines.append(f"_対象 {digest.get('feed_count', 0)} フィード — 直近 {hours}h に新規エントリなし_")
    return "\n".join(lines), digest


def _pdb_section(
    *,
    slot: str,
    topic: str,
    source_mode: str,
    wm_tier: str,
    llm_summary: bool,
    max_scenarios: int,
) -> dict[str, Any]:
    pkg = plugin_loader.load_plugin_modules(
        "worldmonitor-osint",
        (
            "api",
            "auth_setup",
            "free_web",
            "threat_extract",
            "milspec_prose",
            "primary_backfill",
            "egov_primary",
            "core",
            "situation_report",
        ),
    )
    sitrep = sys.modules[f"{pkg}.situation_report"]
    return sitrep.generate_situation_report(
        slot=slot,
        topic=topic,
        source_mode=source_mode,
        wm_tier=wm_tier,
        llm_summary=llm_summary,
        save=False,
        use_primary_backfill=True,
        fetch_egov=True,
        fetch_github=True,
        fetch_gov_feeds=False,
    )


def build_integrated_markdown(
    *,
    slot: str,
    pdb: dict[str, Any],
    sitdeck: dict[str, Any],
    gov_md: str,
    mhlw: dict[str, Any],
) -> str:
    label, clock = SLOT_LABELS.get(slot, ("定時", slot))
    generated = datetime.now(timezone.utc).astimezone()
    parts = [
        "# 統合 OSINT ブリーフィング",
        "",
        f"- **配信**: {label}（{clock} 想定）",
        f"- **生成**: {generated.isoformat()}",
        "- **ソース**: SitDeck + World Monitor Free + 官公庁 RSS + 厚労省指定薬物",
        "- **分類**: オープンソース統合（非機密）",
        "",
        "---",
        "",
        pdb.get("markdown") or "_PDB セクション未取得_",
        "",
        "---",
        "",
    ]
    if sitdeck.get("success") and sitdeck.get("digest"):
        parts.extend([sitdeck["digest"], "", "---", ""])
    elif not sitdeck.get("skipped"):
        parts.extend(["## SitDeck", "", f"_取得失敗: {sitdeck.get('reason') or sitdeck}_", "", "---", ""])
    if gov_md:
        parts.extend([gov_md, "", "---", ""])
    if mhlw.get("markdown"):
        parts.extend([mhlw["markdown"], ""])
    parts.append(
        "\n_統合: hermes osint-agent brief — WM Pro MCP 不要 / egov-law MCP は PDB 内参照_"
    )
    return "\n".join(parts)


def generate_integrated_brief(
    *,
    slot: str = "morning",
    topic: str = "日本の安全保障と世界情勢",
    source_mode: str = "real",
    wm_tier: str = "free",
    llm_summary: bool = False,
    max_scenarios: int = 4,
    include_sitdeck: bool = True,
    include_mhlw: bool = True,
    include_gov_feeds: bool = True,
    sitdeck_headless: bool = True,
    save: bool = True,
    cron_mode: bool = False,
) -> dict[str, Any]:
    """Run all OSINT layers and return unified markdown + JSON payload."""
    sitdeck: dict[str, Any] = {"skipped": True}
    gov_md, gov_raw = ("", {"skipped": True})
    mhlw: dict[str, Any] = {"skipped": True}

    futures: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures["pdb"] = pool.submit(
            _pdb_section,
            slot=slot,
            topic=topic,
            source_mode=source_mode,
            wm_tier=wm_tier,
            llm_summary=llm_summary,
            max_scenarios=max_scenarios,
        )
        if include_sitdeck:
            futures["sitdeck"] = pool.submit(_sitdeck_section, headless=sitdeck_headless)
        if include_gov_feeds:
            futures["gov"] = pool.submit(_gov_feeds_markdown, hours=24)
        if include_mhlw:
            futures["mhlw"] = pool.submit(_mhlw_section, cron_mode=cron_mode)

        pdb = futures["pdb"].result()
        if "sitdeck" in futures:
            sitdeck = futures["sitdeck"].result()
        if "gov" in futures:
            gov_md, gov_raw = futures["gov"].result()
        if "mhlw" in futures:
            mhlw = futures["mhlw"].result()

    markdown = build_integrated_markdown(
        slot=slot,
        pdb=pdb,
        sitdeck=sitdeck,
        gov_md=gov_md,
        mhlw=mhlw,
    )

    payload: dict[str, Any] = {
        "success": bool(pdb.get("success")),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "slot": slot,
        "topic": topic,
        "source_mode": source_mode,
        "wm_tier": wm_tier,
        "sections": {
            "pdb": {"success": pdb.get("success")},
            "sitdeck": sitdeck,
            "gov_feeds": gov_raw,
            "mhlw": {k: v for k, v in mhlw.items() if k != "raw"},
        },
        "markdown": markdown,
        "pdb": pdb,
    }

    if save:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = re.sub(r"[^\w\-]+", "_", f"{slot}_{topic}"[:40]).strip("_") or slot
        md_path = _reports_dir() / f"{stamp}_{slug}.md"
        json_path = _reports_dir() / f"{stamp}_{slug}.json"
        md_path.write_text(markdown + "\n", encoding="utf-8")
        json_path.write_text(_json(payload), encoding="utf-8")
        payload["saved_markdown"] = str(md_path)
        payload["saved_json"] = str(json_path)

    return payload


def run_for_cron_stdout(slot: str, **kwargs: Any) -> int:
    kwargs.setdefault("source_mode", "real")
    kwargs.setdefault("wm_tier", "free")
    kwargs.setdefault("save", True)
    result = generate_integrated_brief(slot=slot, cron_mode=True, **kwargs)
    print(result.get("markdown") or "")
    return 0 if result.get("success") else 1
