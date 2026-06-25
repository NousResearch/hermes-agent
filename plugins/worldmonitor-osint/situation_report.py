"""PDB-style 24h national-security situation report (World Monitor + Shinka)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from . import core
from . import milspec_prose
from . import primary_backfill
from .threat_extract import extract_high_threats

SLOT_LABELS = {
    "morning": ("朝次", "08:00"),
    "evening": ("夕次", "18:00"),
}


def _reports_dir() -> Path:
    path = get_hermes_home() / "worldmonitor-osint" / "situation_reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _top_lines(threats: dict[str, Any], fusion: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    n_high = threats.get("unique_high_threat_count") or 0
    if n_high:
        lines.append(
            f"過去24時間で World Monitor が HIGH 脅威シグナル {n_high} 件を検知"
            "（二次集約；各項目は出典 URL で裏取り要）。"
        )
    headlines = threats.get("high_threat_headlines") or []
    for item in headlines[:3]:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "")[:100]
        url = (item.get("url") or "").strip()
        tier = milspec_prose.classify_source_tier(url)
        cite = f"[出典: {url}]" if url else "[出典: 要一次資料裏取り]"
        lines.append(f"[{tier}] {title} — {cite}")
    shinka = fusion.get("shinka_milspec") or {}
    if shinka.get("success"):
        runs = shinka.get("runs") or []
        if runs:
            lines.append(
                "Shinka MILSPEC 評価を統合"
                "（政府コーパス・evidence_blocks；[出典: Shinka evaluate]）。"
            )
    if not lines:
        lines.append(
            "顕著な HIGH シグナルは限定的。"
            "一次資料（政府公表・e-Gov）による新規事実の追加なし — 継続監視。"
        )
    return lines[:5]


def build_pdb_markdown(
    *,
    slot: str,
    fusion: dict[str, Any],
    threats: dict[str, Any],
    window_hours: int = 24,
    llm_summary: str = "",
    executive_summary_meta: dict[str, Any] | None = None,
    enrichment: dict[str, Any] | None = None,
) -> str:
    slot_key = (slot or "morning").strip().lower()
    label, clock = SLOT_LABELS.get(slot_key, ("定時", slot_key))
    generated = datetime.now(timezone.utc).astimezone()

    lines = [
        "# 安全保障シチュエーションレポート（PDB型 / MILSPEC準拠）",
        "",
        f"- **配信**: {label}ブリーフィング（現地 {clock} 想定）",
        f"- **対象期間**: 過去 {window_hours} 時間",
        f"- **生成**: {generated.isoformat()}",
        "- **分類**: オープンソース統合（非機密）",
        "- **記述規律**: 事実は一次資料優先；未裏取りは UNVERIFIED 明示",
        "",
        "## TOP LINE",
        "",
    ]
    for tl in _top_lines(threats, fusion):
        lines.append(f"- {tl}")

    lines.extend(
        ["", "## KEY DEVELOPMENTS（出典付き；HIGH のみ）", ""]
    )
    lines.extend(milspec_prose.build_key_developments_lines(threats))

    lines.extend(["## ELEVATED CII（combinedScore ≥ 55）", ""])
    cii = threats.get("elevated_cii_regions") or []
    if cii:
        for row in cii[:10]:
            lines.append(
                f"- {row.get('region')}: score={row.get('combinedScore')} "
                f"({row.get('trend', '')}) — [出典: World Monitor risk_scores]"
            )
    else:
        lines.append("_該当リージョンなし_")

    lines.extend(["", "## SHINKA MILSPEC（evidence_blocks）", ""])
    lines.extend(milspec_prose.build_shinka_evidence_lines(fusion))

    enrich = enrichment or {}
    if enrich:
        lines.extend(milspec_prose.build_egov_citations_block(enrich))
        lines.extend(milspec_prose.build_gov_feeds_block(enrich))
        lines.extend(milspec_prose.build_backfill_notes_block(enrich))
        lines.extend(milspec_prose.build_github_provenance_block(enrich))

    exec_meta = executive_summary_meta or {}
    if llm_summary:
        lines.extend(["", "## EXECUTIVE SUMMARY（LLM / 一次資料規律）", "", llm_summary])
        if exec_meta.get("provider_id"):
            lines.append(
                f"\n_モデル: {exec_meta.get('provider_id')}/{exec_meta.get('model')}; "
                f"milspec_primary_source_rule=true_"
            )
    elif exec_meta.get("skipped"):
        lines.extend(
            [
                "",
                "## EXECUTIVE SUMMARY（LLM）",
                "",
                f"_スキップ: {exec_meta.get('reason', 'LLM未使用')}_",
            ]
        )

    lines.extend(milspec_prose.build_provenance_section(fusion, threats, enrichment=enrich))

    lines.extend(["## NEXT 24h WATCHLIST（根拠トレース可能項目のみ）", ""])
    for item in milspec_prose.derive_watchlist(threats, fusion):
        lines.append(item)

    lines.extend(
        [
            "",
            "---",
            "_World Monitor OSINT + ShinkaEvolve MILSPEC — Hermes cron — 一次資料規律適用_",
        ]
    )
    return "\n".join(lines)


def generate_situation_report(
    *,
    slot: str = "morning",
    topic: str = "日本の安全保障と世界情勢",
    country_code: str = "JP",
    max_scenarios: int = 4,
    source_mode: str = "mock",
    wm_tier: str = "auto",
    llm_summary: bool = False,
    save: bool = True,
    window_hours: int = 24,
    use_primary_backfill: bool = True,
    fetch_egov: bool = True,
    fetch_github: bool = True,
    fetch_gov_feeds: bool = True,
    max_headline_backfill: int = 5,
) -> dict[str, Any]:
    fusion = core.fusion_report(
        topic=topic,
        country_code=country_code,
        max_scenarios=max(1, min(max_scenarios, 8)),
        source_mode=source_mode,
        save_report=False,
        wm_tier=wm_tier,
        llm_summary=False,
    )
    wm = fusion.get("worldmonitor") or {}
    threats = extract_high_threats(wm)

    enrichment: dict[str, Any] = {}
    if use_primary_backfill:
        enrichment = primary_backfill.enrich_primary_sources(
            threats,
            topic=topic,
            max_headline_backfill=max_headline_backfill,
            fetch_egov=fetch_egov,
            fetch_github=fetch_github,
            fetch_gov_feeds=fetch_gov_feeds,
        )
        threats = primary_backfill.apply_enrichment_to_threats(threats, enrichment)

    exec_meta: dict[str, Any] = {}
    llm_text = ""
    if llm_summary:
        exec_meta = milspec_prose.synthesize_pdb_executive_summary(
            topic=topic,
            slot=slot,
            threats=threats,
            fusion=fusion,
            enrichment=enrichment,
        )
        llm_text = milspec_prose.extract_executive_summary_text(exec_meta)
        fusion["pdb_executive_summary"] = exec_meta

    markdown = build_pdb_markdown(
        slot=slot,
        fusion=fusion,
        threats=threats,
        window_hours=window_hours,
        llm_summary=llm_text,
        executive_summary_meta=exec_meta,
        enrichment=enrichment,
    )

    payload: dict[str, Any] = {
        "success": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "slot": slot,
        "window_hours": window_hours,
        "topic": topic,
        "source_mode": source_mode,
        "wm_tier": wm_tier,
        "milspec_primary_source_rule": True,
        "primary_enrichment": enrichment or None,
        "high_threat_digest": threats,
        "fusion": fusion,
        "pdb_executive_summary": exec_meta or None,
        "markdown": markdown,
    }

    if save:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = re.sub(r"[^\w\-]+", "_", f"{slot}_{topic}"[:48]).strip("_") or slot
        json_path = _reports_dir() / f"{stamp}_{slug}.json"
        md_path = _reports_dir() / f"{stamp}_{slug}.md"
        json_path.write_text(_json(payload), encoding="utf-8")
        md_path.write_text(markdown + "\n", encoding="utf-8")
        payload["saved_json"] = str(json_path)
        payload["saved_markdown"] = str(md_path)

    return payload


def run_for_cron_stdout(slot: str, **kwargs: Any) -> int:
    """Generate report and print markdown for no-agent cron delivery."""
    save = kwargs.pop("save", True)
    result = generate_situation_report(slot=slot, save=save, **kwargs)
    print(result.get("markdown") or "")
    return 0 if result.get("success") else 1
