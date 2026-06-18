#!/usr/bin/env python3
"""Aggregate high-threat WM JSON + Deep Research + Shinka MILSPEC into a forecast report."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Domain → Shinka scenario slugs (milspec_security_jp)
DOMAIN_SCENARIOS: dict[str, list[str]] = {
    "認知戦": [
        "cognitive_warfare_response",
        "disinformation_detection",
        "strategic_communication",
    ],
    "経済安全保障": [
        "supply_chain_focus",
        "tech_transfer_control",
        "national_security_framework",
    ],
    "AI": [
        "ai_defense_ethics",
        "ai_risk_classification",
        "laws_prohibition_detail",
    ],
    "日本の安全保障": [
        "taiwan_contingency_overview",
        "southwest_islands_defense",
        "cyber_defense_posture",
        "russia_nuclear_threat",
    ],
    "中東情勢": [
        "national_security_framework",
        "supply_chain_focus",
    ],
}

DEEP_RESEARCH: dict[str, dict[str, Any]] = {
    "中東・ホルムズ": {
        "summary_ja": (
            "2026年6月17日時点、米国とイランは暫定合意（スイスで金曜署名予定）で軍事行動停止を目指すが、"
            "公開文書が未開示のため解釈が分岐。イランは「イスラエルのレバノン撤退」を条件と主張し、"
            "米側は撤退義務なしと説明。ホルムズ海峡の再開・制裁緩和・ウラン希釈は60日交渉枠の中核。"
            "合意破綻時はエネルギー・海運リスクが再燃し、日本のエネルギー輸入と経済安全保障に直撃する。"
        ),
        "sources": [
            "https://www.pbs.org/newshour/world/iran-says-the-deal-to-end-the-war-with-the-u-s-requires-israel-to-withdraw-from-lebanon",
            "https://www.bbc.com/news/articles/ce8mv6l6eezo",
            "https://www.dw.com/en/us-iran-hezbollah-spar-over-murky-terms-of-ceasefire-deal/live-77573728",
        ],
        "confidence": "HIGH",
        "horizon": "0-60日（暫定枠）",
    },
    "認知戦・偽情報": {
        "summary_ja": (
            "日本は2026年5月、国家情報会議（NIC）・国家情報局（NIB）設立法を成立し、"
            "外圧による偽情報・影響工作への政府横断対応を強化。防衛省はAI活用OSINT・SNS真偽判定・"
            "将来予測機能を2027年度までに整備する方針。内閣官房の外国関連偽情報ポータル事例では、"
            "自衛隊・艦艇沈没等のデマが拡散。G7迅速対応メカニズムとの連携が継続課題。"
        ),
        "sources": [
            "https://www.mod.go.jp/en/images/ed8cf86c9f9cad56f540d58d782f0e5dc50bc272.pdf",
            "https://www.straitstimes.com/asia/east-asia/japan-overhauls-post-war-intelligence-system-amid-rising-security-threats",
            "https://www.cas.go.jp/jp/seisaku/boueiryoku_kaigi/sogoteki_dai1/siryou3_e.pdf",
        ],
        "confidence": "HIGH",
        "horizon": "2026-2027（制度実装期）",
    },
    "経済安全保障・半導体": {
        "summary_ja": (
            "日米中の技術覇権競争下、日本の半導体装置輸出（METI 23品目）と中国向け商流・"
            "レアアース依存が同時リスク。MATCH Act等で米国と同等規制への圧力が強まり、"
            "2026年11月前後はBIS関連ルール・希土類輸出規制の再調整ウィンドウ。"
            "経済安保推進法に基づく代替調達・国内生産拡大が急務。"
        ),
        "sources": [
            "https://www.jiia.or.jp/eng/report/2026/06/Outlook2026en07.html",
            "https://www.nippon.com/en/in-depth/d01126/",
            "https://timewell.jp/en/columns/match-act-us-china-semiconductor-export-japan-impact-2026",
        ],
        "confidence": "MODERATE",
        "horizon": "2026下半期（規制・供給網）",
    },
}

ISOLATED_RUNNER = r"""
import importlib.util
import json
import os
import sys
from pathlib import Path

payload = json.loads(sys.stdin.read())
root = Path(os.environ["SHINKA_OSINT_ROOT"]).resolve()
example = (payload.get("arguments") or {}).get("example", "")
example_dir = root / "examples" / example if example else root
sys.path[:0] = [str(example_dir), str(root)]
os.chdir(str(example_dir))
spec = importlib.util.spec_from_file_location("shinka_mcp_server", root / "shinka_mcp_server.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
handler = module.TOOL_HANDLERS[payload["tool"]]
result = handler(payload.get("arguments") or {})
print(json.dumps(result, ensure_ascii=False, default=str))
"""

def _hermes_home() -> Path:
    raw = (os.environ.get("HERMES_HOME") or "").strip()
    return Path(raw) if raw else Path.home() / ".hermes"


def _desktop_dir() -> Path:
    if os.name == "nt":
        try:
            import winreg

            key_path = r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                raw, _ = winreg.QueryValueEx(key, "Desktop")
            return Path(os.path.expandvars(str(raw)))
        except Exception:
            pass
    return Path.home() / "Desktop"


DEFAULT_SHINKA_ROOT = _desktop_dir() / "ShinkaEvolve-OSINT-main" / "ShinkaEvolve-OSINT-main"


def _shinka_root() -> Path:
    for key in ("SHINKA_OSINT_ROOT",):
        val = (os.environ.get(key) or "").strip()
        if val:
            return Path(val)
    cfg = _hermes_home() / "shinka-osint" / "config.json"
    if cfg.is_file():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            root = (data.get("root") or "").strip()
            if root:
                return Path(root)
        except json.JSONDecodeError:
            pass
    return DEFAULT_SHINKA_ROOT


def _python_argv() -> list[str]:
    override = (os.environ.get("SHINKA_OSINT_PYTHON") or "").strip()
    if override:
        return override.split()
    for argv in (["py", "-3"], [sys.executable]):
        try:
            proc = subprocess.run(
                [*argv, "-c", "import anthropic"],
                capture_output=True,
                timeout=20,
            )
            if proc.returncode == 0:
                return argv
        except Exception:
            continue
    return ["py", "-3"]


def shinka_call(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    root = _shinka_root()
    example = arguments.get("example") or "milspec_security_jp"
    example_dir = root / "examples" / example
    env = os.environ.copy()
    env["SHINKA_OSINT_ROOT"] = str(root)
    env["SHINKA_DISABLE_GEMINI_EMBEDDING"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(example_dir), str(root)])
    proc = subprocess.run(
        [*_python_argv(), "-c", ISOLATED_RUNNER],
        input=json.dumps({"tool": tool, "arguments": arguments}),
        cwd=str(example_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        return {"success": False, "error": (proc.stderr or proc.stdout)[-2500:]}
    return json.loads(proc.stdout)


def extract_high_threats(reports_dir: Path) -> dict[str, Any]:
    high_items: list[dict[str, Any]] = []
    cii_high: list[dict[str, Any]] = []
    topic_counts: dict[str, int] = {}

    for p in sorted(reports_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        topic = str(d.get("topic") or "?")
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        wm = d.get("worldmonitor") or {}
        sections = wm.get("sections") or {}
        nd = sections.get("news_digest") or {}
        for cat, block in (nd.get("categories") or {}).items():
            if not isinstance(block, dict):
                continue
            for it in block.get("items") or []:
                th = it.get("threat") or {}
                if th.get("level") != "THREAT_LEVEL_HIGH":
                    continue
                high_items.append(
                    {
                        "report": p.name,
                        "topic": topic,
                        "category": cat,
                        "threat_category": th.get("category"),
                        "title": it.get("title") or "",
                        "url": it.get("url") or it.get("link") or "",
                    }
                )
        rs = sections.get("risk_scores") or {}
        for row in rs.get("ciiScores") or []:
            cs = row.get("combinedScore")
            if cs is not None and float(cs) >= 55:
                cii_high.append(
                    {
                        "region": row.get("region"),
                        "combinedScore": cs,
                        "trend": row.get("trend"),
                        "report": p.name,
                    }
                )

    seen: set[str] = set()
    unique_high: list[dict] = []
    for it in high_items:
        title = it["title"]
        if not title or title in seen:
            continue
        seen.add(title)
        unique_high.append(it)

    by_threat_cat: dict[str, list[str]] = {}
    for it in unique_high:
        cat = str(it.get("threat_category") or "unknown")
        by_threat_cat.setdefault(cat, []).append(it["title"])

    return {
        "reports_scanned": len(list(reports_dir.glob("*.json"))),
        "topic_counts": topic_counts,
        "unique_high_threat_count": len(unique_high),
        "high_threat_headlines": unique_high[:40],
        "high_threat_by_category": by_threat_cat,
        "elevated_cii_regions": sorted(
            {r["region"]: r for r in cii_high}.values(),
            key=lambda x: float(x.get("combinedScore") or 0),
            reverse=True,
        )[:12],
    }


def run_shinka_forecast(topics: list[str]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for topic in topics:
        for sid in DOMAIN_SCENARIOS.get(topic, []):
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            result = shinka_call(
                "shinka_evaluate",
                {
                    "example": "milspec_security_jp",
                    "scenario_id": sid,
                    "source_mode": "mock",
                },
            )
            score = 0.0
            if isinstance(result.get("score"), dict):
                score = float(result["score"].get("total") or 0)
            runs.append(
                {
                    "topic": topic,
                    "scenario_id": sid,
                    "milspec_score": score,
                    "evidence_blocks": result.get("evidence_blocks"),
                    "key_judgments": result.get("key_judgments"),
                    "error": result.get("error"),
                    "raw": result if result.get("error") else None,
                }
            )
    return runs


def build_forecast_narrative(
    digest: dict[str, Any], shinka_runs: list[dict[str, Any]]
) -> str:
    lines = [
        "# 今後の脅威レポート（Shinka OSINT × Deep Research）",
        "",
        f"生成: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 1. JSON高脅威度シグナル（World Monitor fusion）",
        f"- スキャン報告書: {digest.get('reports_scanned')} 件",
        f"- THREAT_LEVEL_HIGH ユニーク見出し: {digest.get('unique_high_threat_count')} 件",
        "",
    ]
    for cat, titles in (digest.get("high_threat_by_category") or {}).items():
        lines.append(f"### {cat}")
        for t in titles[:6]:
            lines.append(f"- {t}")
        lines.append("")

    lines.append("## 2. Deep Research 要約（一次・準一次資料）")
    for name, block in DEEP_RESEARCH.items():
        lines.append(f"### {name}")
        lines.append(block["summary_ja"])
        lines.append(f"- 信頼度: {block['confidence']} / 展望: {block['horizon']}")
        for url in block["sources"][:3]:
            lines.append(f"- 出典: {url}")
        lines.append("")

    lines.append("## 3. Shinka MILSPEC シナリオ評価（mock corpus）")
    for run in shinka_runs:
        sid = run["scenario_id"]
        sc = run["milspec_score"]
        err = run.get("error")
        flag = "⚠" if err else ("🔴" if sc >= 70 else ("🟡" if sc >= 40 else "🟢"))
        lines.append(f"- {flag} `{sid}` score={sc}" + (f" — {err}" if err else ""))

    lines.extend(
        [
            "",
            "## 4. 今後90日の監視優先事項",
            "1. **中東**: 米イラン暫定合意の条文公開とホルムズ通航・レバノン停火の実効性",
            "2. **認知戦**: NIB稼働後の偽情報初動対応と防衛省AI-OSINTパイロット",
            "3. **経済安保**: MATCH Act追随規制・希土類代替調達・半導体装置輸出審査",
            "4. **周辺情勢**: 台湾・南西諸島・GPS妨害（東アジアセル）の同時監視",
            "",
            "## 5. 推奨コマンド（更新）",
            "```powershell",
            "py -3 scripts/osint/generate_future_threat_report.py --llm-summary",
            "py -3 scripts/osint/extract_high_threats.py",
            "```",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    reports_dir = _hermes_home() / "worldmonitor-osint" / "reports"
    out_dir = _hermes_home() / "shinka-osint" / "future_threat_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    digest = extract_high_threats(reports_dir)
    topics = sorted(
        digest.get("topic_counts") or {},
        key=lambda t: digest["topic_counts"][t],
        reverse=True,
    )
    if not topics:
        topics = list(DOMAIN_SCENARIOS.keys())

    shinka_runs = run_shinka_forecast(topics[:5])
    narrative = build_forecast_narrative(digest, shinka_runs)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "success": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": (
            "World Monitor fusion JSON (THREAT_LEVEL_HIGH + CII) + "
            "Deep Research synthesis + ShinkaEvolve MILSPEC mock evaluate"
        ),
        "high_threat_digest": digest,
        "deep_research": DEEP_RESEARCH,
        "shinka_forecast_runs": shinka_runs,
        "narrative_markdown": narrative,
    }

    json_path = out_dir / f"{stamp}_future_threat_forecast.json"
    md_path = out_dir / f"{stamp}_future_threat_forecast.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(narrative + "\n", encoding="utf-8")

    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
