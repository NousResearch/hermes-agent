#!/usr/bin/env python3
"""わらしべ: 日本Xで話題になりやすい穴場カテゴリの公開価格スキャン (no_agent cron).

- 購入・ログイン・出品なし
- CloakBrowser経由の公開ページのみ (hermes warashibe price)
- カテゴリ回転で負荷を抑える (1回あたり最大 KEYWORDS_PER_RUN 件)
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# --- config ---
JST = ZoneInfo("Asia/Tokyo")
KEYWORDS_PER_RUN = 4
LIMIT = 6
PLATFORMS = ["mercari"]  # 速度優先。必要なら yahoo_auction を追加
TIMEOUT_SEC = 180
HERMES_CMD = os.environ.get("HERMES_BIN", "hermes")
OUT_DIR = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "warashibe" / "price-scans"

# カテゴリ別キーワード (X話題・穴場寄り。具体商品名を優先)
CATEGORY_POOL: dict[str, list[str]] = {
    "gpu": [
        "RTX 3060 グラボ",
        "RTX 4060 単体",
        "RTX 4070 グラボ",
        "RX 7600 グラボ",
        "RTX 5060 グラボ",
        "RTX 5070 グラボ",
    ],
    "gunpla": [
        "MG エピオン",
        "プレバン ガンプラ",
        "ブラックナイトスコード ガンプラ",
        "RG サザビー",
        "MGEX ユニコーン",
        "ガンプラ 完成品",
    ],
    "pokeka": [
        "ポケカ SAR",
        "ニンフィアex SAR",
        "ブラッキーex SAR",
        "ポケカ スタートデッキ",
        "レックウザVMAX",
        "ポケカ 未開封 ボックス",
    ],
    "lefty_iron": [
        "左利き アイアン ゴルフ",
        "レフティ アイアン XXIO",
        "左利き ゴルフ クラブ セット",
        "レフティ ゼクシオ",
        "左利き用 はさみ",
    ],
    "niche": [
        "カグラバチ カード",
        "メタキラカード",
        "左利き マウス",
        "左利き 包丁",
        "ベイブレード 限定",
        "プレバン 限定",
    ],
}

CATEGORY_ORDER = ["gpu", "gunpla", "pokeka", "lefty_iron", "niche"]


def _pick_keywords(now: datetime) -> list[tuple[str, str]]:
    """日次回転: 各カテゴリから均等に選び、合計 KEYWORDS_PER_RUN 件。"""
    day_index = int(now.strftime("%Y%m%d"))
    hour_bucket = now.hour // 12  # 0=午前枠, 1=午後枠
    seed = day_index * 2 + hour_bucket
    picks: list[tuple[str, str]] = []
    for offset, cat in enumerate(CATEGORY_ORDER):
        pool = CATEGORY_POOL[cat]
        kw = pool[(seed + offset) % len(pool)]
        picks.append((cat, kw))
    return picks[:KEYWORDS_PER_RUN]


def _run_price(keyword: str) -> dict:
    cmd = [
        HERMES_CMD,
        "warashibe",
        "price",
        "--keyword",
        keyword,
        "--limit",
        str(LIMIT),
        "--platforms",
        *PLATFORMS,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SEC,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        return {"keyword": keyword, "error": f"timeout>{TIMEOUT_SEC}s"}
    except FileNotFoundError:
        return {"keyword": keyword, "error": f"command not found: {HERMES_CMD}"}

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return {
            "keyword": keyword,
            "error": f"exit={proc.returncode}",
            "stderr": err[:500],
            "stdout": out[:500],
        }
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"keyword": keyword, "error": "invalid_json", "stdout": out[:800]}


def _prices_from_result(payload: dict) -> list[int]:
    prices: list[int] = []
    for block in payload.get("results") or []:
        for item in block.get("items") or []:
            p = item.get("price")
            if isinstance(p, int) and p > 0:
                prices.append(p)
    return prices


def _sample_titles(payload: dict, n: int = 3) -> list[str]:
    titles: list[str] = []
    for block in payload.get("results") or []:
        for item in block.get("items") or []:
            t = (item.get("title") or "").strip()
            p = item.get("price")
            if t:
                titles.append(f"{t} / ¥{p:,}" if isinstance(p, int) else t)
            if len(titles) >= n:
                return titles
    return titles


def _fmt_yen(v: int | None) -> str:
    return f"¥{v:,}" if isinstance(v, int) else "—"


def main() -> int:
    now = datetime.now(JST)
    picks = _pick_keywords(now)
    scans: list[dict] = []

    for cat, kw in picks:
        payload = _run_price(kw)
        prices = _prices_from_result(payload) if "error" not in payload else []
        scans.append(
            {
                "category": cat,
                "keyword": kw,
                "payload": payload,
                "count": len(prices),
                "min": min(prices) if prices else None,
                "median": int(statistics.median(prices)) if prices else None,
                "max": max(prices) if prices else None,
                "samples": _sample_titles(payload),
            }
        )
        time.sleep(1.0)  # 軽い間隔

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    raw_path = OUT_DIR / f"scan-{stamp}.json"
    raw_path.write_text(
        json.dumps(
            {
                "retrieved_at": now.isoformat(),
                "platforms": PLATFORMS,
                "picks": [{"category": c, "keyword": k} for c, k in picks],
                "scans": scans,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cat_label = {
        "gpu": "GPU/グラボ",
        "gunpla": "ガンプラ",
        "pokeka": "ポケカ",
        "lefty_iron": "左利き/レフティ",
        "niche": "穴場その他",
    }

    lines = [
        f"🛒 わらしべ穴場価格スキャン {now.strftime('%Y-%m-%d %H:%M')} JST",
        f"公開相場のみ / 購入なし / 保存: `{raw_path}`",
        "",
        "対象カテゴリ: GPU・ガンプラ・ポケカ・左利きアイアン/用品・その他穴場",
        "選定: 日本Xで話題・転売議論が出やすい語を日次回転",
        "",
    ]

    hit_any = False
    for s in scans:
        hit_any = hit_any or (s["count"] > 0)
        label = cat_label.get(s["category"], s["category"])
        lines.append(f"### {label}: `{s['keyword']}`")
        if "error" in (s.get("payload") or {}):
            lines.append(f"- 失敗: {s['payload'].get('error')}")
            if s["payload"].get("stderr"):
                lines.append(f"- stderr: {s['payload']['stderr'][:160]}")
        else:
            lines.append(
                f"- 件数 {s['count']} / 最安 {_fmt_yen(s['min'])} / 中央 {_fmt_yen(s['median'])} / 最高 {_fmt_yen(s['max'])}"
            )
            for t in s["samples"]:
                lines.append(f"  - {t}")
            if s["count"] == 0:
                lines.append("  - (一覧取得0件 — セレクタ変更・bot検知・在庫薄の可能性)")
        lines.append("")

    lines.extend(
        [
            "はくあメモ:",
            "- ポケカ/ガンプラは相場透明化が進み、**価値誤認出品**と完成品/箱傷限定が穴場寄り",
            "- GPU中古は1万円超で故障リスク大。型番+保証/動作確認を必須に",
            "- 左利きアイアン(XXIOレフティ等)は供給薄で競合少なめ、回転は遅め",
            "- 仕入れ判断は古物商ルール・規約順守。無在庫転売は対象外",
            "",
            "再実行: `hermes warashibe price -k \"キーワード\" --limit 6 --platforms mercari`",
        ]
    )

    print("\n".join(lines))
    # no_agent: 空stdoutは配信なし。失敗でもダイジェストは出す
    return 0 if hit_any or all("error" not in (s.get("payload") or {}) for s in scans) else 1


if __name__ == "__main__":
    sys.exit(main())
