#!/usr/bin/env python3
"""わらしべ 毎時 黒字ルート監視 (no_agent).

- 予算 10,000円スタート
- ガンプラ: 未完成品/箱/ジャンク/完成品
- GPU: 8GB/12GB/16GB/24GB 高VRAM帯
- 左利きゴルフ/ポケカ/プレバン
- 日本安 -> eBay高 (輸出プレミアム)
- KPI: 利益>=¥500 かつ 利益率>=30%
- no_agent: 黒字0件なら stdout 空（配信なし）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
HERMES_CMD = os.environ.get("HERMES_BIN", "hermes")
OUT_DIR = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "warashibe" / "arb-scans"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1万円スタート・最近の売れ筋・高VRAM・ガンプラ未完成品/箱
KEYWORD_POOL = [
    "ガンプラ 未完成品",
    "ガンプラ 箱",
    "ガンプラ ジャンク",
    "ガンプラ 完成品",
    "RTX 3060 12GB",
    "RTX 3070 8GB",
    "RTX 3080 10GB",
    "RTX 3090 24GB",
    "RTX 4060 8GB",
    "RTX 4070 12GB",
    "RTX 4080 16GB",
    "RTX 4090 24GB",
    "RTX 5060 8GB",
    "RTX 5070 12GB",
    "RTX 5080 16GB",
    "RTX 5090 32GB",
    "RX 6800 16GB",
    "RX 7800 16GB",
    "RX 7900 24GB",
    "ポケモンカード BOX",
    "ポケカ プロモ",
    "レフティ アイアン ゴルフ",
    "レフティ ゼクシオ",
    "プレバン ガンプラ",
]


def run_arb(keyword: str) -> dict | None:
    """Run hermes warashibe arb for one keyword. Returns parsed JSON or None."""
    budget = os.environ.get("WARASHIBE_ARB_BUDGET", "80000")
    limit = os.environ.get("WARASHIBE_ARB_LIMIT", "6")
    platforms = os.environ.get(
        "WARASHIBE_PLATFORMS",
        "mercari,yahoo_auction,ebay,amazon_jp",
    )
    plat_list = [p.strip() for p in platforms.split(",") if p.strip()]

    cmd = [
        HERMES_CMD,
        "warashibe",
        "arb",
        "-k",
        keyword,
        "--platforms",
        *plat_list,
        "--budget",
        str(budget),
        "--limit",
        str(limit),
        "--min-profit",
        os.environ.get("WARASHIBE_MIN_PROFIT", "500"),
        "--min-rate",
        os.environ.get("WARASHIBE_MIN_RATE", "0.3"),
    ]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return {"keyword": keyword, "error": str(e), "winners": [], "combos": []}

    if r.returncode != 0:
        return {
            "keyword": keyword,
            "error": f"exit={r.returncode}",
            "stderr": (r.stderr or "")[:500],
            "winners": [],
            "combos": [],
        }
    try:
        data = json.loads((r.stdout or "").strip() or "{}")
    except json.JSONDecodeError:
        return {
            "keyword": keyword,
            "error": "invalid_json",
            "stdout": (r.stdout or "")[:500],
            "winners": [],
            "combos": [],
        }
    if isinstance(data, dict) and "keyword" not in data:
        data["keyword"] = keyword
    return data


def _fmt_yen(v) -> str:
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return str(v)


def _winners_from(res: dict) -> list[dict]:
    """Extract winners that pass go=True KPI."""
    if not isinstance(res, dict):
        return []
    winners = res.get("winners")
    if isinstance(winners, list) and winners:
        return [w for w in winners if w.get("go", True)]
    combos = res.get("combos") or []
    return [c for c in combos if c.get("go")]


def _market_summary(res: dict) -> list[str]:
    lines: list[str] = []
    market = res.get("market") or {}
    results = market.get("results") if isinstance(market, dict) else None
    if not results:
        results = res.get("results") or []
    for block in results or []:
        plat = block.get("platform") or block.get("platform_name") or "?"
        items = block.get("items") or []
        prices = [i.get("price") for i in items if isinstance(i.get("price"), (int, float))]
        if not prices and block.get("error"):
            lines.append(f"- {plat}: SKIP/ERR ({str(block.get('error'))[:80]})")
            continue
        if not prices:
            lines.append(f"- {plat}: n=0")
            continue
        prices_i = [int(p) for p in prices]
        mid = sorted(prices_i)[len(prices_i) // 2]
        lines.append(
            f"- {plat}: n={len(prices_i)} 最安¥{_fmt_yen(min(prices_i))} 中央¥{_fmt_yen(mid)}"
        )
    return lines


def main() -> int:
    now = datetime.now(JST)
    n = int(os.environ.get("WARASHIBE_ARB_KEYWORDS", "2"))
    seed = now.hour  # rotate hourly
    kws = [KEYWORD_POOL[(seed + i) % len(KEYWORD_POOL)] for i in range(max(1, n))]

    env_kws = os.environ.get("WARASHIBE_KEYWORDS", "").strip()
    if env_kws:
        kws = [k.strip() for k in env_kws.split(",") if k.strip()][: max(1, n)]

    reports: list[dict] = []
    all_winners: list[dict] = []
    picks: list[list[str]] = []

    for i, kw in enumerate(kws):
        res = run_arb(kw) or {"keyword": kw, "winners": [], "combos": [], "error": "null"}
        res["_category"] = "rotated"
        reports.append(res)
        wins = _winners_from(res)
        for w in wins:
            w = dict(w)
            w.setdefault("keyword", kw)
            all_winners.append(w)
        picks.append(["rotated", kw])
        if i + 1 < len(kws):
            time.sleep(1.0)

    budget = os.environ.get("WARASHIBE_ARB_BUDGET", "80000")
    stamp = now.strftime("%Y%m%d-%H%M%S")
    md_path = OUT_DIR / f"arb-{stamp}.md"

    # no_agent は stdout が空だとTelegramへ配信されないため、候補なしも通知する。
    if not all_winners:
        note = (
            f"💹 わらしべ黒字ルート {now.strftime('%Y-%m-%d %H:%M')} JST\n"
            f"候補なし（調査: {' / '.join(kws)}）\n"
            f"KPI: 利益≥¥500 かつ 利益率≥30% / 予算: ¥{budget}\n"
            f"保存: {md_path}\n"
            "公開相場のみ・購入なし。"
        )
        md_path.write_text(note + "\n", encoding="utf-8")
        print(note)
        return 0

    # Sort winners by profit desc
    all_winners.sort(key=lambda w: float(w.get("profit") or 0), reverse=True)

    lines = [
        f"💹 わらしべ黒字ルート速報 {now.strftime('%Y-%m-%d %H:%M')} JST",
        f"候補 {len(all_winners)}件 / 公開相場のみ・購入なし",
        f"保存: {md_path}",
        "",
    ]

    for w in all_winners:
        kw = w.get("keyword") or "?"
        buy_p = w.get("buy_platform") or "?"
        sell_p = w.get("sell_platform") or "?"
        buy_price = w.get("buy_price")
        sell_price = w.get("sell_price")
        profit = w.get("profit")
        rate = float(w.get("profit_rate") or 0) * 100
        ship = w.get("shipping_out_est")
        fee = w.get("platform_fee")
        title = w.get("buy_title") or ""
        url = w.get("buy_url") or ""

        lines.append(f"{kw}  [{buy_p} → {sell_p}]")
        lines.append(
            f"- 仕入¥{_fmt_yen(buy_price)} → 想定売¥{_fmt_yen(sell_price)} / "
            f"利益¥{_fmt_yen(profit)} ({rate:.1f}%)  【KPI: 利益≥¥500 かつ 利益率≥30%】"
        )
        if ship is not None or fee is not None:
            lines.append(
                f"- 送料概算¥{_fmt_yen(ship or 0)} / 手数料¥{_fmt_yen(fee or 0)}"
            )
        if title:
            lines.append(f"- 仕入候補: {title}")
        if url:
            lines.append(f"- {url}")
        lines.append("")

    lines.append("市場サマリ:")
    for res in reports:
        kw = res.get("keyword") or "?"
        lines.append(kw)
        ms = _market_summary(res)
        if ms:
            lines.extend(ms)
        elif res.get("error"):
            lines.append(f"- error: {res.get('error')}")
        else:
            lines.append("- (市場詳細なし)")
        lines.append("")

    lines.extend(
        [
            "注意: 送料・関税・状態差・規約で利益は変動。Amazonは公式API未設定ならスキップ。",
            "仕入れ実行は人手確認後のみ。",
        ]
    )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
