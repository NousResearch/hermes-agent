#!/usr/bin/env python3
"""Weekly Notion Patrol cron wrapper.

Read-only: invokes the local notion-patrol Node script, summarizes the CSV,
and prints a Slack-ready notification. No Notion writes are performed here.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MENTION = "<!subteam^S02FLCBKU0P>"
ROOT_IDS = [
    "106173b008788028ac4efd380a88308c",
    "8db170cdc5ba4c488ef9302e4e58cede",
]
ROOT_LABELS = [
    "法務内マニュアル（106173b008788028ac4efd380a88308c）",
    "法務関連DB（8db170cdc5ba4c488ef9302e4e58cede）",
]

HOME = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()
SKILL_DIR = HOME / "skills" / "productivity" / "notion-patrol"
PATROL_JS = SKILL_DIR / "scripts" / "patrol.js"
OUTPUT_DIR = HOME / "cron" / "output" / "notion-patrol"


def now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def clip(text: str, limit: int = 1200) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...（以下省略）"


def latest_csv_path(stdout: str) -> Path | None:
    for line in reversed((stdout or "").splitlines()):
        if line.startswith("CSV:"):
            return Path(line.split(":", 1)[1].strip())
    files = sorted(OUTPUT_DIR.glob("link_check_test_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def summarize(csv_path: Path) -> str:
    rows = read_rows(csv_path)
    total = len(rows)
    ok = sum(1 for r in rows if r.get("判定") == "OK")
    ng_rows = [r for r in rows if r.get("判定") == "NG"]
    lines = [
        MENTION,
        "【Notionパトロール君】週次リンクチェック結果",
        f"実行日時: {now_text()}",
        "対象:",
        *[f"- {label}" for label in ROOT_LABELS],
        f"チェック件数: {total}",
        f"OK: {ok}",
        f"NG: {len(ng_rows)}",
        f"CSV: {csv_path}",
        f"MEDIA:{csv_path}",
    ]
    if ng_rows:
        lines.append("")
        lines.append("NGリンク（最大20件）:")
        for r in ng_rows[:20]:
            lines.append(
                f"- [{r.get('ステータスコード','')}] {r.get('URL','')} / "
                f"{r.get('ページ名','')} / {clip(r.get('Context（文脈）',''), 160)}"
            )
        if len(ng_rows) > 20:
            lines.append(f"...ほか {len(ng_rows) - 20} 件（CSV参照）")
    else:
        lines.append("NGリンクは検出されませんでした。")
    return "\n".join(lines)


def test_notification() -> str:
    return "\n".join([
        MENTION,
        "【Notionパトロール君】週次リンクチェック結果（テスト通知）",
        f"実行日時: {now_text()}",
        "対象:",
        *[f"- {label}" for label in ROOT_LABELS],
        "チェック件数: 12",
        "OK: 11",
        "NG: 1",
        "CSV: /home/kawazoe_taishi/.hermes/cron/output/notion-patrol/link_check_test_YYYYMMDD.csv",
        "",
        "NGリンク（最大20件）:",
        "- [404] https://example.invalid/dead-link / サンプルページ / サンプル文脈",
        "※これは通知形式確認用のテスト通知です。実際のNotion巡回は実行していません。",
    ])


def run_patrol() -> str:
    if not PATROL_JS.exists():
        return "\n".join([
            MENTION,
            "【Notionパトロール君】週次リンクチェック失敗",
            f"実行日時: {now_text()}",
            f"理由: patrol.js が見つかりません: {PATROL_JS}",
        ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["node", str(PATROL_JS), "--output-dir", str(OUTPUT_DIR)]
    for root in ROOT_IDS:
        cmd.extend(["--root", root])

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(SKILL_DIR),
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("NOTION_PATROL_TIMEOUT_SECONDS", "110")),
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return "\n".join([
            MENTION,
            "【Notionパトロール君】週次リンクチェック失敗",
            f"実行日時: {now_text()}",
            "理由: タイムアウトしました。",
        ])

    if proc.returncode != 0:
        return "\n".join([
            MENTION,
            "【Notionパトロール君】週次リンクチェック失敗",
            f"実行日時: {now_text()}",
            f"終了コード: {proc.returncode}",
            "stderr:",
            clip(proc.stderr, 1800),
            "stdout:",
            clip(proc.stdout, 1800),
        ])

    csv_path = latest_csv_path(proc.stdout)
    if not csv_path or not csv_path.exists():
        return "\n".join([
            MENTION,
            "【Notionパトロール君】週次リンクチェック失敗",
            f"実行日時: {now_text()}",
            "理由: CSV出力を確認できませんでした。",
            "stdout:",
            clip(proc.stdout, 1800),
        ])

    return summarize(csv_path)


def main(argv: list[str]) -> int:
    if "--test-notification" in argv:
        print(test_notification())
        return 0
    print(run_patrol())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

