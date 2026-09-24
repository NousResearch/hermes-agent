from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "scripts" / "social_comment_pipeline.py"
WATCH = ROOT / "scripts" / "social_comment_watch.py"


def test_social_comment_pipeline_generates_archives_and_kanban_commands(tmp_path: Path) -> None:
    input_file = tmp_path / "douyin_comments.jsonl"
    input_file.write_text(
        "\n".join(
            [
                json.dumps({"platform": "douyin", "post_id": "p1", "id": "c1", "text": "希望支持导出 Excel 报表，手工统计太麻烦了"}, ensure_ascii=False),
                json.dumps({"platform": "douyin", "post_id": "p1", "id": "c2", "text": "登录验证码经常失败，账号打不开"}, ensure_ascii=False),
                json.dumps({"platform": "xiaohongshu", "post_id": "n1", "id": "c3", "text": "搜索筛选太复杂，不知道怎么找到历史内容"}, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )
    out_root = tmp_path / "runs"

    proc = subprocess.run(
        [
            sys.executable,
            str(PIPELINE),
            "--input",
            str(input_file),
            "--output",
            str(out_root),
            "--run-name",
            "case1",
            "--dry-run-kanban",
            "--max-requirements",
            "2",
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )

    summary = json.loads(proc.stdout)
    run_dir = out_root / "case1"
    assert summary["comment_count"] == 3
    assert summary["insight_count"] >= 2
    assert summary["task_count"] == 8
    assert (run_dir / "normalized_comments.jsonl").exists()
    assert (run_dir / "insights.md").read_text(encoding="utf-8").startswith("# 社交平台评论需求洞察")
    assert "给产品经理 Agent" in (run_dir / "product_manager_brief.md").read_text(encoding="utf-8")

    tasks = json.loads((run_dir / "kanban_tasks.json").read_text(encoding="utf-8"))
    assert {task["role"] for task in tasks} == {"product_manager", "developer", "tester", "acceptance"}
    dispatch = json.loads((run_dir / "kanban_dispatch_results.json").read_text(encoding="utf-8"))
    assert dispatch and dispatch[0]["dry_run"] is True
    assert dispatch[0]["command"][:3] == ["hermes", "kanban", "create"]


def test_social_comment_watch_processes_new_files_once(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    export = inbox / "comments.csv"
    export.write_text(
        "platform,post_id,id,text\n"
        "weibo,p1,c1,希望增加通知筛选功能\n"
        "weibo,p1,c2,消息提醒太多很麻烦\n",
        encoding="utf-8",
    )
    out_root = tmp_path / "archive"
    state = tmp_path / "state.json"

    cmd = [
        sys.executable,
        str(WATCH),
        "--input-dir",
        str(inbox),
        "--output",
        str(out_root),
        "--state",
        str(state),
        "--dry-run-kanban",
    ]
    first = subprocess.run(cmd, text=True, capture_output=True, timeout=30, check=True)
    assert "处理 comments.csv" in first.stdout
    assert state.exists()
    assert list(out_root.iterdir())

    second = subprocess.run(cmd, text=True, capture_output=True, timeout=30, check=True)
    assert second.stdout == ""
