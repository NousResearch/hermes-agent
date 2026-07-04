import argparse
import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = PROJECT_ROOT / "skills" / "research" / "bilibili-hermes-agent-weekly-top-feishu"
FETCH_SCRIPT = SKILL_DIR / "scripts" / "fetch_bilibili_hermes_agent_weekly_top.py"
SETUP_SCRIPT = SKILL_DIR / "scripts" / "setup_daily_bilibili_hermes_agent_top.py"
CN_TZ = timezone(timedelta(hours=8), "Asia/Shanghai")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def ts(year: int, month: int, day: int, hour: int = 12) -> int:
    return int(datetime(year, month, day, hour, tzinfo=CN_TZ).timestamp())


def test_parse_int_and_clean_html():
    mod = load_module(FETCH_SCRIPT, "bilibili_fetch_helpers")

    assert mod.parse_int("1.2万") == 12000
    assert mod.parse_int("3亿") == 300000000
    assert mod.clean_html('【<em class="keyword">Hermes</em> <em>Agent</em>】') == "【Hermes Agent】"


def test_rank_weekly_videos_filters_dedupes_and_sorts():
    mod = load_module(FETCH_SCRIPT, "bilibili_fetch_rank")
    now = datetime(2026, 6, 14, 23, 40, tzinfo=CN_TZ)
    week_start, _ = mod.week_window(now)

    videos = [
        {
            "bvid": "BV_OLD",
            "title": "Hermes Agent old high play",
            "play": 999999,
            "like": 1,
            "favorites": 1,
            "pubdate": ts(2026, 6, 7),
            "url": "https://www.bilibili.com/video/BV_OLD/",
        },
        {
            "bvid": "BV_IRRELEVANT",
            "title": "Agent workflow without target product",
            "tags": "Hermes",
            "play": 800000,
            "like": 1,
            "favorites": 1,
            "pubdate": ts(2026, 6, 12),
            "url": "https://www.bilibili.com/video/BV_IRRELEVANT/",
        },
        {
            "bvid": "BV_DUP",
            "title": "Hermes Agent intro",
            "play": 100,
            "like": 10,
            "favorites": 2,
            "pubdate": ts(2026, 6, 12),
            "url": "https://www.bilibili.com/video/BV_DUP/",
            "source_queries": [{"keyword": "HermesAgent", "order": "pubdate", "page": 1}],
        },
        {
            "bvid": "BV_DUP",
            "title": "Hermes Agent intro",
            "play": 700,
            "like": 9,
            "favorites": 3,
            "pubdate": ts(2026, 6, 12),
            "url": "https://www.bilibili.com/video/BV_DUP/",
            "source_queries": [{"keyword": "Hermes Agent", "order": "click", "page": 1}],
        },
        {
            "bvid": "BV_SECOND",
            "title": "Hermes Agent deep dive",
            "play": 500,
            "like": 20,
            "favorites": 9,
            "pubdate": ts(2026, 6, 13),
            "url": "https://www.bilibili.com/video/BV_SECOND/",
        },
    ]

    ranked = mod.rank_weekly_videos(videos, week_start=week_start, now=now, limit=10)

    assert [item["bvid"] for item in ranked] == ["BV_DUP", "BV_SECOND"]
    assert ranked[0]["rank"] == 1
    assert ranked[0]["play"] == 700
    assert len(ranked[0]["source_queries"]) == 2


def test_build_digest_with_mocked_bilibili_response(monkeypatch):
    mod = load_module(FETCH_SCRIPT, "bilibili_fetch_digest")

    def fake_fetch_search_page(**kwargs):
        assert kwargs["keyword"] == "HermesAgent"
        return {
            "code": 0,
            "data": {
                "result": [
                    {
                        "type": "video",
                        "bvid": "BV123",
                        "aid": 123,
                        "title": '<em class="keyword">Hermes</em> Agent 上手指南',
                        "description": "Hermes Agent tutorial",
                        "author": "tester",
                        "play": "1.2万",
                        "like": 30,
                        "favorites": 10,
                        "review": 4,
                        "video_review": 5,
                        "pubdate": ts(2026, 6, 14, 9),
                        "duration": "5:20",
                        "typename": "计算机技术",
                        "tag": "HermesAgent,AI",
                        "hit_columns": ["title", "tag"],
                    }
                ]
            },
        }

    monkeypatch.setattr(mod, "fetch_search_page", fake_fetch_search_page)
    args = argparse.Namespace(
        limit=10,
        pages=1,
        page_size=20,
        keyword=["HermesAgent"],
        order=["click"],
        now="2026-06-14T23:40:00+08:00",
        server_time_filter=False,
    )

    digest = mod.build_digest(args)

    assert digest["matched_count"] == 1
    assert digest["items"][0]["title"] == "Hermes Agent 上手指南"
    assert digest["items"][0]["play"] == 12000
    assert digest["items"][0]["url"] == "https://www.bilibili.com/video/BV123/"
    assert digest["items"][0]["source_queries"] == [{"keyword": "HermesAgent", "order": "click", "page": 1}]


def test_setup_installs_runtime_script_and_creates_cron_job(monkeypatch, capsys):
    mod = load_module(SETUP_SCRIPT, "bilibili_setup")
    hermes_home = Path(os.environ["HERMES_HOME"])
    (hermes_home / "cron").mkdir(exist_ok=True)
    (hermes_home / "cron" / "output").mkdir(parents=True, exist_ok=True)
    (hermes_home / "scripts").mkdir(exist_ok=True)

    import cron.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    assert mod.main(["--deliver", "local", "--skip-skill-install", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["success"] is True
    assert output["job"]["name"] == mod.JOB_NAME
    assert output["job"]["schedule_display"] == "40 23 * * *"
    assert output["job"]["deliver"] == "local"
    assert output["job"]["skills"] == [mod.SKILL_NAME]
    assert output["job"]["script"] == mod.RUNTIME_FETCH_SCRIPT
    assert (hermes_home / "scripts" / mod.RUNTIME_FETCH_SCRIPT).exists()
