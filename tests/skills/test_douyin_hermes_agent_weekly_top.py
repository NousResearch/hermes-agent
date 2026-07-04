import argparse
import importlib.util
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = PROJECT_ROOT / "skills" / "research" / "douyin-hermes-agent-weekly-top-feishu"
FETCH_SCRIPT = SKILL_DIR / "scripts" / "fetch_douyin_hermes_agent_weekly_top.py"
SETUP_SCRIPT = SKILL_DIR / "scripts" / "setup_daily_douyin_hermes_agent_top.py"
CN_TZ = timezone(timedelta(hours=8), "Asia/Shanghai")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def ts(year: int, month: int, day: int, hour: int = 12) -> int:
    return int(datetime(year, month, day, hour, tzinfo=CN_TZ).timestamp())


def create_firefox_cookie_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE moz_cookies (
                name TEXT,
                value TEXT,
                host TEXT,
                path TEXT,
                expiry INTEGER
            )
            """
        )
        conn.executemany(
            "INSERT INTO moz_cookies (name, value, host, path, expiry) VALUES (?, ?, ?, ?, ?)",
            [
                ("sessionid", "fake_session", ".douyin.com", "/", 2_000_000_000),
                ("ttwid", "fake_ttwid", "www.douyin.com", "/", 2_000_000_000),
                ("expired", "old", ".douyin.com", "/", 1),
                ("other", "skip", ".example.com", "/", 2_000_000_000),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def create_firefox_profile(root: Path) -> Path:
    profile = root / "hm8p2iee.default"
    profile.mkdir(parents=True)
    (root / "profiles.ini").write_text(
        "\n".join(
            [
                "[Profile0]",
                "Name=default",
                "IsRelative=1",
                "Path=hm8p2iee.default",
                "Default=1",
            ]
        ),
        encoding="utf-8",
    )
    create_firefox_cookie_db(profile / "cookies.sqlite")
    return profile


def test_parse_int_and_clean_text():
    mod = load_module(FETCH_SCRIPT, "douyin_fetch_helpers")

    assert mod.parse_int("1.2万") == 12000
    assert mod.parse_int("3亿") == 300000000
    assert mod.parse_int("4.5w") == 45000
    assert mod.clean_text('【<em class="keyword">Hermes</em> <em>Agent</em>】') == "【Hermes Agent】"


def test_cookie_header_reads_default_firefox_profile(monkeypatch, tmp_path):
    mod = load_module(FETCH_SCRIPT, "douyin_fetch_firefox_cookie")
    firefox_root = tmp_path / ".mozilla" / "firefox"
    create_firefox_profile(firefox_root)

    monkeypatch.delenv("DOUYIN_COOKIE", raising=False)
    monkeypatch.delenv("DOUYIN_SEARCH_COOKIE", raising=False)
    monkeypatch.delenv("DOUYIN_WEB_COOKIE", raising=False)
    monkeypatch.setenv("DOUYIN_FIREFOX_ROOT", str(firefox_root))

    header = mod.cookie_header()

    assert header == "sessionid=fake_session; ttwid=fake_ttwid"


def test_explicit_cookie_overrides_firefox_profile(monkeypatch, tmp_path):
    mod = load_module(FETCH_SCRIPT, "douyin_fetch_cookie_override")
    firefox_root = tmp_path / ".mozilla" / "firefox"
    create_firefox_profile(firefox_root)

    monkeypatch.setenv("DOUYIN_COOKIE", "manual_cookie=1")
    monkeypatch.setenv("DOUYIN_FIREFOX_ROOT", str(firefox_root))

    assert mod.cookie_header() == "manual_cookie=1"


def test_build_search_url_includes_pc_browser_params(monkeypatch):
    mod = load_module(FETCH_SCRIPT, "douyin_fetch_search_url")

    monkeypatch.delenv("DOUYIN_BROWSER_PLATFORM", raising=False)
    url = mod.build_search_url(
        "HermesAgent",
        sort_type="0",
        offset=0,
        count=10,
        publish_time="7",
    )

    assert "keyword=HermesAgent" in url
    assert "pc_client_type=1" in url
    assert "version_code=290100" in url
    assert "browser_platform=Win32" in url
    assert "browser_name=Edge" in url
    assert "engine_name=Blink" in url
    assert "os_name=Windows" in url


def test_build_search_url_includes_tracking_params():
    mod = load_module(FETCH_SCRIPT, "douyin_fetch_tracking_params")

    url = mod.build_search_url(
        "HermesAgent",
        sort_type="0",
        offset=0,
        count=10,
        publish_time="7",
        tracking_params={"webid": "7652756661757429289", "msToken": "token=="},
    )

    assert "webid=7652756661757429289" in url
    assert "msToken=token%3D%3D" in url


def test_douyin_ms_token_prefers_env_then_cookie(monkeypatch, tmp_path):
    mod = load_module(FETCH_SCRIPT, "douyin_fetch_ms_token")
    firefox_root = tmp_path / ".mozilla" / "firefox"
    profile = create_firefox_profile(firefox_root)
    conn = sqlite3.connect(profile / "cookies.sqlite")
    try:
        conn.execute(
            "INSERT INTO moz_cookies (name, value, host, path, expiry) VALUES (?, ?, ?, ?, ?)",
            ("msToken", "cookie_token", ".douyin.com", "/", 2_000_000_000),
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.delenv("DOUYIN_COOKIE", raising=False)
    monkeypatch.delenv("DOUYIN_MSTOKEN", raising=False)
    monkeypatch.delenv("DOUYIN_MS_TOKEN", raising=False)
    monkeypatch.setenv("DOUYIN_FIREFOX_ROOT", str(firefox_root))

    assert mod.douyin_ms_token() == "cookie_token"

    monkeypatch.setenv("DOUYIN_MSTOKEN", "env_token")
    assert mod.douyin_ms_token() == "env_token"


def test_collect_search_results_retries_verify_check_with_new_tracking(monkeypatch):
    mod = load_module(FETCH_SCRIPT, "douyin_fetch_verify_retry")
    tracking_values = iter(
        [
            {"msToken": "token-1", "webid": "web-1"},
            {"msToken": "token-2", "webid": "web-2"},
        ]
    )
    seen_tokens = []

    def fake_tracking_search_params():
        return next(tracking_values)

    def fake_fetch_search_page(**kwargs):
        seen_tokens.append(kwargs["tracking_params"]["msToken"])
        if kwargs["tracking_params"]["msToken"] == "token-1":
            return {
                "status_code": 0,
                "data": [],
                "search_nil_info": {
                    "search_nil_type": "verify_check",
                    "search_nil_item": "verify_check",
                },
            }
        return {
            "status_code": 0,
            "has_more": False,
            "data": [
                {
                    "aweme_info": {
                        "aweme_id": "7333333333333333333",
                        "desc": "Hermes Agent 上手指南",
                        "create_time": ts(2026, 6, 14, 9),
                        "author": {"nickname": "tester", "uid": "100"},
                        "statistics": {"digg_count": 30},
                    }
                }
            ],
        }

    monkeypatch.setenv("DOUYIN_COOKIE", "sessionid=fake")
    monkeypatch.setattr(mod, "tracking_search_params", fake_tracking_search_params)
    monkeypatch.setattr(mod, "fetch_search_page", fake_fetch_search_page)

    videos, errors, raw_count = mod.collect_search_results(
        keywords=["HermesAgent"],
        sort_types=["0"],
        pages=1,
        page_size=20,
        publish_time="7",
    )

    assert seen_tokens == ["token-1", "token-2"]
    assert raw_count == 1
    assert errors == []
    assert videos[0]["aweme_id"] == "7333333333333333333"


def test_rank_weekly_videos_filters_dedupes_and_sorts():
    mod = load_module(FETCH_SCRIPT, "douyin_fetch_rank")
    now = datetime(2026, 6, 14, 23, 10, tzinfo=CN_TZ)
    week_start, _ = mod.week_window(now)

    videos = [
        {
            "aweme_id": "OLD",
            "title": "Hermes Agent old high play",
            "play": 999999,
            "like": 1,
            "create_time": ts(2026, 6, 7),
            "url": "https://www.douyin.com/video/OLD",
        },
        {
            "aweme_id": "IRRELEVANT",
            "title": "Agent workflow without target product",
            "hashtags": ["Hermes"],
            "play": 800000,
            "like": 1,
            "create_time": ts(2026, 6, 12),
            "url": "https://www.douyin.com/video/IRRELEVANT",
        },
        {
            "aweme_id": "DUP",
            "title": "Hermes Agent intro",
            "play": 100,
            "like": 10,
            "create_time": ts(2026, 6, 12),
            "url": "https://www.douyin.com/video/DUP",
            "source_queries": [{"keyword": "HermesAgent", "sort_type": "0", "offset": 0}],
        },
        {
            "aweme_id": "DUP",
            "title": "Hermes Agent intro",
            "play": 700,
            "like": 9,
            "create_time": ts(2026, 6, 12),
            "url": "https://www.douyin.com/video/DUP",
            "source_queries": [{"keyword": "Hermes Agent", "sort_type": "1", "offset": 20}],
        },
        {
            "aweme_id": "SECOND",
            "title": "Hermes Agent deep dive",
            "play": 500,
            "like": 20,
            "create_time": ts(2026, 6, 13),
            "url": "https://www.douyin.com/video/SECOND",
        },
    ]

    ranked = mod.rank_weekly_videos(videos, week_start=week_start, now=now, limit=10)

    assert [item["aweme_id"] for item in ranked] == ["DUP", "SECOND"]
    assert ranked[0]["rank"] == 1
    assert ranked[0]["play"] == 700
    assert len(ranked[0]["source_queries"]) == 2


def test_build_digest_with_mocked_douyin_response(monkeypatch):
    mod = load_module(FETCH_SCRIPT, "douyin_fetch_digest")

    def fake_fetch_search_page(**kwargs):
        assert kwargs["keyword"] == "HermesAgent"
        return {
            "status_code": 0,
            "has_more": False,
            "data": [
                {
                    "aweme_info": {
                        "aweme_id": "7333333333333333333",
                        "desc": "Hermes Agent 上手指南",
                        "create_time": ts(2026, 6, 14, 9),
                        "author": {"nickname": "tester", "uid": "100"},
                        "statistics": {
                            "play_count": "1.2万",
                            "digg_count": 30,
                            "comment_count": 4,
                            "share_count": 5,
                            "collect_count": 6,
                        },
                        "text_extra": [{"hashtag_name": "HermesAgent"}],
                    }
                }
            ],
        }

    monkeypatch.setenv("DOUYIN_COOKIE", "sessionid=fake")
    monkeypatch.setattr(mod, "fetch_search_page", fake_fetch_search_page)
    args = argparse.Namespace(
        limit=10,
        pages=1,
        page_size=20,
        keyword=["HermesAgent"],
        sort_type=["0"],
        publish_time="7",
        now="2026-06-14T23:10:00+08:00",
    )

    digest = mod.build_digest(args)

    assert digest["matched_count"] == 1
    assert digest["items"][0]["title"] == "Hermes Agent 上手指南"
    assert digest["items"][0]["play"] == 12000
    assert digest["items"][0]["url"] == "https://www.douyin.com/video/7333333333333333333"
    assert digest["items"][0]["source_queries"] == [{"keyword": "HermesAgent", "sort_type": "0", "offset": 0}]


def test_build_digest_stops_after_enough_weekly_matches(monkeypatch):
    mod = load_module(FETCH_SCRIPT, "douyin_fetch_early_stop")
    offsets = []

    def fake_fetch_search_page(**kwargs):
        offsets.append(kwargs["offset"])
        assert kwargs["offset"] == 0
        return {
            "status_code": 0,
            "has_more": True,
            "data": [
                {
                    "aweme_info": {
                        "aweme_id": f"73333333333333333{index:02d}",
                        "desc": f"Hermes Agent 视频 {index}",
                        "create_time": ts(2026, 6, 18, 9),
                        "author": {"nickname": f"tester-{index}", "uid": str(index)},
                        "statistics": {"digg_count": 100 - index},
                    }
                }
                for index in range(10)
            ],
        }

    monkeypatch.setenv("DOUYIN_COOKIE", "sessionid=fake")
    monkeypatch.setenv("DOUYIN_WEBID", "webid")
    monkeypatch.setattr(mod, "douyin_ms_token", lambda: "token")
    monkeypatch.setattr(mod, "fetch_search_page", fake_fetch_search_page)
    args = argparse.Namespace(
        limit=10,
        pages=3,
        page_size=20,
        keyword=["HermesAgent", "Hermes Agent"],
        sort_type=["0", "1"],
        publish_time="7",
        now="2026-06-18T23:10:00+08:00",
    )

    digest = mod.build_digest(args)

    assert offsets == [0]
    assert digest["raw_result_count"] == 10
    assert digest["matched_count"] == 10
    assert digest["errors"] == []


def test_build_digest_reports_douyin_verify_check(monkeypatch):
    mod = load_module(FETCH_SCRIPT, "douyin_fetch_verify_check")

    def fake_fetch_search_page(**_kwargs):
        return {
            "status_code": 0,
            "data": [],
            "has_more": 0,
            "search_nil_info": {
                "search_nil_type": "verify_check",
                "search_nil_item": "verify_check",
            },
        }

    monkeypatch.setenv("DOUYIN_COOKIE", "sessionid=fake")
    monkeypatch.setattr(mod, "fetch_search_page", fake_fetch_search_page)
    args = argparse.Namespace(
        limit=10,
        pages=1,
        page_size=20,
        keyword=["HermesAgent"],
        sort_type=["0"],
        publish_time="7",
        now="2026-06-14T23:10:00+08:00",
    )

    digest = mod.build_digest(args)

    assert digest["matched_count"] == 0
    assert "search_nil_type=verify_check" in digest["errors"][0]


def test_setup_installs_runtime_script_and_creates_cron_job(monkeypatch, capsys):
    mod = load_module(SETUP_SCRIPT, "douyin_setup")
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
    assert output["job"]["schedule_display"] == "10 23 * * *"
    assert output["job"]["deliver"] == "local"
    assert output["job"]["skills"] == [mod.SKILL_NAME]
    assert output["job"]["script"] == mod.RUNTIME_FETCH_SCRIPT
    assert (hermes_home / "scripts" / mod.RUNTIME_FETCH_SCRIPT).exists()


def test_setup_uses_single_feishu_channel_directory_target(monkeypatch, tmp_path):
    mod = load_module(SETUP_SCRIPT, "douyin_setup_channel")
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "channel_directory.json").write_text(
        json.dumps({"platforms": {"feishu": [{"id": "oc_test", "type": "dm"}]}}),
        encoding="utf-8",
    )
    monkeypatch.delenv("FEISHU_HOME_CHANNEL", raising=False)

    deliver, warnings = mod.resolve_default_deliver("feishu", hermes_home, {})

    assert deliver == "feishu:oc_test"
    assert warnings == ["FEISHU_HOME_CHANNEL is not set; using channel_directory target feishu:oc_test."]
