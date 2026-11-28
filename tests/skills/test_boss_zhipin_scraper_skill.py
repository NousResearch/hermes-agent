"""boss-zhipin-scraper optional skill: SKILL.md standards, importability, pure-function I/O.

Zero live network: Chrome/CDP/websocket are never touched — only the pure
parsing helpers of the bundled scripts and the shipped city code table.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / "optional-skills" / "productivity" / "boss-zhipin-scraper"

EXPECTED_SECTIONS = [
    "When to Use",
    "Prerequisites",
    "How to Run",
    "Quick Reference",
    "Procedure",
    "Pitfalls",
    "Verification",
]
# Authoring standard #2: prose must name Hermes tools, not wrapped shell utilities.
WRAPPED_SHELL_UTILITIES = re.compile(r"\b(grep|cat|head|tail|sed|awk|find|ls)\b", re.I)


def _frontmatter(text: str) -> dict:
    assert text.startswith("---"), "SKILL.md must open with frontmatter"
    end = text.index("\n---", 3)
    fields = {}
    for line in text[3:end].splitlines():
        m = re.match(r"^([a-z_]+): (.*)$", line)
        if m:
            fields[m.group(1)] = m.group(2)
    return fields


@pytest.fixture(scope="module")
def skill_text() -> str:
    return (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def boss():
    spec = importlib.util.spec_from_file_location(
        "boss_cdp_raw", SKILL_DIR / "scripts" / "boss_cdp_raw.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["boss_cdp_raw"] = module
    return module


@pytest.fixture(scope="module")
def summary(boss):
    spec = importlib.util.spec_from_file_location(
        "job_summary", SKILL_DIR / "scripts" / "job_summary.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── SKILL.md authoring standards ────────────────────────────────────────────

def test_frontmatter_meets_hardline(skill_text):
    fm = _frontmatter(skill_text)
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert fm.get(field), f"missing frontmatter field: {field}"
    assert fm["name"] == SKILL_DIR.name
    description = fm["description"]
    assert len(description) <= 60, f"description is {len(description)} chars (max 60)"
    assert description.endswith(".")
    assert fm["author"].split(",")[0].strip() == "eatmoreduck"
    assert {p.strip() for p in fm["platforms"].strip("[]").split(",")} == {"macos", "linux"}


def test_hermes_metadata_declares_category(skill_text):
    assert re.search(r"^    category: productivity$", skill_text, re.M)
    assert re.search(r"^    tags: \[.+\]$", skill_text, re.M)
    assert re.search(r"^    related_skills: \[.+\]$", skill_text, re.M)


def test_section_order_is_modern(skill_text):
    body = skill_text.split("\n---\n", 1)[1]
    assert body.lstrip().splitlines()[0] == "# BOSS直聘 Scraper Skill"
    sections = re.findall(r"^## (.+)$", body, re.M)
    assert sections == EXPECTED_SECTIONS


def test_skill_length_within_budget(skill_text):
    lines = skill_text.count("\n")
    assert 100 <= lines <= 200, f"SKILL.md is {lines} lines (target 100-200)"


def test_prose_names_hermes_tools_not_shell_utilities(skill_text):
    prose = re.sub(r"```.*?```", "", skill_text, flags=re.S)  # code fences are exempt
    m = WRAPPED_SHELL_UTILITIES.search(prose)
    assert not m, f"prose names wrapped shell utility {m.group(0)!r}"
    assert "`terminal`" in prose, "prose should reference the terminal tool"
    # Scripts are referenced by skill-relative path (standard #6), fence or prose.
    assert "scripts/boss_cdp_raw.py" in skill_text and "scripts/job_summary.py" in skill_text


# ── bundled scripts import cleanly without optional deps ────────────────────

def test_scripts_import_and_versions_align(boss, summary, skill_text):
    # requests/websocket-client are absent in this env; import must still work
    # because the script resolves them lazily via require_runtime_dependencies().
    assert boss.__version__ == "2.2.0"
    assert _frontmatter(skill_text)["version"] == boss.__version__
    assert callable(boss.map_api_job)
    assert callable(summary.split_tags)


# ── core parsing: API payload → normalized job (plaintext salary) ────────────

RAW_JOB = {
    "encryptJobId": "c4420e8bce3a6e25",
    "encryptBrandId": "brand123",
    "jobName": "AI Agent工程师",
    "salaryDesc": "30-60K·15薪",
    "cityName": "上海",
    "areaDistrict": "闵行区",
    "businessDistrict": "虹桥",
    "jobExperience": "5-10年",
    "jobDegree": "本科",
    "brandName": "SHEIN",
    "brandScaleName": "10000人以上",
    "brandStageName": "D轮及以上",
    "brandIndustry": "电子商务",
    "skills": ["Java", "Spring"],
    "welfareList": ["节日福利", "定期体检"],
    "securityId": "sid-1",
    "lid": "lid-1",
}


def test_map_api_job_keeps_plaintext_salary_and_builds_links(boss):
    job = boss.map_api_job(RAW_JOB)
    assert job["salary"] == "30-60K·15薪"
    assert job["salary_source"] == "api"
    assert job["job_link"] == "https://www.zhipin.com/job_detail/c4420e8bce3a6e25.html"
    assert job["company_link"] == "https://www.zhipin.com/gongsi/brand123.html"
    assert job["location"] == "上海·闵行区·虹桥"
    assert job["skills"] == "Java | Spring"
    assert job["welfare"] == "节日福利 | 定期体检"


def test_map_api_job_drops_placeholder_tags(boss):
    # Only the exact placeholder "不限" is dropped; "经验不限" is real data and stays.
    job = boss.map_api_job({**RAW_JOB, "jobExperience": "不限"})
    assert job["tags"] == "本科"
    both_unlimited = boss.map_api_job({**RAW_JOB, "jobExperience": "不限", "jobDegree": "不限"})
    assert both_unlimited["tags"] == ""
    assert boss.map_api_job(RAW_JOB)["tags"] == "5-10年 | 本科"


def test_map_api_job_flags_missing_salary(boss):
    job = boss.map_api_job({k: v for k, v in RAW_JOB.items() if k != "salaryDesc"})
    assert job["salary"] == ""
    assert job["salary_source"] == "api_empty"


def test_map_api_jobs_extracts_joblist_and_skips_junk(boss):
    payload = {"zpData": {"jobList": [RAW_JOB, "junk-entry", None]}}
    jobs = boss.map_api_jobs(payload)
    assert len(jobs) == 1 and jobs[0]["title"] == "AI Agent工程师"
    assert boss.map_api_jobs("not-a-dict") == []
    assert boss.map_api_job(["not-a-dict"]) is None


# ── login/risk-control probe classification (pure) ───────────────────────────

def test_login_probe_classifies_http_and_risk_codes(boss):
    unauth = boss.classify_login_probe_response({}, http_status=401)
    assert unauth.status is boss.LoginProbeStatus.UNAUTHENTICATED

    for status in (403, 429):
        restricted = boss.classify_login_probe_response({}, http_status=status)
        assert restricted.status is boss.LoginProbeStatus.RESTRICTED

    server_error = boss.classify_login_probe_response({}, http_status=500)
    assert server_error.status is boss.LoginProbeStatus.RESPONSE_ERROR
    assert server_error.retryable

    # API code 37 = BOSS risk control, even with HTTP 200.
    risk = boss.classify_login_probe_response({"code": 37, "message": "环境存在异常"})
    assert risk.status is boss.LoginProbeStatus.RESTRICTED

    # Unknown non-zero code with risk-control wording still maps to RESTRICTED.
    keyword = boss.classify_login_probe_response({"code": 999, "message": "请完成滑块验证"})
    assert keyword.status is boss.LoginProbeStatus.RESTRICTED

    unknown = boss.classify_login_probe_response({"code": 999, "message": "boom"})
    assert unknown.status is boss.LoginProbeStatus.RESPONSE_ERROR
    assert not unknown.retryable


def test_login_probe_classifies_payload_shapes(boss):
    ok = {"code": 0, "zpData": {"jobList": [{"salaryDesc": "20-40K·14薪"}]}}
    assert boss.classify_login_probe_response(ok).status is boss.LoginProbeStatus.AVAILABLE
    assert boss.is_logged_in_search_response(ok)

    logged_out = {"code": 0, "zpData": {"jobList": [{"salaryDesc": ""}]}}
    assert (
        boss.classify_login_probe_response(logged_out).status
        is boss.LoginProbeStatus.UNAUTHENTICATED
    )

    empty = {"code": 0, "zpData": {"jobList": []}}
    assert boss.classify_login_probe_response(empty).status is boss.LoginProbeStatus.EMPTY

    broken = boss.classify_login_probe_response({"code": 0})
    assert broken.status is boss.LoginProbeStatus.RESPONSE_ERROR
    assert broken.retryable


# ── city resolution: bundled table, offline ──────────────────────────────────

def test_resolve_city_uses_bundled_table_without_network(boss, monkeypatch):
    # The live-BOSS fallback must not fire for names the bundled table knows.
    live = mock.Mock(side_effect=AssertionError("network fallback must not be hit"))
    monkeypatch.setattr(boss, "load_live_city_maps", live)
    assert boss.resolve_city("上海") == ("上海", "101020100")
    assert boss.resolve_city("101020100") == ("上海", "101020100")
    live.assert_not_called()


def test_resolve_city_fallback_and_rejection(boss, monkeypatch):
    monkeypatch.setattr(boss, "load_live_city_maps", lambda timeout=10: ({}, {}))
    assert boss.resolve_city("123456789") == ("123456789", "123456789")  # bare 9-digit code
    with pytest.raises(boss.CityResolutionError):
        boss.resolve_city("Atlantis")


# ── merge dedup and URL building (pure) ──────────────────────────────────────

def test_merge_unique_keeps_existing_by_default(boss):
    existing = [{"job_id": "a", "salary": "old"}]
    incoming = [{"job_id": "a", "salary": "new"}, {"job_id": "b", "salary": "b"}]
    assert boss.merge_unique(existing, incoming) == [
        {"job_id": "a", "salary": "old"},
        {"job_id": "b", "salary": "b"},
    ]
    assert boss.merge_unique(existing, incoming, new_overrides=True) == [
        {"job_id": "a", "salary": "new"},
        {"job_id": "b", "salary": "b"},
    ]


def test_url_builders(boss):
    url = boss.build_search_url("Java", "101020100", 2, {"salary": "406"})
    assert url == "https://www.zhipin.com/web/geek/job?query=Java&city=101020100&page=2&salary=406"

    job = {"job_link": "https://www.zhipin.com/job_detail/abc.html", "lid": "L1", "security_id": "S1"}
    detail = boss.build_detail_url(job)
    assert "lid=L1" in detail and "securityId=S1" in detail

    already = {"job_link": "https://www.zhipin.com/job_detail/abc.html?lid=L0", "lid": "L1"}
    assert "lid=L0" in boss.build_detail_url(already)  # existing param is kept
    assert boss.build_detail_url({"job_link": ""}) == ""


# ── job_summary pure helpers ─────────────────────────────────────────────────

def test_summary_split_tags(summary):
    assert summary.split_tags("Java｜Spring | AI") == ["Java", "Spring", "AI"]
    assert summary.split_tags(["Java", " Spring "]) == ["Java", "Spring"]
    assert summary.split_tags(None) == []
    assert summary.split_tags("  ") == []


def test_summary_loads_jobs_file(summary, tmp_path):
    path = tmp_path / "boss_jobs_test.json"
    path.write_text(
        json.dumps({"keyword": "AI", "city": "上海", "jobs": [{"job_id": "a"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    jobs, meta = summary.load_jobs_file(str(path))
    assert jobs == [{"job_id": "a"}]
    assert meta["keyword"] == "AI" and meta["city"] == "上海"


# ── optional-catalog discoverability (needs skills-hub deps; skips otherwise) ─

def test_optional_catalog_discovers_and_bundles_skill():
    pytest.importorskip("httpx", reason="skills hub needs httpx")
    try:
        from tools.skills_hub_official import OptionalSkillSource
    except ImportError as exc:  # partial dev env without full agent deps
        pytest.skip(f"skills hub import failed: {exc}")

    source = OptionalSkillSource()
    source._optional_dir = ROOT / "optional-skills"
    metas = [m for m in source.list_local() if m.name == "boss-zhipin-scraper"]
    assert metas, "skill must be discoverable in the optional catalog"
    assert "zhipin" in [t.lower() for t in metas[0].tags]

    bundle = source.fetch("productivity/boss-zhipin-scraper")
    assert bundle is not None
    for rel in (
        "SKILL.md",
        "scripts/boss_cdp_raw.py",
        "scripts/job_summary.py",
        "data/city_codes.json",
    ):
        assert rel in bundle.files, f"bundle is missing {rel}"
