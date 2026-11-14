"""scripts/ci/skills_index_probe.py judges a saved skills-index.json the way the freshness watchdog does."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "ci" / "skills_index_probe.py"
sys.path.insert(0, str(SCRIPT.parent))

import skills_index_probe as probe

NOW = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
HEALTHY_COUNTS = {"skills-sh": 60, "skills.sh": 400, "lobehub": 500, "clawhub": 300,
                  "official": 120, "github": 80, "browse-sh": 90, "community": 50}


def _index(counts: dict[str, int], *, age: timedelta = timedelta(hours=3), total: int | None = None) -> dict:
    skills = [{"name": f"{src}-{i}", "source": src} for src, n in counts.items() for i in range(n)]
    return {
        "generated_at": (NOW - age).isoformat().replace("+00:00", "Z"),
        "skill_count": len(skills) if total is None else total,
        "skills": skills,
    }


def _fields(lines: list[str]) -> dict[str, str]:
    return dict(line.split("=", 1) for line in lines)


def test_healthy_index_is_ok_with_a_source_summary():
    out = _fields(probe.evaluate(_index(HEALTHY_COUNTS), NOW))
    assert out["status"] == "ok"
    assert out["detail"] == "Index OK — 1600 skills, generated 2026-11-09T09:00:00Z"
    assert out["summary"].startswith("lobehub=500, skills.sh=400, clawhub=300")


def test_stale_collapsed_and_small_index_lists_every_problem():
    counts = {**HEALTHY_COUNTS, "skills.sh": 30, "skills-sh": 30, "github": 0}
    out = _fields(probe.evaluate(_index(counts, age=timedelta(hours=30), total=1200), NOW))
    assert out == {
        "status": "degraded",
        "detail": "Index is 30.0h old (limit 26h); skills.sh: 60 < 100; github: 0 < 30; "
                  "total skills: 1200 < 1500",
    }


def test_unreadable_timestamp_skips_the_age_check_only():
    data = {**_index(HEALTHY_COUNTS), "generated_at": "yesterday"}
    assert _fields(probe.evaluate(data, NOW))["status"] == "ok"


def test_non_list_skills_is_invalid_shape():
    out = _fields(probe.evaluate({"skills": {"a": 1}}, NOW))
    assert out == {"status": "invalid-shape", "detail": "skills field is not a list (got dict)"}


def test_remote_newlines_cannot_add_output_lines():
    data = _index(HEALTHY_COUNTS)
    data["generated_at"] += "\nstatus=forged"
    data["skills"].append({"name": "x", "source": "lobehub\r\nstatus=forged"})
    lines = probe.evaluate(data, NOW)
    assert all("\n" not in line and "\r" not in line for line in lines)
    assert [line.split("=", 1)[0] for line in lines] == ["status", "detail", "summary"]


def test_script_reports_parse_failure_and_exits_zero(tmp_path):
    index = tmp_path / "skills-index.json"
    index.write_text("<html>502</html>", encoding="utf-8")
    run = subprocess.run([sys.executable, str(SCRIPT), str(index)], capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, check=False)
    assert run.returncode == 0
    status, detail = run.stdout.splitlines()
    assert status == "status=parse-failed"
    assert detail.startswith("detail=JSON decode error: Expecting value")


def test_script_judges_a_fresh_saved_response(tmp_path):
    index = tmp_path / "skills-index.json"
    data = _index(HEALTHY_COUNTS)
    data["generated_at"] = datetime.now(UTC).isoformat()
    index.write_text(json.dumps(data), encoding="utf-8")
    run = subprocess.run([sys.executable, str(SCRIPT), str(index)], capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, check=True)
    assert run.stdout.splitlines()[0] == "status=ok"
