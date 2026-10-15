"""Exercise the workflow's actual Python probe against synthetic index files (#139372)."""

import io
import json
import textwrap
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest


@pytest.fixture
def probe():
    # The workflow is the executable artifact: run its heredoc, not a copy of its policy.
    workflow = Path(__file__).resolve().parents[2] / ".github/workflows/skills-index-freshness.yml"
    code = textwrap.dedent(workflow.read_text(encoding="utf-8").split(
        "python3 <<'PY' >> \"$GITHUB_OUTPUT\"\n", 1
    )[1].split("          PY\n", 1)[0])

    def run(counts, *, age=0, raw=None):
        skills = [{"source": source} for source, count in counts.items() for _ in range(count)]
        data = json.dumps({
            "generated_at": (datetime.now(timezone.utc) - timedelta(hours=age)).isoformat(),
            "skill_count": len(skills),
            "skills": skills,
        }) if raw is None else raw
        output = io.StringIO()
        with patch("builtins.open", mock_open(read_data=data)), redirect_stdout(output):
            try:
                exec(compile(code, str(workflow), "exec"), {"__name__": "__main__"})
            except SystemExit as exc:
                assert exc.code == 0
        return dict(line.split("=", 1) for line in output.getvalue().splitlines())

    return run


@pytest.mark.parametrize("source,count", [("skills.sh", 200), ("clawhub", 1000)])
def test_major_source_collapse_is_degraded_despite_healthy_total(probe, source, count):
    counts = {"skills.sh": 18504, "clawhub": 80250, "lobehub": 505,
              "official": 153, "github": 1076, "browse-sh": 478}
    assert probe(counts)["status"] == "ok"
    counts[source] = count
    result = probe(counts)
    assert result["status"] == "degraded"
    assert f"{source}: {count} < " in result["detail"]
    assert "total skills:" not in result["detail"]


def test_floor_boundaries_alias_and_existing_failure_paths(probe):
    counts = {"skills.sh": 5000, "skills-sh": 5000, "clawhub": 20000,
              "lobehub": 100, "official": 50, "github": 30, "browse-sh": 50}
    assert probe(counts)["status"] == "ok"
    assert probe(counts, age=27)["status"] == "degraded"
    assert probe({})["status"] == "degraded"
    assert probe({}, raw="not json")["status"] == "parse-failed"
    assert probe({}, raw='{"skills": {}}')["status"] == "invalid-shape"
    for source in counts:
        below = dict(counts)
        below[source] -= 1
        assert probe(below)["status"] == "degraded"
