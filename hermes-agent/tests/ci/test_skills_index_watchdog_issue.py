"""scripts/ci/skills_index_watchdog_issue.py against a stub ``gh`` that records argv and env."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "ci" / "skills_index_watchdog_issue.py"

_STUB = """#!{python}
import json, os, sys
with open(os.environ["STUB_CALLS"], "a") as f:
    f.write(json.dumps({{"argv": sys.argv[1:], "token": os.environ.get("GH_TOKEN")}}) + "\\n")
if sys.argv[1:3] == ["issue", "list"]:
    if os.environ.get("STUB_LIST_FAILS"):
        sys.exit(1)
    print(os.environ["STUB_ISSUES"])
"""


def _run(tmp_path: Path, watchdog: str, issues: list[dict], *, list_fails: bool = False) -> tuple[str, list[dict]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(_STUB.format(python=sys.executable), encoding="utf-8")
    gh.chmod(0o755)
    calls = tmp_path / "calls"
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GH_TOKEN": "ghs_secret",
        "GITHUB_REPOSITORY": "NousResearch/hermes-agent",
        "STATUS": "degraded",
        "DETAIL": "github: 0 < 30",
        "STUB_CALLS": str(calls),
        "STUB_ISSUES": json.dumps(issues),
        **({"STUB_LIST_FAILS": "1"} if list_fails else {}),
    }
    run = subprocess.run([sys.executable, str(SCRIPT), watchdog], env=env, capture_output=True,
                         text=True, stdin=subprocess.DEVNULL, check=True)
    return run.stdout, [json.loads(line) for line in calls.read_text().splitlines()]


@pytest.mark.platforms("linux", "macos")
def test_no_open_issue_opens_one_with_the_watchdog_prefix(tmp_path):
    stdout, calls = _run(tmp_path, "skills-index", [])
    assert stdout == "Opening new watchdog issue\n"
    listing, create = calls
    assert listing["argv"][listing["argv"].index("--search") + 1] == 'in:title "[skills-index-watchdog]"'
    argv = create["argv"]
    assert argv[:5] == ["issue", "create", "--repo", "NousResearch/hermes-agent", "--title"]
    assert argv[5] == "[skills-index-watchdog] Skills index is stale or degraded (degraded)"
    body = argv[argv.index("--body") + 1]
    assert "**Status:** `degraded`\n**Detail:** github: 0 < 30" in body
    assert "https://github.com/NousResearch/hermes-agent/actions/workflows/skills-index.yml" in body
    assert all(c["token"] == "ghs_secret" and "ghs_secret" not in c["argv"] for c in calls)


@pytest.mark.platforms("linux", "macos")
def test_open_issue_with_the_prefix_gets_a_comment(tmp_path):
    issues = [{"number": 7, "title": "Re: [plugin-catalog-cors-watchdog] mentioned mid-title"},
              {"number": 9, "title": "[plugin-catalog-cors-watchdog] Desktop cannot read the plugin catalog (blocked)"}]
    stdout, calls = _run(tmp_path, "plugin-catalog-cors", issues)
    assert stdout == "Appending to existing issue #9\n"
    argv = calls[1]["argv"]
    assert argv[:6] == ["issue", "comment", "9", "--repo", "NousResearch/hermes-agent", "--body"]
    assert argv[6].startswith("Probe still failing at ")
    assert argv[6].endswith("Z: `degraded` — github: 0 < 30")


@pytest.mark.platforms("linux", "macos")
def test_failed_listing_files_a_new_issue(tmp_path):
    stdout, calls = _run(tmp_path, "plugin-catalog-cors", [], list_fails=True)
    assert stdout == "Opening new watchdog issue\n"
    assert calls[1]["argv"][5] == "[plugin-catalog-cors-watchdog] Desktop cannot read the plugin catalog (degraded)"
