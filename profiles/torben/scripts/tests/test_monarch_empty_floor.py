from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT", "/Users/ericfreeman/.hermes/hermes-agent"))
SCRIPT = SCRIPTS_DIR / "torben_monarch_savings.py"


def _empty_fixture(path: Path) -> Path:
    fixture = {
        "source": "monarch-money-mcp",
        "source_window": {"transactions_start_date": "2026-07-01", "transactions_end_date": "2026-07-06"},
        "read_tool_calls": [
            {"tool": "GetRecurring", "status": "ok", "item_count": 0},
            {"tool": "GetTransactions", "status": "ok", "item_count": 0},
            {"tool": "GetSpendingByCategory", "status": "ok", "item_count": 0},
            {"tool": "GetBudget", "status": "ok", "item_count": 0},
            {"tool": "GetCashFlow", "status": "ok", "item_count": 0},
        ],
        "blocked_tool_calls": [],
        "tool_errors": [],
    }
    path.write_text(json.dumps(fixture), encoding="utf-8")
    return path


def _run_monarch(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env["HERMES_REPO_ROOT"] = str(REPO_ROOT)
    return subprocess.run(
        ["uv", "run", "python", str(SCRIPT), *args],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_all_empty_monarch_read_fails_instead_of_quiet(tmp_path: Path) -> None:
    home = tmp_path / "profile"
    state = home / "state"
    state.mkdir(parents=True)
    fixture = _empty_fixture(tmp_path / "empty-monarch.json")

    result = _run_monarch(home, "--loop", "daily", "--fixture", str(fixture), "--state-dir", str(state), "--json")
    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert payload["status"] == "failed"
    assert payload["reason"] == "monarch_live_read_all_empty"
    assert payload["empty_floor"]["total_item_count"] == 0
    assert payload["empty_floor"]["read_call_count"] == 5
    assert payload["wakeAgent"] is True
    assert payload["actions"] == []
    assert "mcp login monarch-money-mcp" in payload["text"]


def test_monarch_diagnose_reports_reauth_step_for_all_empty(tmp_path: Path) -> None:
    home = tmp_path / "profile"
    (home / "state").mkdir(parents=True)
    fixture = _empty_fixture(tmp_path / "empty-monarch.json")

    result = _run_monarch(home, "--loop", "daily", "--fixture", str(fixture), "--diagnose")
    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert payload["status"] == "requires_reauth"
    assert payload["reason"] == "all_monarch_read_tools_returned_zero_items"
    assert payload["total_item_count"] == 0
    assert payload["login_step"]["command"].endswith("torben-hermes mcp login monarch-money-mcp")
    assert payload["login_step"]["endpoint"] == "https://api.monarch.com/mcp"
