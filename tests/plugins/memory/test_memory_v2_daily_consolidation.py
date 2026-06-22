"""Daily consolidation/report tests for Memory v2."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from plugins.memory.memory_v2 import MemoryV2Provider
from plugins.memory.memory_v2.daily_consolidation import run_daily_consolidation_report


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _new_provider(tmp_path):
    provider = MemoryV2Provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="discord")
    return provider


def test_daily_consolidation_report_promotes_pending_candidates_and_writes_report_files(tmp_path):
    provider = _new_provider(tmp_path)
    provider.sync_turn("Remember that Alex prefers daily memory reports with concrete IDs.", "Queued.")
    provider.sync_turn("Remember to follow up on Memory v2 report UX tomorrow.", "Tracked.")

    report = run_daily_consolidation_report(provider.store, provider.index, date="2026-06-01")

    report_path = tmp_path / "memory_v2" / report["report_path"]
    episode_path = tmp_path / "memory_v2" / report["daily_episode_path"]
    persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
    daily_episode = yaml.safe_load(episode_path.read_text(encoding="utf-8"))

    assert report["success"] is True
    assert report["date"] == "2026-06-01"
    assert report["consolidation"]["promoted"] == 1
    assert report["consolidation"]["archived_only"] == 1
    assert report["after_counts"]["pending_candidates"] == 0
    assert report["after_counts"]["memory_items"] == 1
    assert report["after_counts"]["open_loops"] == 1
    assert report["report_path"] == "reports/daily_consolidation/2026-06-01.json"
    assert report["daily_episode_path"] == "episodic/daily/2026-06-01.yaml"
    assert persisted_report == report
    assert daily_episode["kind"] == "daily_memory_consolidation"
    assert daily_episode["date"] == "2026-06-01"
    assert daily_episode["consolidation"]["promoted_ids"] == report["consolidation"]["promoted_ids"]
    assert daily_episode["open_loops"][0]["status"] == "open"


def test_daily_report_provider_tool_is_exposed_and_returns_json(tmp_path):
    provider = _new_provider(tmp_path)
    provider.sync_turn("Remember that Alex prefers report commands to be auditable.", "Queued.")

    schemas = {schema["name"] for schema in provider.get_tool_schemas()}
    result = json.loads(provider.handle_tool_call("memory_v2_daily_report", {"date": "2026-06-02"}))

    assert "memory_v2_daily_report" in schemas
    assert result["success"] is True
    assert result["date"] == "2026-06-02"
    assert result["after_counts"]["pending_candidates"] == 0
    assert (tmp_path / "memory_v2" / result["report_path"]).is_file()


def test_daily_consolidation_module_cli_runs_against_profile_home(tmp_path):
    provider = _new_provider(tmp_path)
    provider.sync_turn("Remember that Memory v2 daily CLI should be profile scoped.", "Queued.")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "plugins.memory.memory_v2.daily_consolidation",
            "--hermes-home",
            str(tmp_path),
            "--date",
            "2026-06-03",
            "--session-id",
            "daily-cli-test",
        ],
        check=False,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert payload["success"] is True
    assert payload["date"] == "2026-06-03"
    assert payload["after_counts"]["pending_candidates"] == 0
    assert (tmp_path / "memory_v2" / payload["report_path"]).is_file()


def test_daily_consolidation_cli_updates_provider_index_path(tmp_path):
    provider = _new_provider(tmp_path)
    provider.sync_turn("Remember that Alex prefers daily reports to stay unified.", "Queued.")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "plugins.memory.memory_v2.daily_consolidation",
            "--hermes-home",
            str(tmp_path),
            "--date",
            "2026-06-04",
        ],
        check=False,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    reloaded = _new_provider(tmp_path)

    assert completed.returncode == 0
    results = reloaded.index.search("daily reports stay unified", route="preference_recall", limit=5)
    assert results[0]["type"] == "preference"
    assert results[0]["status"] == "active"
