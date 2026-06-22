"""Repeatable dogfood harness tests for Memory v2."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from plugins.memory.memory_v2.dogfood import prepare_dogfood_profile, run_dogfood_scenario_tests


def _write_source_context(source_home: Path) -> None:
    source_home.mkdir(parents=True, exist_ok=True)
    (source_home / "USER.md").write_text(
        """
Prefers direct, source-grounded answers.
Prefers Memory v2 dogfood reports with concrete failure IDs.
Wants robust low-compute memory.
""".strip(),
        encoding="utf-8",
    )
    (source_home / "SOUL.md").write_text(
        """
Be truthful rather than merely confident.
Do keep private things private.
Do handle external actions with care.
""".strip(),
        encoding="utf-8",
    )
    (source_home / "AGENTS.md").write_text(
        """
Use tools when checking facts.
Do not expose credentials.
Memory writes should be source-grounded and gated.
""".strip(),
        encoding="utf-8",
    )
    (source_home / "TOOLS.md").write_text("example-cli is installed in the project toolchain.\n", encoding="utf-8")
    (source_home / "MEMORY.md").write_text("Hermes runs in WSL for this profile.\n", encoding="utf-8")


def test_prepare_dogfood_profile_fresh_resets_memory_v2_and_imports_core(tmp_path: Path):
    source_home = tmp_path / "source"
    target_home = tmp_path / "dogfood"
    _write_source_context(source_home)
    target_home.mkdir()
    (target_home / "config.yaml").write_text("memory:\n  memory_enabled: true\n  provider: ''\n", encoding="utf-8")
    stale = target_home / "memory_v2" / "inbox" / "raw_events.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"id":"stale"}\n', encoding="utf-8")
    stale_item = target_home / "memory_v2" / "semantic" / "items" / "stale.yaml"
    stale_item.parent.mkdir(parents=True)
    stale_item.write_text("id: stale\n", encoding="utf-8")

    report = prepare_dogfood_profile(
        target_hermes_home=target_home,
        source_hermes_home=source_home,
        fresh=True,
        core_budget=8,
        category_minimums={"user": 2, "assistant_identity": 2, "operating_rule": 2, "environment": 1},
    )

    config = yaml.safe_load((target_home / "config.yaml").read_text(encoding="utf-8"))
    assert report["fresh_reset"] is True
    assert config["memory"]["provider"] == "memory_v2"
    assert config["memory"]["memory_enabled"] is True
    assert not stale_item.exists()
    assert (target_home / "memory_v2" / "inbox" / "raw_events.jsonl").read_text(encoding="utf-8") == ""
    assert report["import_report"]["records_written"] == 8
    assert report["core_counts"]["user"] >= 2
    assert report["core_counts"]["assistant_identity"] >= 2


def test_fresh_dogfood_scenario_runs_start_from_clean_state_each_time(tmp_path: Path):
    source_home = tmp_path / "source"
    target_home = tmp_path / "dogfood"
    _write_source_context(source_home)
    target_home.mkdir()
    (target_home / "config.yaml").write_text("memory:\n  memory_enabled: true\n  provider: ''\n", encoding="utf-8")

    first = run_dogfood_scenario_tests(
        target_hermes_home=target_home,
        source_hermes_home=source_home,
        default_hermes_home=tmp_path / "default",
        fresh=True,
        core_budget=8,
        category_minimums={"user": 2, "assistant_identity": 2, "operating_rule": 2, "environment": 1},
    )
    second = run_dogfood_scenario_tests(
        target_hermes_home=target_home,
        source_hermes_home=source_home,
        default_hermes_home=tmp_path / "default",
        fresh=True,
        core_budget=8,
        category_minimums={"user": 2, "assistant_identity": 2, "operating_rule": 2, "environment": 1},
    )

    assert first["success"] is True
    assert second["success"] is True
    assert first["initial_counts"] == second["initial_counts"] == {
        "raw_events": 0,
        "pending_candidates": 0,
        "open_loops": 0,
        "memory_items": 0,
    }
    assert first["final_counts"] == second["final_counts"] == {
        "raw_events": 4,
        "pending_candidates": 1,
        "open_loops": 1,
        "memory_items": 1,
    }
    report_files = sorted((target_home / "memory_v2" / "reports").glob("dogfood_report_*.json"))
    assert len(report_files) == 1
    persisted = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert persisted["success"] is True
    assert persisted["fresh_reset"] is True


def test_fresh_dogfood_allows_preexisting_default_memory_v2_store(tmp_path: Path):
    source_home = tmp_path / "source"
    target_home = tmp_path / "dogfood"
    default_home = tmp_path / "default"
    _write_source_context(source_home)
    (default_home / "memory_v2" / "inbox").mkdir(parents=True)
    sentinel = default_home / "memory_v2" / "inbox" / "sentinel.txt"
    sentinel.write_text("preexisting default profile memory_v2 store", encoding="utf-8")

    report = run_dogfood_scenario_tests(
        target_hermes_home=target_home,
        source_hermes_home=source_home,
        default_hermes_home=default_home,
        fresh=True,
        core_budget=8,
        category_minimums={"user": 2, "assistant_identity": 2, "operating_rule": 2, "environment": 1},
    )

    assert report["success"] is True
    assert sentinel.read_text(encoding="utf-8") == "preexisting default profile memory_v2 store"


def test_dogfood_can_run_local_eval_and_persist_summary(tmp_path: Path):
    source_home = tmp_path / "source"
    target_home = tmp_path / "dogfood"
    _write_source_context(source_home)

    report = run_dogfood_scenario_tests(
        target_hermes_home=target_home,
        source_hermes_home=source_home,
        default_hermes_home=tmp_path / "default",
        fresh=True,
        core_budget=8,
        category_minimums={"user": 2, "assistant_identity": 2, "operating_rule": 2, "environment": 1},
        run_local_eval=True,
    )

    assert report["success"] is True
    assert report["local_eval"]["dataset"] == "local_memory_eval_v1"
    assert report["local_eval"]["external_baselines"] == []
    assert set(report["local_eval"]["baselines"]) == {"memory_v2", "no_memory", "raw_fts"}
    assert report["local_eval"]["summary"]["memory_v2"]["query_count"] == 3
    assert any(
        result["name"] == "local_eval_fixture_runs" and result["ok"]
        for result in report["results"]
    )

    persisted = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert persisted["local_eval"]["summary"] == report["local_eval"]["summary"]
