from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _write_profile(home: Path, name: str, config: dict) -> Path:
    profile_dir = home / "profiles" / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    return profile_dir


def _write_raw_profile_config(home: Path, name: str, text: str) -> Path:
    profile_dir = home / "profiles" / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text(text, encoding="utf-8")
    return profile_dir


def _run_config(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["HOME"] = str(home.parent)
    env["USERPROFILE"] = str(home.parent)
    env["PYTHONPATH"] = str(ROOT)
    env["HERMES_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "config", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_config_check_all_profiles_kanban_workers_flags_zero_budget_workers(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    zero_worker = {
        "toolsets": ["terminal", "file"],
        "agent": {"max_turns": 0},
        "delegation": {"max_iterations": 0},
        "goals": {"max_turns": 0},
    }
    _write_profile(home, "reels", zero_worker)
    _write_profile(home, "mapasocial", zero_worker)
    _write_profile(home, "researchswarm", zero_worker)
    _write_profile(
        home,
        "archive-only",
        {
            "toolsets": [],
            "agent": {"max_turns": 0},
            "delegation": {"max_iterations": 0},
            "kanban": {"worker": {"enabled": False, "reason": "notification-only"}},
        },
    )

    res = _run_config(home, "check", "--all-profiles", "--kanban-workers")

    combined = res.stdout + res.stderr
    assert res.returncode == 1
    assert "reels" in combined
    assert "mapasocial" in combined
    assert "researchswarm" in combined
    assert "agent.max_turns" in combined
    assert "delegation.max_iterations" in combined
    assert "goals.max_turns" in combined
    assert "archive-only" not in combined


def test_config_check_all_profiles_kanban_workers_fails_closed_on_malformed_profile_config(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    _write_raw_profile_config(home, "broken", "agent: [unterminated")

    res = _run_config(home, "check", "--all-profiles", "--kanban-workers")

    combined = res.stdout + res.stderr
    assert res.returncode != 0
    assert "broken" in combined
    assert "invalid" in combined.lower() or "unreadable" in combined.lower()
    assert "unterminated" not in combined


def test_config_repair_worker_budgets_dry_run_and_profile_scoped_set(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    zero_worker = {
        "toolsets": ["terminal", "file"],
        "agent": {"max_turns": 0},
        "delegation": {"max_iterations": 0},
        "goals": {"max_turns": 0},
    }
    reels_cfg = _write_profile(home, "reels", zero_worker) / "config.yaml"
    mapasocial_cfg = _write_profile(home, "mapasocial", zero_worker) / "config.yaml"
    before_reels = reels_cfg.read_text(encoding="utf-8")
    before_mapasocial = mapasocial_cfg.read_text(encoding="utf-8")

    dry = _run_config(
        home,
        "repair",
        "worker-budgets",
        "--all-profiles",
        "--dry-run",
        "--set",
        "120",
    )

    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert "dry-run" in (dry.stdout + dry.stderr).lower()
    assert reels_cfg.read_text(encoding="utf-8") == before_reels
    assert mapasocial_cfg.read_text(encoding="utf-8") == before_mapasocial

    fixed = _run_config(
        home,
        "repair",
        "worker-budgets",
        "--profile",
        "reels",
        "--set",
        "120",
    )

    assert fixed.returncode == 0, fixed.stdout + fixed.stderr
    repaired = yaml.safe_load(reels_cfg.read_text(encoding="utf-8"))
    untouched = yaml.safe_load(mapasocial_cfg.read_text(encoding="utf-8"))
    assert repaired["agent"]["max_turns"] == 120
    assert repaired["delegation"]["max_iterations"] == 120
    assert repaired["goals"]["max_turns"] == 120
    assert untouched["agent"]["max_turns"] == 0
    assert untouched["delegation"]["max_iterations"] == 0


def test_config_repair_worker_budgets_fails_closed_on_malformed_profile_config(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    broken_cfg = _write_raw_profile_config(home, "broken", "agent: [unterminated") / "config.yaml"
    before = broken_cfg.read_text(encoding="utf-8")

    res = _run_config(home, "repair", "worker-budgets", "--all-profiles")

    combined = res.stdout + res.stderr
    assert res.returncode != 0
    assert "broken" in combined
    assert "invalid" in combined.lower() or "unreadable" in combined.lower()
    assert "unterminated" not in combined
    assert broken_cfg.read_text(encoding="utf-8") == before
