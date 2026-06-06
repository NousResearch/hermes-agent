from pathlib import Path

import pytest
import yaml

from hermes_cli.project_watchdog_bootstrap import (
    LIVE_TEMPLATE_DIR,
    ProjectWatchdogValidationError,
    backfill_project_watchdogs,
    validate_project_watchdog_index,
)


def _write_index(path: Path, projects: list[dict]) -> None:
    path.write_text(yaml.safe_dump({"projects": projects}, sort_keys=True), encoding="utf-8")


def _active_project(tag: str = "alpha") -> dict:
    return {
        "project_tag": tag,
        "status": "active",
        "board_slug": tag,
        "coordinator_profile": f"coordinator_{tag}",
        "matrix": {
            "room_alias": f"#h-yunuen-montelongo--p-coordinator-{tag}--b-{tag}:matrix.example.org",
            "space_alias": f"#space-{tag}:matrix.example.org",
            "encryption": "encrypted",
        },
        "github": {"repository": f"yunuenmf/{tag}", "default_branch": "main"},
        "repository": {"path": f"repos/{tag}", "worktree_root": f"repos/{tag}/.worktrees"},
        "worker_profiles": [f"developer_{tag}", f"researcher_{tag}"],
        "metadata": {
            "dispatch_owner": f"coordinator_{tag}",
            "watchdog_owner": f"coordinator_{tag}",
            "pa_audit_owner": "pa_yunuen",
        },
    }


def test_backfill_project_watchdogs_emits_required_artifacts_for_every_active_project(tmp_path: Path):
    index_path = tmp_path / "project_index.yaml"
    _write_index(
        index_path,
        [
            _active_project("alpha"),
            {"project_tag": "retired", "status": "archived"},
        ],
    )

    result = backfill_project_watchdogs(index_path=index_path, output_root=tmp_path / "coordination")

    assert [project["project_tag"] for project in result.projects] == ["alpha"]
    project_root = tmp_path / "coordination" / "projects" / "alpha" / "watchdog"
    expected_paths = {
        "policy": project_root / "policy.yaml",
        "runtime_config": project_root / "runtime_config.yaml",
        "state_dir": project_root / "state",
        "latest_report_dir": project_root / "latest-report",
        "systemd_recipe": project_root / "recipes" / "project-watchdog.service",
        "cron_recipe": project_root / "recipes" / "project-watchdog.crontab",
        "hermes_cron_recipe": project_root / "recipes" / "hermes-cron.yaml",
        "validation_gates": project_root / "validation_gates.yaml",
    }
    for path in expected_paths.values():
        assert path.exists(), path

    policy = yaml.safe_load(expected_paths["policy"].read_text(encoding="utf-8"))
    runtime = yaml.safe_load(expected_paths["runtime_config"].read_text(encoding="utf-8"))
    gates = yaml.safe_load(expected_paths["validation_gates"].read_text(encoding="utf-8"))

    assert policy["project_tag"] == "alpha"
    assert policy["owners"] == {
        "dispatch_owner": "coordinator_alpha",
        "watchdog_owner": "coordinator_alpha",
        "pa_audit_owner": "pa_yunuen",
    }
    assert policy["owners"]["dispatch_owner"] == policy["coordinator_profile"]
    assert runtime["state_dir"] == "projects/alpha/watchdog/state"
    assert runtime["latest_report_dir"] == "projects/alpha/watchdog/latest-report"
    assert runtime["scheduler"]["owner_profile"] == "coordinator_alpha"
    assert {gate["name"] for gate in gates["gates"]} >= {
        "kanban_board_bound",
        "matrix_route_bound",
        "github_repository_bound",
        "exactly_one_dispatch_owner",
        "coordinator_owned_scheduler",
    }

    validate_project_watchdog_index(index_path=index_path, output_root=tmp_path / "coordination")


def test_validation_rejects_missing_or_ambiguous_ownership(tmp_path: Path):
    index_path = tmp_path / "project_index.yaml"
    project = _active_project("alpha")
    project["metadata"]["dispatch_owner"] = ["coordinator_alpha", "pa_yunuen"]
    _write_index(index_path, [project])

    with pytest.raises(ProjectWatchdogValidationError, match="exactly one dispatch_owner"):
        validate_project_watchdog_index(index_path=index_path, output_root=tmp_path / "coordination")


def test_reusable_templates_are_sanitized():
    forbidden_fragments = [
        "@yunuen.montelongo:matrix.org",
        "!",
        "#h-yunuen",
        "/home/engs2272",
        "sk-",
        "ghp_",
        "github_pat_",
    ]

    for template_path in LIVE_TEMPLATE_DIR.glob("*.yaml"):
        text = template_path.read_text(encoding="utf-8")
        assert "{{" in text and "}}" in text, template_path
        assert not any(fragment in text for fragment in forbidden_fragments), template_path
