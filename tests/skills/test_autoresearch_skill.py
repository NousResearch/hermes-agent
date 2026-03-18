from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "research"
    / "autoresearch"
    / "scripts"
    / "autoresearch.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("autoresearch_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_action_dispatches_research_cycle():
    mod = load_module()

    with patch.object(
        mod.runtime,
        "research_cycle",
        return_value={"run_id": "run-1", "status": "completed"},
    ) as mock_run:
        payload = mod.run_action(
            "research_cycle",
            family_id="params",
            population=3,
            survivors=2,
            seed=11,
            model="test-model",
        )

    assert payload["success"] is True
    assert payload["run_id"] == "run-1"
    mock_run.assert_called_once_with(
        project_root=None,
        family_id="params",
        population=3,
        survivors=2,
        seed=11,
        model="test-model",
        task_id=None,
    )


def test_run_action_requires_family_id_for_research_cycle():
    mod = load_module()

    payload = mod.run_action("research_cycle")

    assert payload["success"] is False
    assert "family_id" in payload["error"]


def test_main_prints_json_payload(capsys):
    mod = load_module()

    with patch.object(mod, "run_action", return_value={"success": True, "count": 1}) as mock_run:
        exit_code = mod.main(["list-projects", "--project-root", "/tmp/demo"])

    stdout = capsys.readouterr().out
    rendered = json.loads(stdout)

    assert exit_code == 0
    assert rendered["count"] == 1
    mock_run.assert_called_once_with("list_projects", project_root="/tmp/demo")
