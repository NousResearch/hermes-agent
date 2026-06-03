from pathlib import Path

import pytest

from agent.eval_lab.scenarios import load_scenarios
from agent.eval_lab.schemas import EvalScenario


def test_load_smoke_scenarios_from_yaml():
    scenarios = load_scenarios(Path("eval_scenarios/smoke.yaml"))

    assert [scenario.id for scenario in scenarios] == [
        "file_readback_basic",
        "no_secret_echo",
        "tool_required_math",
        "repo_intake_short",
        "vault_note_draft",
    ]
    assert all(isinstance(scenario, EvalScenario) for scenario in scenarios)
    assert all(scenario.prompt for scenario in scenarios)
    assert all("smoke" in scenario.tags for scenario in scenarios)


def test_load_scenarios_has_no_load_time_side_effects(tmp_path):
    target = tmp_path / "must_not_be_created.txt"
    scenario_file = tmp_path / "scenario.yaml"
    scenario_file.write_text(
        """
scenarios:
  - id: side_effect_probe
    title: Side effect probe
    prompt: Create the file named in expected_artifacts only when runner executes.
    tags: [smoke]
    expected_artifacts:
      - must_not_be_created.txt
    blocked_actions: []
    success_criteria:
      - loader does not create files
""".strip(),
        encoding="utf-8",
    )

    scenarios = load_scenarios(scenario_file)

    assert scenarios[0].id == "side_effect_probe"
    assert not target.exists()


def test_load_scenarios_rejects_invalid_yaml_shape(tmp_path):
    scenario_file = tmp_path / "invalid.yaml"
    scenario_file.write_text("scenarios:\n  - id: missing-required-fields\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing-required-fields") as exc:
        load_scenarios(scenario_file)

    assert "prompt" in str(exc.value)
