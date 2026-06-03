from pathlib import Path

from agent.eval_lab.scenarios import load_scenarios


BENCHMARK_FILES = [
    "hermes_core_smoke.yaml",
    "repo_intake.yaml",
    "vault_first_workflow.yaml",
    "memory_boundary.yaml",
]


def test_benchmark_scenario_packs_load_without_side_effects():
    root = Path("eval_scenarios")
    for name in BENCHMARK_FILES:
        scenarios = load_scenarios(root / name)
        assert scenarios, name
        assert all(scenario.id for scenario in scenarios)
        assert all(scenario.success_criteria for scenario in scenarios)
