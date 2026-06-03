"""YAML scenario loading for the Hermes eval lab."""

from __future__ import annotations

from pathlib import Path

import yaml

from agent.eval_lab.schemas import EvalScenario


def load_scenarios(path: str | Path) -> list[EvalScenario]:
    """Load and validate eval scenarios from a YAML file.

    The loader is intentionally read-only: it parses YAML and constructs
    dataclasses, but never creates expected artifacts or performs runner work.
    """
    scenario_path = Path(path)
    try:
        raw = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Scenario file not found: {scenario_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid scenario YAML {scenario_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Scenario file {scenario_path} must contain a mapping")

    items = raw.get("scenarios")
    if not isinstance(items, list):
        raise ValueError(f"Scenario file {scenario_path} must contain scenarios: list")

    scenarios: list[EvalScenario] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        scenario_id = item.get("id", f"index-{index}") if isinstance(item, dict) else f"index-{index}"
        try:
            scenario = EvalScenario.from_dict(item)
        except ValueError as exc:
            raise ValueError(f"Invalid scenario {scenario_id}: {exc}") from exc
        if scenario.id in seen_ids:
            raise ValueError(f"Duplicate scenario id: {scenario.id}")
        seen_ids.add(scenario.id)
        scenarios.append(scenario)

    return scenarios
