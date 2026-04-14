from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    module_path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hermes_routing_governance_snapshot_script_json_output(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  provider: openai-codex
  default: gpt-5.4
fallback_providers:
  - provider: google
    model: gemini-2.0-flash
""".strip()
    )
    module = _load_script("hermes_routing_governance_snapshot")
    monkeypatch.setattr(
        module,
        "build_routing_governance_snapshot",
        lambda **_: {
            "policy": {"policy_id": "session_topology_routing_policy_v1_2026-04-04"},
            "available_routes": [{"provider": "openai-codex", "model": "gpt-5.4"}],
            "parity_validation": {"tasks_with_selected_route": 9, "tasks_without_any_policy_candidate": []},
        },
    )

    exit_code = module.main(["--config", str(config_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["policy"]["policy_id"] == "session_topology_routing_policy_v1_2026-04-04"
    assert payload["parity_validation"]["tasks_with_selected_route"] == 9
