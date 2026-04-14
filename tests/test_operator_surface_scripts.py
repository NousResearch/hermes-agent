from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_script(name: str):
    module_path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_operator_mission_control_snapshot_script_json_output(monkeypatch, capsys):
    module = _load_script("operator_mission_control_snapshot")
    monkeypatch.setattr(
        module,
        "build_operator_mission_surface",
        lambda: {"schema": "hermes.operator_mission_surface.v1", "headline": "Gateway healthy"},
    )

    exit_code = module.main(["--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["headline"] == "Gateway healthy"


def test_operator_triage_console_snapshot_script_text_output(monkeypatch, capsys):
    module = _load_script("operator_triage_console_snapshot")
    monkeypatch.setattr(
        module,
        "build_operator_triage_surface",
        lambda: {
            "schema": "hermes.operator_triage_surface.v1",
            "severity": "warning",
            "summary": "Operator action required",
            "issues": [{"summary": "Gateway restart has been requested", "severity": "warning"}],
        },
    )

    exit_code = module.main([])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Severity: warning" in output
    assert "Gateway restart has been requested" in output
