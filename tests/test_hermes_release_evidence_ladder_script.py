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


def test_hermes_release_evidence_ladder_script_json_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    module = _load_script("hermes_release_evidence_ladder")

    monkeypatch.setattr(
        module,
        "build_release_evidence_bundle",
        lambda **_: {"release_id": "rel_wave3_cli", "activation_mode": "shadow"},
    )
    monkeypatch.setattr(
        module,
        "evaluate_release_evidence_ladder",
        lambda **_: {"verdict": "pass", "gate_results": [{"gate_id": "schema", "status": "pass"}]},
    )

    exit_code = module.main(["--release-id", "rel_wave3_cli", "--activation-mode", "shadow", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["bundle"]["release_id"] == "rel_wave3_cli"
    assert payload["decision"]["verdict"] == "pass"
